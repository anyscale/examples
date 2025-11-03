import base64
import concurrent.futures
import os
import ray
import requests
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO
import pyarrow.fs as pafs
from requests.adapters import HTTPAdapter
import urllib3
import logging
import warnings

logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# Disable SSL warnings since we're disabling verification for misconfigured image hosts
# Suppress urllib3 connection pool warnings (timeout, connection errors, etc.)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# SCALABILITY CONFIGURATION FOR 2B+ IMAGES
# ============================================================================
# num_images = 100
num_model_replicas = 32
tensor_parallelism = 1
max_concurrent_downloads = 10  # Reduced to minimize memory spikes (was 10)

from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

output_path = f"/mnt/shared_storage/process_images_output/{timestamp}"


def create_session():
    """
    Create a requests session for image downloads without automatic retries.
    """
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=50,
        pool_maxsize=50,
        # Keep connections alive longer
        pool_block=False,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_image(url, session=None):
    """
    Fetch image with validation and error handling.

    If the download or validation fails, the image is marked as invalid without retrying.
    """
    # Validate URL format first
    if not url or not isinstance(url, str):
        return None, False, "Invalid URL: empty or not a string"

    # Parse URL to properly validate its structure
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)

        # Check if URL has a valid scheme (http or https)
        if parsed.scheme not in ("http", "https"):
            return (
                None,
                False,
                f"Invalid URL",
            )

        # Check if URL has a valid hostname/netloc
        # netloc is the network location (domain/IP), e.g., "example.com" or "192.168.1.1"
        if not parsed.netloc or len(parsed.netloc.strip()) < 3:
            return None, False, f"Invalid URL: missing or invalid hostname: {url[:100]}"

        # Check if netloc contains at least a dot (for domain) or is localhost/IP
        # This catches URLs like "https://albumart/image.jpg" which have no valid domain
        if "." not in parsed.netloc and parsed.netloc not in ("localhost", "127.0.0.1"):
            # Allow if it looks like an IPv6 address (contains colons)
            if ":" not in parsed.netloc:
                return (
                    None,
                    False,
                    f"Invalid URL: malformed hostname (missing domain): {url[:100]}",
                )

    except Exception as e:
        return None, False, f"Invalid URL: failed to parse: {str(e)[:100]}"

    # Create session if not provided (will be reused within batch)
    if session is None:
        session = create_session()

    try:
        response = session.get(
            url,
            timeout=(10, 20),  # (connect_timeout=30s, read_timeout=60s)
            verify=False,  # Disable SSL verification for broken certs
            allow_redirects=True,  # Follow redirects
            stream=False,  # Download entire response
        )
    except Exception as e:
        return None, False, f"Error: {str(e)[:10]}"

    if response.status_code == 200:
        ctype = response.headers.get("Content-Type", "")
        if ctype.startswith("image"):
            image_bytes = response.content
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=UserWarning)
                    warnings.filterwarnings(
                        "error", category=Image.DecompressionBombWarning
                    )
                    # First verify the image format
                    img = Image.open(BytesIO(image_bytes))
                    img.verify()  # This checks if file is broken
                    # After verify(), we need to reopen to actually load the image
                    img = Image.open(BytesIO(image_bytes))
                    img.load()  # Force full image loading to detect truncation
                    img.close()
            except (OSError, IOError) as e:
                # Catch truncated images and other IO errors
                error_msg = str(e)[:100]
                if "truncated" in error_msg.lower():
                    return None, False, f"Truncated image: {error_msg}"
                return None, False, f"Image IO error: {error_msg}"
            except Exception as e:
                return None, False, f"Image validation error: {str(e)[:100]}"
            return image_bytes, True, None
        return None, False, f"Content-Type is not an image: {ctype}"

    return None, False, f"Status code: {response.status_code}"


def is_jpeg_format(row):
    """Memory-efficient JPEG format check without keeping Image object in memory."""
    try:
        image_data = row.get("image_base64")
        if image_data is None:
            return False
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        with Image.open(BytesIO(image_data)) as img:
            return img.format == "JPEG"
    except:
        return False


def resize_image(row):
    """Resize image to 128x128 pixels and standardize RGB values."""
    try:
        image_data = row.get("image_base64")
        if image_data is None:
            return row

        # Decode base64 string to bytes
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        # Open image, convert to RGB, resize, and save back
        with Image.open(BytesIO(image_bytes)) as img:
            # Convert to RGB mode to ensure consistent 3-channel format
            # This handles CMYK, grayscale, RGBA, etc.
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to 128x128 using high-quality Lanczos resampling
            resized_img = img.resize((128, 128), Image.Resampling.LANCZOS)

            # Save resized image to bytes
            output_buffer = BytesIO()
            resized_img.save(output_buffer, format="JPEG", quality=95)
            resized_bytes = output_buffer.getvalue()

        # Encode back to base64 string
        row["image_base64"] = base64.b64encode(resized_bytes).decode("ascii")
        return row
    except Exception as e:
        # If resize fails, keep original image
        return row


def fetch_images_batch_threaded(batch):
    """Fetch images in parallel with increased concurrency for network throughput."""
    # Disable SSL warnings in each Ray worker process
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Create a single shared session across threads in this batch
    # Note: requests.Session is thread-safe for reading
    session = create_session()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrent_downloads
    ) as executor:
        # Pass session to each fetch_image call
        results = list(
            executor.map(lambda url: fetch_image(url, session), batch["url"])
        )

    batch["image_base64"] = [
        (
            base64.b64encode(result[0]).decode("ascii")
            if (result[0] is not None and result[1])
            else None
        )
        for result in results
    ]
    batch["success"] = [result[1] for result in results]
    return batch


vision_processor_config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=tensor_parallelism,
        pipeline_parallel_size=1,
        max_model_len=32768,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
    ),
    # Override Ray's runtime env to include the Hugging Face token. Ray Data uses Ray under the hood to orchestrate the inference pipeline.
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",
            VLLM_DISABLE_COMPILE_CACHE="1",
        ),
    ),
    batch_size=8,  # Reduced from 16 to lower memory usage
    max_concurrent_batches=16,  # Increased to saturate vLLM engine (8 * 16 = 128)
    accelerator_type="A10G",
    concurrency=num_model_replicas,
    has_image=True,
)


def vision_preprocess(row: dict) -> dict:
    # Keep image data as base64 string for Arrow serialization
    # The vLLM engine will handle the conversion internally
    image_data = row["image_base64"]

    image_data = f"data:image;base64,{image_data}"
    return dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_data,
                    },
                ],
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=150,
            detokenize=False,
        ),
    )


def vision_postprocess(row: dict) -> dict:
    row.pop("image_base64")
    return row


vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)


# Initialize Ray with S3 spilling configuration
# This ensures Ray can access S3 for object spilling on all workers
if not ray.is_initialized():
    ray.init()

ray.data.DataContext.get_current().retried_io_errors.extend(
    [
        # Network connectivity errors
        "Temporary failure in name resolution",
        "Name or service not known",
        "Max retries exceeded with url",
        "Failed to establish a new connection",
        "Connection refused",
        "Connection timed out",
        "Read timed out",
        "ConnectTimeoutError",
        "connect timeout",
        "HTTPSConnectionPool",
        "Remote end closed connection",
        "Connection broken",
        # SSL/TLS errors
        "SSLError",
        "SSL: CERTIFICATE_VERIFY_FAILED",
        "hostname mismatch",
        "certificate verify failed",
        # Rate limiting
        "429 Client Error: Too Many Requests",
        "We had to rate limit you",
    ]
)
num_cpu = 512
tasks_per_cpu = 1
concurrency = num_cpu * tasks_per_cpu
ctx = ray.data.DataContext.get_current()
target_block_size_mb = 128
ctx.target_max_block_size = target_block_size_mb * 1024 * 1024
ctx.use_push_based_shuffle = False


# Data pipeline with scalability optimizations
dataset = (
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        columns=["url"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=concurrency,
        num_cpus=2,
        memory=int(4 * 1024**3),
    )
    .map_batches(
        fetch_images_batch_threaded,
        batch_size=50,
        memory=int(2 * 1024**3),
        num_cpus=2,
    )  # removed partition to reduce memory usage and redundency
    .filter(lambda row: row["success"])
    .filter(is_jpeg_format)
    .map(resize_image)
    .drop_columns(["success"])
)  # Drop success column early to reduce memory


# Apply vision processing with scaled replicas
# Note: image_base64 column is dropped in vision_postprocess to avoid Arrow serialization issues
dataset = vision_processor(dataset)

# Write with optimizations for throughput and fault tolerance
dataset.write_parquet(
    output_path,
    max_rows_per_file=100000,  # ~100K rows per file for manageable file sizes
)

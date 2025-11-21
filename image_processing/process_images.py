import os
import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import time
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SCALABILITY CONFIGURATION FOR 2B+ IMAGES
# ============================================================================
num_images_to_process = 10**7
# Target 64 concurrent L4 replicas on g6.xlarge workers.

num_gpu = 64
num_cpu = 384 * 2
tensor_parallelism = 1

# Download configuration - optimized for thread pool
download_threads_per_batch = 20  # Number of threads per batch (was semaphore)
download_timeout = 10  # Increased from 5s to be more tolerant
max_retries = 1  # Number of retries for transient failures
batch_size = 50  # Reduced from 100 for better memory management

# Create a session for connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=0,  # We handle retries manually
)
session.mount("http://", adapter)
session.mount("https://", adapter)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_path = f"/mnt/shared_storage/process_images_output/{timestamp}"


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and properly formatted."""
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")


def download_single_image_with_retry(url: str) -> Dict[str, Any]:
    """
    Download a single image with retry logic and detailed error tracking.

    Returns:
        Dict with keys:
        - 'content': Image bytes or None
        - 'status': Success/failure status
        - 'url': Original URL for tracking
    """
    if not is_valid_url(url):
        return {"content": None, "status": "invalid_url", "url": url}

    last_error = None
    for attempt in range(max_retries):
        try:
            # Use the global session for connection pooling
            response = session.get(url, timeout=download_timeout, stream=True)

            if response.status_code == 200:
                # Read content in chunks to handle large images efficiently
                content = response.content
                return {"content": content, "status": "success", "url": url}
            elif response.status_code == 404:
                # Don't retry 404s - they're permanent failures
                return {
                    "content": None,
                    "status": f"http_{response.status_code}",
                    "url": url,
                }
            else:
                last_error = f"http_{response.status_code}"

        except requests.Timeout:
            last_error = "timeout"
            if attempt < max_retries - 1:
                # Exponential backoff for retries
                time.sleep(2**attempt)
                continue

        except requests.ConnectionError as e:
            last_error = "connection_error"
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue

        except Exception as e:
            last_error = f"error_{type(e).__name__}"
            logger.debug(f"Unexpected error downloading {url}: {e}")
            break

    # All retries exhausted
    return {"content": None, "status": last_error, "url": url}


def image_download(batch: Dict[str, List]) -> Dict[str, List]:
    """
    Download images in batch using thread pool for parallelism.

    This replaces the complex async implementation with a simpler,
    more robust thread-based approach that's equally performant for I/O.
    """
    urls = batch["url"]

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=download_threads_per_batch) as executor:
        # Map download function over all URLs
        results = list(executor.map(download_single_image_with_retry, urls))

    # Extract content and status for downstream processing
    batch["bytes"] = [r["content"] for r in results]
    batch["download_status"] = [r["status"] for r in results]

    # Log statistics for monitoring
    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)
    failure_types = {}
    for r in results:
        if r["status"] != "success":
            failure_types[r["status"]] = failure_types.get(r["status"], 0) + 1

    if total_count > 0:
        success_rate = (success_count / total_count) * 100
        logger.info(
            f"Batch download: {success_count}/{total_count} succeeded ({success_rate:.1f}%)"
        )
        if failure_types:
            logger.info(f"Failure breakdown: {failure_types}")

    return batch


def process_single_image(image_bytes: Optional[bytes]) -> Optional[bytes]:
    """
    Process a single image: validate, convert to RGB, and resize.

    Args:
        image_bytes: Raw image bytes or None

    Returns:
        Processed image bytes as JPEG or None if processing fails
    """
    if image_bytes is None:
        return None

    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()  # Force load to detect corrupt images early

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to target dimensions
        img = img.resize((128, 128), Image.Resampling.LANCZOS)

        # Save as JPEG with high quality
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()

    except Exception as e:
        logger.debug(f"Failed to process image: {type(e).__name__}")
        return None


def process_image_bytes(batch):
    image_bytes_list = batch["bytes"]

    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(process_single_image, image_bytes_list))

    batch["bytes"] = results
    return batch


vision_processor_config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=tensor_parallelism,
        pipeline_parallel_size=1,
        max_model_len=32768,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
        distributed_executor_backend="mp",
    ),
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",
            VLLM_DISABLE_COMPILE_CACHE="1",
        ),
    ),
    batch_size=128,
    max_concurrent_batches=128,
    accelerator_type="A10G",
    concurrency=num_gpu,
    has_image=True,
)


def vision_preprocess(row):
    image_bytes = row["bytes"]
    return dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(BytesIO(image_bytes)),
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


def vision_postprocess(row):
    row.pop("bytes")
    return row


vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)

tasks_per_cpu = 1
concurrency = num_cpu * tasks_per_cpu

ctx = ray.data.DataContext.get_current()
target_block_size_mb = 128
ctx.target_max_block_size = target_block_size_mb * 1024 * 1024
ctx.use_push_based_shuffle = False

# The data processing pipeline includes the following steps:
# 1. Read the 2B images dataset with url column
# 2. Download images using thread pool with retry logic
# 3. Validate and resize images to 128x128
# 4. Process images with the vision model
# 5. Write results to the output path
dataset = (
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        columns=["url"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=concurrency,
        num_cpus=2,
        memory=int(4 * 1024**3),
    )  # Read dataset with memory allocation to avoid OOM errors
    .limit(num_images_to_process)
    .repartition(num_cpu)
    .map_batches(
        image_download,
        batch_size=batch_size,  # Use optimized batch size
        num_cpus=1,
        concurrency=num_cpu,
    )
    .drop_columns(["url"])  # Drop URL after download to save memory
    .map_batches(
        process_image_bytes,
        batch_size=batch_size,  # Consistent batch size
        num_cpus=1,
        concurrency=num_cpu,
    )
    .filter(
        lambda row: row["bytes"] is not None
    )  # Filter out failed downloads/processing
    .drop_columns(["download_status"])  # Drop status column after filtering
)

dataset = vision_processor(dataset)

dataset.write_parquet(
    output_path,
    max_rows_per_file=100000,
)

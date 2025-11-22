import os
import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List


# ============================================================================
# SCALABILITY CONFIGURATION FOR 2B+ IMAGES
# ============================================================================
num_images_to_process = 10**6
# Target 64 concurrent L4 replicas on g6.xlarge workers.

num_gpus = 64

# Download configuration - optimized for thread pool
download_threads_per_batch = 50  # Number of threads per batch
download_timeout = 5  # Timeout for each download request

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_path = f"/mnt/shared_storage/process_images_output/{timestamp}"


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and properly formatted."""
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")


def download_single_image(url: str, session: requests.Session) -> Dict[str, Any]:
    """Download a single image."""
    if not is_valid_url(url):
        return {"content": None, "status": "invalid_url", "url": url}

    try:
        # Use the provided session for connection pooling
        response = session.get(url, timeout=download_timeout, stream=True)

        if response.status_code == 200:
            # Read content
            content = response.content
            return {"content": content, "status": "success", "url": url}
        else:
            # Return HTTP status code for non-200 responses
            return {
                "content": None,
                "status": f"http_{response.status_code}",
                "url": url,
            }

    except requests.Timeout:
        return {"content": None, "status": "timeout", "url": url}

    except requests.ConnectionError:
        return {"content": None, "status": "connection_error", "url": url}

    except Exception as e:
        return {"content": None, "status": f"error_{type(e).__name__}", "url": url}


def image_download(batch: Dict[str, List]) -> Dict[str, List]:
    """Download a batch of images using a thread pool for parallelism."""
    urls = batch["url"]

    # Create a session for connection pooling within this batch
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100,
        pool_maxsize=100,
        max_retries=0,  # No automatic retries
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=download_threads_per_batch) as executor:
        # Map download function over all URLs with the session
        results = list(executor.map(lambda url: download_single_image(url, session), urls))

    # Extract content and status for downstream processing
    batch["bytes"] = [r["content"] for r in results]

    return batch


def process_single_image(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single image: validate, convert to RGB, and resize.

    Args:
        image_bytes: Raw image bytes or None

    Returns:
        Processed image bytes as JPEG or None if processing fails
    """
    def process_image_bytes(image_bytes: Optional[bytes]) -> Optional[bytes]:
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
            return None
    row["bytes"] = process_image_bytes(row["bytes"])
    return row


vision_processor_config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.95,
    ),
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",
            VLLM_DISABLE_COMPILE_CACHE="1",
        ),
    ),
    batch_size=64,
    max_concurrent_batches=8,
    accelerator_type="A10G",
    concurrency=num_gpus,
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

dataset = (
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        columns=["url"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=10,
        ray_remote_args={"memory": int(4 * 10**9)},
    )  # Read dataset with memory allocation to avoid OOM errors
    .limit(num_images_to_process)
    .repartition(target_num_rows_per_block=1000)
    .map_batches(
        image_download,
        batch_size=100,  # Use optimized batch size
        memory=(10**9),
    )
    .drop_columns(["url"])  # Drop URL after download to save memory
    .map(process_single_image)
    .filter(lambda row: row["bytes"] is not None)  # Filter out failed downloads/processing
)

dataset = vision_processor(dataset)

dataset.write_parquet(output_path)

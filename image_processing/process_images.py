"""
Large-Scale Image Processing Pipeline with Vision Language Models

This script demonstrates best practices for processing billions of images using:
- Ray Data for distributed data processing
- Async I/O for efficient image downloading
- vLLM for high-performance vision model inference
- Automatic GPU autoscaling for cost optimization

Pipeline stages:
1. Read URLs from HuggingFace dataset (2B+ rows)
2. Download images asynchronously with fault tolerance
3. Preprocess and standardize images (resize, convert to RGB)
4. Run vision model inference with vLLM
5. Write results to Parquet files

Best Practices Demonstrated:
- Async/await for I/O-bound operations (network downloads)
- ThreadPoolExecutor for CPU-bound operations (image processing)
- Batch processing for optimal throughput
- Resource allocation tuning (CPU, memory, GPU)
- Graceful error handling and filtering
- Timestamped outputs for reproducibility
"""

import os
import asyncio
import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


# ============================================================================
# SCALABILITY CONFIGURATION FOR 2B+ IMAGES
# ============================================================================
# These parameters control the scale and performance of the pipeline

# GPU Autoscaling Configuration
# Best Practice: Use autoscaling ranges instead of fixed counts for cost efficiency
max_gpu_num = 96  # Maximum GPU replicas (scales up during high load)
min_gpu_num = 32  # Minimum GPU replicas (scales down to save costs)
tensor_parallelism = 1  # Number of GPUs per model replica (1 = no tensor parallelism)

# Network Download Configuration
# Best Practice: High concurrency for I/O-bound operations
download_concurrency = 1000  # Number of simultaneous downloads (tune based on network capacity)
download_timeout = 5  # Timeout per image download in seconds (fail fast for broken URLs)

# Output Configuration
# Best Practice: Use timestamps to avoid overwriting previous runs
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_path = f"/mnt/shared_storage/process_images_output/{timestamp}"


# ============================================================================
# STAGE 1: IMAGE DOWNLOADING (ASYNC I/O)
# ============================================================================
# Best Practice: Use async/await for I/O-bound operations to maximize throughput


def is_valid_url(url):
    """
    Validate URL format before attempting download.
    
    Best Practice: Fail fast - validate inputs early to avoid wasted work.
    This prevents network calls for obviously invalid URLs.
    """
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")


async def download_single_image(session, url, semaphore):
    """
    Download a single image asynchronously with timeout and error handling.
    
    Args:
        session: Shared aiohttp session for connection pooling
        url: Image URL to download
        semaphore: Limits concurrent downloads to avoid overwhelming the network
    
    Best Practices:
    - Use semaphore to control concurrency and prevent resource exhaustion
    - Reuse session for connection pooling (reduces overhead)
    - Set timeouts to fail fast on slow/broken connections
    - Return None on errors (graceful degradation)
    """
    async with semaphore:  # Acquire semaphore slot (blocks if at limit)
        if not is_valid_url(url):
            return None

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=download_timeout)
            ) as response:
                if response.status == 200:
                    content = await response.read()
                    return content
                return None
        except Exception:
            # Best Practice: Silent failures for individual images in batch processing
            # Logs would be too verbose at billion-scale; filter out None results later
            return None


async def download_images_async(urls):
    """
    Download multiple images concurrently using asyncio.
    
    Best Practices Demonstrated:
    1. Connection pooling with TCPConnector for efficiency
    2. Per-host limits to be respectful to servers
    3. DNS caching to reduce lookup overhead
    4. Proper connection cleanup with context manager
    5. Exception handling with gather(return_exceptions=True)
    """
    semaphore = asyncio.Semaphore(download_concurrency)

    # Configure TCP connector for optimal performance
    connector = aiohttp.TCPConnector(
        limit=download_concurrency,  # Total connection pool size
        limit_per_host=100,  # Limit per domain (be respectful, avoid rate limiting)
        ttl_dns_cache=300,  # Cache DNS lookups for 5 minutes
        enable_cleanup_closed=True,  # Clean up closed connections automatically
    )

    # Set connection and total timeouts
    timeout_config = aiohttp.ClientTimeout(total=download_timeout, connect=3)

    # Best Practice: Use context manager for automatic resource cleanup
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout_config
    ) as session:
        # Create all download tasks
        tasks = [download_single_image(session, url, semaphore) for url in urls]
        # Execute all tasks concurrently, capturing exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to None for consistent error handling
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results


def image_download(batch):
    """
    Ray Data UDF for downloading image batches.
    
    This function is called by Ray Data's map_batches() operation.
    
    Best Practices:
    - Handle event loop creation for Ray worker processes
    - Process entire batches for efficiency (amortizes overhead)
    - Return batch dict with new/modified columns
    
    Args:
        batch: Dict with "url" column containing image URLs
    
    Returns:
        Dict with "bytes" column containing downloaded image bytes (or None)
    """
    urls = batch["url"]

    # Best Practice: Ray workers may not have an event loop, so create one
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Execute async download function from sync context
    results = loop.run_until_complete(download_images_async(urls))
    batch["bytes"] = results
    return batch


# ============================================================================
# STAGE 2: IMAGE PREPROCESSING (CPU-BOUND)
# ============================================================================
# Best Practice: Use ThreadPoolExecutor for CPU-bound operations within batches


def process_single_image(image_bytes):
    """
    Process and standardize a single image.
    
    Operations:
    1. Validate image can be loaded
    2. Convert to RGB color space (handles RGBA, grayscale, CMYK, etc.)
    3. Resize to 128x128 (standardize dimensions for model)
    4. Save as JPEG with high quality
    
    Best Practices:
    - Standardize format (RGB JPEG) for consistent model input
    - Use LANCZOS resampling for high-quality resizing
    - Return None on errors (graceful degradation)
    - Force img.load() to catch truncated images early
    """
    if image_bytes is None:
        return None

    try:
        # Open and validate image
        img = Image.open(BytesIO(image_bytes))
        img.load()  # Force loading to detect truncated/corrupted images

        # Standardize color space to RGB (3 channels)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to fixed dimensions (LANCZOS = highest quality)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)

        # Re-encode as JPEG with high quality
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()
    except Exception:
        # Failed to process (corrupted, unsupported format, etc.)
        return None


def process_image_bytes(batch):
    """
    Ray Data UDF for processing image batches in parallel.
    
    Best Practices:
    - Use ThreadPoolExecutor for CPU-bound image processing
    - Process entire batch in parallel (50 threads)
    - Thread pool auto-closes with context manager
    
    Note: Python's GIL is released during PIL operations, so threads are effective here.
    
    Args:
        batch: Dict with "bytes" column containing raw image bytes
    
    Returns:
        Dict with "bytes" column containing processed JPEG bytes (or None)
    """
    image_bytes_list = batch["bytes"]

    # Best Practice: Use thread pool for parallelizing CPU-bound work within a batch
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(process_single_image, image_bytes_list))

    batch["bytes"] = results
    return batch


# ============================================================================
# STAGE 3: VISION MODEL INFERENCE (GPU-ACCELERATED)
# ============================================================================
# Best Practice: Use vLLM for high-throughput, production-grade inference

vision_processor_config = vLLMEngineProcessorConfig(
    # Model Configuration
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",  # Vision-language model from HuggingFace
    
    # vLLM Engine Settings
    engine_kwargs=dict(
        tensor_parallel_size=tensor_parallelism,  # GPUs per replica (1 = no TP)
        pipeline_parallel_size=1,  # Pipeline parallelism (1 = disabled)
        max_model_len=32768,  # Maximum context length
        enable_chunked_prefill=True,  # Improves throughput for long contexts
        max_num_batched_tokens=2048,  # Maximum tokens per forward pass
    ),
    
    # Runtime Environment
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",  # Use vLLM v1 API (better performance)
            VLLM_DISABLE_COMPILE_CACHE="1",  # Disable compile cache (avoid disk issues)
        ),
    ),
    
    # Batching Configuration
    # Best Practice: Tune batch_size × max_concurrent_batches to saturate GPU
    batch_size=8,  # Images per batch (tune based on GPU memory)
    max_concurrent_batches=16,  # Concurrent batches per GPU (8 × 16 = 128 in-flight)
    
    # Infrastructure
    accelerator_type="A10G",  # GPU type to use
    concurrency=(min_gpu_num, max_gpu_num),  # Autoscaling range (32-96 GPUs)
    has_image=True,  # Enable vision model support
)


def vision_preprocess(row):
    """
    Prepare image for vision model inference.
    
    Converts processed image bytes into the format expected by vLLM.
    
    Best Practice: Keep preprocessing minimal - heavy work should be done
    in earlier pipeline stages to avoid duplicating work on autoscaling.
    """
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
            temperature=0.3,  # Low temperature for more deterministic outputs
            max_tokens=150,  # Maximum response length
            detokenize=False,  # Return token IDs (faster, use True for text)
        ),
    )


def vision_postprocess(row):
    """
    Clean up after inference.
    
    Best Practice: Remove large columns (like image bytes) after use
    to reduce memory footprint and output size.
    """
    row.pop("bytes")  # Remove image bytes (no longer needed)
    return row


# Build the vision processor with Ray Data integration
# Best Practice: Wrap vLLM with Ray Data for automatic scaling and fault tolerance
vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)

# ============================================================================
# PIPELINE ORCHESTRATION WITH RAY DATA
# ============================================================================
# Ray Data provides distributed, fault-tolerant data processing at scale

# Configure read parallelism
num_cpu = 512  # Number of CPUs for parallel parquet reading
tasks_per_cpu = 1  # Tasks per CPU (1 = one task per CPU)
concurrency = num_cpu * tasks_per_cpu

# Ray Data performance tuning
# Best Practice: Configure block size and shuffle for your workload
ctx = ray.data.DataContext.get_current()
target_block_size_mb = 128  # Target size per data block (affects parallelism)
ctx.target_max_block_size = target_block_size_mb * 1024 * 1024
ctx.use_push_based_shuffle = False  # Use pull-based shuffle (more stable for large-scale)

# Build the complete data pipeline
# Best Practice: Chain operations in a single pipeline for optimal execution planning
dataset = (
    # STAGE 0: Read dataset from HuggingFace
    # Best Practice: Only read columns you need, allocate memory to avoid OOM
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],  # Only read parquet files
        columns=["url"],  # Only load URL column (reduces memory)
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),  # HF authentication
        concurrency=concurrency,  # Parallel read tasks (512)
        num_cpus=2,  # CPUs per read task
        memory=int(4 * 1024**3),  # 4GB memory per task (prevents OOM)
    )
    # STAGE 1: Download images asynchronously
    # Best Practice: High concurrency for I/O-bound operations, low CPU allocation
    .map_batches(
        image_download,
        batch_size=50,  # Process 50 URLs per batch
        num_cpus=0.5,  # Low CPU (I/O-bound, not CPU-bound)
        concurrency=1024,  # High task concurrency (many parallel downloads)
    )
    # STAGE 2: Drop URL column (no longer needed, saves memory)
    .drop_columns(["url"])
    # STAGE 3: Preprocess images (resize, convert to RGB)
    # Best Practice: Higher CPU for CPU-bound operations
    .map_batches(
        process_image_bytes,
        batch_size=50,  # Process 50 images per batch
        num_cpus=1,  # More CPU for image processing
    )
    # STAGE 4: Filter out failed downloads/processing
    # Best Practice: Filter early to avoid wasting GPU resources on invalid data
    .filter(lambda row: row["bytes"] is not None)
)

# STAGE 5: Run vision model inference with autoscaling
# Best Practice: vLLM processor handles batching, GPU allocation, and autoscaling automatically
dataset = vision_processor(dataset)

# STAGE 6: Write results to Parquet files
# Best Practice: Partition output into manageable file sizes
dataset.write_parquet(
    output_path,
    max_rows_per_file=100000,  # 100K rows per file (balance between file count and size)
)
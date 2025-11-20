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
# num_images = 100
# Target 64 concurrent L4 replicas on g6.xlarge workers.

num_gpu = 16
num_cpu = 256  # Should be able to do around 4.5K rows / s
# num_gpu = 32
# num_cpu = 256
tensor_parallelism = 1
download_concurrency = 256
download_timeout = 5
MAX_IMAGES_TO_PROCESS = 1000000

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_path = f"/mnt/shared_storage/process_images_output/{timestamp}"


def is_valid_url(url):
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")


async def download_single_image(session, url, semaphore):
    async with semaphore:
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
            return None


async def download_images_async(urls):
    semaphore = asyncio.Semaphore(download_concurrency)

    connector = aiohttp.TCPConnector(
        limit=download_concurrency,
        limit_per_host=100,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    timeout_config = aiohttp.ClientTimeout(total=download_timeout, connect=3)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout_config
    ) as session:
        tasks = [download_single_image(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results


def image_download(batch):
    urls = batch["url"]

    # Use a dedicated event loop per batch to avoid interfering with any
    # existing asyncio loops that Ray or vLLM may be running in this process.
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(download_images_async(urls))
    finally:
        # Comprehensive cleanup to prevent resource leaks
        try:
            # Cancel all pending tasks in this specific loop
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Wait for all task cancellations to complete
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            # Shut down async generators
            loop.run_until_complete(loop.shutdown_asyncgens())

            # Shut down the default executor (thread pool)
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            # Best-effort cleanup; failures here should not impact the worker
            pass
        finally:
            # Close the loop without affecting the global event loop
            loop.close()

    batch["bytes"] = results
    return batch


def process_single_image(image_bytes):
    if image_bytes is None:
        return None

    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((128, 128), Image.Resampling.LANCZOS)

        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()
    except Exception:
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
    batch_size=8,
    max_concurrent_batches=16,
    accelerator_type="L40S",
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

# The data data processing include the following steps:
# 1. Download the 2B images dataset with url column
# 2. Download the images from the url in parallel async jobs
# 3. Check the image is valid and resize the images to 128x128
# 4. Process the images with the vision model
# 5. Write the images to the output path
dataset = (
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        columns=["url"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=concurrency,
        num_cpus=2,
        memory=int(4 * 1024**3),
    )  # Download the dataset with memory allocation to avoid OOM errors
    .limit(MAX_IMAGES_TO_PROCESS)
    .repartition(target_rows_per_block=1000)
    .map_batches(image_download, batch_size=50, num_cpus=0.5, concurrency=num_cpu * 2)
    .drop_columns(["url"])
    .map_batches(process_image_bytes, batch_size=50, num_cpus=0.5, concurrency=num_cpu * 2)
    .filter(lambda row: row["bytes"] is not None)
)


dataset = vision_processor(dataset)

dataset.write_parquet(
    output_path,
    max_rows_per_file=100000,
)
import concurrent.futures
import os
import ray
import requests
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO


num_images = 100
num_model_replicas = 1
tensor_parallelism = 1

output_path = os.path.join(os.environ["ANYSCALE_ARTIFACT_STORAGE"], "rkn/process_images_output")


def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            ctype = response.headers.get("Content-Type", "")
            if ctype.startswith("image"):
                image_bytes = response.content
                Image.open(BytesIO(image_bytes))  # Validate the image formatting.
                success = True
                error = None
            else:
                image_bytes = None
                success = False
                error = f"Content-Type is not an image: {ctype}"
        else:
            image_bytes = None
            success = False
            error = f"Status code: {response.status_code}"
    except Exception as e:
        image_bytes = None
        success = False
        error = str(e)

    return image_bytes, success, error


def convert_to_pil_image(row):
    row["pil_image"] = Image.open(BytesIO(row["image_bytes"]))
    return row


def fetch_images_batch_threaded(batch):
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(fetch_image, batch["url"]))
    batch["image_bytes"] = [result[0] for result in results]
    batch["success"] = [result[1] for result in results]
    batch["error"] = [result[2] for result in results]
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
    batch_size=16,
    accelerator_type="A10G",
    concurrency=num_model_replicas,
    has_image=True,
)


def vision_preprocess(row: dict) -> dict:
    return dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # Ray Data accepts PIL Image or image URL.
                        # "image": row["pil_image"],
                        "image": Image.open(BytesIO(row["image_bytes"]))
                        # "image": row["image_bytes"],
                    },
                ]
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=150,
            detokenize=False,
        ),
    )


def vision_postprocess(row: dict) -> dict:
    return row


vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)


ray.data.DataContext.get_current().retried_io_errors.extend(
    [
        "Temporary failure in name resolution",
        "Max retries exceeded with url",
        "Failed to establish a new connection",
        "HTTPSConnectionPool",
    ]
)


dataset = ray.data \
    .read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        ray_remote_args={"memory": 10*10**9},
    ) \
    .limit(num_images) \
    .repartition(target_num_rows_per_block=1000) \
    .map_batches(
        fetch_images_batch_threaded,
        batch_size=200,
    ) \
    .filter(lambda row: row["success"]) \
    .filter(lambda row: Image.open(BytesIO(row["image_bytes"])).format == "JPEG")

dataset = vision_processor(dataset)
dataset = dataset.drop_columns(["image_bytes"])
dataset.write_parquet(output_path)


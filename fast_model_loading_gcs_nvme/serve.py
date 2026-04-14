"""
Ray Serve deployment with fast model loading from GCS.

Uses ray.anyscale.safetensors to stream model weights from GCS directly,
downloading one shard at a time to local NVMe. This caps peak RAM at ~5 GB
(one shard) and lets multiple tensor-parallel workers share pages via mmap.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path

from ray.serve.llm import LLMConfig, build_openai_app

logger = logging.getLogger("ray.serve")

GCS_MODEL_URI = os.environ.get(
    "GCS_MODEL_URI",
    "gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Llama-70B",
)
GCS_REGION = os.environ.get("ANYSCALE_CLOUD_STORAGE_BUCKET_REGION", "us-west1")
LOCAL_MODEL_PATH = "/mnt/local_storage/model"
DOWNLOAD_MARKER = os.path.join(LOCAL_MODEL_PATH, ".download_complete")
LOCK_FILE = "/mnt/local_storage/.model_download_lock"


def cache_model_on_nvme():
    """Download model shards from GCS to local NVMe one at a time.

    Peak memory = 1 shard (~5 GB), not the full model. Saves to local NVMe so
    that vLLM can mmap the files — tensor-parallel workers share physical pages.
    """
    if os.path.exists(DOWNLOAD_MARKER):
        logger.info("Model already cached at %s", LOCAL_MODEL_PATH)
        return

    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        logger.info("Another replica is downloading, waiting...")
        while not os.path.exists(DOWNLOAD_MARKER):
            time.sleep(10)
        return

    try:
        from ray.anyscale.safetensors._common import get_http_downloader_for_uri
        from safetensors.torch import save_file

        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        # Copy config/tokenizer files from the GCS URI.
        # These are small (<10 MB), use google-cloud-storage directly.
        _copy_metadata_files()

        index_path = os.path.join(LOCAL_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))

        logger.info("Downloading %d shards from %s", len(shard_files), GCS_MODEL_URI)
        t0 = time.monotonic()

        for i, shard in enumerate(shard_files):
            uri = f"{GCS_MODEL_URI}/{shard}"
            d, url = get_http_downloader_for_uri(uri, region=GCS_REGION)
            sd, _ = d.restore_state_dict_from_http(url, device="cpu")
            save_file(sd, os.path.join(LOCAL_MODEL_PATH, shard))
            del sd
            gc.collect()
            logger.info("  [%d/%d] %s saved", i + 1, len(shard_files), shard)

        elapsed = time.monotonic() - t0
        total_bytes = sum(
            f.stat().st_size
            for f in Path(LOCAL_MODEL_PATH).glob("*.safetensors")
        )
        logger.info(
            "Download complete: %.1f GB in %.0fs (%.2f GB/s)",
            total_bytes / 1e9,
            elapsed,
            total_bytes / elapsed / 1e9,
        )

        Path(DOWNLOAD_MARKER).touch()
    except Exception:
        import subprocess
        subprocess.run(["rm", "-rf", LOCAL_MODEL_PATH], check=False)
        raise
    finally:
        try:
            os.unlink(LOCK_FILE)
        except FileNotFoundError:
            pass


def _copy_metadata_files():
    """Copy config.json, tokenizer files, and shard index from GCS."""
    from google.cloud import storage

    bucket_name = GCS_MODEL_URI.split("/")[2]
    prefix = "/".join(GCS_MODEL_URI.split("/")[3:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    for blob in blobs:
        name = blob.name[len(prefix):].lstrip("/")
        if not name or name.endswith(".safetensors"):
            continue
        dest = os.path.join(LOCAL_MODEL_PATH, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        logger.info("  copied %s (%d bytes)", name, blob.size or 0)


cache_model_on_nvme()

model_id = "my-model"

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id=model_id,
        model_source=LOCAL_MODEL_PATH,
    ),
    accelerator_type="A100",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=4,
        )
    ),
    engine_kwargs=dict(
        max_model_len=32768,
        tensor_parallel_size=8,
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

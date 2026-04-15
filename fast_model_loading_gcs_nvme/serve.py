"""Two-phase model loading: GCS → NVMe (cached) → GPU.

Phase 1: A Ray Serve callback downloads model files from GCS to local NVMe
          using Run:ai Model Streamer, with file locking for multi-process safety.
Phase 2: vLLM loads from NVMe using either runai_streamer (~4.3 GB/s) or
          the default HuggingFace/safetensors loader (~1.5 GB/s).
"""

import logging
import os
import time

from ray.llm._internal.common.callbacks.base import CallbackBase
from ray.serve.llm import LLMConfig, build_openai_app

logger = logging.getLogger(__name__)

NVME_MODEL_DIR = "/mnt/local_storage/model"


class NVMeCacheCallback(CallbackBase):
    """Download model from GCS to local NVMe before vLLM loads it.

    Uses ObjectStorageModel which handles file locking (fcntl.flock) and
    idempotency (.runai_complete sentinel). First process on a node downloads;
    subsequent processes skip. Each node has its own NVMe so no cross-node
    coordination is needed.
    """

    def on_before_download_model_files_distributed(self):
        from runai_model_streamer import ObjectStorageModel

        gcs_uri = os.environ["GCS_MODEL_URI"]
        logger.info(f"Downloading {gcs_uri} → {NVME_MODEL_DIR}")
        start = time.time()

        with ObjectStorageModel(model_path=gcs_uri, dst=NVME_MODEL_DIR) as model:
            model.pull_files(allow_pattern=["*.safetensors"])
            model.pull_files(ignore_pattern=["*.safetensors"])

        elapsed = time.time() - start
        logger.info(f"Model ready at {NVME_MODEL_DIR} ({elapsed:.1f}s)")


# Load format: "runai_streamer" (fast, ~4.3 GB/s) or "auto" (HF/safetensors, ~1.5 GB/s)
load_format = os.environ.get("LOAD_FORMAT", "runai_streamer")

engine_kwargs = dict(
    max_model_len=32768,
    tensor_parallel_size=8,
)
if load_format != "auto":
    engine_kwargs["load_format"] = load_format

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-model",
        model_source=NVME_MODEL_DIR,
    ),
    accelerator_type="A100",
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=4),
    ),
    callback_config=dict(callback_class=NVMeCacheCallback),
    engine_kwargs=engine_kwargs,
)

app = build_openai_app({"llm_configs": [llm_config]})

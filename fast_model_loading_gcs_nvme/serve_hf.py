"""Two-phase model loading with HuggingFace Transformers: GCS → NVMe (cached) → GPU.

Phase 1: Download model files from GCS to local NVMe using Run:ai Model Streamer,
          with file locking for multi-process safety.
Phase 2: Load from NVMe into GPU memory via AutoModelForCausalLM.from_pretrained.

Use this template when you need HuggingFace Transformers inference (custom forward
passes, non-vLLM features). For standard OpenAI-compatible serving, prefer serve.py
which uses vLLM and gets runai_streamer (~4.3 GB/s) and tensor parallelism for free.
"""

import logging
import os
import time

from ray import serve

logger = logging.getLogger(__name__)

NVME_MODEL_DIR = "/mnt/local_storage/model"


@serve.deployment(ray_actor_options={"num_gpus": 4})
class LLMDeployment:
    def __init__(self):
        # Phase 1: GCS → NVMe.
        # ObjectStorageModel handles file locking (fcntl.flock) and idempotency
        # (.runai_complete sentinel) so only one process per node downloads;
        # subsequent replicas skip straight to Phase 2.
        from runai_model_streamer import ObjectStorageModel

        gcs_uri = os.environ["GCS_MODEL_URI"]
        logger.info(f"Downloading {gcs_uri} → {NVME_MODEL_DIR}")
        start = time.time()
        with ObjectStorageModel(model_path=gcs_uri, dst=NVME_MODEL_DIR) as model:
            model.pull_files(allow_pattern=["*.safetensors"])
            model.pull_files(ignore_pattern=["*.safetensors"])
        logger.info(f"Model ready at {NVME_MODEL_DIR} ({time.time() - start:.1f}s)")

        # Phase 2: NVMe → GPU via HuggingFace Transformers.
        # model_source is the local NVMe path — no HuggingFace Hub download.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(NVME_MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            NVME_MODEL_DIR,
            torch_dtype="auto",
            device_map="auto",
        )
        logger.info("Model loaded to GPU.")

    async def __call__(self, request):
        data = await request.json()
        inputs = self.tokenizer(data["prompt"], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text}


app = LLMDeployment.bind()

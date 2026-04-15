"""
Ray Serve deployment with fast model loading from GCS via Run:ai Model Streamer.

The streamer uses concurrent C++ threads to read safetensor shards from GCS
directly to GPU memory at ~3 GB/s — no intermediate disk writes needed.
vLLM's built-in runai_streamer integration handles this automatically.
"""

import os

from ray.serve.llm import LLMConfig, build_openai_app

GCS_MODEL_URI = os.environ.get(
    "GCS_MODEL_URI",
    "gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Llama-70B",
)

model_id = "my-model"

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id=model_id,
        model_source=GCS_MODEL_URI,
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
        load_format="runai_streamer",
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

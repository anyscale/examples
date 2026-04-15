"""Model loading with vLLM's built-in runai_streamer: GCS → GPU directly.

vLLM's runai_streamer load format streams model weights from GCS to GPU memory
using concurrent C++ threads (~3.2 GB/s from GCS). No NVMe caching or custom
callback needed — vLLM handles everything natively via runai-model-streamer-gcs.

Use this when:
- You want the simplest possible setup (no NVMe disks required)
- Replicas per node is low (each replica re-streams from GCS)

For multi-replica deployments on the same node, see serve_nvme.py which caches
model weights to NVMe so subsequent replicas skip the GCS download entirely.
"""

import os

from ray.serve.llm import LLMConfig, build_openai_app

# Load format: "runai_streamer" (GCS → GPU, ~3.2 GB/s) or "auto" (HF/safetensors)
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
        model_source=os.environ["GCS_MODEL_URI"],  # vLLM streams directly from GCS
    ),
    accelerator_type="A100",
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=4),
    ),
    engine_kwargs=engine_kwargs,
)

app = build_openai_app({"llm_configs": [llm_config]})

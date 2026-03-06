"""Ray Serve deployment for SGLang inference.

This deployment uses Ray Serve's placement_group_bundles to reserve GPUs
across multiple nodes for tensor-parallel inference with SGLang.

Based on the Ray Serve LLM + SGLang integration pattern from:
https://github.com/ray-project/ray/pull/58366
"""

import os
import signal

from fastapi import FastAPI
from ray import serve

# Configuration from environment (same defaults as driver_offline.py)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-1.7B")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
PP_SIZE = int(os.environ.get("PP_SIZE", "2"))
NUM_NODES_PER_REPLICA = int(os.environ.get("NUM_NODES_PER_REPLICA", "2"))

gpus_per_node = (TP_SIZE * PP_SIZE) // NUM_NODES_PER_REPLICA

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 4,
    },
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
    },
    # Reserve resources across multiple nodes for tensor parallelism.
    # Each bundle reserves GPUs on one node.
    placement_group_bundles=[{"CPU": 1, "GPU": gpus_per_node}] * NUM_NODES_PER_REPLICA,
)
@serve.ingress(app)
class SGLangDeployment:
    def __init__(self):
        # Import sglang inside the actor because it initializes CUDA and
        # cannot be imported on the CPU-only head node where the Serve
        # controller runs.
        from sglang import Engine

        # Monkey patch signal.signal to avoid "signal only works in main thread"
        # error. SGLang tries to register signal handlers for graceful shutdown,
        # but Ray Serve runs user code in a separate thread.
        original_signal = signal.signal

        def noop_signal_handler(sig, action):
            return signal.SIG_DFL

        try:
            signal.signal = noop_signal_handler
            self.engine = Engine(
                model_path=MODEL_PATH,
                tp_size=TP_SIZE,
                pp_size=PP_SIZE,
                nnodes=NUM_NODES_PER_REPLICA,
                use_ray=True,
            )
        finally:
            signal.signal = original_signal

    @app.post("/")
    async def generate(self, request: dict) -> dict:
        text = request.get("text", "")
        sampling_params = request.get("sampling_params", {"max_new_tokens": 64})

        # Set stream=False to get a single result instead of an AsyncGenerator.
        # Without it, the await would hang (generators need async for/anext).
        result = await self.engine.async_generate(
            prompt=text,
            sampling_params=sampling_params,
            stream=False
        )
        return {"text": result["text"]}


app_deploy = SGLangDeployment.bind()

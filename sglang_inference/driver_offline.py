import os
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Configuration from environment (same as serve.py)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-1.7B")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
PP_SIZE = int(os.environ.get("PP_SIZE", "2"))
NUM_NODES_PER_REPLICA = int(os.environ.get("NUM_NODES_PER_REPLICA", "2"))


@ray.remote
class EngineActor:
    """Thin wrapper that creates an sglang.Engine inside a Ray actor.

    We import sglang inside the actor because it initializes CUDA and
    cannot be imported on the CPU-only head node where the driver runs.
    """

    def __init__(self, **kwargs):
        from sglang import Engine

        self.engine = Engine(**kwargs)

    def generate(self, prompts, sampling_params):
        return [
            self.engine.generate(prompt=p, sampling_params=sampling_params)
            for p in prompts
        ]

    def shutdown(self):
        self.engine.shutdown()


gpus_per_node = (TP_SIZE * PP_SIZE) // NUM_NODES_PER_REPLICA

print(f"Configuration: MODEL_PATH={MODEL_PATH}, TP={TP_SIZE}, PP={PP_SIZE}, NUM_NODES_PER_REPLICA={NUM_NODES_PER_REPLICA}")
print(f"GPUs per node: {gpus_per_node}")

# Reserve GPUs across nodes
pg = placement_group(
    bundles=[{"CPU": 1, "GPU": gpus_per_node}] * NUM_NODES_PER_REPLICA,
)
ray.get(pg.ready())
print("Placement group ready.")

# Start engine actor on the first bundle
engine = EngineActor.options(
    num_cpus=1,
    num_gpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=0,
    ),
).remote(
    model_path=MODEL_PATH,
    tp_size=TP_SIZE,
    pp_size=PP_SIZE,
    nnodes=NUM_NODES_PER_REPLICA,
    use_ray=True,
)

# Wait for engine to be ready (model loaded)
print("Loading model...")
ray.get(engine.generate.remote(["warmup"], {"max_new_tokens": 1}))
print("Engine ready.")

# Batch generate
prompts = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a haiku about programming:",
    "What is 2 + 2?",
]

t0 = time.time()
results = ray.get(
    engine.generate.remote(prompts, {"max_new_tokens": 64, "temperature": 0.0})
)
print(f"Generated {len(results)} responses in {time.time() - t0:.2f}s\n")

for prompt, result in zip(prompts, results):
    print(f"Prompt:   {prompt}")
    print(f"Response: {result['text'][:200]}\n")

# Cleanup
ray.get(engine.shutdown.remote())
ray.util.remove_placement_group(pg)

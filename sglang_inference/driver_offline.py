import os
import random
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
    """Thin wrapper that creates an sglang.srt.ray.engine.RayEngine inside a Ray actor.

    We import sglang inside the actor because it initializes CUDA and
    cannot be imported on the CPU-only head node where the driver runs.
    """

    def __init__(self, **kwargs):
        from sglang.srt.ray.engine import RayEngine

        self.engine = RayEngine(**kwargs)

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

# Start engine actor on the first bundle.
# RayEngine spawns SchedulerActor children (one per GPU rank) and distributes them across nodes.
engine = EngineActor.options(
    num_cpus=1,
    num_gpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
        placement_group_capture_child_tasks=True,  # Create the child actors in the same placement group.
    ),
).remote(
    model_path=MODEL_PATH,
    tp_size=TP_SIZE,
    pp_size=PP_SIZE,
    nnodes=NUM_NODES_PER_REPLICA,
)

# Wait for engine to be ready (model loaded)
print("Loading model...")
ray.get(engine.generate.remote(["warmup"], {"max_new_tokens": 1}))
print("Engine ready.")

# Generate a large batch of prompts
base_prompts = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a haiku about programming:",
    "What is the meaning of life?",
    "Describe the water cycle:",
    "What are the benefits of exercise?",
    "Explain photosynthesis:",
    "Write a short story about a robot:",
    "What is artificial intelligence?",
    "Describe the solar system:",
    "What causes seasons on Earth?",
    "Explain the theory of relativity:",
    "What is machine learning?",
    "Describe DNA structure:",
    "What is climate change?",
    "Explain how computers work:",
    "What is the internet?",
    "Describe the human brain:",
    "What is evolution?",
    "Explain gravity:",
]

num_prompts = 250
prompts = [random.choice(base_prompts) for _ in range(num_prompts)]
batch_size = 20
sampling_params = {"max_new_tokens": 512, "temperature": 0.8}

print(f"Generating {len(prompts)} responses in batches of {batch_size}...\n")

t0 = time.time()
all_results = []

# Submit all batches upfront
futures = [(engine.generate.remote(prompts[i:i+batch_size], sampling_params), i)
           for i in range(0, len(prompts), batch_size)]

# Process results in order
for future, start_idx in futures:
    results = ray.get(future)
    all_results.extend(results)

    # Print full prompt and response for first element of batch
    print(f"\nPrompt: {prompts[start_idx]}")
    print(f"Response: {results[0]['text']}")
    print(f"Completed {len(all_results)}/{len(prompts)} responses ({len(all_results)/(time.time()-t0):.1f} resp/sec)\n")

elapsed = time.time() - t0
print(f"Generated {len(all_results)} responses in {elapsed:.2f}s ({len(all_results)/elapsed:.2f} resp/sec)")

# Cleanup
ray.get(engine.shutdown.remote())
ray.util.remove_placement_group(pg)

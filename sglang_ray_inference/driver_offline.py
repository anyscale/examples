"""
Offline (Batch) Inference with SGLang on Ray

Runs sglang.Engine inside a Ray actor for batch generation.
The head node needs no GPU — sglang is imported only inside the actor.

Usage:
    python driver_offline.py --model-path Qwen/Qwen3-1.7B --tp-size 8 --nnodes 2
"""

import argparse
import sys
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def main():
    parser = argparse.ArgumentParser(description="SGLang offline inference via Ray")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    world_size = args.tp_size * args.pp_size
    gpus_per_node = world_size // args.nnodes

    # --- Ray init ---
    print(f"Model={args.model_path}  TP={args.tp_size}  PP={args.pp_size}  "
          f"nodes={args.nnodes}  GPUs/node={gpus_per_node}")

    # --- Placement group: one bundle per node ---
    strategy = "STRICT_PACK" if args.nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(
        name="engine_group",
        bundles=[{"CPU": 1, "GPU": gpus_per_node}] * args.nnodes,
        strategy=strategy,
    )
    ray.get(pg.ready())
    print("Placement group ready.")

    # --- Engine actor (sglang imported inside, not on head node) ---
    @ray.remote
    class EngineActor:
        def __init__(self, **kwargs):
            from sglang import Engine
            self.engine = Engine(**kwargs)

        def is_ready(self):
            return True

        def generate(self, prompts, sampling_params):
            return [
                {"prompt": p, "text": self.engine.generate(prompt=p, sampling_params=sampling_params)["text"]}
                for p in prompts
            ]

        def shutdown(self):
            if self.engine:
                self.engine.shutdown()
                self.engine = None

    engine = EngineActor.options(
        num_cpus=1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0,
        ),
    ).remote(
        model_path=args.model_path,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        nnodes=args.nnodes,
        port=args.port,
        use_ray=True,
    )

    print("Waiting for engine...")
    ray.get(engine.is_ready.remote())
    print("Engine ready.\n")

    # --- Generate ---
    prompts = [
        "The capital of France is",
        "Explain quantum computing in simple terms:",
        "Write a haiku about programming:",
        "What is 2 + 2?",
    ]

    t0 = time.time()
    results = ray.get(engine.generate.remote(prompts, {"max_new_tokens": 64, "temperature": 0.0}))
    print(f"Generated {len(results)} responses in {time.time() - t0:.2f}s\n")

    for r in results:
        print(f"Prompt:    {r['prompt']}")
        print(f"Response:  {r['text'][:200]}")
        print("-" * 60)

    # --- Cleanup ---
    ray.get(engine.shutdown.remote())
    ray.util.remove_placement_group(pg)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())

"""
Offline (batch) inference with SGLang on Ray.

Wraps sglang.Engine in a Ray actor for multi-node batch generation.
The driver (head node) needs no GPU — sglang is imported only inside the actor.

Usage:
    python driver_offline.py --model-path Qwen/Qwen3-1.7B --tp-size 4 --nnodes 1
"""

import argparse
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class EngineActor:
    """Thin wrapper that creates an sglang.Engine inside a Ray actor."""

    def __init__(self, **kwargs):
        from sglang import Engine

        self.engine = Engine(**kwargs)

    def ready(self):
        return True

    def generate(self, prompts, sampling_params):
        return [
            self.engine.generate(prompt=p, sampling_params=sampling_params)
            for p in prompts
        ]

    def shutdown(self):
        self.engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    gpus_per_node = (args.tp_size * args.pp_size) // args.nnodes

    # Reserve GPUs across nodes
    pg = placement_group(
        bundles=[{"CPU": 1, "GPU": gpus_per_node}] * args.nnodes,
        strategy="STRICT_PACK" if args.nnodes == 1 else "STRICT_SPREAD",
    )
    ray.get(pg.ready())

    # Start engine actor on the first bundle
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

    ray.get(engine.ready.remote())
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


if __name__ == "__main__":
    main()

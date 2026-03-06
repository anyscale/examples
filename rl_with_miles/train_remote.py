#!/usr/bin/env python
"""Ray remote wrapper for training - ensures it runs on GPU workers."""
import sys
import subprocess
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

@ray.remote
def run_training(cmd_args):
    """Run training on GPU workers.

    Uses node label scheduling to ensure placement on H100 GPU nodes.
    The label must match the accelerator-type in job.yaml compute_config.

    Note: We don't reserve GPUs (num_gpus) because the MILES training script
    internally uses Ray to allocate GPUs for training and rollout.
    Reserving GPUs in the wrapper would conflict with the subprocess's
    GPU allocation.
    """
    result = subprocess.run(
        ["python", "/tmp/miles/train_async.py"] + cmd_args,
        capture_output=False,  # Stream output directly
        text=True
    )
    return result.returncode

if __name__ == "__main__":
    # Pass through all command-line arguments
    cmd_args = sys.argv[1:]

    # Get a GPU worker node (matches ray.io/accelerator-type: H100 from job.yaml)
    gpu_nodes = [node for node in ray.nodes() if node.get("Resources", {}).get("GPU", 0) > 0]
    if not gpu_nodes:
        raise RuntimeError("No GPU nodes available")

    # Schedule on a GPU node with H100 accelerators
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        node_id=gpu_nodes[0]["NodeID"],
        soft=False,
    )

    # Run training on GPU workers
    returncode = ray.get(
        run_training.options(scheduling_strategy=scheduling_strategy).remote(cmd_args)
    )

    sys.exit(returncode)

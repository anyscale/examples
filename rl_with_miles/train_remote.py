#!/usr/bin/env python
"""Ray remote wrapper for training - ensures it runs on GPU workers."""
import sys
import subprocess
import ray


@ray.remote(label_selector={"ray.io/accelerator-type": "H100"})
def run_training(cmd_args):
    """Run training on GPU workers.

    Uses label selector to ensure placement on H100 GPU nodes.
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


sys.exit(ray.get(run_training.remote(sys.argv[1:])))

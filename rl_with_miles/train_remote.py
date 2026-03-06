#!/usr/bin/env python
"""Ray remote wrapper for training - ensures it runs on GPU workers."""
import sys
import subprocess
import ray

@ray.remote(resources={"accelerator_type_H100": 0.001})
def run_training(cmd_args):
    """Run training on GPU workers.

    Uses accelerator_type_H100 resource to ensure scheduling on H100 GPU nodes.
    This must match the accelerator-type label in job.yaml compute_config.

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

    # Run training on GPU workers
    returncode = ray.get(run_training.remote(cmd_args))

    sys.exit(returncode)

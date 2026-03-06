#!/usr/bin/env python
"""Ray remote wrapper for training - ensures it runs on GPU workers."""
import sys
import subprocess
import ray

@ray.remote(num_gpus=8)  # Reserves all 8 GPUs (4 for training + 4 for rollout)
def run_training(cmd_args):
    """Run training on GPU workers."""
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

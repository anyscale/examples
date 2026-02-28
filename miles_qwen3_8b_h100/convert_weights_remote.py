#!/usr/bin/env python
"""Ray remote wrapper for weight conversion - ensures it runs on a GPU worker."""
import sys
import subprocess
import ray

@ray.remote(num_gpus=1)
def convert_weights(cmd_args):
    """Run weight conversion on a GPU worker."""
    result = subprocess.run(
        ["python", "/tmp/miles/tools/convert_hf_to_torch_dist.py"] + cmd_args,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr

if __name__ == "__main__":
    # Pass through all command-line arguments
    cmd_args = sys.argv[1:]

    # Run conversion on GPU worker
    returncode, stdout, stderr = ray.get(convert_weights.remote(cmd_args))

    # Print output
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)

    sys.exit(returncode)

#!/usr/bin/env python
"""Ray remote wrapper for weight conversion - ensures it runs on a GPU worker."""
import sys
import subprocess
import ray

@ray.remote(num_gpus=1, label_selector={"ray.io/accelerator-type": "H100"})
def convert_weights(cmd_args):
    """Run weight conversion on a GPU worker.

    Uses label selector to ensure placement on H100 GPU nodes.
    The label must match the accelerator-type in job.yaml compute_config.

    Reserves 1 GPU for the Megatron weight conversion process.
    """
    result = subprocess.run(
        ["python", "/tmp/miles/tools/convert_hf_to_torch_dist.py"] + cmd_args,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr

returncode, stdout, stderr = ray.get(convert_weights.remote(sys.argv[1:]))
if stdout:
    print(stdout, end="")
if stderr:
    print(stderr, end="", file=sys.stderr)
sys.exit(returncode)

import ray
import subprocess
from time import perf_counter as pc

SCRIPT = """
set -e
cp /mnt/user_storage/cosmos-config.yaml /cosmos_curate/config/cosmos_curate.yaml
# Hello World
pixi run -e model-download python -m cosmos_curate.core.managers.model_cli download --models gpt2
# Reference Video Pipeline
pixi run -e model-download python -m cosmos_curate.core.managers.model_cli download --models qwen2.5_vl,transnetv2,internvideo2_mm,bert
"""

@ray.remote(num_cpus=0)
def run_init():
    try:
        return subprocess.check_output(SCRIPT, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Init script failed (exit code {e.returncode}):\n{e.output.decode()}") from None

if __name__ == "__main__":
    t = pc()
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"]]
    tasks = [
        run_init.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=n["NodeID"], soft=False
            )
        ).remote()
        for n in nodes
    ]
    print(f"Downloading models on {len(tasks)} nodes...")
    ray.get(tasks)
    dur = pc() - t
    print(f"Done. ({dur:0.1f}s)")

"""Simulate engine failure by killing a GPU worker process.

Deployed as a separate Ray Serve application so it doesn't interfere with the WideEP application.

Usage:
    curl -X POST -H "Authorization: Bearer $SERVICE_TOKEN" $SERVICE_URL/simulate-fault
"""

import random
import subprocess

import ray
from ray import serve
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from starlette.requests import Request
from starlette.responses import JSONResponse


@ray.remote(num_cpus=0)
def _find_and_kill_gpu_process():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"error": f"nvidia-smi failed: {result.stderr}"}

    pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip()]
    if not pids:
        return {"error": "No GPU processes found on this node"}

    victim_pid = pids[0]
    kill_result = subprocess.run(["kill", "-9", str(victim_pid)])
    if kill_result.returncode != 0:
        return {"error": f"Failed to kill PID {victim_pid}"}

    return {"killed_pid": victim_pid, "node_ip": ray.util.get_node_ip_address()}


@serve.deployment(num_replicas=1)
class KillWorkerProc:
    async def __call__(self, request: Request):
        if request.method != "POST":
            return JSONResponse(
                {"error": "Use POST"}, status_code=405, headers={"Allow": "POST"}
            )

        # Find nodes that have GPUs.
        gpu_nodes = [
            n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0
        ]
        if not gpu_nodes:
            return JSONResponse({"error": "No live GPU nodes found"}, status_code=404)

        target = random.choice(gpu_nodes)
        result = await _find_and_kill_gpu_process.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=target["NodeID"], soft=False
            )
        ).remote()

        return JSONResponse(result)


app = KillWorkerProc.bind()

"""DP group fault tolerance repro.

Deploys microsoft/Phi-tiny-MoE-instruct with dp_size=2, num_replicas=2 (4 total
Serve replicas), then kills one GPU process via nvidia-smi to trigger gang recovery.
Watch the Ray Dashboard to observe the effect.
"""

import asyncio
import subprocess
import time

import ray
from ray import serve
from ray._common.test_utils import wait_for_condition
from ray.serve._private.constants import SERVE_DEFAULT_APP_NAME
from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_dp_deployment
from ray.serve.schema import ApplicationStatus, ReplicaState
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


def is_default_app_running():
    try:
        app = serve.status().applications[SERVE_DEFAULT_APP_NAME]
        return app.status == ApplicationStatus.RUNNING
    except (KeyError, AttributeError):
        return False


def get_num_running_replicas(deployment_name: str) -> int:
    dep = (
        serve.status().applications[SERVE_DEFAULT_APP_NAME].deployments[deployment_name]
    )
    return dep.replica_states.get(ReplicaState.RUNNING, 0)


def kill_one_gpu_process():
    """Kill one GPU process found via nvidia-smi.

    Uses nvidia-smi to find a PID using a GPU, then kills it with SIGKILL.
    Returns the killed PID.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip()]
    assert len(pids) > 0, "No GPU processes found via nvidia-smi"

    victim_pid = pids[0]
    log(f"Killing GPU process with PID {victim_pid}")
    subprocess.run(["kill", "-9", str(victim_pid)], check=True)
    return victim_pid


def log(msg):
    print(msg, flush=True)


def main():
    deployment_name = "DPServer:microsoft--Phi-tiny-MoE-instruct"
    dp_size = 2
    num_replicas = 2
    expected_serve_replicas = num_replicas * dp_size

    llm_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id="microsoft/Phi-tiny-MoE-instruct",
            model_source="microsoft/Phi-tiny-MoE-instruct",
        ),
        deployment_config=dict(
            num_replicas=num_replicas,
            health_check_period_s=1,
            health_check_timeout_s=5,
        ),
        engine_kwargs=dict(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=dp_size,
            distributed_executor_backend="ray",
            max_model_len=1024,
            max_num_seqs=32,
            enforce_eager=True,
        ),
        runtime_env={
            "env_vars": {
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "RAY_CGRAPH_get_timeout": "30",
            },
        },
    )

    # --- Deploy ---
    log("Deploying LLM service with dp_size=2, num_replicas=2 ...")
    handle = serve.run(build_dp_deployment(llm_config), blocking=False)
    wait_for_condition(is_default_app_running, timeout=300)
    log(f"All {expected_serve_replicas} replicas RUNNING.")

    # --- Background request sender ---
    @ray.remote
    class RequestSender:
        def __init__(self, h):
            self._handle = h.options(stream=True)
            self.total = 0
            self.errors = []
            self._stop = False

        async def _send_loop(self):
            req = CompletionRequest(
                model="microsoft/Phi-tiny-MoE-instruct",
                prompt="Hello, world!",
                max_tokens=5,
            )
            while not self._stop:
                try:
                    async for _ in self._handle.completions.remote(req):
                        pass
                    self.total += 1
                except Exception as e:
                    self.errors.append(str(e))
                    self.total += 1
                await asyncio.sleep(0.1)

        async def run(self, concurrency=10):
            await asyncio.gather(*[self._send_loop() for _ in range(concurrency)])

        def stop(self):
            self._stop = True

        def get_results(self):
            return self.total, self.errors

    sender = RequestSender.remote(handle)
    sender.run.remote()
    log("Background request sender started. Warming up for 2 min ...")
    time.sleep(120)

    # --- Kill a GPU process ---
    log("Killing one GPU process via nvidia-smi ...")
    killed_pid = kill_one_gpu_process()
    log(f"Killed PID {killed_pid}. Watch the Ray Dashboard for recovery.")

    # --- Wait for gang teardown ---
    log("Waiting for faulty gang teardown (replica count to drop) ...")
    wait_for_condition(
        lambda: get_num_running_replicas(deployment_name) < expected_serve_replicas,
        timeout=90,
    )
    current = get_num_running_replicas(deployment_name)
    log(
        f"Gang teardown detected. Running replicas: {current}/{expected_serve_replicas}"
    )

    # --- Wait for full recovery ---
    log("Waiting for full recovery ...")
    wait_for_condition(
        lambda: get_num_running_replicas(deployment_name) == expected_serve_replicas,
        timeout=180,
    )
    log(f"All {expected_serve_replicas} replicas recovered and RUNNING.")

    # --- Check request sender results ---
    ray.get(sender.stop.remote())
    total, errors = ray.get(sender.get_results.remote())
    log(f"Request sender results: {total} total requests, {len(errors)} errors")
    if errors:
        log(f"Errors: {errors[:5]}")

    # --- Keep service alive for dashboard inspection ---
    log("\nService is still running. Press Ctrl+C to shut down.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        log("Shutting down ...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()

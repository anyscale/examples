"""
Online (HTTP Server) Inference with SGLang on Ray

Launches the SGLang HTTP server as a Ray remote function.
The head node needs no GPU — sglang is imported only inside the task.

Usage:
    python driver_online.py --model-path Qwen/Qwen3-1.7B --tp-size 8 --nnodes 2
"""

import argparse
import sys
import time

import ray
import requests
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
def _get_node_ip():
    return ray.util.get_node_ip_address()


@ray.remote
def _launch_server(**server_kwargs):
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    launch_server(ServerArgs(**server_kwargs))


def main():
    parser = argparse.ArgumentParser(description="SGLang HTTP server via Ray")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    world_size = args.tp_size * args.pp_size
    gpus_per_node = world_size // args.nnodes

    print(f"Model={args.model_path}  TP={args.tp_size}  PP={args.pp_size}  "
          f"nodes={args.nnodes}  GPUs/node={gpus_per_node}")

    # --- Placement group: one bundle per node ---
    strategy = "STRICT_PACK" if args.nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(
        name="engine_group",
        bundles=[{"CPU": 1, "GPU": gpus_per_node}] * args.nnodes,
        strategy=strategy,
    )
    ray.get(pg.ready())
    print("Placement group ready.")

    pg_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=0,
    )

    # --- Resolve the node IP where the server will run ---
    node_ip = ray.get(
        _get_node_ip.options(
            num_cpus=0,
            scheduling_strategy=pg_strategy,
        ).remote()
    )
    url = f"http://{node_ip}:{args.port}"
    print(f"Server URL: {url}")

    # --- Launch the HTTP server as a Ray task (blocks until server exits) ---
    server_ref = _launch_server.options(
        num_cpus=1,
        num_gpus=0,
        scheduling_strategy=pg_strategy,
    ).remote(
        model_path=args.model_path,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        nnodes=args.nnodes,
        port=args.port,
        host="0.0.0.0",
        use_ray=True,
    )

    # --- Health check ---
    print("Waiting for server to be healthy...")
    t0 = time.time()
    timeout = 600
    healthy = False
    while time.time() - t0 < timeout:
        # Check if the task crashed early
        ready, _ = ray.wait([server_ref], timeout=0)
        if ready:
            # Task finished unexpectedly — surface the error
            ray.get(server_ref)
            print("ERROR: server task exited before becoming healthy.")
            ray.util.remove_placement_group(pg)
            return 1
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                healthy = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
        elapsed = int(time.time() - t0)
        if elapsed % 30 == 0:
            print(f"  {elapsed}s elapsed...")

    if not healthy:
        print("ERROR: server did not become healthy within timeout.")
        ray.cancel(server_ref, force=True)
        ray.util.remove_placement_group(pg)
        return 1

    print(f"Server healthy ({int(time.time() - t0)}s).")

    # --- Test request ---
    try:
        resp = requests.post(
            f"{url}/generate",
            json={"text": "The capital of France is",
                  "sampling_params": {"max_new_tokens": 32, "temperature": 0.0}},
            timeout=60,
        )
        resp.raise_for_status()
        print(f"Test response: {resp.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Warning: test request failed: {e}")

    # --- Shutdown ---
    ray.cancel(server_ref, force=True)
    ray.util.remove_placement_group(pg)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

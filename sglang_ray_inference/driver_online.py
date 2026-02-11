"""
Online (HTTP server) inference with SGLang on Ray.

Launches the SGLang HTTP server inside a Ray task for multi-node serving.
The driver (head node) needs no GPU — sglang is imported only inside the task.

Usage:
    python driver_online.py --model-path Qwen/Qwen3-1.7B --tp-size 4 --nnodes 1
"""

import argparse
import time

import ray
import requests
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
def get_node_ip():
    """Return the IP of the node this task lands on."""
    return ray.util.get_node_ip_address()


@ray.remote
def launch_server(**kwargs):
    """Start the SGLang HTTP server (blocks until the server exits)."""
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    launch_server(ServerArgs(**kwargs))


def wait_for_healthy(url, server_ref, timeout=600, poll=5):
    """Poll the health endpoint until the server is ready or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        # If the task already finished, it crashed
        done, _ = ray.wait([server_ref], timeout=0)
        if done:
            ray.get(server_ref)  # raises on error
            raise RuntimeError("Server exited before becoming healthy.")
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(poll)
    raise TimeoutError(f"Server not healthy after {timeout}s.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    gpus_per_node = (args.tp_size * args.pp_size) // args.nnodes

    # Reserve GPUs across nodes
    pg = placement_group(
        bundles=[{"CPU": 1, "GPU": gpus_per_node}] * args.nnodes,
        strategy="STRICT_PACK" if args.nnodes == 1 else "STRICT_SPREAD",
    )
    ray.get(pg.ready())

    pg_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=0,
    )

    # Resolve the IP of the node where the server will run
    node_ip = ray.get(
        get_node_ip.options(num_cpus=0, scheduling_strategy=pg_strategy).remote()
    )
    url = f"http://{node_ip}:{args.port}"
    print(f"Server URL: {url}")

    # Launch the HTTP server (runs until cancelled)
    server_ref = launch_server.options(
        num_cpus=1, num_gpus=0, scheduling_strategy=pg_strategy,
    ).remote(
        model_path=args.model_path,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        nnodes=args.nnodes,
        port=args.port,
        host="0.0.0.0",
        use_ray=True,
    )

    wait_for_healthy(url, server_ref)
    print("Server healthy.")

    # Test request
    resp = requests.post(
        f"{url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
        },
        timeout=60,
    )
    resp.raise_for_status()
    print(f"Test response: {resp.json()}")

    # Cleanup
    ray.cancel(server_ref, force=True)
    ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    main()

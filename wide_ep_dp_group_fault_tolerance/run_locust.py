"""Run locust load test with a shaped traffic pattern.

Traffic shape (14 min total):
  baseline 2m -> ramp up 4m -> peak 2m -> ramp down 4m -> baseline 2m

Usage:
    python run_locust.py --host https://my-service.anyscale.com \
        --token $ANYSCALE_API_TOKEN --baseline-users 10 --peak-users 50
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_locust_headless(
    host: str,
    locustfile: str,
    token: str,
    route_prefix: str,
    max_tokens: int,
    prompt: str,
    output_dir: str,
    baseline_users: int,
    peak_users: int,
    spawn_rate: int,
    processes: int,
):
    """Run a single locust headless session with the TrafficShape."""
    results_file = Path(output_dir) / f"results_b{baseline_users}_p{peak_users}"

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "--headless",
        "--host",
        host,
        "--locustfile",
        locustfile,
        "--csv",
        str(results_file),
        "--token",
        token,
        "--route-prefix",
        route_prefix,
        "--max-tokens",
        str(max_tokens),
        "--prompt",
        prompt,
        "--baseline-users",
        str(baseline_users),
        "--peak-users",
        str(peak_users),
        "--ramp-rate",
        str(spawn_rate),
    ]

    if processes > 1:
        cmd.extend(["--processes", str(processes)])

    print(f"\n{'='*60}")
    print(
        f"Traffic shape: baseline={baseline_users}, peak={peak_users}, spawn_rate={spawn_rate}"
    )
    print(f"  0:00-2:00  baseline ({baseline_users} users)")
    print(f"  2:00-6:00  ramp up -> {peak_users} users")
    print(f"  6:00-8:00  peak ({peak_users} users)")
    print(f"  8:00-12:00 ramp down -> {baseline_users} users")
    print(f"  12:00-14:00 baseline ({baseline_users} users)")
    print(f"Results: {results_file}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run locust load tests against an LLM service"
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Target host URL (e.g. https://my-service.anyscale.com)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("ANYSCALE_API_TOKEN", ""),
        help="Bearer token for auth (or set ANYSCALE_API_TOKEN env var)",
    )
    parser.add_argument(
        "--route-prefix",
        type=str,
        default="/v1",
        help="Route prefix for the service (default: /v1)",
    )
    parser.add_argument(
        "--baseline-users",
        type=int,
        default=10,
        help="Baseline number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--peak-users",
        type=int,
        default=50,
        help="Peak number of concurrent users (default: 50)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=5,
        help="Users spawned/despawned per second during ramps (default: 5)",
    )
    parser.add_argument(
        "--locustfile",
        type=str,
        default="locustfile.py",
        help="Path to locustfile (default: locustfile.py)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of locust worker processes (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per LLM request (default: 256)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short paragraph about distributed systems.",
        help="Prompt to send to the LLM",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for CSV result files (default: results/)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Host: {args.host}")
    print(
        f"Traffic shape: baseline={args.baseline_users} -> peak={args.peak_users} -> baseline={args.baseline_users}"
    )
    print("Total duration: 14 minutes")
    print(f"Output dir: {args.output_dir}")

    rc = run_locust_headless(
        host=args.host,
        locustfile=args.locustfile,
        token=args.token,
        route_prefix=args.route_prefix,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        output_dir=args.output_dir,
        baseline_users=args.baseline_users,
        peak_users=args.peak_users,
        spawn_rate=args.spawn_rate,
        processes=args.processes,
    )

    if rc != 0:
        print(f"WARNING: Locust exited with code {rc}")

    print("\nRun complete. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()

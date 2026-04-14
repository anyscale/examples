"""Locust load test for an OpenAI-compatible LLM service.

Supports two traffic patterns (--traffic-pattern):
- constant: Steady traffic at --baseline-users (runs indefinitely, Ctrl+C to stop)
- varying: Shaped 14-min pattern: baseline 2m -> ramp up 4m -> peak 2m -> ramp down 4m -> baseline 2m
"""

import json
import os

from locust import HttpUser, LoadTestShape, task, constant, events


@events.init_command_line_parser.add_listener
def add_custom_args(parser):
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("ANYSCALE_API_TOKEN", ""),
        help="Bearer token for Anyscale service auth",
    )
    parser.add_argument(
        "--route-prefix",
        type=str,
        default="/v1",
        help="Route prefix for the service",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per request",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short paragraph about distributed systems.",
        help="Prompt to send to the LLM",
    )
    parser.add_argument(
        "--baseline-users",
        type=int,
        default=10,
        help="Baseline number of users (default: 10)",
    )
    parser.add_argument(
        "--peak-users",
        type=int,
        default=50,
        help="Peak number of users (default: 50)",
    )
    parser.add_argument(
        "--ramp-rate",
        type=float,
        default=5,
        help="Users spawned/despawned per second during ramps (default: 5)",
    )
    parser.add_argument(
        "--traffic-pattern",
        type=str,
        required=True,
        choices=["constant", "varying"],
        help="Traffic pattern: 'constant' for steady load, 'varying' for shaped 14-min pattern",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-tiny-MoE-instruct",
        help="Model ID to send in chat completion requests",
    )


class TrafficShape(LoadTestShape):
    """
    Supports two modes based on --traffic-pattern:

    constant:
      Holds --baseline-users forever (Ctrl+C to stop).

    varying (14-min shaped pattern):
      0:00  - 2:00   baseline (steady)
      2:00  - 6:00   ramp up from baseline to peak
      6:00  - 8:00   peak (steady)
      8:00  - 12:00  ramp down from peak to baseline
      12:00 - 14:00  baseline (steady)
      14:00          stop
    """

    def tick(self):
        opts = self.runner.environment.parsed_options
        baseline = opts.baseline_users
        peak = opts.peak_users
        spawn_rate = opts.ramp_rate

        if opts.traffic_pattern == "constant":
            return baseline, spawn_rate

        # --- varying: 14-min shaped pattern ---
        run_time = self.get_run_time()

        t_baseline1 = 120  # 2 min
        t_ramp_up = 240  # 4 min
        t_peak = 120  # 2 min
        t_ramp_down = 240  # 4 min
        t_baseline2 = 120  # 2 min

        c1 = t_baseline1
        c2 = c1 + t_ramp_up
        c3 = c2 + t_peak
        c4 = c3 + t_ramp_down
        c5 = c4 + t_baseline2

        if run_time < c1:
            return baseline, spawn_rate
        elif run_time < c2:
            progress = (run_time - c1) / t_ramp_up
            users = int(baseline + (peak - baseline) * progress)
            return users, spawn_rate
        elif run_time < c3:
            return peak, spawn_rate
        elif run_time < c4:
            progress = (run_time - c3) / t_ramp_down
            users = int(peak - (peak - baseline) * progress)
            return users, spawn_rate
        elif run_time < c5:
            return baseline, spawn_rate
        else:
            return None


class LLMUser(HttpUser):
    """Simulates a user sending chat completion requests to an LLM service."""

    wait_time = constant(0)

    def on_start(self):
        token = self.environment.parsed_options.token
        self.route_prefix = self.environment.parsed_options.route_prefix
        self.max_tokens = self.environment.parsed_options.max_tokens
        self.prompt = self.environment.parsed_options.prompt

        self.model = self.environment.parsed_options.model
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    @task
    def chat_completion(self):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }

        with self.client.post(
            f"{self.route_prefix}/chat/completions",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" not in data:
                        response.failure("Response missing 'choices' field")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

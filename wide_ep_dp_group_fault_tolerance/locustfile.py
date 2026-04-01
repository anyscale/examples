"""Locust load test for an OpenAI-compatible LLM service.

Traffic shape (14 min total):
  baseline 2m -> ramp up 4m -> peak 2m -> ramp down 4m -> baseline 2m

Custom CLI args:  --token, --route-prefix, --max-tokens, --prompt,
                  --baseline-users, --peak-users, --spawn-rate
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


class TrafficShape(LoadTestShape):
    """
    Custom traffic shape:
      0:00 - 2:00  baseline (steady)
      2:00 - 6:00  ramp up from baseline to peak
      6:00 - 8:00  peak (steady)
      8:00 - 12:00 ramp down from peak to baseline
      12:00 - 14:00 baseline (steady)
      14:00        stop
    """

    def tick(self):
        run_time = self.get_run_time()
        opts = self.runner.environment.parsed_options

        baseline = opts.baseline_users
        peak = opts.peak_users
        spawn_rate = opts.ramp_rate

        # Phase durations in seconds
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
            # Phase 1: baseline
            return baseline, spawn_rate
        elif run_time < c2:
            # Phase 2: ramp up
            progress = (run_time - c1) / t_ramp_up
            users = int(baseline + (peak - baseline) * progress)
            return users, spawn_rate
        elif run_time < c3:
            # Phase 3: peak
            return peak, spawn_rate
        elif run_time < c4:
            # Phase 4: ramp down
            progress = (run_time - c3) / t_ramp_down
            users = int(peak - (peak - baseline) * progress)
            return users, spawn_rate
        elif run_time < c5:
            # Phase 5: baseline
            return baseline, spawn_rate
        else:
            # Done
            return None


class LLMUser(HttpUser):
    """Simulates a user sending chat completion requests to an LLM service."""

    # Set to constant(0) for max throughput, or between(1, 3) for realistic pacing
    wait_time = constant(0)

    def on_start(self):
        token = self.environment.parsed_options.token
        self.route_prefix = self.environment.parsed_options.route_prefix
        self.max_tokens = self.environment.parsed_options.max_tokens
        self.prompt = self.environment.parsed_options.prompt

        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    @task
    def chat_completion(self):
        payload = {
            "model": "microsoft/Phi-tiny-MoE-instruct",
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

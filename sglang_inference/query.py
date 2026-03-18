"""Query the SGLang inference service."""

import os
import requests

SERVICE_URL = os.environ.get("SERVICE_URL")
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN")

if not SERVICE_URL or not SERVICE_TOKEN:
    print("Set SERVICE_URL and SERVICE_TOKEN from 'anyscale service deploy' output")
    raise SystemExit(1)

prompts = [
    "The capital of France is",
    "Explain quantum computing in one sentence:",
    "Write a haiku about programming:",
    "What is 2 + 2?",
    "The largest planet in our solar system is",
]

for prompt in prompts:
    response = requests.post(
        SERVICE_URL,
        headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
        json={"text": prompt, "sampling_params": {"max_new_tokens": 32}},
        timeout=120,
    )
    response.raise_for_status()
    print(f"{prompt}{response.json()['text']}\n")

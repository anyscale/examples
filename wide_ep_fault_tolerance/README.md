# Wide EP DP Group Fault Tolerance

This example demonstrates data-parallel (DP) group fault tolerance and autoscaling for LLM serving with Ray Serve. It uses gang-scheduled DP deployments (`DPServer`), where all workers in a DP group are restarted atomically when one fails.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Clone the example

```bash
git clone https://github.com/anyscale/examples.git
cd examples/wide_ep_fault_tolerance
```

## Demo 1: Autoscaling service

Deploy an OpenAI-compatible endpoint that autoscales DP groups based on traffic:

```bash
anyscale service deploy -f service.yaml
```

Wait for the service to be ready:

```bash
anyscale service wait --name dp-group-fault-tolerance --state RUNNING --timeout-s 600
```

The `anyscale service deploy` command outputs a line that looks like:

```text
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <SERVICE_URL>
```

Set the environment variables and send a test request:

```bash
export SERVICE_URL=<SERVICE_URL>
export SERVICE_TOKEN=<SERVICE_TOKEN>

curl -H "Authorization: Bearer $SERVICE_TOKEN" \
     -H "Content-Type: application/json" \
     $SERVICE_URL/v1/chat/completions \
     -d '{"model": "microsoft/Phi-tiny-MoE-instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Generate load with Locust

Install dependencies and run the load test:

```bash
pip install -r requirements.txt

python run_locust.py \
    --host $SERVICE_URL \
    --token $SERVICE_TOKEN \
    --baseline-users 10 \
    --peak-users 50
```

The load test runs a 14-minute shaped traffic pattern (baseline → ramp up → peak → ramp down → baseline) designed to trigger autoscaling. Watch replica count change in the [services tab](https://console.anyscale.com/services).

### Shutdown

```bash
anyscale service terminate --name dp-group-fault-tolerance
```

## Demo 2: Fault tolerance

Run [`fault_tolerance_demo.py`](https://github.com/anyscale/examples/blob/main/wide_ep_fault_tolerance/fault_tolerance_demo.py) as an Anyscale job. The script deploys the model with `dp_size=2, num_replicas=2`, sends continuous traffic, kills a GPU process to simulate a real-world failure, and verifies that the DP group recovers:

```bash
anyscale job submit -f job.yaml
```

View job logs in the [jobs tab](https://console.anyscale.com/jobs). The output reports how many requests were served and how many errored during the fault window.

## Understanding the example

- [`service.yaml`](https://github.com/anyscale/examples/blob/main/wide_ep_fault_tolerance/service.yaml) deploys an OpenAI-compatible endpoint via `ray.serve.llm:build_dp_openai_app` with `data_parallel_size: 2`. Each replica spans 2 GPU workers scheduled as a placement group.
- `num_replicas: auto` enables autoscaling between 1 and 4 DP groups based on ongoing request count (`target_ongoing_requests: 5`).
- Gang scheduling ensures that if one worker in a DP group fails, the entire group is torn down and restarted together — preventing partial failures from leaving the deployment in an inconsistent state.
- The fault tolerance demo uses `dp_size=2, num_replicas=2`, creating 4 total Ray Serve replicas (2 DP groups × 2 workers each). Killing one GPU process causes the entire group to tear down while the other group continues serving, then the killed group restarts atomically.

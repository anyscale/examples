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

This demo triggers a GPU failure against the live service while traffic is running, and observes gang recovery.

### Step 1: Start a background load test

Keep traffic flowing so you can observe requests succeed, dip during recovery, and succeed again:

```bash
pip install -r requirements.txt

python run_locust.py \
    --host $SERVICE_URL \
    --token $SERVICE_TOKEN \
    --baseline-users 10 \
    --peak-users 10
```

### Step 2: Kill a GPU process

Open the [Anyscale console](https://console.anyscale.com/services), navigate to your service, and click on the **Nodes** tab. Click on any worker node and open a **Terminal**. Then run:

```bash
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | head -1)
```

### Step 3: Observe recovery

- The **Locust output** will show a brief spike in latency or errors as the affected DP group tears down.
- The **Ray Serve dashboard** (accessible from the service page) shows replica count drop from 4 to 2, then recover back to 4.
- The surviving DP group continues serving requests throughout — only requests in-flight on the killed group are affected.

## Understanding the example

- [`service.yaml`](https://github.com/anyscale/examples/blob/main/wide_ep_fault_tolerance/service.yaml) deploys an OpenAI-compatible endpoint via `ray.serve.llm:build_dp_openai_app` with `data_parallel_size: 2`. Each replica spans 2 GPU workers scheduled as a placement group.
- `num_replicas: auto` enables autoscaling between 1 and 4 DP groups based on ongoing request count (`target_ongoing_requests: 5`).
- Gang scheduling ensures that if one worker in a DP group fails, the entire group is torn down and restarted together — preventing partial failures from leaving the deployment in an inconsistent state.
- The fault tolerance demo uses `dp_size=2, num_replicas=2`, creating 4 total Ray Serve replicas (2 DP groups × 2 workers each). Killing one GPU process causes the entire group to tear down while the other group continues serving, then the killed group restarts atomically.

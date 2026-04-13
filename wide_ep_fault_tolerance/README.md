# Wide EP Fault Tolerance

Demonstrates data-parallel (DP) group fault tolerance and autoscaling for MoE LLM serving with Ray Serve. Uses gang-scheduled DP deployments where all workers in a DP group are restarted atomically when one fails.

Check out the [blog post](https://www.anyscale.com/blog/dp-group-fault-tolerance-vllm-wideep-ray-serve-llm) for a detailed walkthrough of the Wide EP Fault Tolerance feature.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/wide_ep_fault_tolerance
```

Deploy the service. By default it uses `microsoft/Phi-tiny-MoE-instruct` with autoscaling enabled (`num_replicas: auto`).

```bash
anyscale service deploy -f service.yaml
anyscale service wait --name wide-ep-fault-tolerance --state RUNNING --timeout-s 600
```

Set `SERVICE_URL` and `SERVICE_TOKEN` from the deploy output:

```bash
export SERVICE_URL=<SERVICE_URL>
export SERVICE_TOKEN=<SERVICE_TOKEN>
```

## Fault tolerance demo

Start constant traffic in one terminal:

```bash
uv run --with locust --with requests run_locust.py \
    --host $SERVICE_URL \
    --token $SERVICE_TOKEN \
    --traffic-pattern constant \
    --baseline-users 10
```

In another terminal, kill a random GPU worker process via the service's `/simulate-fault` endpoint:

```bash
curl -X POST -H "Authorization: Bearer $SERVICE_TOKEN" $SERVICE_URL/simulate-fault
```

Observe recovery:

- The **Locust output** shows a brief spike in errors as the affected DP group tears down.
- The **Service dashboard** shows replica count drop then recover.
- The surviving DP group continues serving requests throughout.

## Autoscaling demo

Run a shaped traffic pattern to trigger scale-up/down:

```bash
uv run --with locust --with requests run_locust.py \
    --host $SERVICE_URL \
    --token $SERVICE_TOKEN \
    --traffic-pattern varying \
    --baseline-users 5 \
    --peak-users 40
```

The load test runs a 14-minute shaped traffic pattern (baseline -> ramp up -> peak -> ramp down -> baseline). The service autoscales when traffic pattern shifts. Watch replica count change in the services tab.

## Understanding the example

- `service.yaml` deploys `microsoft/Phi-tiny-MoE-instruct` with `data_parallel_size: 2` and `num_replicas: auto` (autoscaling between 1-4 DP groups, 2 ranks per group).
- `kill_worker_proc.py` is deployed as a separate Ray Serve application at `/simulate-fault`. It uses `nvidia-smi` to find a GPU process on a random worker node and kills it with `SIGKILL`.
- Ray Serve gang scheduling ensures that if one worker in a DP group fails, the entire group is torn down and restarted together — preventing partial failures from leaving the deployment in an inconsistent state.

## Shutdown

```bash
anyscale service terminate --name wide-ep-fault-tolerance
```

# Wide EP Fault Tolerance

Demonstrates data-parallel (DP) group fault tolerance and autoscaling for LLM serving with Ray Serve. Uses gang-scheduled DP deployments where all workers in a DP group are restarted atomically when one fails.

Check out https://www.anyscale.com/blog/dp-group-fault-tolerance-vllm-wideep-ray-serve-llm for a detailed
walkthrough of the Wide EP Fault Tolerance feature.

## Setup

```bash
pip install -U anyscale
anyscale login
git clone https://github.com/anyscale/examples.git
cd examples/wide_ep_fault_tolerance
pip install -r requirements.txt
```

## Demo 1: Autoscaling

Deploy an autoscaling service and generate a shaped traffic pattern to trigger scale-up/down:

```bash
anyscale service deploy -f autoscaling/service.yaml
anyscale service wait --name wide-ep-autoscaling --state RUNNING --timeout-s 600
```

Set `SERVICE_URL` and `SERVICE_TOKEN` from the deploy output, then run the load test:

```bash
export SERVICE_URL=<SERVICE_URL>
export SERVICE_TOKEN=<SERVICE_TOKEN>

python run_locust.py \
    --host $SERVICE_URL \
    --token $SERVICE_TOKEN \
    --traffic-pattern varying \
    --baseline-users 5 \
    --peak-users 40
```

The load test runs a 14-minute shaped traffic pattern (baseline -> ramp up -> peak -> ramp down -> baseline). Watch replica count change in the [services tab](https://console.anyscale.com/services).

```bash
anyscale service terminate --name wide-ep-autoscaling
```

## Demo 2: Fault tolerance

Deploy a fixed-replica service and kill a GPU worker to observe gang recovery:

```bash
anyscale service deploy -f fault_tolerance/service.yaml
anyscale service wait --name wide-ep-fault-tolerance --state RUNNING --timeout-s 600
```

Set `SERVICE_URL` and `SERVICE_TOKEN` from the deploy output:

```bash
export SERVICE_URL=<SERVICE_URL>
export SERVICE_TOKEN=<SERVICE_TOKEN>
```

### Step 1: Start constant traffic

In one terminal, start a steady load:

```bash
python run_locust.py \
    --host $SERVICE_URL \
    --token $SERVICE_TOKEN \
    --traffic-pattern constant \
    --baseline-users 10
```

### Step 2: Kill a GPU worker process

In another terminal, kill a random GPU worker process via the service's debug endpoint:

```bash
curl -X POST -H "Authorization: Bearer $SERVICE_TOKEN" $SERVICE_URL/simulate-fault
```

### Step 3: Observe recovery

- The **Locust output** shows a brief spike in errors as the affected DP group tears down.
- The **Ray Serve dashboard** (accessible from the service page) shows replica count drop then recover.
- The surviving DP group continues serving requests throughout.

```bash
anyscale service terminate --name wide-ep-fault-tolerance
```

## Understanding the example

- [`fault_tolerance/service.yaml`](fault_tolerance/service.yaml) deploys `microsoft/Phi-tiny-MoE-instruct` with `data_parallel_size: 2` and `num_replicas: 2` (2 DP groups x 2 workers = 4 GPU workers). [`autoscaling/service.yaml`](autoscaling/service.yaml) uses `num_replicas: auto` to enable autoscaling between 1-4 DP groups.
- [`fault_tolerance/kill_worker_proc.py`](fault_tolerance/kill_worker_proc.py) is deployed as a separate Ray Serve application at `/simulate-fault`. It uses `nvidia-smi` to find a GPU process on a random worker node and kills it with `SIGKILL`.
- Ray Serve gang scheduling ensures that if one worker in a DP group fails, the entire group is torn down and restarted together — preventing partial failures from leaving the deployment in an inconsistent state.

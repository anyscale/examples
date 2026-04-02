# DP Group Fault Tolerance for LLM Serving

Demonstrate data-parallel (DP) group fault tolerance and autoscaling on Ray Serve LLM deployments. Both demos use gang-scheduled data-parallel deployments (`DPServer`), where all workers in a DP group are restarted atomically on failure.

## Repository Structure

```
├── dp_group_fault_tolerance_demo.py   # Demo 1: fault tolerance via Python builders
├── dp_group_autoscaling_service.yaml  # Demo 2: autoscaling via declarative YAML config
├── locustfile.py                      # Locust load test with shaped traffic pattern
├── run_locust.py                      # CLI wrapper for running the load test
├── requirements.txt                   # Python dependencies
```

## Prerequisites

- Python 3.10+
- An [Anyscale](https://www.anyscale.com/) account with API access
- `anyscale` CLI installed and authenticated
- A Ray cluster with GPUs (for Demo 1) or Anyscale platform access (for Demo 2).

```bash
pip install -r requirements.txt
```

## Note
You can either use Python builder or declarative YAML config pattern to spin up a service. The DP group fault tolerance and autoscaling features are agnostic to the builder pattern. Both features are fully supported in Ray OSS 2.55.

---

## Demo 1: Fault Tolerance (Python Builders)

This demo uses `dp_group_fault_tolerance_demo.py` to deploy a DP group locally on a Ray cluster using Python builders, send continuous traffic, kill a GPU process simulating real-world GPU failures, and observe DP group recovery.

### How it works

The script uses `build_dp_deployment` from `ray.serve.llm` to construct a `DPServer` deployment programmatically:

```python
from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_dp_deployment

llm_config = LLMConfig(
    model_loading_config=ModelLoadingConfig(
        model_id="microsoft/Phi-tiny-MoE-instruct",
        model_source="microsoft/Phi-tiny-MoE-instruct",
    ),
    deployment_config=dict(
        num_replicas=2,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=2,
        distributed_executor_backend="ray",
        max_model_len=1024,
        max_num_seqs=32,
        enforce_eager=True,
    ),
    runtime_env={
        "env_vars": {
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        },
    },
)

handle = serve.run(build_dp_deployment(llm_config), blocking=False)
```

With `dp_size=2` and `num_replicas=2`, this creates **4 total Ray Serve replicas** (2 DP groups (`num_replicas`) x 2 workers (`dp_size`) each).

### What the script does

1. **Deploy** — calls `serve.run(build_dp_deployment(llm_config))` and waits for all 4 replicas to be `RUNNING`.
2. **Send traffic** — spawns a `RequestSender` Ray actor that sends 10 concurrent completion requests in a loop, then warms up for 2 minutes.
3. **Kill a GPU process** — uses `nvidia-smi --query-compute-apps=pid` to find a GPU process and kills it with `SIGKILL`.
4. **Observe gang teardown** — waits for the running replica count to drop below 4 (the entire DP group containing the killed worker is torn down).
5. **Observe recovery** — waits for all 4 replicas to return to `RUNNING` (the gang is restarted atomically).
6. **Report results** — prints total requests sent and errors encountered during the fault.

### Run it

On a Ray cluster with at least 4 GPUs:

```bash
python dp_group_fault_tolerance_demo.py
```

The script keeps the service alive after recovery so you can inspect the Ray Dashboard. Press `Ctrl+C` to shut down.


### What to expect

- After killing a GPU process, the **entire DP group** containing that worker is torn down (replica count drops from 4 to 2).
- The surviving DP group continues serving requests.
- The killed DP group is restarted atomically — both workers come back together.
- Replica count returns to 4.
- The `RequestSender` reports errors only for requests that were in-flight on the killed group.

---

## Demo 2: Autoscaling (Declarative YAML)

This demo deploys the same model on Anyscale using a declarative `dp_group_autoscaling_service.yaml`, then uses Locust to drive shaped traffic that triggers autoscaling.

### Deploy

```bash
anyscale service deploy -f dp_group_autoscaling_service.yaml
```

Note the service URL and auth token from the output.

### dp_group_autoscaling_service.yaml configuration reference

The service deploys an OpenAI-compatible LLM endpoint via `ray.serve.llm:build_dp_openai_app`, which constructs a Ray Serve application with a gang-scheduled `DPServer`. Unlike Demo 1 which uses a fixed replica count, this config uses `num_replicas: auto` to enable autoscaling.


### Verify the service

```bash
anyscale service status --name dp-group-fault-tolerance
```

Wait until the service state is `RUNNING`.

### Send a test request

```bash
curl -H "Authorization: Bearer <TOKEN>" \
     -H "Content-Type: application/json" \
     https://<SERVICE_URL>/v1/chat/completions \
     -d '{"model": "microsoft/Phi-tiny-MoE-instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Generate load with Locust

The load test uses a fixed **14-minute shaped traffic pattern** designed to trigger autoscaling:

```
  0:00 -  2:00   baseline (steady at --baseline-users)
  2:00 -  6:00   ramp up to --peak-users
  6:00 -  8:00   peak (steady at --peak-users)
  8:00 - 12:00   ramp down to --baseline-users
 12:00 - 14:00   baseline (steady at --baseline-users)
```

This shape is defined by the `TrafficShape` class in `locustfile.py`.

#### Basic run

```bash
python run_locust.py \
    --host https://<SERVICE_URL> \
    --token <TOKEN> \
    --baseline-users 10 \
    --peak-users 50
```

#### Higher peak with shorter outputs

```bash
python run_locust.py \
    --host https://<SERVICE_URL> \
    --token <TOKEN> \
    --baseline-users 10 \
    --peak-users 200 \
    --max-tokens 32 \
    --spawn-rate 10
```

### What to expect

- During the ramp-up phase, `target_ongoing_requests: 5` is exceeded and the autoscaler adds DP groups (after `upscale_delay_s: 10`).
- During the ramp-down phase, the autoscaler removes DP groups (after `downscale_delay_s: 20`).
- Check the Anyscale console / Ray Serve dashboard for replica count changes.

### Cleanup

```bash
anyscale service terminate --name dp-group-fault-tolerance
```

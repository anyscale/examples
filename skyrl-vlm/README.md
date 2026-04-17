# Multi-turn GRPO for a Vision-Language Model with SkyRL

This example uses [SkyRL](https://github.com/NovaSky-AI/SkyRL) to run multi-turn GRPO training on the [Geometry-3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset with the vision-language model `Qwen/Qwen3-VL-8B-Instruct`.

Geometry-3K contains 3,002 geometry problems with diagrams; the model must reason over both the image and the problem text. The reward is 1.0 for a correct answer (extracted from `\boxed{}`) and 0.0 otherwise.

This is the multimodal counterpart to the text-only GRPO example in [`../skyrl`](../skyrl).

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/skyrl-vlm
```

Submit the job.

```bash
anyscale job submit -f job.yaml
```

## Understanding the example

- The [job.yaml](./job.yaml) hyperparameters mirror [`run_geometry3k.sh`](https://github.com/NovaSky-AI/SkyRL/blob/main/examples/train/geometry3k/run_geometry3k.sh) from the upstream SkyRL repo verbatim, so results should reproduce the upstream example.
- The entrypoint first runs `examples/train/geometry3k/geometry_3k_dataset.py` to materialize the dataset under `/mnt/cluster_storage/data/geometry_3k`. The `/mnt/cluster_storage/` directory is an ephemeral shared filesystem attached to the cluster for the duration of the job, which lets all workers read the parquet files.
- Training runs via `examples/train/geometry3k/geometry3k_entrypoint.py` rather than SkyRL's `main_base` entrypoint. The geometry3k entrypoint registers a custom `geometry3k` SkyRL-Gym environment (for answer extraction and reward) before launching `BasePPOExp`.
- The base image [`novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`](https://hub.docker.com/r/novaskyai/skyrl-train-ray) ships Ray 2.51.1, CUDA 12.8, and SkyRL's training dependencies. Because SkyRL's `pyproject.toml` also pins `ray==2.51.1`, the `uv run --isolated` venv matches the cluster Ray automatically — no `--with ray@...` override is needed.
- The job requests a single `p5.48xlarge` worker (8× H100 80GB). All eight GPUs are colocated for the policy, critic/ref, and vLLM inference engines (`trainer.placement.colocate_all=true`, `generator.inference_engine.num_engines=8`, `tensor_parallel_size=1`).
- Checkpoints and exports go to `/mnt/cluster_storage/geometry3k_vlm/...`, which is shared across workers but lost when the cluster stops. To persist to blob storage, set `trainer.ckpt_path=$ANYSCALE_ARTIFACT_STORAGE/geometry3k_vlm/checkpoints` and add `--with s3fs` (AWS) or `--with gcsfs` (GCP) to the second `uv run` command. Read more about [Anyscale storage options](https://docs.anyscale.com/configuration/storage).

## Enabling Weights & Biases

The default logger is `console`. To log to W&B, set your API key and flip the logger:

```yaml
env_vars:
  WANDB_API_KEY: "<your_key>"
```

Then change `trainer.logger="console"` to `trainer.logger="wandb"` in the entrypoint.

## Key hyperparameters

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Algorithm | GRPO |
| Strategy | FSDP2, all roles colocated |
| Train batch size | 256 |
| Policy mini-batch | 64 |
| `n_samples_per_prompt` | 8 |
| `max_turns` | 3 |
| Learning rate | 1e-6 |
| Epochs | 6 |

## View the job

View the job in the [jobs tab](https://console.anyscale.com/jobs) of the Anyscale console.

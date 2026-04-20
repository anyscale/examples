# GRPO with SkyRL

This directory has two [SkyRL](https://github.com/NovaSky-AI/SkyRL) examples:

- **GSM8K** — single-turn GRPO training of `Qwen2.5-1.5B-Instruct` on the GSM8K math dataset.
- **Geometry-3K (VLM)** — multi-turn GRPO training of `Qwen3-VL-2B-Instruct` on the Geometry-3K visual reasoning dataset.

SkyRL is a modular and extensible reinforcement learning library for training large language models. It supports RL algorithms like PPO, GRPO, and DAPO, tool-use tasks, and multi-turn agentic workflows.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Clone the repo

```bash
git clone https://github.com/anyscale/examples.git
cd examples/skyrl
```

## Run the GSM8K example

```bash
anyscale job submit -f job.yaml
```

This runs on a single 4-GPU worker (`g6.12xlarge` on AWS, `g2-standard-48-nvidia-l4-4` on GCP) using the image built from `Dockerfile`.

## Run the Geometry-3K (VLM) example

```bash
anyscale job submit -f vlm-job.yaml
```

This runs on a single 4×L40S worker and trains `Qwen3-VL-2B-Instruct`. To train the larger `Qwen3-VL-8B-Instruct`, bump the worker to 8×H100 (or similar) and raise `policy_num_gpus_per_node` / `num_engines` in the entrypoint to match.

Checkpoints and exports land in blob storage at `$ANYSCALE_ARTIFACT_STORAGE/skyrl_vlm_checkpoints` and `$ANYSCALE_ARTIFACT_STORAGE/skyrl_vlm_exports`.

## Understanding the examples

- Both examples first run a dataset script to materialize parquet files under `/mnt/cluster_storage/data/...`. `/mnt/cluster_storage/` is an ephemeral shared filesystem attached to the cluster for the duration of the job, which ensures all workers see the same data.
- Both examples invoke the SkyRL entrypoint via `uv run`, which picks up the relevant `pyproject.toml` from the SkyRL checkout at `$HOME/SkyRL`. The GSM8K job `cd`s into `$HOME/SkyRL/skyrl-train` (nested pyproject) and the VLM job `cd`s into `$HOME/SkyRL` (root pyproject). Because `uv` looks for `pyproject.toml` in the working directory, neither job yaml sets `working_dir` — if it did, `uv` would look in the wrong place.
- The GSM8K entrypoint passes `--with ray@http://localhost:9478/ray/ray-<version>-cp312-cp312-manylinux2014_x86_64.whl` so the uv venv uses the same Ray that ships in the cluster image. Keep this URL's version in sync with the `FROM anyscale/ray:<version>-...` line in `Dockerfile` whenever you bump the base image.
- Checkpoints can go in a few places — read more about [Anyscale storage options](https://docs.anyscale.com/configuration/storage):
  - `/mnt/cluster_storage/...` — ephemeral, gone when the job ends.
  - `/mnt/shared_storage/...` — persistent shared filesystem across jobs.
  - `$ANYSCALE_ARTIFACT_STORAGE/...` — blob storage (S3/GCS). Pass `--with s3fs` (AWS) or `--with gcsfs` (GCP) to the `uv run` command so the training process can write there. The VLM example already writes here.

## View the job

View the job in the [jobs tab](https://console.anyscale.com/jobs) of the Anyscale console.

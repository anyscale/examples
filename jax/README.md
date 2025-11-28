# Train a model with Jax on GPUs

This example uses Jax to train a small model on 16 T4 GPUs.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone this repository and submit the job with

```bash
anyscale job submit -f job.yaml
```

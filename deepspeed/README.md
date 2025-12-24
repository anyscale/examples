# Train a model with DeepSpeed and Ray Train

This example uses Jax to train a small model on 16 T4 GPUs.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone this repository

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples
git checkout deepspeed
cd deepspeed
```

Submit the job with

```bash
anyscale job submit -f job.yaml
```

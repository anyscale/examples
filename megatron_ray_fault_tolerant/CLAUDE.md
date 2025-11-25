# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PPO-style distributed training example using Megatron-Core and Ray with fault tolerance capabilities. The system can automatically recover from actor failures by utilizing backup actors and re-initializing NCCL process groups.

## Commands

### Running the example
```bash
uv run --isolated main.py
```

### Submit to Anyscale
```bash
anyscale job submit -f job.yaml
```

### Linting and formatting
```bash
ruff check --fix .
black .
```

## Architecture

### Core Components

**MegatronActor** (`megatron_actor.py:60-642`): Ray remote actor wrapping a Megatron model. Each actor:
- Owns one GPU and participates in distributed training
- Manages its own NCCL process group membership
- Handles model init, forward/backward, checkpointing, and recovery

**MegatronActorGroup** (`megatron_actor.py:644-935`): Manages a collection of MegatronActors:
- Creates and places actors on placement group bundles
- Coordinates distributed operations via dispatch system
- Implements fault recovery by replacing dead actors with backups and re-initializing process groups

**Dispatch System** (`dispatch.py`): Handles data distribution and result collection:
- `MeshDispatch`: Shards data across DP dimension, collects from primary ranks (SP=0, TP=0, PP=last)
- `PassThroughDispatch`: Broadcasts same data/commands to all actors
- Extensible via `register_dispatch_type()`

**MegatronModelWrapper** (`megatron_model_wrapper.py`): Wraps Megatron model for PPO training:
- Handles micro-batch accumulation via Megatron's `forward_backward_func`
- Computes PPO policy loss with clipping

### Data Flow

1. `TrainingInputBatch` is dispatched to actors (sharded by DP rank for `MeshDispatch`)
2. Each actor runs `ppo_train()` which iterates micro-batches and accumulates gradients
3. Megatron handles gradient sync across TP/PP/DP dimensions
4. Results collected from primary DP ranks and concatenated

### Parallelism Dimensions

- **DP (Data Parallel)**: Distributes data across groups
- **TP (Tensor Parallel)**: Splits model tensors within a node
- **PP (Pipeline Parallel)**: Distributes model layers across stages
- **SP/CP (Context Parallel)**: Sequence parallelism for long contexts

`MeshRank` dataclass tracks each actor's position in the 4D mesh.

### Fault Recovery Flow

1. Detect dead actors via `_check_actor_alive()`
2. Destroy old NCCL process groups on healthy actors
3. Pop backup actors from `backup_actor_group`
4. Update world size and reassign ranks
5. Re-initialize process groups and model components
6. Load checkpoint to restore state

### Cloud Storage

`file_io.py` provides unified file I/O for local/S3/GCS:
- `local_work_dir()`: Context manager for checkpoint saving (auto-uploads from temp dir)
- `local_read_dir()`: Context manager for checkpoint loading (auto-downloads to temp dir)

## Key Configuration

In `main.py`, the `Config` dataclass controls:
- `num_nodes`, `num_gpus_per_node`: Active worker topology
- `num_spare_gpus`: Backup actors for fault tolerance
- `megatron_config`: Parallelism settings (TP, PP, CP sizes)
- `ckpt_dir`: Checkpoint location (supports S3/GCS paths)

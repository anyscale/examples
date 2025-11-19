# Megatron + Ray Fault Tolerant Training

This example implements PPO-style distributed training using Megatron and Ray with comprehensive fault tolerance capabilities. The system can automatically recover from actor failures during training by utilizing backup actors and re-initializing process groups.

## Key Features

### Fault Tolerance Mechanisms

1. **Actor Health Monitoring**: Continuously monitors the health of distributed training actors
2. **Backup Actor Pool**: Pre-allocated backup actors ready to replace failed workers
3. **Automatic Recovery**: Seamlessly recovers from failures by:
   - Detecting dead actors
   - Destroying old process groups
   - Replacing failed actors with backup actors
   - Re-initializing process groups with new world size
   - Reloading model and optimizer state from checkpoints

4. **Distributed Checkpointing**: Implements efficient sharded checkpoint saving/loading using Megatron's distributed checkpointing
5. **Process Group Management**: Handles NCCL process group initialization, destruction, and re-initialization

### Parallelism Support

- **Data Parallelism (DP)**: Distributes training data across multiple GPUs
- **Tensor Parallelism (TP)**: Splits model tensors across GPUs
- **Pipeline Parallelism (PP)**: Distributes model layers across GPUs
- **Context Parallelism (CP)**: Enables sequence parallelism for long contexts

### Advanced Training Features

- **PPO Training**: Implements Proximal Policy Optimization with micro-batch accumulation
- **Mixed Precision**: Supports BF16 training for improved performance
- **Gradient Accumulation**: Handles micro-batches with automatic gradient accumulation
- **Distributed Optimizer**: Uses Megatron's distributed optimizer for memory efficiency

## Architecture

### Core Components

1. **MegatronActor** (`megatron_actor.py`): 
   - Individual training actor wrapping Megatron models
   - Handles model initialization, forward/backward passes, and checkpointing
   - Supports dynamic process group re-initialization

2. **MegatronActorGroup** (`megatron_actor.py`):
   - Manages a group of distributed actors
   - Implements fault recovery logic
   - Coordinates distributed training operations

3. **Dispatch System** (`dispatch.py`):
   - **MeshDispatch**: Distributes data across the device mesh (DP, SP, TP, PP)
   - **PassThroughDispatch**: Broadcasts same data/commands to all actors
   - Handles data sharding and result collection

4. **Training Batch** (`training_batch.py`):
   - Defines input/output batch structures for PPO training
   - Supports chunking and concatenation for distributed operations

5. **Checkpoint I/O** (`file_io.py`):
   - Cloud-aware file I/O supporting S3, GCS, and local storage
   - Efficient checkpoint upload/download with parallel transfers

## Getting Started

### Quick Start

```bash
uv run --isolated main.py
```

This will:
1. Create a placement group with workers and backup GPUs
2. Initialize the actor group and model
3. Run a training step
4. Save a checkpoint
5. Simulate a failure by killing actors
6. Recover from the failure using backup actors
7. Resume training after recovery

### Configuration

Edit the `Config` class in `main.py` to customize:

```python
@dataclass
class Config:
    model: str = "Qwen/Qwen3-0.6B"  # HuggingFace model name
    num_nodes: int = 1
    num_gpus_per_node: int = 4
    num_spare_gpus: int = 4  # Backup actors for fault tolerance
    mini_batch_size: int = 16
    micro_train_batch_size_per_gpu: int = 2
    
    # Megatron parallelism settings
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
```

### Megatron Parallelism Configuration

```python
@dataclass
class MegatronConfig:
    tensor_model_parallel_size: int = 1  # TP degree
    pipeline_model_parallel_size: int = 1  # PP degree
    context_parallel_size: int = 1  # CP degree
    expert_model_parallel_size: int = 1  # For MoE models
```

## Fault Recovery Workflow

1. **Training Phase**:
   - Actors perform distributed training using Megatron
   - Periodic checkpoints saved to cloud storage

2. **Failure Detection**:
   - System detects actor failures via health checks
   - Identifies affected data parallel groups

3. **Recovery Process**:
   - Destroy old process groups on healthy actors
   - Pop backup actors from the backup pool
   - Insert backup actors at failed ranks
   - Update world size and reassign ranks
   - Re-initialize process groups with new configuration
   - Reload model/optimizer state from checkpoint

4. **Resume Training**:
   - Continue training with recovered actor group
   - No loss of training progress (from last checkpoint)

## Advanced Usage

### Custom Dispatch Types

Register custom dispatch strategies:

```python
from dispatch import register_dispatch_type, Dispatch

class CustomDispatch(Dispatch):
    # Implement dispatch, collect, and validate methods
    pass

register_dispatch_type("custom", CustomDispatch)
```

### CPU Offloading (Experimental)

For faster recovery, offload model/optimizer state to CPU memory:

```python
# Before failure
ray.get(actor_group.async_run_ray_method("pass_through", "offload_to_cpu"))

# After recovery, on healthy actors
ray.get(actor_group.async_run_ray_method("pass_through", "backload_to_gpu"))
```

## Dependencies

See `pyproject.toml` for full dependency list. Key dependencies:
- Ray for distributed orchestration
- Megatron-Core for model parallelism
- PyTorch with CUDA support
- Transformers for model loading
- vLLM and related libraries

## Running on Anyscale

Submit the job using:

```bash
anyscale job submit -f job.yaml
```

The job configuration in `job.yaml` specifies:
- Container image with dependencies
- GPU instance types (g6e.12xlarge with 4xL4)
- Resource limits and scaling
- Environment variables for NCCL configuration

## Limitations and Future Work

- Virtual pipeline parallelism not yet supported
- CPU offloading optimization in progress
- Async checkpoint saving planned for future releases

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Ray Documentation](https://docs.ray.io/)
- [Anyscale Platform](https://docs.anyscale.com/)

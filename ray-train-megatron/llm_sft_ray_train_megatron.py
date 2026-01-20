"""
Ray Train integration for Megatron-Bridge finetuning.

Launches Megatron-Bridge finetuning using Ray Train's TorchTrainer
for distributed orchestration. Supports tensor, pipeline, and data parallelism.

Usage:
    # Basic usage with 8 GPUs (TP=2, PP=2, DP=2)
    python finetune_decoder_ray.py \
        --hf_model_path meta-llama/Meta-Llama-3.1-8B \
        --num_workers 8 \
        --storage_path /mnt/cluster_storage

    # Custom parallelism
    python finetune_decoder_ray.py \
        --hf_model_path meta-llama/Meta-Llama-3.1-8B \
        --num_workers 8 \
        --tensor_parallel_size 2 \
        --pipeline_parallel_size 2 \
        --train_iters 1000

    # With custom training parameters
    python finetune_decoder_ray.py \
        --hf_model_path meta-llama/Meta-Llama-3.1-8B \
        --num_workers 8 \
        --global_batch_size 64 \
        --micro_batch_size 1 \
        --learning_rate 5e-6
"""

import argparse
import logging
import os
import sys
import uuid
from typing import Any, Dict

# Enable Ray Train V2
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

# Required for sequence parallelism with tensor parallelism > 1
# Without this, communication and computation cannot properly overlap
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# PyTorch memory optimization - reduces fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Get Megatron-Bridge and Megatron-LM paths for workers
# When running as a Ray job with working_dir sync, MEGATRON_BRIDGE_ROOT env var should be set
# Otherwise, compute paths relative to script location
_MEGATRON_BRIDGE_ROOT = os.environ.get("MEGATRON_BRIDGE_ROOT")
if _MEGATRON_BRIDGE_ROOT is None:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _MEGATRON_BRIDGE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

_MEGATRON_BRIDGE_SRC = os.path.join(_MEGATRON_BRIDGE_ROOT, "src")
# Use bundled Megatron-LM from 3rdparty (has correct version for Megatron-Bridge)
_MEGATRON_LM_ROOT = os.path.join(_MEGATRON_BRIDGE_ROOT, "3rdparty", "Megatron-LM")

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

logger = logging.getLogger(__name__)


def log_rank0(message: str) -> None:
    """Log message only on rank 0."""
    if ray.train.get_context().get_world_rank() == 0:
        logger.info(message)


def create_megatron_config(
    hf_model_path: str,
    output_dir: str,
    tensor_parallel_size: int = 2,
    pipeline_parallel_size: int = 2,
    train_iters: int = 1000,
    global_batch_size: int = 64,
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    learning_rate: float = 5e-6,
    eval_interval: int = 100,
    save_interval: int = 100,
) -> "ConfigContainer":
    """Create Megatron-Bridge ConfigContainer for Ray Train finetuning.

    Key differences from standard finetune config:
    1. Uses AutoBridge with load_weights=True for HF weight loading
    2. Sets external_gpu_device_mapping=True (Ray handles GPU assignment)
    3. Supports tensor and pipeline parallelism

    Args:
        hf_model_path: HuggingFace model path (e.g., "meta-llama/Meta-Llama-3.1-8B")
        output_dir: Directory for checkpoints and logs
        tensor_parallel_size: Tensor parallelism degree
        pipeline_parallel_size: Pipeline parallelism degree
        train_iters: Number of training iterations
        global_batch_size: Global batch size across all workers
        micro_batch_size: Micro batch size per GPU
        seq_length: Sequence length for training
        learning_rate: Learning rate for finetuning
        eval_interval: Evaluation interval in iterations
        save_interval: Checkpoint save interval in iterations

    Returns:
        ConfigContainer ready for Megatron-Bridge training
    """
    # Import Megatron-Bridge modules
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
    from megatron.bridge.recipes.utils.optimizer_utils import (
        distributed_fused_adam_with_cosine_annealing,
    )
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        DistributedDataParallelConfig,
        DistributedInitConfig,
        LoggerConfig,
        RNGConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    # Setup directories
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, "tb_logs")

    # Create model config from HuggingFace with weight loading
    bridge = AutoBridge.from_hf_pretrained(hf_model_path)
    model_cfg = bridge.to_megatron_provider(load_weights=True)

    # Set parallelism configuration
    model_cfg.tensor_model_parallel_size = tensor_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_parallel_size
    model_cfg.context_parallel_size = 1
    model_cfg.sequence_parallel = tensor_parallel_size > 1
    model_cfg.seq_length = seq_length

    # For pipeline parallelism > 1, may need virtual pipeline parallel
    if pipeline_parallel_size > 1:
        # Optional: Enable virtual pipeline parallelism for better efficiency
        # model_cfg.virtual_pipeline_model_parallel_size = 2
        pass

    # Optimizer configuration
    # Ensure warmup_iters < decay_iters (which defaults to train_iters)
    lr_warmup_iters = min(50, max(1, train_iters // 10))
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=learning_rate,
        min_lr=0.0,
        adam_beta2=0.98,
    )

    # Training configuration
    train_cfg = TrainingConfig(
        train_iters=train_iters,
        eval_interval=eval_interval,
        eval_iters=32,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        manual_gc=True,
        manual_gc_interval=100,
        manual_gc_eval=100,
    )

    # DDP configuration
    # Disable overlap features that require coalesced ops not supported with Ray Train's process groups
    ddp_cfg = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,  # Disable - requires NCCL coalesced ops
        overlap_param_gather=False,  # Disable - requires NCCL coalesced ops
        average_in_collective=True,
    )

    # Distributed initialization config for Ray Train
    dist_cfg = DistributedInitConfig(
        external_gpu_device_mapping=True,  # Ray handles GPU assignment via CUDA_VISIBLE_DEVICES
        use_gloo_process_groups=False,  # Disable Gloo groups - Ray Train handles this
    )

    # Logger configuration with enhanced metrics
    logger_cfg = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        log_memory_to_tensorboard=True,  # Track GPU memory usage
        log_throughput_to_tensorboard=True,  # Track training throughput
    )

    # Tokenizer configuration
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_model_path,
    )

    # Checkpoint configuration
    # Note: We don't set pretrained_checkpoint since weights are loaded via AutoBridge
    checkpoint_cfg = CheckpointConfig(
        save_interval=save_interval,
        save=checkpoint_dir,
        load=checkpoint_dir,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )

    # Create ConfigContainer
    config = ConfigContainer(
        model=model_cfg,
        train=train_cfg,
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
        dist=dist_cfg,
        dataset=default_squad_config(seq_length, packed_sequence=False),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=checkpoint_cfg,
        rng=RNGConfig(seed=5678),
        peft=None,  # Full finetuning, no LoRA/PEFT
        inprocess_restart=None,  # Must be None for Ray Train compatibility
        mixed_precision="bf16_mixed",
    )

    return config


def train_loop(config: Dict[str, Any]) -> None:
    """Per-worker training loop for Megatron-Bridge finetuning.

    This function is called by Ray Train on each worker. It:
    1. Creates the Megatron-Bridge configuration
    2. Calls pretrain() directly (bypasses finetune() assertion since
       weights are loaded via AutoBridge with load_weights=True)

    Args:
        config: Training configuration dict from Ray Train containing:
            - hf_model_path: HuggingFace model path
            - output_dir: Output directory for checkpoints
            - tensor_parallel_size: TP degree
            - pipeline_parallel_size: PP degree
            - train_iters: Number of training iterations
            - global_batch_size: Global batch size
            - micro_batch_size: Micro batch size
            - seq_length: Sequence length
            - learning_rate: Learning rate
            - megatron_bridge_src: Path to Megatron-Bridge src directory
            - megatron_lm_root: Path to Megatron-LM root directory
            - nemo_datasets_cache: Path for dataset caching (for multi-node)
    """
    # CRITICAL: Set NEMO_DATASETS_CACHE for multi-node compatibility
    # In multi-node setups, the default ~/.cache/nemo/datasets is local to each node
    # This causes race conditions when workers on different nodes try to access
    # dataset index files. Setting this to shared storage solves the issue.
    nemo_datasets_cache = config.get("nemo_datasets_cache")
    if nemo_datasets_cache:
        os.environ["NEMO_DATASETS_CACHE"] = nemo_datasets_cache
        os.environ["NEMO_HOME"] = os.path.dirname(nemo_datasets_cache)

    # Add Megatron-LM and Megatron-Bridge to Python path for workers
    megatron_lm_root = config.get("megatron_lm_root")
    if megatron_lm_root and megatron_lm_root not in sys.path:
        sys.path.insert(0, megatron_lm_root)

    megatron_bridge_src = config.get("megatron_bridge_src")
    if megatron_bridge_src and megatron_bridge_src not in sys.path:
        sys.path.insert(0, megatron_bridge_src)

    # Import Megatron-Bridge modules inside worker for proper CUDA context
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    # Get Ray Train context for logging
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    if world_rank == 0:
        logger.info(f"Starting Megatron-Bridge training with {world_size} workers")
        logger.info(f"Model: {config['hf_model_path']}")
        logger.info(
            f"Parallelism: TP={config['tensor_parallel_size']}, "
            f"PP={config['pipeline_parallel_size']}, "
            f"DP={world_size // (config['tensor_parallel_size'] * config['pipeline_parallel_size'])}"
        )

    # CRITICAL: Synchronize all workers before Megatron initialization
    # Ray Train initializes torch.distributed, but Megatron's initialize_megatron()
    # skips its internal barrier when dist is already initialized. This can cause
    # rank desynchronization during parallel_state.initialize_model_parallel().
    import torch.distributed as dist

    if dist.is_initialized():
        if world_rank == 0:
            logger.info("Synchronizing all workers before Megatron initialization...")
        dist.barrier()
        if world_rank == 0:
            logger.info(f"All {world_size} workers synchronized. Proceeding with initialization.")

    # Create Megatron-Bridge configuration
    # Note: This involves HuggingFace model loading which may take different
    # times on different ranks due to network/disk I/O variations
    if world_rank == 0:
        logger.info("Creating Megatron-Bridge configuration (loading HF model)...")

    megatron_config = create_megatron_config(
        hf_model_path=config["hf_model_path"],
        output_dir=config["output_dir"],
        tensor_parallel_size=config["tensor_parallel_size"],
        pipeline_parallel_size=config["pipeline_parallel_size"],
        train_iters=config["train_iters"],
        global_batch_size=config["global_batch_size"],
        micro_batch_size=config["micro_batch_size"],
        seq_length=config["seq_length"],
        learning_rate=config["learning_rate"],
        eval_interval=config.get("eval_interval", 100),
        save_interval=config.get("save_interval", 100),
    )

    # CRITICAL: Synchronize all workers after config creation
    # The HuggingFace model loading in create_megatron_config() can take different
    # times on different ranks. Without this barrier, some ranks may start
    # pretrain() while others are still loading, causing collective mismatches.
    if dist.is_initialized():
        if world_rank == 0:
            logger.info("Config created. Synchronizing all workers before pretrain()...")
        dist.barrier()
        if world_rank == 0:
            logger.info("All workers synchronized. Starting pretrain()...")

    # Run training using pretrain() directly
    # We bypass finetune() because it asserts pretrained_checkpoint is not None,
    # but we load weights directly via AutoBridge with load_weights=True
    pretrain(config=megatron_config, forward_step_func=forward_step)

    # Report final metrics to Ray Train dashboard
    if world_rank == 0:
        import torch

        gpu_mem_allocated = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0
        )
        gpu_mem_reserved = (
            torch.cuda.max_memory_reserved() / (1024**3)
            if torch.cuda.is_available()
            else 0
        )

        ray.train.report(
            {
                "status": "completed",
                "gpu_memory_allocated_gb": round(gpu_mem_allocated, 2),
                "gpu_memory_reserved_gb": round(gpu_mem_reserved, 2),
            }
        )
        logger.info("Training completed successfully")


def main():
    """Main entry point for Ray Train Megatron-Bridge finetuning."""
    args = parse_args()

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Validate parallelism configuration
    tp = args.tensor_parallel_size
    pp = args.pipeline_parallel_size
    total_parallel = tp * pp
    if args.num_workers % total_parallel != 0:
        raise ValueError(
            f"num_workers ({args.num_workers}) must be divisible by "
            f"TP * PP ({tp} * {pp} = {total_parallel})"
        )

    dp = args.num_workers // total_parallel
    print(f"Parallelism configuration: TP={tp}, PP={pp}, DP={dp}")
    print(f"Total workers: {args.num_workers}")

    # Ray Train scaling configuration
    # Use PACK strategy to colocate workers on same nodes for efficient TP communication
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
        accelerator_type="L4", 
        placement_strategy="PACK",
    )

    # Training loop configuration
    # Set NEMO_DATASETS_CACHE on shared storage for multi-node compatibility
    nemo_datasets_cache = os.path.join(args.storage_path, ".cache", "nemo", "datasets")
    print(f"Dataset cache path (shared): {nemo_datasets_cache}")

    train_loop_config = {
        "hf_model_path": args.hf_model_path,
        "output_dir": args.output_dir,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "train_iters": args.train_iters,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "megatron_bridge_src": _MEGATRON_BRIDGE_SRC,  # Path for workers
        "megatron_lm_root": _MEGATRON_LM_ROOT,  # Megatron-LM path for workers
        "nemo_datasets_cache": nemo_datasets_cache,  # Shared storage for multi-node
    }

    # Experiment name
    name = (
        f"megatron_ray_{uuid.uuid4().hex[:8]}"
        if args.experiment_name is None
        else args.experiment_name
    )
    print(f"Experiment name: {name}")

    # Ray Train run configuration
    run_config = RunConfig(
        storage_path=args.storage_path,
        name=name,
    )

    # Create TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    # Run training
    print("Starting Ray Train with Megatron-Bridge...")
    result = trainer.fit()
    print(f"Training finished. Result: {result}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Megatron-Bridge finetuning with Ray Train",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="HuggingFace model path (default: meta-llama/Meta-Llama-3.1-8B)",
    )

    # Parallelism configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of Ray Train workers (total GPUs) (default: 8)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=2,
        help="Pipeline parallelism degree (default: 2)",
    )

    # Training configuration
    parser.add_argument(
        "--train_iters",
        type=int,
        default=1000,
        help="Number of training iterations (default: 1000)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=64,
        help="Global batch size (default: 64)",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size per GPU (default: 1)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )

    # Checkpointing
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="Evaluation interval in iterations (default: 100)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Checkpoint save interval in iterations (default: 100)",
    )

    # Output configuration
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/cluster_storage",
        help="Ray Train storage path for checkpoints (default: /mnt/cluster_storage)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for Megatron checkpoints (default: storage_path/megatron_outputs)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )

    args = parser.parse_args()

    # Set default output_dir if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.storage_path, "megatron_outputs")

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
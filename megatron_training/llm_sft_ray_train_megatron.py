"""
Ray Train + Megatron-Bridge Tutorial
====================================

Distributed LLM finetuning combining:
- Ray Train: Orchestrates distributed workers across GPUs/nodes
- Megatron-Bridge: Efficient tensor and pipeline parallelism

Quick Start:
    python llm_sft_ray_train_megatron.py --num_workers 8

Architecture:
    main() → TorchTrainer → train_loop() on each GPU
                              ↓
                         create_megatron_config()
                              ↓
                         pretrain()

Default Configuration:
    - Tensor Parallelism (TP) = 2
    - Pipeline Parallelism (PP) = 2
    - Data Parallelism (DP) = num_workers / (TP * PP)
    - Model: Qwen/Qwen2.5-7B (7B parameter model for production training)
    - Dataset: wikitext-2-raw-v1 from HuggingFace (dataset for tutorial)
"""

import argparse
import logging
import os
import sys
import uuid
from typing import Any, Dict

# ============================================================
# SECTION 1: ENVIRONMENT SETUP
# Required environment variables for Megatron + Ray Train
# ============================================================

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Required for sequence parallelism
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Megatron paths - MEGATRON_BRIDGE_ROOT must be set (e.g., /app/Megatron-Bridge)
MEGATRON_BRIDGE_ROOT = os.environ["MEGATRON_BRIDGE_ROOT"]
MEGATRON_BRIDGE_SRC = os.path.join(MEGATRON_BRIDGE_ROOT, "src")
MEGATRON_LM_ROOT = os.path.join(MEGATRON_BRIDGE_ROOT, "3rdparty", "Megatron-LM")

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

logger = logging.getLogger(__name__)

# ============================================================
# SECTION 2: MEGATRON CONFIGURATION
# Creates the config container for Megatron-Bridge training
# ============================================================

# Tutorial defaults 
TENSOR_PARALLEL_SIZE = 2
PIPELINE_PARALLEL_SIZE = 2
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 512
LEARNING_RATE = 5e-6
EVAL_INTERVAL = 100
SAVE_INTERVAL = 100


def create_megatron_config(
    hf_model_path: str,
    output_dir: str,
    train_iters: int,
    hf_dataset_name: str = "wikitext",
    hf_dataset_config: str = "wikitext-2-raw-v1",
) -> "ConfigContainer":
    """Create Megatron-Bridge config with tutorial defaults.

    Uses hardcoded parallelism settings: TP=2, PP=2, batch=32.
    Loads dataset from HuggingFace instead of local files.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.utils.optimizer_utils import (
        distributed_fused_adam_with_cosine_annealing,
    )
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        FinetuningDatasetConfig,
        DistributedDataParallelConfig,
        DistributedInitConfig,
        LoggerConfig,
        RNGConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, "tb_logs")

    # Load model from HuggingFace and convert to Megatron format
    bridge = AutoBridge.from_hf_pretrained(hf_model_path)
    model_cfg = bridge.to_megatron_provider(load_weights=True)

    # Parallelism: TP splits layers across GPUs, PP splits model stages
    model_cfg.tensor_model_parallel_size = TENSOR_PARALLEL_SIZE
    model_cfg.pipeline_model_parallel_size = PIPELINE_PARALLEL_SIZE
    model_cfg.context_parallel_size = 1
    model_cfg.sequence_parallel = TENSOR_PARALLEL_SIZE > 1
    model_cfg.seq_length = SEQ_LENGTH

    # Optimizer with cosine annealing schedule
    lr_warmup_iters = min(50, max(1, train_iters // 10))
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=LEARNING_RATE,
        min_lr=0.0,
        adam_beta2=0.98,
    )

    # Training parameters
    train_cfg = TrainingConfig(
        train_iters=train_iters,
        eval_interval=EVAL_INTERVAL,
        eval_iters=32,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=MICRO_BATCH_SIZE,
        manual_gc=True,
        manual_gc_interval=100,
        manual_gc_eval=100,
    )

    # DDP: Disable overlap features (not supported with Ray Train process groups)
    ddp_cfg = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        average_in_collective=True,
    )

    # Distributed init: Ray handles GPU assignment via CUDA_VISIBLE_DEVICES
    dist_cfg = DistributedInitConfig(
        external_gpu_device_mapping=True,
        use_gloo_process_groups=False,
    )

    # Logging: TensorBoard with memory and throughput tracking
    logger_cfg = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        log_memory_to_tensorboard=True,
        log_throughput_to_tensorboard=True,
    )

    # Checkpointing: Distributed format for efficient save/load
    checkpoint_cfg = CheckpointConfig(
        save_interval=SAVE_INTERVAL,
        save=checkpoint_dir,
        load=checkpoint_dir,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )

    # Dataset config: Use HuggingFace dataset via FinetuningDatasetConfig
    # In the latest API, HF datasets are loaded via dataset_kwargs
    dataset_cfg = FinetuningDatasetConfig(
        seq_length=SEQ_LENGTH,
        dataloader_type="batch",
        num_workers=2,
        dataset_kwargs={
            "hf_dataset": True,
            "dataset_name": hf_dataset_name,
            "dataset_config_name": hf_dataset_config,
            "split": "train",
        },
        do_validation=True,
        do_test=False,
    )

    return ConfigContainer(
        model=model_cfg,
        train=train_cfg,
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
        dist=dist_cfg,
        dataset=dataset_cfg,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_path,
        ),
        checkpoint=checkpoint_cfg,
        rng=RNGConfig(seed=5678),
        peft=None,
        inprocess_restart=None,
        mixed_precision="bf16_mixed",
    )


# ============================================================
# SECTION 3: TRAINING LOOP
# This function runs on each Ray worker (one per GPU)
# ============================================================


def train_loop(config: Dict[str, Any]) -> None:
    """Per-worker training loop for Megatron-Bridge.

    Ray Train calls this on each worker. It:
    1. Sets up Python paths for Megatron imports
    2. Synchronizes workers before initialization
    3. Creates Megatron config and runs training
    """
    # Set dataset cache to shared storage (required for multi-node)
    nemo_datasets_cache = config.get("nemo_datasets_cache")
    if nemo_datasets_cache:
        os.environ["NEMO_DATASETS_CACHE"] = nemo_datasets_cache
        os.environ["NEMO_HOME"] = os.path.dirname(nemo_datasets_cache)

    # Add Megatron to Python path for this worker
    if MEGATRON_LM_ROOT not in sys.path:
        sys.path.insert(0, MEGATRON_LM_ROOT)
    if MEGATRON_BRIDGE_SRC not in sys.path:
        sys.path.insert(0, MEGATRON_BRIDGE_SRC)

    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    if world_rank == 0:
        dp = world_size // (TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE)
        logger.info(f"Training with {world_size} GPUs: TP={TENSOR_PARALLEL_SIZE}, PP={PIPELINE_PARALLEL_SIZE}, DP={dp}")

    # Synchronize workers before Megatron initialization
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Create config (involves HF model and dataset download)
    megatron_config = create_megatron_config(
        hf_model_path=config["hf_model_path"],
        output_dir=config["output_dir"],
        train_iters=config["train_iters"],
        hf_dataset_name=config.get("hf_dataset_name", "wikitext"),
        hf_dataset_config=config.get("hf_dataset_config", "wikitext-2-raw-v1"),
    )

    # Synchronize after config creation (HF download times vary)
    if dist.is_initialized():
        dist.barrier()

    # Run training
    pretrain(config=megatron_config, forward_step_func=forward_step)

    if world_rank == 0:
        logger.info("Training completed successfully")


# ============================================================
# SECTION 4: MAIN ENTRY POINT
# Sets up Ray Train and launches distributed training
# ============================================================


def main():
    """Launch Ray Train distributed training."""
    args = parse_args()

    # Validate: num_workers must be divisible by TP * PP
    total_parallel = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE
    if args.num_workers % total_parallel != 0:
        raise ValueError(
            f"num_workers ({args.num_workers}) must be divisible by "
            f"TP * PP ({TENSOR_PARALLEL_SIZE} * {PIPELINE_PARALLEL_SIZE} = {total_parallel})"
        )

    dp = args.num_workers // total_parallel
    print(f"Configuration: TP={TENSOR_PARALLEL_SIZE}, PP={PIPELINE_PARALLEL_SIZE}, DP={dp}")
    print(f"Total workers: {args.num_workers}")
    print(f"Model: {args.hf_model_path}")
    print(f"Dataset: {args.hf_dataset_name} ({args.hf_dataset_config})")

    # Ray Train scaling: PACK strategy keeps TP workers on same node
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
        placement_strategy="PACK",
    )

    output_dir = os.path.join(args.storage_path, "megatron_outputs")
    nemo_datasets_cache = os.path.join(args.storage_path, ".cache", "nemo", "datasets")

    train_loop_config = {
        "hf_model_path": args.hf_model_path,
        "output_dir": output_dir,
        "train_iters": args.train_iters,
        "nemo_datasets_cache": nemo_datasets_cache,
        "hf_dataset_name": args.hf_dataset_name,
        "hf_dataset_config": args.hf_dataset_config,
    }

    experiment_name = f"megatron_ray_{uuid.uuid4().hex[:8]}"
    print(f"Experiment: {experiment_name}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=RunConfig(storage_path=args.storage_path, name=experiment_name),
    )

    print("Starting Ray Train...")
    result = trainer.fit()
    print(f"Training finished. Result: {result}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (simplified for tutorial)."""
    parser = argparse.ArgumentParser(
        description="Ray Train + Megatron-Bridge Tutorial",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="HuggingFace model (default: Qwen/Qwen2.5-7B)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of GPUs (default: 8, must be divisible by TP*PP=4)",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/local_storage",
        help="Storage path for checkpoints (default: /mnt/local_storage)",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=100,
        help="Training iterations (default: 100)",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--hf_dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="HuggingFace dataset config/subset (default: wikitext-2-raw-v1)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()

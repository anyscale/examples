"""
Ray Train + DeepSpeed AutoTP Tensor Parallelism Training.

This script demonstrates training with DeepSpeed AutoTP tensor parallelism
using Ray Train's TorchTrainer for distributed execution and checkpointing.

Example usage:
    # 8 GPUs: 4-way tensor parallelism, 2-way data parallelism
    python train_deepspeed.py \
        --model_name Qwen/Qwen2-7B \
        --tp_size 4 \
        --dp_size 2 \
        --num_workers 8 \
        --dataset_name wikitext \
        --batch_size 2 \
        --seq_length 2048 \
        --num_epochs 3

    # 4 GPUs: 4-way tensor parallelism only
    python train_deepspeed.py \
        --model_name Qwen/Qwen2-7B \
        --tp_size 4 \
        --dp_size 1 \
        --num_workers 4 \
        --dataset_name wikitext \
        --num_epochs 3
"""

import argparse
import os
from typing import Any, Dict

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import deepspeed
import torch

import ray.train
import ray.train.torch

from autotp_strategy import RayAutoTPStrategy
from common import (
    add_common_args,
    get_common_train_config,
    log_rank0,
    run_trainer,
    run_training_loop,
)


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    """
    Main training loop executed by each Ray Train worker.

    Args:
        config: Training configuration dict
    """
    # Get Ray Train context
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    log_rank0(f"Worker started: world_rank={world_rank}, world_size={world_size}")

    # Initialize DeepSpeed distributed (will detect and use existing process group from Ray Train)
    deepspeed.init_distributed()

    # Create and setup the AutoTP strategy
    strategy = RayAutoTPStrategy(
        tp_size=config["tp_size"],
        dp_size=config["dp_size"],
    )

    strategy.setup(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        config={
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.01),
            "max_grad_norm": config.get("max_grad_norm", 1.0),
            "zero_stage": config["zero_stage"],
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
            "num_layers": config.get("num_layers", 0),
            "attn_impl": config.get("attn_impl", "sdpa"),
            "activation_checkpointing": config.get("activation_checkpointing", False),
            "vocab_parallel": config.get("vocab_parallel", False),
            "init_weights_path": config.get("init_weights_path"),
        },
    )

    # Run common training loop
    run_training_loop(strategy, config)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Train + DeepSpeed AutoTP Tensor Parallelism Training"
    )

    # Add common arguments
    add_common_args(parser)

    # DeepSpeed-specific arguments
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="DeepSpeed ZeRO stage (0-2, ZeRO-3 not supported with AutoTP)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = get_args()

    # Build train_loop_config
    train_loop_config = get_common_train_config(args)
    train_loop_config["zero_stage"] = args.zero_stage
    train_loop_config["impl_name"] = "deepspeed"

    # Run trainer
    run_trainer(
        args=args,
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        experiment_prefix="deepspeed_autotp",
    )


if __name__ == "__main__":
    main()

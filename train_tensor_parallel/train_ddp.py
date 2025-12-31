"""
Ray Train + DDP (Distributed Data Parallel) Training.

This script serves as a baseline reference to verify the correctness of
tensor parallelism implementations. It uses standard PyTorch DDP for
data parallelism only (no tensor parallelism).

Example usage:
    # 4 GPUs: 4-way data parallelism
    python train_ddp.py \
        --model_name Qwen/Qwen2-7B \
        --num_workers 4 \
        --dataset_name wikitext \
        --batch_size 2 \
        --seq_length 2048 \
        --num_epochs 3

    # 8 GPUs: 8-way data parallelism
    python train_ddp.py \
        --model_name Qwen/Qwen2-7B \
        --num_workers 8 \
        --dataset_name wikitext \
        --batch_size 1 \
        --seq_length 2048 \
        --num_epochs 3
"""

import argparse
import os
import uuid
from typing import Any, Dict

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import torch

import ray.train
import ray.train.torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from common import (
    log_rank0,
    save_checkpoint,
    load_checkpoint,
)
from data import create_tp_aware_dataloader
from ddp_strategy import RayDDPStrategy


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

    # Create and setup the DDP strategy
    strategy = RayDDPStrategy()

    strategy.setup(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        config={
            "learning_rate": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.01),
            "num_layers": config.get("num_layers", 0),
            "attn_impl": config.get("attn_impl", "sdpa"),
            "activation_checkpointing": config.get("activation_checkpointing", False),
            "autocast": config.get("autocast", True),
            "init_weights_path": config.get("init_weights_path"),
            "save_init_weights": config.get("save_init_weights", False),
        },
    )

    # Run training loop
    run_ddp_training_loop(strategy, config)


def run_ddp_training_loop(
    strategy: RayDDPStrategy,
    config: Dict[str, Any],
) -> None:
    """
    Run the training loop for DDP.

    This is similar to run_training_loop in common.py but adapted for DDP
    where all workers have the full model and use standard data sharding.

    Args:
        strategy: The DDP strategy (already set up)
        config: Training configuration dict
    """
    world_rank = ray.train.get_context().get_world_rank()
    device = ray.train.torch.get_device()

    # Create TP-aware dataloader (uses dp_rank/dp_size)
    # For DDP: dp_rank = world_rank, dp_size = world_size
    dataloader = create_tp_aware_dataloader(
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        dp_rank=strategy.dp_rank,
        dp_size=strategy.dp_size,
        seed=config.get("seed", 42),
        dataset_percentage=config.get("dataset_percentage", 10.0),
    )

    steps_per_epoch = len(dataloader)
    log_rank0(f"Dataloader created: {steps_per_epoch} steps per epoch")

    # Load checkpoint if resuming
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        metadata = load_checkpoint(strategy, checkpoint)
        start_epoch = metadata.get("epoch", 0) + 1
        log_rank0(f"Resuming training from epoch {start_epoch}")

    # Set model to training mode
    strategy.train()

    # Loss history tracking for verification
    loss_history = []

    # Training loop
    for epoch in range(start_epoch, config["num_epochs"]):
        # Set sampler epoch for different shuffling each epoch
        dataloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients
            strategy.zero_grad()

            # Forward pass
            loss = strategy.forward(batch)

            # Backward pass
            strategy.backward(loss)

            # Track per-step loss
            loss_value = loss.item()
            loss_history.append(loss_value)

            # Log progress
            if world_rank == 0 and step % config.get("log_interval", 10) == 0:
                log_rank0(
                    f"Epoch: {epoch} Step: {step + 1}/{steps_per_epoch} Loss: {loss_value:.4f}"
                )

            # Optimizer step
            strategy.optimizer_step()

            running_loss += loss_value
            num_batches += 1

            # Debug mode: stop early for testing
            if config.get("debug_steps", 0) > 0 and step + 1 >= config["debug_steps"]:
                log_rank0(f"Debug steps finished. Stopping epoch {epoch}.")
                break

        # Calculate average loss for epoch
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # Save checkpoint at end of epoch
        save_checkpoint(
            strategy=strategy,
            epoch=epoch,
            step=step,
            metrics={"loss": avg_loss, "epoch": epoch},
        )

        log_rank0(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    # Save loss history if output_dir is specified (rank 0 only)
    output_dir = config.get("loss_output_dir")
    if output_dir and world_rank == 0:
        import json
        os.makedirs(output_dir, exist_ok=True)
        loss_file = os.path.join(output_dir, "loss_ddp.json")
        with open(loss_file, "w") as f:
            json.dump({"implementation": "ddp", "loss_history": loss_history}, f)
        log_rank0(f"Loss history saved to {loss_file}")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Train + DDP Distributed Data Parallel Training (Baseline)"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-7B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=0,
        help="Override number of layers (0 = use model default)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation",
    )

    # Parallelism configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Number of workers (data parallelism degree)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=10.0,
        help="Percentage of dataset to use (0-100)",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU micro batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Enable activation/gradient checkpointing",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        default=True,
        help="Enable torch.autocast for mixed precision (default: True)",
    )

    # Checkpointing configuration
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/cluster_storage",
        help="Storage path for checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Experiment name to resume from",
    )

    # Logging and debugging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--debug_steps",
        type=int,
        default=0,
        help="Stop after this many steps per epoch (0 = run full epoch)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--loss_output_dir",
        type=str,
        default=None,
        help="Directory to save per-step loss history JSON (for verification)",
    )
    parser.add_argument(
        "--init_weights_path",
        type=str,
        default=None,
        help="Path to load initial model weights from (for verification across implementations)",
    )
    parser.add_argument(
        "--save_init_weights",
        action="store_true",
        help="Save initial model weights to --init_weights_path before training",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = get_args()

    print(f"Configuration: {args}")

    # Build train_loop_config
    train_loop_config = {
        # Model
        "model_name": args.model_name,
        "num_layers": args.num_layers,
        "attn_impl": args.attn_impl,
        # Parallelism - DDP has no TP, only DP
        "tp_size": 1,
        "dp_size": args.num_workers,
        # Dataset
        "dataset_name": args.dataset_name,
        "dataset_percentage": args.dataset_percentage,
        # Training
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "activation_checkpointing": args.activation_checkpointing,
        "autocast": args.autocast,
        # Logging/debugging
        "log_interval": args.log_interval,
        "debug_steps": args.debug_steps,
        "seed": args.seed,
        # Loss output for verification
        "loss_output_dir": args.loss_output_dir,
        # Init weights for verification
        "init_weights_path": args.init_weights_path,
        "save_init_weights": args.save_init_weights,
    }

    # Configure Ray Train
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    # Generate experiment name
    name = args.experiment_name
    if name is None:
        if args.resume_from is not None:
            name = args.resume_from
        else:
            name = f"ddp_dp{args.num_workers}_{uuid.uuid4().hex[:8]}"

    print(f"Experiment name: {name}")

    run_config = RunConfig(
        storage_path=args.storage_path,
        name=name,
    )

    # Create and run trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Training finished. Result: {result}")


if __name__ == "__main__":
    main()

"""
Common utilities shared between DeepSpeed AutoTP and FSDP+DTensor training scripts.
"""

import argparse
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Callable, Dict, Protocol

import torch

import ray
import ray.train
import ray.train.torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from data import create_tp_aware_dataloader

logger = logging.getLogger(__name__)


class TPStrategy(Protocol):
    """Protocol defining the interface for tensor parallelism strategies."""

    tp_rank: int
    dp_rank: int
    tp_size: int
    dp_size: int

    def setup(self, model_name: str, device: torch.device, dtype: torch.dtype, config: Dict[str, Any]) -> None:
        ...

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    def backward(self, loss: torch.Tensor) -> None:
        ...

    def forward_backward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Optional: Combined forward+backward for strategies that need it (e.g., DTensor with loss_parallel)."""
        ...

    def optimizer_step(self) -> None:
        ...

    def zero_grad(self) -> None:
        ...

    def train(self) -> None:
        ...

    def save_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        ...

    def load_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        ...


def log_rank0(message: str) -> None:
    """Log message only from rank 0."""
    if ray.train.get_context().get_world_rank() == 0:
        logger.info(message)


def save_checkpoint(
    strategy: TPStrategy,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    """
    Save checkpoint and report to Ray Train.

    All workers save their checkpoint shards, then report to Ray Train.
    Ray Train aggregates checkpoints from all workers properly.

    Args:
        strategy: The TP strategy containing the model/engine
        epoch: Current epoch number
        step: Current step number
        metrics: Training metrics to report
    """
    world_rank = ray.train.get_context().get_world_rank()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save checkpoint (each rank saves its shard)
        strategy.save_checkpoint(checkpoint_dir, tag="model")

        # Save metadata (from world rank 0 only)
        if world_rank == 0:
            metadata = {"epoch": epoch, "step": step}
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

        # All workers must call report() with their checkpoint
        checkpoint = Checkpoint.from_directory(tmp_dir)
        ray.train.report(metrics, checkpoint=checkpoint)

    if world_rank == 0:
        experiment_name = ray.train.get_context().get_experiment_name()
        log_rank0(f"Checkpoint saved for experiment {experiment_name}. Metrics: {metrics}")


def load_checkpoint(
    strategy: TPStrategy,
    checkpoint: ray.train.Checkpoint,
) -> Dict[str, Any]:
    """
    Load checkpoint for resuming training.

    Each rank loads its corresponding checkpoint shard.

    Args:
        strategy: The TP strategy containing the model/engine
        checkpoint: Ray Train checkpoint to load

    Returns:
        Metadata dict with epoch and step info
    """
    metadata = {"epoch": 0, "step": 0}

    try:
        with checkpoint.as_directory() as checkpoint_dir:
            log_rank0(f"Loading checkpoint from {checkpoint_dir}")
            ckpt_dir = os.path.join(checkpoint_dir, "checkpoint")
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = checkpoint_dir

            # Load checkpoint (each rank loads its shard)
            strategy.load_checkpoint(ckpt_dir, tag="model")

            # Read metadata
            metadata_file = os.path.join(ckpt_dir, "metadata.json")
            if os.path.isfile(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            # Synchronize across all workers
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        log_rank0(f"Successfully loaded checkpoint. Epoch: {metadata.get('epoch', 0)}")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    return metadata


def run_training_loop(
    strategy: TPStrategy,
    config: Dict[str, Any],
) -> None:
    """
    Run the common training loop for any TP strategy.

    Args:
        strategy: The TP strategy (already set up)
        config: Training configuration dict
    """
    world_rank = ray.train.get_context().get_world_rank()
    device = ray.train.torch.get_device()

    # Create TP-aware dataloader
    # Uses dp_rank/dp_size for sharding, not world_rank/world_size
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

        # Check if strategy has combined forward_backward (needed for DTensor with loss_parallel)
        use_forward_backward = hasattr(strategy, "forward_backward") and callable(
            getattr(strategy, "forward_backward", None)
        )

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients (for FSDP, DeepSpeed handles this internally)
            strategy.zero_grad()

            if use_forward_backward:
                # Combined forward+backward (required for DTensor with loss_parallel)
                loss = strategy.forward_backward(batch)
            else:
                # Separate forward and backward passes
                loss = strategy.forward(batch)
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
    impl_name = config.get("impl_name", "unknown")
    if output_dir and world_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        loss_file = os.path.join(output_dir, f"loss_{impl_name}.json")
        with open(loss_file, "w") as f:
            json.dump({"implementation": impl_name, "loss_history": loss_history}, f)
        log_rank0(f"Loss history saved to {loss_file}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all training scripts."""
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
        "--tp_size",
        type=int,
        required=True,
        help="Tensor parallel degree",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data parallel degree",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Total number of workers (must equal tp_size * dp_size)",
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
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Enable activation/gradient checkpointing",
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


def get_common_train_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build common train_loop_config from parsed args."""
    return {
        # Model
        "model_name": args.model_name,
        "num_layers": args.num_layers,
        "attn_impl": args.attn_impl,
        # Parallelism
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        # Dataset
        "dataset_name": args.dataset_name,
        "dataset_percentage": args.dataset_percentage,
        # Training
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "activation_checkpointing": args.activation_checkpointing,
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


def run_trainer(
    args: argparse.Namespace,
    train_loop_per_worker: Callable[[Dict[str, Any]], None],
    train_loop_config: Dict[str, Any],
    experiment_prefix: str,
) -> None:
    """
    Common trainer setup and execution.

    Args:
        args: Parsed command line arguments
        train_loop_per_worker: The training loop function
        train_loop_config: Configuration dict for training
        experiment_prefix: Prefix for auto-generated experiment names
    """
    # Validate parallelism configuration
    if args.tp_size * args.dp_size != args.num_workers:
        raise ValueError(
            f"tp_size ({args.tp_size}) * dp_size ({args.dp_size}) "
            f"must equal num_workers ({args.num_workers})"
        )

    print(f"Configuration: {args}")

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
            name = f"{experiment_prefix}_tp{args.tp_size}_dp{args.dp_size}_{uuid.uuid4().hex[:8]}"

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

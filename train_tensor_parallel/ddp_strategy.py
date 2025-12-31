"""DDP (Distributed Data Parallel) strategy for Ray Train as a baseline reference."""

import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from model_builder import create_model, get_model_config


class RayDDPStrategy:
    """
    Standard DDP strategy as a baseline reference for comparison.

    This is the simplest distributed training approach:
    - Each worker has a full copy of the model
    - Data is sharded across workers (data parallelism only)
    - Gradients are all-reduced across workers

    No tensor parallelism is used - this serves as a correctness baseline
    to verify that the TP implementations produce similar loss curves.
    """

    def __init__(self):
        self.model = None
        self.ddp_model = None
        self.optimizer = None
        self.rank = None
        self.world_size = None
        self.device = None
        # For compatibility with TPStrategy interface
        self.tp_size = 1
        self.dp_size = None  # Will be set to world_size
        self.tp_rank = 0
        self.dp_rank = None  # Will be set to world_rank
        self.use_autocast = False
        self.autocast_dtype = None

    def setup(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
    ) -> None:
        """
        Set up DDP training.

        Args:
            model_name: HuggingFace model name
            device: Target device
            dtype: Target dtype (bfloat16, float16, float32)
            config: Training configuration dict
        """
        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # For DDP: dp_size = world_size, tp_size = 1
        self.dp_size = self.world_size
        self.dp_rank = self.rank

        if self.rank == 0:
            print(f"[DDP] Setting up with world_size={self.world_size}")

        # Create model
        model = create_model(
            model_name=model_name,
            device=device,
            dtype=dtype,
            num_layers=config.get("num_layers", 0),
            attn_impl=config.get("attn_impl", "sdpa"),
        )

        # Apply activation checkpointing if requested
        if config.get("activation_checkpointing", False):
            model.gradient_checkpointing_enable()
            if self.rank == 0:
                print("[DDP] Enabled activation checkpointing")

        # Handle initial weights loading/saving for verification
        init_weights_path = config.get("init_weights_path")
        save_init_weights = config.get("save_init_weights", False)

        if init_weights_path:
            if save_init_weights:
                # Save initial weights (rank 0 only)
                if self.rank == 0:
                    os.makedirs(os.path.dirname(init_weights_path), exist_ok=True)
                    torch.save(model.state_dict(), init_weights_path)
                    print(f"[DDP] Saved initial weights to {init_weights_path}")
                # Synchronize to ensure file is written before other processes read
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            elif os.path.exists(init_weights_path):
                # Load initial weights
                state_dict = torch.load(init_weights_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                if self.rank == 0:
                    print(f"[DDP] Loaded initial weights from {init_weights_path}")

        # Wrap with DDP
        self.model = model
        self.ddp_model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

        # Create optimizer
        learning_rate = config.get("learning_rate", 1e-5)
        weight_decay = config.get("weight_decay", 0.01)
        self.optimizer = torch.optim.AdamW(
            self.ddp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        if self.rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"[DDP] Model initialized with {num_params:,} parameters")
            print(f"[DDP] Data parallel across {self.world_size} workers")

        # Configure autocast
        self.use_autocast = config.get("autocast", dtype in (torch.bfloat16, torch.float16))
        if self.use_autocast:
            self.autocast_dtype = dtype
            if self.rank == 0:
                dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
                print(f"[DDP] torch.autocast enabled with dtype={dtype_name}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass with DDP model."""
        from contextlib import nullcontext

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.use_autocast
            else nullcontext()
        )

        with autocast_ctx:
            outputs = self.ddp_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
        return outputs.loss

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass."""
        loss.backward()

    def optimizer_step(self) -> None:
        """Run optimizer step."""
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=True)

    def train(self) -> None:
        """Set model to training mode."""
        self.ddp_model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.ddp_model.eval()

    def save_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        """
        Save checkpoint for DDP model.

        For DDP, only rank 0 needs to save since all ranks have identical weights.
        """
        if self.rank != 0:
            return

        # Create tag directory
        tag_dir = os.path.join(checkpoint_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        # Save model state dict (unwrap DDP module)
        model_path = os.path.join(tag_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer state dict
        optim_path = os.path.join(tag_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optim_path)

    def load_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        """
        Load checkpoint for DDP model.

        All ranks load from the same checkpoint file.
        """
        tag_dir = os.path.join(checkpoint_dir, tag)

        # Load model state dict
        model_path = os.path.join(tag_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        # Load optimizer state dict
        optim_path = os.path.join(tag_dir, "optimizer.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(
                torch.load(optim_path, map_location=self.device, weights_only=True)
            )

        # Ensure all ranks are synchronized after loading
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

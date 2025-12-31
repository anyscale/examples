"""FSDP2 + DTensor tensor parallelism strategy adapted for Ray Train."""

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from model_builder import (
    create_model,
    get_model_config,
    get_transformer_layers,
)


class RayFSDPStrategy:
    """
    FSDP2 + DTensor tensor parallelism strategy adapted for Ray Train.

    Uses a 2D device mesh where:
    - First dimension (dp) is for data parallelism via FSDP2 (fully_shard)
    - Second dimension (tp) is for tensor parallelism via DTensor

    For example, with 8 GPUs, dp_size=2, tp_size=4:
    - Device mesh shape: (2, 4)
    - TP groups (same row): {0,1,2,3}, {4,5,6,7}
    - DP groups (same column): {0,4}, {1,5}, {2,6}, {3,7}
    """

    def __init__(self, tp_size: int, dp_size: int = 1):
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.model = None
        self.optimizer = None
        self.device_mesh = None
        self.tp_mesh = None
        self.dp_mesh = None
        self.tp_group = None
        self.tp_rank = None
        self.dp_rank = None
        self.rank = None
        self.world_size = None
        self.device = None
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
        Set up FSDP2 + DTensor with 2D device mesh.

        The mesh is organized as (dp, tp) where:
        - dp dimension: FSDP2 shards optimizer states and gradients
        - tp dimension: DTensor shards model weights for tensor parallelism
        """
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Calculate TP and DP rank (same layout as AutoTP)
        self.tp_rank = self.rank % self.tp_size
        self.dp_rank = self.rank // self.tp_size

        # Validate configuration
        if self.dp_size * self.tp_size != self.world_size:
            raise ValueError(
                f"dp_size ({self.dp_size}) * tp_size ({self.tp_size}) must equal "
                f"world_size ({self.world_size})"
            )

        model_config = get_model_config(model_name)
        if model_config.num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                f"TP size {self.tp_size} must divide num_key_value_heads "
                f"{model_config.num_key_value_heads}"
            )

        if self.rank == 0:
            print(
                f"[FSDP2+DTensor] Setting up 2D mesh: dp_size={self.dp_size}, tp_size={self.tp_size}"
            )

        # Create 2D device mesh: (dp, tp)
        self.device_mesh = init_device_mesh(
            "cuda", (self.dp_size, self.tp_size), mesh_dim_names=("dp", "tp")
        )
        self.tp_mesh = self.device_mesh["tp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_group = self.tp_mesh.get_group()

        if self.rank == 0:
            print(f"[FSDP2+DTensor] Device mesh created: {self.device_mesh}")

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
                print("[FSDP2+DTensor] Enabled activation checkpointing")

        # Handle initial weights loading for verification
        # This must happen BEFORE TP/FSDP sharding
        init_weights_path = config.get("init_weights_path")
        if init_weights_path and os.path.exists(init_weights_path):
            state_dict = torch.load(init_weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            if self.rank == 0:
                print(f"[FSDP2+DTensor] Loaded initial weights from {init_weights_path}")

        # Get transformer layers
        layers = get_transformer_layers(model)
        if layers is None:
            raise ValueError("Could not find transformer layers in model")

        # TP mapping for transformer layers (Qwen/Llama-style models)
        tp_mapping = {
            # Attention projections
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            # MLP projections
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

        if self.rank == 0:
            print(f"[FSDP2+DTensor] Applying DTensor TP to {len(layers)} layers")

        # Apply DTensor TP to transformer layers
        for layer in layers:
            parallelize_module(layer, self.tp_mesh, tp_mapping)

        # Apply FSDP2 (fully_shard) if dp_size > 1
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        if self.dp_size > 1:
            if self.rank == 0:
                print("[FSDP2+DTensor] Applying FSDP2 to transformer layers")

            for layer in layers:
                fully_shard(layer, mesh=self.dp_mesh, mp_policy=mp_policy)

            # Apply to the whole model
            fully_shard(model, mesh=self.dp_mesh, mp_policy=mp_policy)
        else:
            if self.rank == 0:
                print("[FSDP2+DTensor] dp_size=1, skipping FSDP sharding (TP only)")

        self.model = model

        # Create optimizer
        # Note: Use foreach=False because DTensor doesn't support fused optimizer ops
        learning_rate = config.get("learning_rate", 1e-5)
        weight_decay = config.get("weight_decay", 0.01)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=False,  # Required for DTensor compatibility
        )

        if self.rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"[FSDP2+DTensor] Model initialized with {num_params:,} parameters")
            if self.dp_size > 1:
                print(f"[FSDP2+DTensor] 2D parallelism: {self.dp_size} DP x {self.tp_size} TP")

        # Always enable autocast for mixed precision dtypes (bf16/fp16)
        self.use_autocast = dtype in (torch.bfloat16, torch.float16)
        if self.use_autocast:
            self.autocast_dtype = dtype
            if self.rank == 0:
                dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
                print(f"[FSDP2+DTensor] torch.autocast enabled with dtype={dtype_name}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass with FSDP2+DTensor model."""
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.use_autocast
            else nullcontext()
        )

        with autocast_ctx:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
        return outputs.loss

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass."""
        loss.backward()

    def forward_backward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run forward and backward pass together.

        Returns:
            The loss value (for logging)
        """
        loss = self.forward(batch)
        loss.backward()
        return loss

    def optimizer_step(self) -> None:
        """Run optimizer step."""
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=True)

    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def save_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        """
        Save checkpoint for FSDP2+DTensor model.

        For FSDP2, we use torch.save on the model state dict.
        Each rank saves its own shard.
        """
        import os

        # Create tag directory
        tag_dir = os.path.join(checkpoint_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        # Save model state dict (each rank saves its shard)
        model_path = os.path.join(tag_dir, f"model_rank{self.rank}.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer state dict
        optim_path = os.path.join(tag_dir, f"optimizer_rank{self.rank}.pt")
        torch.save(self.optimizer.state_dict(), optim_path)

    def load_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        """
        Load checkpoint for FSDP2+DTensor model.

        Each rank loads its corresponding shard.
        """
        import os

        tag_dir = os.path.join(checkpoint_dir, tag)

        # Load model state dict
        model_path = os.path.join(tag_dir, f"model_rank{self.rank}.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

        # Load optimizer state dict
        optim_path = os.path.join(tag_dir, f"optimizer_rank{self.rank}.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(torch.load(optim_path, weights_only=True))

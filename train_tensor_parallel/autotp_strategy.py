"""DeepSpeed AutoTP tensor parallelism strategy adapted for Ray Train."""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from model_builder import (
    ModelConfig,
    create_model,
    get_model_config,
)


class RayAutoTPStrategy:
    """
    DeepSpeed AutoTP tensor parallelism strategy adapted for Ray Train.

    Uses DeepSpeed's automatic tensor parallelism via:
    - set_autotp_mode() for instrumentation
    - deepspeed.tp_model_init() for automatic model sharding
    - DeepSpeed engine for training with ZeRO optimization

    When dp_size > 1, creates a 2D parallelism configuration:
    - TP groups (same row): processes that share tensor-parallel shards
    - DP groups (same column): processes that handle different data batches

    For example, with 8 GPUs, dp_size=2, tp_size=4:
    - Device arrangement: [[0,1,2,3], [4,5,6,7]]
    - TP groups: {0,1,2,3}, {4,5,6,7}
    - DP groups: {0,4}, {1,5}, {2,6}, {3,7}
    """

    def __init__(self, tp_size: int, dp_size: int = 1):
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.engine = None
        self.optimizer = None
        self.model = None
        self.tp_group = None
        self.dp_group = None
        self.tp_rank = None
        self.dp_rank = None
        self.rank = None
        self.world_size = None
        self.device = None
        self._model_config: Optional[ModelConfig] = None

    def _create_process_groups(self) -> None:
        """
        Create TP and DP process groups for 2D parallelism.

        Organizes GPUs into a 2D grid where:
        - Rows are TP groups (processes that shard model weights)
        - Columns are DP groups (processes that handle different data)

        Example with 8 GPUs (dp_size=2, tp_size=4):
          GPU layout: [[0, 1, 2, 3],
                       [4, 5, 6, 7]]
          TP groups: {0,1,2,3}, {4,5,6,7}
          DP groups: {0,4}, {1,5}, {2,6}, {3,7}
        """
        # Calculate which TP and DP group this rank belongs to
        self.tp_rank = self.rank % self.tp_size  # Position within TP group
        self.dp_rank = self.rank // self.tp_size  # Which DP replica

        # Create TP groups (processes in the same row)
        # Each TP group contains tp_size consecutive ranks
        tp_groups: List[List[int]] = []
        for dp_idx in range(self.dp_size):
            tp_group_ranks = list(range(dp_idx * self.tp_size, (dp_idx + 1) * self.tp_size))
            tp_groups.append(tp_group_ranks)
            group = dist.new_group(tp_group_ranks)
            if self.rank in tp_group_ranks:
                self.tp_group = group

        # Create DP groups (processes in the same column)
        # Each DP group contains ranks at the same position across TP groups
        dp_groups: List[List[int]] = []
        for tp_idx in range(self.tp_size):
            dp_group_ranks = [tp_idx + dp_idx * self.tp_size for dp_idx in range(self.dp_size)]
            dp_groups.append(dp_group_ranks)
            group = dist.new_group(dp_group_ranks)
            if self.rank in dp_group_ranks:
                self.dp_group = group

        if self.rank == 0:
            print(f"[AutoTP] TP groups: {tp_groups}")
            print(f"[AutoTP] DP groups: {dp_groups}")
            print(f"[AutoTP] Rank {self.rank}: tp_rank={self.tp_rank}, dp_rank={self.dp_rank}")

    def _sync_model_weights(self, model: nn.Module) -> None:
        """
        Synchronize all model weights within each TP group.

        This is necessary because each rank creates the model with different
        random initializations, but DeepSpeed tp_model_init() expects all
        ranks to have identical weights before sharding.

        With 2D parallelism (DP x TP), we have multiple TP groups:
        - TP group 0: global ranks [0, 1, ..., tp_size-1]
        - TP group 1: global ranks [tp_size, tp_size+1, ..., 2*tp_size-1]
        - etc.

        Each TP group broadcasts from its first member (tp_rank=0).

        Args:
            model: The model to synchronize
        """
        if self.tp_group is None or self.tp_size <= 1:
            return

        # The source rank for broadcast is the first rank in this TP group
        # With 2D layout: TP group i contains ranks [i*tp_size, (i+1)*tp_size)
        # So the first rank in this TP group is dp_rank * tp_size
        tp_group_first_rank = self.dp_rank * self.tp_size

        if self.rank == 0:
            print(f"[AutoTP] Synchronizing model weights within each TP group...")

        with torch.no_grad():
            for name, param in model.named_parameters():
                dist.broadcast(param.data, src=tp_group_first_rank, group=self.tp_group)

        if self.tp_rank == 0:
            print(f"[AutoTP] Model weights synchronized in TP group (src_rank={tp_group_first_rank})")

    def _create_mpu(self) -> "ModelParallelUnit":
        """
        Create a Model Parallel Unit (MPU) for DeepSpeed.

        DeepSpeed uses the MPU to understand the parallelism topology and
        perform gradient all-reduce only across the data parallel group.
        """
        return ModelParallelUnit(
            tp_group=self.tp_group,
            dp_group=self.dp_group,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
        )

    def setup(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
    ) -> None:
        """
        Set up DeepSpeed AutoTP with optional data parallelism.

        Args:
            model_name: HuggingFace model name
            device: Target device
            dtype: Target dtype (bfloat16, float16, float32)
            config: Training configuration dict
        """
        import deepspeed
        from deepspeed.module_inject.layers import set_autotp_mode

        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._model_config = get_model_config(model_name)

        # Validate 2D configuration
        if self.dp_size * self.tp_size != self.world_size:
            raise ValueError(
                f"dp_size ({self.dp_size}) * tp_size ({self.tp_size}) must equal "
                f"world_size ({self.world_size})"
            )

        # Validate model configuration
        if self._model_config.num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                f"TP size {self.tp_size} must divide num_key_value_heads "
                f"{self._model_config.num_key_value_heads}"
            )

        if self.rank == 0:
            print(f"[AutoTP] Setting up with dp_size={self.dp_size}, tp_size={self.tp_size}")

        # Create TP and DP process groups for 2D parallelism
        self._create_process_groups()

        # Enable AutoTP instrumentation BEFORE model creation
        set_autotp_mode(training=True)

        # Create model on actual device with proper initialization
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
                print("[AutoTP] Enabled activation checkpointing")

        # Handle initial weights loading for verification
        # This must happen BEFORE weight synchronization so all ranks load the same weights
        init_weights_path = config.get("init_weights_path")
        if init_weights_path and os.path.exists(init_weights_path):
            state_dict = torch.load(init_weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            if self.rank == 0:
                print(f"[AutoTP] Loaded initial weights from {init_weights_path}")
            # Skip weight synchronization since all ranks now have identical weights
            skip_sync = True
        else:
            skip_sync = False

        # CRITICAL: Broadcast ALL model weights from rank 0 to all TP ranks before sharding.
        # DeepSpeed tp_model_init() shards weights but doesn't synchronize them across ranks.
        # Without this, each rank would have different random initializations, leading to
        # incorrect model behavior after sharding.
        # Skip if we loaded weights from checkpoint (all ranks already have identical weights)
        if not skip_sync:
            self._sync_model_weights(model)

        # Apply TP sharding with deepspeed.tp_model_init()
        # Pass the TP group so DeepSpeed knows which ranks share model shards
        model = deepspeed.tp_model_init(
            model,
            tp_size=self.tp_size,
            dtype=dtype,
            tp_group=self.tp_group,
        )

        # Get all parameters
        params = list(model.parameters())

        # Build DeepSpeed config
        zero_stage = config.get("zero_stage", 1)
        batch_size = config.get("batch_size", 1)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # train_batch_size calculation:
        # When MPU is provided (tp_size > 1): DeepSpeed uses mpu.get_data_parallel_world_size()
        effective_dp = self.dp_size if self.tp_size > 1 else self.world_size
        ds_config = {
            "train_batch_size": batch_size * effective_dp * gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": config.get("max_grad_norm", 1.0),
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
            },
            "tensor_parallel": {
                "autotp_size": self.tp_size,
            },
            "data_parallel_size": self.dp_size,
            "zero_allow_untested_optimizer": True,
            "steps_per_print": 2000,
            "wall_clock_breakdown": False,
        }

        # Add precision config
        if dtype == torch.bfloat16:
            ds_config["bf16"] = {
                "enabled": True,
                "bf16_master_weights_and_grads": True,
                "bf16_optimizer_states": True,
            }
            # Enable PyTorch native autocast for bfloat16
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "bfloat16",
            }
        elif dtype == torch.float16:
            ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
            # Enable PyTorch native autocast for float16
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "float16",
            }

        # Create optimizer
        learning_rate = config.get("learning_rate", 1e-5)
        weight_decay = config.get("weight_decay", 0.01)
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

        # Initialize DeepSpeed engine
        # Pass the MPU so DeepSpeed knows the parallelism topology
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
            mpu=self._create_mpu() if self.tp_size > 1 else None,
        )

        self.model = self.engine.module

        if self.rank == 0:
            num_params = sum(p.numel() for p in params)
            print(f"[AutoTP] Model initialized with {num_params:,} parameters")
            print(f"[AutoTP] DeepSpeed engine created with ZeRO-{zero_stage}")
            if self.dp_size > 1:
                print(f"[AutoTP] 2D parallelism: {self.dp_size} DP x {self.tp_size} TP")
            if "torch_autocast" in ds_config:
                print(f"[AutoTP] torch.autocast enabled with dtype={ds_config['torch_autocast']['dtype']}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass with AutoTP model."""
        outputs = self.engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        return outputs.loss

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass through DeepSpeed engine."""
        self.engine.backward(loss)

    def optimizer_step(self) -> None:
        """Run optimizer step through DeepSpeed engine."""
        self.engine.step()

    def zero_grad(self) -> None:
        """Zero gradients - handled internally by DeepSpeed."""
        pass

    def train(self) -> None:
        """Set model to training mode."""
        self.engine.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.engine.eval()

    def save_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        """
        Save checkpoint using DeepSpeed.

        Args:
            checkpoint_dir: Directory to save checkpoint
            tag: Checkpoint tag/name
        """
        self.engine.save_checkpoint(checkpoint_dir, tag=tag)

    def load_checkpoint(self, checkpoint_dir: str, tag: str = "model") -> None:
        """
        Load checkpoint using DeepSpeed.

        Args:
            checkpoint_dir: Directory containing checkpoint
            tag: Checkpoint tag/name
        """
        self.engine.load_checkpoint(checkpoint_dir, tag=tag)


class ModelParallelUnit:
    """
    Model Parallel Unit (MPU) interface for DeepSpeed.

    This class implements the interface DeepSpeed expects for understanding
    the parallelism topology when using custom TP+DP configurations.

    DeepSpeed calls methods like get_data_parallel_group() to know which
    ranks should participate in gradient all-reduce operations.
    """

    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        dp_group: dist.ProcessGroup,
        tp_size: int,
        dp_size: int,
        tp_rank: int,
        dp_rank: int,
    ):
        self._tp_group = tp_group
        self._dp_group = dp_group
        self._tp_size = tp_size
        self._dp_size = dp_size
        self._tp_rank = tp_rank
        self._dp_rank = dp_rank

    def get_data_parallel_group(self) -> dist.ProcessGroup:
        """Return the data parallel process group."""
        return self._dp_group

    def get_model_parallel_group(self) -> dist.ProcessGroup:
        """Return the model/tensor parallel process group."""
        return self._tp_group

    def get_data_parallel_world_size(self) -> int:
        """Return the number of data parallel ranks."""
        return self._dp_size

    def get_model_parallel_world_size(self) -> int:
        """Return the number of model/tensor parallel ranks."""
        return self._tp_size

    def get_data_parallel_rank(self) -> int:
        """Return this rank's position in the data parallel group."""
        return self._dp_rank

    def get_model_parallel_rank(self) -> int:
        """Return this rank's position in the model/tensor parallel group."""
        return self._tp_rank

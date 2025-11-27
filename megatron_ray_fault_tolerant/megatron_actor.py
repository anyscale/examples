import logging
import os
import random
import socket
from dataclasses import asdict
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist
import ray
from ray import ObjectRef
from ray.util.placement_group import (
    PlacementGroup,
    PlacementGroupSchedulingStrategy,
    placement_group_table,
)
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from loguru import logger

# megatron
from megatron.bridge import AutoBridge
import megatron.core.parallel_state as mpu
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.strategies import base as ckpt_base
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)

# local imports
import file_io as io  # local io module to support cloud storage for checkpointing
from training_batch import TrainingOutputBatch
from optimizer import (
    init_megatron_optim_config,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from megatron_model_wrapper import MegatronModelWrapper
from megatron_utils import (
    offload_megatron_model_to_cpu,
    snapshot_optimizer_state_cpu,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    apply_optimizer_state_snapshot,
    offload_megatron_grads_to_cpu,
    load_megatron_grads_to_gpu,
)
from utils import BatchIterator
from dispatch import DispatchRegistry, Dispatch, ActorInfo, MeshRank


@ray.remote(num_gpus=1)
class MegatronActor:
    def __init__(
        self,
        world_size,
        rank,
        local_rank,
        master_addr,
        master_port,
        megatron_config,
        seed,
        cfg,
    ):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        os.environ["LOCAL_RANK"] = "0"
        self.megatron_config = megatron_config
        self.seed = seed
        self.cfg = cfg

    def get_node_local_rank(self):
        return self._local_rank

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def init_worker_process_group(self):
        """Initialize worker process group and megatron model parallel."""
        # Destroy any existing process group first to ensure clean state
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass  # Ignore errors if already destroyed

        # Initialize process group using environment variables
        torch.distributed.init_process_group(backend="nccl")

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            expert_model_parallel_size=self.megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.megatron_config.expert_tensor_parallel_size,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
            order="tp-pp-dp",
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()
        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def get_mesh_rank(self):
        return self.mesh_rank

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]

    def print(self, *msg):
        """Print only on rank 0"""
        if dist.get_rank() == 0:
            logger.info(*msg)

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    def get_ray_node_id(self):
        return ray.get_runtime_context().get_node_id()

    @staticmethod
    def get_rng_state():
        """Get current RNG state for reproducibility"""
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        # Only save CUDA RNG state if CUDA is available and being used
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            rng_state["cuda"] = torch.cuda.get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        """Load RNG state for reproducibility"""
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        # Only restore CUDA RNG state if it was saved and CUDA is available
        if (
            "cuda" in rng_state
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
        ):
            torch.cuda.set_rng_state(rng_state["cuda"])

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def destroy_worker_process_group(self):
        mpu.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Clear stale env vars
        for env_var in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]:
            if env_var in os.environ:
                del os.environ[env_var]

    def reinit_model_after_recovery(self):
        """Re-initialize model and optimizer after process group recovery.

        This is needed because the model and optimizer were created with the old
        process group and still have references to old NCCL communicators.

        We need to fully reinitialize the provider and model to ensure they use
        the new process group.
        """
        if not hasattr(self, "_model_path") or self._model_path is None:
            # Fall back to cfg.model if _model_path not set
            if hasattr(self.cfg, "model"):
                model_path = self.cfg.model
            else:
                logger.warning("No model path found, cannot re-initialize model")
                return
        else:
            model_path = self._model_path

        num_training_steps = getattr(self, "_num_training_steps", 1e9)

        logger.info("Re-initializing model components after process group recovery...")

        # Re-initialize the bridge and provider with the new process group
        # This ensures all NCCL communicators are created fresh
        self.init_configs(
            model_path,
            megatron_config=self.cfg.megatron_config,
            transformer_config=self.cfg.megatron_config.transformer_config,
            bf16=True,
            flash_attn=True,
        )

        # Recreate the DDP-wrapped module with the new process group
        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=True,
            ddp_config=asdict(self.cfg.megatron_config.ddp_config),
            bf16=True,
        )

        # Recreate optimizer with the new process group
        optim_config = init_megatron_optim_config(
            asdict(self.cfg.megatron_config.optimizer_config)
        )
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        # Recreate scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=asdict(self.cfg.megatron_config.optimizer_config),
            num_training_steps=num_training_steps,
        )

        # Recreate model wrapper
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
        )

        # Re-normalize mini batch size with new world size
        self._normalize_mini_batch_size()

        logger.info("Model components re-initialized successfully")

    def update_world_size(self, new_world_size: int):
        """Update the world_size stored in the actor."""
        self._world_size = new_world_size
        os.environ["WORLD_SIZE"] = str(new_world_size)

    def update_rank(self, new_rank: int):
        """Update the rank stored in the actor."""
        self._rank = new_rank
        os.environ["RANK"] = str(new_rank)

    def update_master_addr_port(self, master_addr: str, master_port: int):
        """Update the master address and port for process group initialization."""
        self._master_addr = master_addr
        self._master_port = master_port
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

    def _normalize_mini_batch_size(self):
        """
        Normalize mini batch sizes to per-gpu mini batch sizes.
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError(
                "mesh_rank must be initialized before calling _normalize_mini_batch_size()"
            )

        dp_size = self.mesh_rank.dp_size
        self.policy_mini_batch_size_per_gpu = self.cfg.mini_batch_size // dp_size

    def ppo_train(self, train_data) -> "TrainingOutputBatch":
        """
        Overrides `PolicyWorkerBase.ppo_train` for megatron.

        Since we want megatron to handle gradient accumulation over micro batches, we directly pass mini batches into the
        worker MegatronModelWrapper.forward_backward_mini_batch method.
        """
        dataloader = BatchIterator(
            train_data,
            sample_batch_size=self.cfg.micro_train_batch_size_per_gpu,
            drop_last=False,
        )

        micro_batches_per_mini_batch = (
            self.policy_mini_batch_size_per_gpu
            // self.cfg.micro_train_batch_size_per_gpu
        )

        self.optimizer.zero_grad()
        pbar = tqdm(
            dataloader,
            desc="ppo train",
            disable=not dist.get_rank() == 0,
        )

        micro_buffer = []
        all_metrics = []
        for local_step, experience in enumerate(pbar):
            experience.to_device(torch.cuda.current_device())
            sequences = experience.sequences
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            micro_buffer.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "old_action_log_probs": experience.action_log_probs,
                    "base_action_log_probs": experience.base_action_log_probs,
                    "advantages": experience.advantages,
                    "loss_mask": experience.loss_mask,
                    "rollout_action_logprobs": experience.rollout_logprobs,
                }
            )

            if len(micro_buffer) == micro_batches_per_mini_batch:
                # run mini-batch forward-backward and then one optimizer step
                self.model.train()
                for chunk in self.actor_module:
                    # if use distributed optimizer, zero grad buffer will be handled by optimizer
                    chunk.zero_grad_buffer()
                seq_len = micro_buffer[0]["sequences"].shape[1]
                micro_bsz = micro_buffer[0]["sequences"].shape[0]

                metrics = self.model.forward_backward_mini_batch(
                    micro_batches=micro_buffer,
                    seq_len=seq_len,
                    micro_batch_size=micro_bsz,
                )
                all_metrics.extend(metrics)
                _, grad_norm, _ = self.optimizer.step()
                self.scheduler.step(1)
                self.optimizer.zero_grad()

        torch.distributed.barrier()

        return all_metrics

    def save_checkpoint(self, ckpt_dir: str):
        # Extract base model.
        model: List[nn.Module] = self.model.actor_module
        optimizer = self.optimizer
        scheduler = self.scheduler
        node_local_rank = self.get_node_local_rank()
        assert (
            len(model) == 1
        ), "Megatron virtual pipeline parallel is not yet supported"
        model = model[0]
        if hasattr(model, "module"):
            model = model.module

        # Create checkpoint directory if it doesn't exist.
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # All ranks wait for the checkpoint directory to be created before saving.
        dist.barrier()

        # Collect the sharded state dicts for model and optimizer, and full state dict for the scheduler.
        sharded_state_dict = {}
        model_sharded_state_dict = model.sharded_state_dict()
        sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer:
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict
            )
        if scheduler:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        # Save RNG state.
        sharded_state_dict["rng"] = self.get_rng_state()

        # Save the checkpoint across ranks in parallel.
        save_strategy = get_default_save_sharded_strategy("torch_dist")
        save_strategy = FullyParallelSaveStrategyWrapper(save_strategy)

        with io.local_work_dir(ckpt_dir) as work_dir:
            # synchronous checkpointing for now
            async_save_request = dist_checkpointing.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=work_dir,
                sharded_strategy=save_strategy,
                async_sharded_save=False,
                validate_access_integrity=True,
            )
            assert (
                async_save_request is None
            ), "Async save is not yet supported for Megatron"

        dist.barrier()
        ckpt_base.async_calls.close()
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)
        self.print(f"Checkpoint successfully saved to {ckpt_dir}")

    def load_checkpoint(
        self,
        ckpt_dir: str,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ):
        if not ckpt_dir or not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract base model.
        model: List[nn.Module] = self.model.actor_module
        optimizer = self.optimizer
        scheduler = self.scheduler
        assert (
            len(model) == 1
        ), "Megatron virtual pipeline parallel is not yet supported"
        unwrapped_model = model[0]
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Extract sharded state dicts.
        sharded_state_dict = {}
        model_sharded_state_dict = unwrapped_model.sharded_state_dict()
        sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer and load_optimizer_states:
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict
            )
        if scheduler and load_lr_scheduler_states:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        prefixes=[f"__{self._rank}_", ".metadata", "common.pt", "metadata.json"]

        # currently, if the ckpt_dir is a cloud path, we download all the contents of the cloud path to a local directory
        # this should be improved to download only the relevant shards for this actor to load
        with io.local_read_dir(ckpt_dir, local_path=self.cfg.local_ckpt_dir, prefixes=prefixes) as read_dir:
            dist.barrier()
            # Load the checkpoint in parallel.
            load_strategy = get_default_load_sharded_strategy(read_dir)
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy)
            state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=read_dir,
                sharded_strategy=load_strategy,
                strict="assume_ok_unexpected",
            )

        # Load the model, optimizer, and scheduler state dicts.
        assert (
            "model" in state_dict
        ), f"Model state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
        model[0].load_state_dict(state_dict["model"], strict=load_module_strict)
        self.print("Loaded model state dict.")

        if optimizer and load_optimizer_states:
            assert (
                "optimizer" in state_dict
            ), f"Optimizer state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            optimizer.load_state_dict(state_dict["optimizer"])
            self.print("Loaded optimizer state dict.")

        if scheduler and load_lr_scheduler_states:
            assert (
                "lr_scheduler" in state_dict
            ), f"LR scheduler state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            scheduler.load_state_dict(state_dict["lr_scheduler"])
            self.print("Loaded LR scheduler state dict.")

        # Load RNG state, if present.
        if "rng" in state_dict:
            self.load_rng_state(state_dict["rng"])

        return ckpt_dir, {}

    def offload_to_cpu(self):
        self.all_buffer_sizes = offload_megatron_grads_to_cpu(self.actor_module)
        self.all_model_buffers_param_data, self.all_model_buffers_param_data_sizes = offload_megatron_model_to_cpu(self.actor_module)
        self.all_optimizer_state_dict = snapshot_optimizer_state_cpu(self.optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self):
        load_megatron_grads_to_gpu(self.actor_module, self.all_buffer_sizes)
        load_megatron_model_to_gpu(self.actor_module, self.all_model_buffers_param_data, self.all_model_buffers_param_data_sizes)
        apply_optimizer_state_snapshot(self.optimizer, self.all_optimizer_state_dict)
        load_megatron_optimizer(self.optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # model init and bridge from huggingface methods:
    def init_configs(
        self,
        model_path,
        megatron_config,
        transformer_config,
        bf16=True,
        flash_attn=True,
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # if flash_attn is enabled, we use flash attention backend, otherwise fall back to fused attention backend
        transformer_config = asdict(transformer_config)
        transformer_config["attention_backend"] = "flash" if flash_attn else "fused"

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        provider = bridge.to_megatron_provider()
        provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = (
            megatron_config.pipeline_model_parallel_size
        )
        provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
        provider.context_parallel_size = megatron_config.context_parallel_size
        provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = (
            megatron_config.expert_tensor_parallel_size
        )
        provider.sequence_parallel = megatron_config.tensor_model_parallel_size > 1
        provider.attention_backend = "flash" if flash_attn else "fused"
        provider.variable_seq_lengths = True
        provider.masked_softmax_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"

        for k, v in transformer_config.items():
            setattr(provider, k, v)
        provider.finalize()

        self.provider = provider
        self.bridge = bridge
        self.tokenizer = tokenizer

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Dict[str, Any]] = None,
        bf16: bool = True,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import (
            DistributedDataParallelConfig,
        )

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in ddp_config.items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        return model

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model, optimizer, and scheduler for the policy worker.
        """
        # Store model path for potential recovery
        self._model_path = model_path
        self._num_training_steps = num_training_steps

        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            megatron_config=self.cfg.megatron_config,
            transformer_config=self.cfg.megatron_config.transformer_config,
            bf16=True,
            flash_attn=True,
        )

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=True,
            ddp_config=asdict(self.cfg.megatron_config.ddp_config),
            bf16=True,
        )

        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        # create optimizer
        optim_config = init_megatron_optim_config(
            asdict(self.cfg.megatron_config.optimizer_config)
        )
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        self._normalize_mini_batch_size()

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=asdict(self.cfg.megatron_config.optimizer_config),
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
        )

        # NOTE: Set Megatron dist checkpoint async backend to persistent to avoid `os.fork()`-ing
        # short-lived background workers, which does not work well with Ray.
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)


class MegatronActorGroup:
    """
    A group of distributed megatron actors
    Functions start with 'async' should return list of object refs

    Args:
        cfg: config object for workers
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        cfg,
        num_nodes,
        num_gpus_per_node,
        pg: PlacementGroup,
        bundle_indices: List[int],
        num_gpus_per_actor: float = 1.0,
        resources: Optional[Dict[str, float]] = None,
        num_resources_per_node: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor, bundle_indices)

    def _initiate_actors(
        self,
        pg: Optional[PlacementGroup],
        num_gpus_per_actor: float,
        bundle_indices: List[int],
    ):
        """Initialize Ray actors in the worker group.

        Args:
            pg: The placement group for the worker group
            num_gpus_per_actor: The number of gpus to allocate per actor.
        """
        world_size = self._num_nodes * self._num_gpus_per_node
        assert pg is not None, "placement group must be provided to MegatronActorGroup"
        pg_data = placement_group_table(pg)
        assert (
            len(pg_data["bundles"]) >= world_size
        ), "the number of bundles in the shared placement group must be greater than or equal to the world size"

        # place master actor on the
        master_actor = MegatronActor.options(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            resources=self._resources,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_indices[0],
            ),
        ).remote(
            world_size=world_size,
            rank=0,
            local_rank=0,
            master_addr=None,
            master_port=None,
            megatron_config=self.cfg.megatron_config,
            seed=42,
            cfg=self.cfg,
        )

        self._actor_handlers = [master_actor]
        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(
                master_actor.get_master_addr_port.remote()
            )
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node

                worker_actor = MegatronActor.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    resources=self._resources,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_indices[rank],
                    ),
                ).remote(
                    world_size=world_size,
                    rank=rank,
                    local_rank=local_rank,
                    master_addr=master_addr,
                    master_port=master_port,
                    megatron_config=self.cfg.megatron_config,
                    seed=42,
                    cfg=self.cfg,
                )
                self._actor_handlers.append(worker_actor)

    def initiate_worker_process_group(self):
        # Initialize process group
        logger.info("Initializing process group for RayActorGroup")
        ray.get(
            [actor.init_worker_process_group.remote() for actor in self._actor_handlers]
        )
        logger.info("Initialized process group for RayActorGroup")
        self.actor_infos = [
            ActorInfo(actor, ray.get(actor.get_mesh_rank.remote()))
            for actor in self._actor_handlers
        ]
        logger.info(
            f"Mesh Ranks: {[actor_info.rank for actor_info in self.actor_infos]}"
        )

    def async_init_model(
        self,
        *args,
        **kwargs,
    ) -> List[ObjectRef]:
        """Asynchronously initialize worker state (model, and optimizer if applicable) from model path on all the workers.

        Returns:
            A list of ray object refs.
        """
        return [
            actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers
        ]

    def async_run_ray_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> List[ObjectRef]:
        """Run a method on all actors using specified dispatch type asynchronously.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            List of object references
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(
            self.actor_infos, method_name, *args, **kwargs
        )
        return object_refs

    def async_run_method_no_dispatch(
        self, method_name: str, actor_ids: List[int] = None, *args, **kwargs
    ) -> List[ObjectRef]:
        """Run a method on all actors without dispatching."""
        if actor_ids is None:
            return [
                getattr(handle, method_name).remote(*args, **kwargs)
                for handle in self._actor_handlers
            ]
        else:
            object_refs = []
            for i, handle in enumerate(self._actor_handlers):
                if i in actor_ids:
                    object_refs.append(getattr(handle, method_name).remote(*args, **kwargs))
            return object_refs

    def _check_actor_alive(self, actor_handle) -> bool:
        """Check if an actor is still alive by attempting to call a simple method."""
        try:
            # Try to get a simple attribute or call a simple method with timeout
            ray.get(actor_handle.get_mesh_rank.remote(), timeout=10)
            return True
        except Exception:
            return False

    def recover_from_failure(
        self, backup_actor_group: Optional["MegatronActorGroup"] = None
    ):
        """Recover from actor failures by removing dead actors and re-initializing process group."""
        logger.info("Starting recovery from actor failure...")

        # Filter out dead actors - both actor_infos and actor_handlers should be in sync
        alive_actor_handlers = []
        num_dead_actors = 0
        dead_actor_ranks = []

        for i, (actor_info, actor_handle) in enumerate(
            zip(self.actor_infos, self._actor_handlers)
        ):
            if self._check_actor_alive(actor_info.handle):
                alive_actor_handlers.append(actor_handle)
            else:
                logger.warning(f"Actor {i} is dead, removing from group")
                num_dead_actors += 1
                dead_actor_ranks.append(i)

        if len(alive_actor_handlers) == 0:
            raise RuntimeError("All actors are dead, cannot recover")

        if len(alive_actor_handlers) == len(self._actor_handlers):
            logger.info("All actors are alive, no recovery needed")
            return

        logger.info(
            f"Recovering with {len(alive_actor_handlers)}/{len(self._actor_handlers)} actors"
        )

        self._actor_handlers = alive_actor_handlers

        # Destroy existing process groups on alive actors first
        logger.info("Destroying old process groups...")
        try:
            ray.get(
                [
                    actor.destroy_worker_process_group.remote()
                    for actor in self._actor_handlers
                ]
            )
        except Exception as e:
            logger.warning(
                f"Some errors during process group destruction (may be expected): {e}"
            )

        # if backup actor group is provided, we pop idle actors from the backup actor group and insert them into the current actor group
        if backup_actor_group is not None:
            logger.info(
                f"Popping {num_dead_actors} idle actors from backup actor group"
            )
            idle_actor_handles = [
                backup_actor_group._actor_handlers.pop() for _ in range(num_dead_actors)
            ]
            # let's assume for now that the dead actors are contiguous in the actor group, so we insert the idle actors at the rank of the first dead actor
            rank_to_insert = min(dead_actor_ranks)
            logger.info(f"Inserting idle actors at rank {rank_to_insert}")
            self._actor_handlers = (
                self._actor_handlers[:rank_to_insert]
                + idle_actor_handles
                + self._actor_handlers[rank_to_insert:]
            )

        # Re-initialize process group with remaining actors
        # Update world_size and ranks to match the number of alive actors
        new_world_size = len(self._actor_handlers)

        # Update world_size and reassign ranks sequentially (0, 1, 2, ...)
        logger.info(f"Updating world_size to {new_world_size} and reassigning ranks...")
        update_tasks = []
        for new_rank, actor in enumerate(self._actor_handlers):
            update_tasks.append(actor.update_world_size.remote(new_world_size))
            update_tasks.append(actor.update_rank.remote(new_rank))
        ray.get(update_tasks)

        # get master address and a new free port for the new process group
        master_addr, _ = ray.get(self._actor_handlers[0].get_master_addr_port.remote())
        master_port = ray.get(self._actor_handlers[0]._get_free_port.remote())
        logger.info(f"Using master_addr={master_addr}, master_port={master_port}")

        # Update master address/port in all actors
        ray.get(
            [
                actor.update_master_addr_port.remote(master_addr, master_port)
                for actor in self._actor_handlers
            ]
        )

        # Re-initialize process groups with new world_size and ranks
        logger.info(
            f"Re-initializing process group with world_size={new_world_size}..."
        )
        ray.get(
            [actor.init_worker_process_group.remote() for actor in self._actor_handlers]
        )

        # Re-initialize model and optimizer with the new process group
        # This is critical because they were created with the old process group
        logger.info("Re-initializing model and optimizer with new process group...")
        ray.get(
            [
                actor.reinit_model_after_recovery.remote()
                for actor in self._actor_handlers
            ]
        )

        # Update actor_infos with new mesh ranks
        self.actor_infos = [
            ActorInfo(actor, ray.get(actor.get_mesh_rank.remote()))
            for actor in self._actor_handlers
        ]
        logger.info(
            f"Recovery complete. New mesh ranks: {[actor_info.rank for actor_info in self.actor_infos]}"
        )

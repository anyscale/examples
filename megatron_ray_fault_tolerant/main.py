import os
from dataclasses import dataclass, field
import ray
from typing import Optional, List
from megatron_actor import MegatronActorGroup
from ray.util.placement_group import placement_group
from ray.runtime_env import RuntimeEnv, RuntimeEnvConfig

import random
import time
from utils import get_test_training_batch, get_reordered_bundle_indices


@dataclass
class DDPConfig:
    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    average_in_collective: bool = True


@dataclass
class OptimizerConfig:
    lr: float = 1.0e-6
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    offload_after_step: bool = True
    num_warmup_steps: int = 0
    scheduler: str = "constant_with_warmup"


@dataclass
class TransformerConfig:
    recompute_granularity: Optional[str] = None
    recompute_modules: List[str] = field(default_factory=lambda: ["core_attn"])
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None


@dataclass
class MegatronConfig:
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = None
    ddp_config: DDPConfig = field(default_factory=DDPConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class Config:
    model: str = "Qwen/Qwen3-4B"
    # TODO: test on actually more than 2 nodes for recovery, where we just want to ditch a whole node and replace it
    num_nodes: int = 8
    num_gpus_per_node: int = 4
    mini_batch_size: int = 16
    num_spare_gpus: int = 4
    micro_train_batch_size_per_gpu: int = 2
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    ckpt_dir: str = (
        os.environ["ANYSCALE_ARTIFACT_STORAGE"] + "/megatron_fault_tolerance/ckpt3/"
    )
    # algorithm config
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    clip_ratio_c: float = 3.0


def main():
    config = Config()
    # create placement group including spare gpus

    pg = placement_group(
        [{"GPU": 1, "CPU": 12}] * config.num_nodes * config.num_gpus_per_node
        + [{"GPU": 1, "CPU": 12}] * config.num_spare_gpus,
        strategy="PACK",
    )

    ray.get(pg.ready(), timeout=1200)
    print("Placement group ready")
    # this is needed because placement group gpu bundle order is not deterministic: https://github.com/ray-project/ray/issues/51117
    reordered_bundle_indices = get_reordered_bundle_indices(pg)

    actor_group = MegatronActorGroup(
        cfg=config,
        num_nodes=config.num_nodes,
        num_gpus_per_node=config.num_gpus_per_node,
        pg=pg,
        bundle_indices=reordered_bundle_indices[: -config.num_spare_gpus],
    )
    actor_group.initiate_worker_process_group()
    ray.get(actor_group.async_init_model(config.model))

    # potentially need some time for dependencies like transformer-engine-torch to build on worker nodes (this is something good to warm start...)
    backup_actor_group = MegatronActorGroup(
        cfg=config,
        num_nodes=config.num_spare_gpus // config.num_gpus_per_node,
        num_gpus_per_node=config.num_gpus_per_node,
        pg=pg,
        bundle_indices=reordered_bundle_indices[-config.num_spare_gpus :],
    )
    # just place but don't initiate the worker process group for the backup actor group
    # call a function to make sure the actors are placed
    ray.get(backup_actor_group.async_run_method_no_dispatch("get_gpu_id"))

    # train on one batch
    batch = get_test_training_batch(config.model, batch_size=32)
    print("Starting training step 1...")
    start_time = time.time()
    ray.get(actor_group.async_run_ray_method("mesh", "ppo_train", batch))
    print(f"Training step 1 took {time.time() - start_time:.2f} seconds")

    # save checkpoint
    start_time = time.time()
    ray.get(
        actor_group.async_run_ray_method(
            "pass_through", "save_checkpoint", ckpt_dir=config.ckpt_dir
        )
    )
    print(f"Checkpoint saving took {time.time() - start_time:.2f} seconds")

    # TODO: add a cpu offload (or cpu save memory) call here
    # in order for the healthy actors to save a copy of the model and optimizer state to cpu memory
    # ray.get(actor_group.async_run_ray_method("pass_through", "offload_to_cpu"))

    # TODO: run another training batch here and save results but don't save checkpoint

    # randomly kill an actor to simulate fault tolerance scenario
    # TODO: go deeper into the actor code and throw an exception on a given node and catch it here
    print("Simulating failure and recovery...")
    start_time = time.time()

    actor_id = random.randint(0, len(actor_group.actor_infos) - 1)
    # get the whole dp group associated with the failed actor
    dp_group_actors = []
    for actor_info in actor_group.actor_infos:
        if actor_info.rank.dp == actor_group.actor_infos[actor_id].rank.dp:
            dp_group_actors.append(actor_info)
    print(
        f"Killing actors {[actor_info.rank for actor_info in dp_group_actors]} to simulate failure..."
    )
    for actor_info in dp_group_actors:
        ray.kill(actor_info.handle)

    # Destroy process groups on all actors (including dead ones, which will fail gracefully)
    print("Destroying old process groups...")
    try:
        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "destroy_worker_process_group"
            )
        )
    except Exception as e:
        print(f"Some actors failed during destroy (expected): {e}")

    for i, actor_info in enumerate(actor_group.actor_infos):
        is_alive = actor_group._check_actor_alive(actor_info.handle)
        print(f"Actor {i} (handle: {actor_info.handle}) is alive: {is_alive}")

    # Recover from failure: remove dead actors and re-initialize process group
    print("Recovering from actor failure...")
    actor_group.recover_from_failure(backup_actor_group)

    # load checkpoint on all actors
    # TODO: improve the logic here
    # we want to only call load checkpoint on the actors that are fresh
    # on previously healthy actors we want to restore weights and optimizer state from cpu memory
    # ray.get(actor_group.async_run_ray_method("pass_through", "backload_to_gpu"), actor_ids=[previously healthy actor ids])
    # only for new actors, we want to load the checkpoint
    ray.get(
        actor_group.async_run_ray_method(
            "pass_through", "load_checkpoint", ckpt_dir=config.ckpt_dir
        )
    )
    print(f"Recovery took {time.time() - start_time:.2f} seconds")

    # TODO: check that results here are the same as before the failure when resuming from checkpoint
    # Test that training still works after recovery
    print("Testing training after recovery...")
    batch_after_recovery = get_test_training_batch(config.model, batch_size=32)
    start_time = time.time()
    ray.get(
        actor_group.async_run_ray_method(
            "pass_through", "ppo_train", batch_after_recovery
        )
    )
    print(
        f"Training step 2 (after recovery) took {time.time() - start_time:.2f} seconds"
    )
    print("Recovery successful! Training works with remaining actors.")


if __name__ == "__main__":
    main()

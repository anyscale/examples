import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Dict

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.v2.jax import JaxTrainer


# ---------------- Model & helpers (plain, serializable) ----------------


class SimpleMLP(nn.Module):
    features: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat, name=f"dense_{i}")(x)
            if i != len(self.features) - 1:
                x = jnp.maximum(x, 0)  # ReLU
        return x


class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, lr, model, input_shape):
    params = model.init(rng, jnp.ones(input_shape))["params"]
    tx = optax.adam(lr)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_synthetic_batch(key, batch_size, input_dim, output_dim):
    k_in, k_noise = jax.random.split(key)
    inputs = jax.random.normal(k_in, (batch_size, input_dim))
    true_w = jnp.arange(input_dim, dtype=jnp.float32).reshape((input_dim, 1))
    labels = inputs @ true_w + 0.5 + 0.1 * jax.random.normal(k_noise, (batch_size, output_dim))
    return {"inputs": inputs, "labels": labels}


def shard_for_pmap(batch: Dict[str, jnp.ndarray], local_device_count: int):
    def _shard(x):
        assert x.shape[0] % local_device_count == 0, "batch not divisible by local_device_count"
        per_dev = x.shape[0] // local_device_count
        return x.reshape((local_device_count, per_dev) + x.shape[1:])
    return jax.tree.map(_shard, batch)


def train_step_fn(state: TrainState, batch: Dict[str, jnp.ndarray]):
    """Pure step (no JAX transforms here)."""
    def loss_fn(p):
        preds = state.apply_fn({"params": p}, batch["inputs"])
        return jnp.mean((preds - batch["labels"]) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, grads


def params_checksum_fn(state: TrainState):
    leaves = jax.tree_util.tree_leaves(state.params)
    return sum([jnp.sum(jnp.abs(x)).astype(jnp.float32) for x in leaves])


# ---------------- Ray Train per-worker loop ----------------


def train_loop_per_worker(config):
    # Basic topology info
    ctx_rank = ray.train.get_context().get_world_rank()
    pidx = jax.process_index()
    nproc = jax.process_count()
    local_devices = jax.local_devices()
    ldc = len(local_devices)
    print(f">>> Ray rank={ctx_rank}, JAX process {pidx}/{nproc}, local_devices={local_devices}")

    # Config
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    per_host_batch = config["batch_size"]  # interpret as per-host batch
    assert per_host_batch % ldc == 0, "batch_size must be divisible by local_device_count"
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    log_every = config.get("log_every", 10)

    # Identical init across hosts (fixed seed) -> no broadcast needed
    model = SimpleMLP([64, 32, output_dim])
    init_key = jax.random.PRNGKey(42)
    state_single = create_train_state(init_key, learning_rate, model, input_shape=(1, input_dim))
    state = jax.device_put_replicated(state_single, local_devices)

    # Host-local RNG for data
    key = jax.random.PRNGKey(1234 + pidx)


    # --- Create pmapped callables *inside* the worker (avoid Ray pickling issues) ---

    def _apply_step(state, batch):
        loss, grads = train_step_fn(state, batch)
        # average across all replicas (devices & hosts)
        # grads = jax.lax.pmean(grads, axis_name="x")
        loss  = jax.lax.pmean(loss,  axis_name="x")
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def _sync_check(state):
        cs = params_checksum_fn(state)
        # if replicas are identical, spread == 0
        spread = jax.lax.pmax(cs, "x") - jax.lax.pmin(cs, "x")
        mean_cs = jax.lax.pmean(cs, "x")
        return mean_cs, spread

    train_step = jax.pmap(_apply_step, axis_name="x")
    sync_check = jax.pmap(_sync_check, axis_name="x")

    # Initial sync check
    mean_cs, spread = sync_check(state)
    mean_cs, spread = map(lambda a: float(jax.device_get(a)[0]), (mean_cs, spread))
    if pidx == 0:
        print(f"[init] checksum {mean_cs:.6f} | spread {spread:.3e}")

    # Train
    for epoch in range(1, num_epochs + 1):
        key, kbatch = jax.random.split(key)
        batch = get_synthetic_batch(kbatch, per_host_batch, input_dim, output_dim)
        batch = shard_for_pmap(batch, ldc)

        state, loss = train_step(state, batch)
        loss_scalar = float(jax.device_get(loss)[0])

        if epoch == 1 or (epoch % log_every == 0):
            mean_cs, spread = sync_check(state)
            mean_cs, spread = map(lambda a: float(jax.device_get(a)[0]), (mean_cs, spread))
            if pidx == 0:
                print(f"Epoch {epoch:04d} | loss {loss_scalar:.6f} | checksum {mean_cs:.6f} | spread {spread:.3e}")
            ray.train.report({"epoch": epoch, "loss": loss_scalar, "checksum": mean_cs, "spread": spread})

    if pidx == 0:
        print("DDP training finished.")


# ---------------- Main: JaxTrainer wiring ----------------


def main():
    print("Starting JAX training...")

    trainer = JaxTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "input_dim": 10,
            "output_dim": 1,
            "batch_size": 1024,     # per-host batch
            "num_epochs": 100,
            "learning_rate": 0.01,
            "log_every": 10,
        },
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=4,
            resources_per_worker={"GPU": 4},
        ),
        run_config=RunConfig(
            name="jax_ddp_pmap",
            worker_runtime_env={
                "env_vars": {
                    "JAX_PLATFORMS": "cuda",
                    # set if needed in your env; comment out if it breaks local libs
                    "LD_LIBRARY_PATH": "",
                },
            },
        ),
    )

    result = trainer.fit()
    print("Training complete!")


if __name__ == "__main__":
    main()

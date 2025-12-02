from typing import Optional, List
from functools import partial
import torch
import torch.nn as nn

from megatron.core.pipeline_parallel import get_forward_backward_func
import megatron.core.parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

from megatron_model_utils import from_parallel_logits_to_logprobs
from megatron_utils import (
    get_model_config,
    make_batch_generator,
    preprocess_packed_seqs,
    postprocess_packed_seqs,
)
from utils import ppo_policy_loss


class MegatronModelWrapper:
    def __init__(
        self,
        config,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_backward_mini_batch(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: A list of micro-batch dicts. Each dict must contain keys:
                "sequences", "attention_mask", "position_ids", "num_actions",
                "old_action_log_probs", "base_action_log_probs", "advantages",
                "loss_mask".
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()

        def loss_func(logits, data):
            sequences = data["sequences"]
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]

            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # temperature normalization
            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=False,
                cp_group=None,  # we handle cp gathering in `postprocess_packed_seqs`
                chunk_size=None,
            )

            action_log_probs = token_logprobs[:, -num_actions:]

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, clip_ratio = ppo_policy_loss(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg,
                loss_mask=loss_mask,
            )

            # no kl loss or entropy loss
            loss = policy_loss

            metrics = {
                "policy_loss": policy_loss.detach().item(),
                "ppo_clip_ratio": clip_ratio,
            }
            return loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)

            new_sequences, packed_seq_params = preprocess_packed_seqs(
                sequences,
                attention_mask,
                pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
            )
            new_attention_mask = None
            new_position_ids = None

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )

            outputs = postprocess_packed_seqs(
                outputs,
                packed_seq_params,
                attention_mask,
                micro_batch_size,
                seq_len,
                post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
            )

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of micro-batches
        batch_generator = make_batch_generator(
            micro_batches, vpp_size=len(self.actor_module)
        )

        metrics_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list

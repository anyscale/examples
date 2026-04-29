import ray
from ray.util.placement_group import (
    PlacementGroup,
    PlacementGroupSchedulingStrategy,
    placement_group_table,
)
import torch
from typing import Any, Optional, Dict, List, Union, Tuple
from dataclasses import dataclass
from jaxtyping import Integer, Float
import math
from transformers import AutoTokenizer


from training_batch import TrainingInputBatch

BasicType = Union[int, float, str, bool]


@ray.remote(num_gpus=1)
class InfoActor:
    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]


def get_reordered_bundle_indices(pg: PlacementGroup):
    """
    Get the reordered bundle indices for a placement group to ensure adjacent ranks are on the same node when possible
    """
    pg_data = placement_group_table(pg)
    num_bundles = len(pg_data["bundles"])
    bundle_to_node_ids = pg_data["bundles_to_node_id"]

    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                num_cpus=0.01,  # set both num_cpus and num_gpus to be small values to enable assignment in colocated case
                num_gpus=0.01,
                resources=None,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
        )

    gpu_ids = ray.get([actor.get_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    # original index, node_id, gpu_id
    bundle_infos = [(i, bundle_to_node_ids[i], gpu_ids[i]) for i in range(num_bundles)]
    pg_reordered_bundle_indices = [
        bundle_info[0]
        for bundle_info in sorted(bundle_infos, key=lambda x: (x[1], x[2]))
    ]  # sort by node_id, then gpu_id
    return pg_reordered_bundle_indices


def to(tensor: Union[torch.Tensor, List[torch.Tensor], BasicType], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    else:
        return tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions/ response length.
    """

    sequences: Integer[torch.Tensor, "batch seq_len"]
    action_log_probs: Float[torch.Tensor, "batch response_len"]
    base_action_log_probs: Optional[Float[torch.Tensor, "batch response_len"]]
    values: Optional[Float[torch.Tensor, "batch response_len"]]
    returns: Optional[Float[torch.Tensor, "batch response_len"]]
    advantages: Optional[Float[torch.Tensor, "batch response_len"]]
    attention_mask: Optional[Integer[torch.LongTensor, "batch seq_len"]]
    loss_mask: Optional[Integer[torch.LongTensor, "batch response_len"]]
    action_mask: Optional[Integer[torch.Tensor, "batch response_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch response_len"]]
    num_actions: int
    info: Optional[dict]
    kl: Optional[Float[torch.Tensor, "batch response_len"]] = None
    metadata: Optional[Dict[str, Any]] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = to(self.base_action_log_probs, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.returns is not None:
            self.returns = to(self.returns, device)
        if self.advantages is not None:
            self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = to(self.attention_mask, device)
        if self.loss_mask is not None:
            self.loss_mask = to(self.loss_mask, device)
        if self.action_mask is not None:
            self.action_mask = to(self.action_mask, device)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = to(self.rollout_logprobs, device)


class BatchIterator:
    """A simple iterator to yield micro batches of data from the training batch."""

    def __init__(
        self, data: TrainingInputBatch, sample_batch_size: int, drop_last: bool = False
    ):
        self.data = data
        self.sample_batch_size = sample_batch_size
        self.total_batch_size = data.batch_size
        self.drop_last = drop_last
        assert not drop_last, "drop_last is not supported yet"
        num_micro_batches = self.total_batch_size / self.sample_batch_size
        self.num_micro_batches = (
            int(num_micro_batches) if drop_last else math.ceil(num_micro_batches)
        )
        # TODO: switch to tensordict.map_iter if possible
        self._chunks = self.data.chunk(self.sample_batch_size)
        self._iter = iter(self._chunks)

    def __len__(self):
        return self.num_micro_batches

    def __iter__(self):
        return self

    def __next__(self) -> Experience:
        try:
            batch = next(self._iter)
            exp = self.batch_to_experience(batch)
            return exp
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

    @staticmethod
    def batch_to_experience(batch: TrainingInputBatch):
        exp = Experience(
            sequences=batch["sequences"],
            action_log_probs=batch["action_log_probs"],
            base_action_log_probs=batch["base_action_log_probs"],
            values=batch["values"],
            returns=batch["returns"],
            advantages=batch["advantages"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            action_mask=batch["response_mask"],
            num_actions=batch.metadata["response_length"],  # int
            rollout_logprobs=(
                batch["rollout_logprobs"] if "rollout_logprobs" in batch else None
            ),
            # additional info
            # can be used to log metrics etc for micro-batches in the worker
            info={},
            # propagate metadata as is
            metadata=batch.metadata,
        )
        return exp


def masked_mean(
    tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


def _safe_exp_delta(
    delta: torch.Tensor, clip: float = 20.0, out_dtype=None
) -> torch.Tensor:
    """
    Clamp the delta before exponentiating to avoid potential overflow.
    """
    y = torch.clamp(delta.to(torch.float32), -clip, clip).exp()
    return y.to(out_dtype or delta.dtype)


def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Compute dual clip PPO policy loss."""
    ratio = _safe_exp_delta(
        log_probs - old_log_probs, clip=20.0, out_dtype=log_probs.dtype
    )
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = (
        masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    )
    clip_pg_losses1 = loss
    pg_losses3 = -advantages * config.clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    loss = loss = masked_mean(loss, loss_mask)
    return loss, clip_ratio


def get_test_training_batch(model_name, batch_size=4) -> TrainingInputBatch:
    """
    Returns a test training batch with padded seqs and attention masks

    Gives a batch of 4 sequences with variable amounts of left padding, and variable response lengths/amounts of right padding
    Attention masks are 1 for non-padding tokens, 0 for padding tokens
    The rest of the fields are filled with dummy data
    """
    assert batch_size % 4 == 0, "batch size must be divisible by 4"
    num_repeats = batch_size // 4
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    sentences = [
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "<|im_start|>user\nThe selling price of a bicycle that had sold $220 last year was increased by 15",
        "What is the new price? Let's think step by step and output the final answer after `####`.<|im_end|>\n",
        "<|im_start|>assistant\nTo find the new price of the bicycle after the increase,",
    ] * num_repeats

    sequences = [tokenizer.encode(sentence) for sentence in sentences]
    attention_masks = [[1] * len(seq) for seq in sequences]
    num_actions = 10
    # max seq len 1 longer than the longest sequence so we always have some padding
    max_seq_length = max([len(seq) for seq in sequences]) + 7

    pad_token_id = tokenizer.pad_token_id
    pad_before = [4, 0, 1, 6] * num_repeats
    pad_after = [
        max_seq_length - len(seq) - pad_before[i] for i, seq in enumerate(sequences)
    ]

    for i, (pad_before, pad_after) in enumerate(zip(pad_before, pad_after)):
        sequences[i] = (
            [pad_token_id] * pad_before + sequences[i] + [pad_token_id] * pad_after
        )
        attention_masks[i] = [0] * pad_before + attention_masks[i] + [0] * pad_after

    attention_masks = torch.tensor(attention_masks)
    sequences = torch.tensor(sequences)

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_masks,
            "action_log_probs": torch.tensor([[0.1] * num_actions] * batch_size),
            "base_action_log_probs": torch.tensor([[0.2] * num_actions] * batch_size),
            "rollout_logprobs": torch.tensor([[0.11] * num_actions] * batch_size),
            "values": torch.tensor([[0.1] * num_actions] * batch_size),
            "returns": torch.tensor([[0.1] * num_actions] * batch_size),
            "advantages": torch.tensor([[0.5] * num_actions] * batch_size),
            "loss_mask": torch.tensor([[1] * num_actions] * batch_size),
            "response_mask": torch.tensor([[1] * num_actions] * batch_size),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data

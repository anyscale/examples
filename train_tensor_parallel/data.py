"""TP-aware data loading for tensor parallelism training."""

from typing import Any

import torch.distributed as dist
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

import ray.train


def _get_world_rank() -> int:
    """Get world rank, handling both distributed and non-distributed cases."""
    try:
        return ray.train.get_context().get_world_rank()
    except Exception:
        return 0


def _barrier():
    """Synchronize all workers."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    """
    Load and configure the tokenizer for the given model.

    Args:
        model_name: Name of the model to load tokenizer for
        trust_remote_code: Whether to trust remote code

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without eos_token
            tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def create_tp_aware_dataloader(
    model_name: str,
    dataset_name: str,
    seq_length: int,
    batch_size: int,
    dp_rank: int,
    dp_size: int,
    seed: int = 42,
    dataset_percentage: float = 10.0,
) -> DataLoader:
    """
    Create dataloader with TP-aware sharding.

    IMPORTANT: This function uses dp_rank and dp_size for data sharding,
    NOT world_rank and world_size. This ensures that all TP ranks within
    the same DP group see identical batches, which is required for
    tensor parallelism to work correctly.

    Do NOT use ray.train.torch.prepare_data_loader() as it uses world_rank
    for sharding, which would give different data to each TP rank.

    Args:
        model_name: HuggingFace model name for tokenizer
        dataset_name: HuggingFace dataset name
        seq_length: Maximum sequence length
        batch_size: Batch size per worker
        dp_rank: Data parallel rank (NOT world rank)
        dp_size: Data parallel size (NOT world size)
        seed: Random seed for shuffling
        dataset_percentage: Percentage of dataset to use (0-100)

    Returns:
        DataLoader with TP-aware sharding
    """
    world_rank = _get_world_rank()

    # Handle datasets that require a config name
    dataset_config = None
    if dataset_name == "wikitext":
        dataset_config = "wikitext-2-raw-v1"

    split_spec = f"train[:{int(dataset_percentage)}%]"

    # Rank 0 downloads tokenizer and dataset first to avoid file handle conflicts
    if world_rank == 0:
        tokenizer = get_tokenizer(model_name)
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split_spec,
            download_config=DownloadConfig(disable_tqdm=True),
        )

    # Wait for rank 0 to finish downloading
    _barrier()

    # Other ranks load from cache
    if world_rank != 0:
        tokenizer = get_tokenizer(model_name)
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split_spec,
            download_config=DownloadConfig(disable_tqdm=True),
        )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )

    # Add labels column (same as input_ids for causal LM training)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(
        add_labels,
        batched=True,
        num_proc=1,
        keep_in_memory=True,
    )

    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Create DistributedSampler with DP rank/size (NOT world rank/size)
    # This ensures all TP ranks in the same DP group get the same data
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=dp_size,  # Use DP size, not world size
        rank=dp_rank,  # Use DP rank, not world rank
        shuffle=True,
        seed=seed,
    )

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )

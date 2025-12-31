"""Simplified model builder for tensor parallelism training."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class ModelConfig:
    """Configuration dataclass for model specifications."""

    name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int


@contextmanager
def use_default_device(device):
    """Context manager to temporarily set default device."""
    prev_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(prev_device)


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration from HuggingFace.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        ModelConfig with model specifications
    """
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Handle models with nested text_config (e.g., Qwen2.5-VL)
    if hasattr(hf_config, "text_config"):
        text_config = hf_config.text_config
    else:
        text_config = hf_config

    # Get hidden_size (GPT-2 uses n_embd)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(text_config, "n_embd", 768)

    # Get num_attention_heads (GPT-2 uses n_head)
    num_attention_heads = getattr(text_config, "num_attention_heads", None)
    if num_attention_heads is None:
        num_attention_heads = getattr(text_config, "n_head", 12)

    # Get num_hidden_layers (GPT-2 uses n_layer)
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = getattr(text_config, "n_layer", 12)

    # Get intermediate_size (GPT-2 uses n_inner, defaults to 4 * hidden_size)
    intermediate_size = getattr(text_config, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = getattr(text_config, "n_inner", None)
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    return ModelConfig(
        name=model_name,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=getattr(
            text_config, "num_key_value_heads", num_attention_heads
        ),
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=text_config.vocab_size,
    )


def create_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int = 0,
    attn_impl: str = "sdpa",
) -> nn.Module:
    """
    Create model with random initialization.

    Args:
        model_name: HuggingFace model name or local path
        device: Device to place the model on
        dtype: Data type for model parameters
        num_layers: Override number of layers (0 = use model default)
        attn_impl: Attention implementation to use

    Returns:
        Initialized model
    """
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Handle models with nested text_config (e.g., Qwen2.5-VL)
    if hasattr(hf_config, "text_config"):
        config_to_modify = hf_config.text_config
    else:
        config_to_modify = hf_config

    # Override number of layers if specified
    if num_layers > 0:
        original_layers = config_to_modify.num_hidden_layers
        config_to_modify.num_hidden_layers = num_layers
        print(f"Overriding num_hidden_layers: {original_layers} -> {num_layers}")

    # Create model with random initialization
    with use_default_device(device):
        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

    # Move to target dtype
    model = model.to(dtype=dtype)

    return model


def get_transformer_layers(model: nn.Module):
    """
    Get the list of transformer layers from the model.

    Args:
        model: The model to get layers from

    Returns:
        List or ModuleList of transformer layers, or None if not found
    """
    # Qwen/Llama/Mistral structure: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT-2 structure: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    return None

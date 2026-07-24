"""Checkpoint loading helpers for released MMC models."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from src.model.config import TransformerConfig
from src.model.transformer import CausalTransformer


_ARCH_KEYS = [
    "n_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "max_seq_len",
    "swa_window_size",
    "full_attention_every_n",
    "n_experts",
    "moe_top_k",
    "expert_d_ff",
    "use_bias_routing",
    "compressed_attn_stride",
    "compressed_attn_chunk",
    "use_split_qkv",
]


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint file with full metadata."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict in {path}, got {type(ckpt).__name__}")
    return ckpt


def checkpoint_config_to_dict(config: Any) -> dict[str, Any]:
    """Normalize a checkpoint config payload into a plain dict."""
    if isinstance(config, TransformerConfig):
        return asdict(config)
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"Unsupported checkpoint config type: {type(config).__name__}")


def config_from_checkpoint(ckpt: dict[str, Any]) -> TransformerConfig:
    """Extract a TransformerConfig from a loaded checkpoint."""
    if "config" not in ckpt:
        raise KeyError("Checkpoint does not contain a saved config")
    cfg_dict = checkpoint_config_to_dict(ckpt["config"])
    return TransformerConfig(**cfg_dict)


def load_model_from_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[dict[str, Any], TransformerConfig, CausalTransformer]:
    """Load checkpoint, reconstruct model from saved config, and load weights."""
    ckpt = load_checkpoint(path, map_location=device)
    config = config_from_checkpoint(ckpt)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {path} does not contain model_state_dict")
    model = CausalTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, config, model


def summarize_config(config: TransformerConfig) -> dict[str, Any]:
    """Return a compact architecture summary for logging and model cards."""
    cfg = checkpoint_config_to_dict(config)
    return {k: cfg[k] for k in _ARCH_KEYS if k in cfg}


def architecture_label(config: TransformerConfig) -> str:
    """Return a short human-readable architecture label."""
    parts: list[str] = []
    if config.compressed_attn_stride > 0:
        parts.append("nsa")
    elif config.swa_window_size > 0:
        parts.append("swa")
    else:
        parts.append("dense")

    if config.n_experts > 1:
        parts.append("moe")
    else:
        parts.append("ffn")

    if config.use_bias_routing:
        parts.append("bias-routing")
    if config.use_split_qkv:
        parts.append("split-qkv")

    return "+".join(parts)

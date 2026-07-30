"""Layer and structure-level statistics for MMC checkpoint weights."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch

from src.model.checkpoints import checkpoint_config_to_dict

_LAYER_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")


def _empty() -> dict[str, Any]:
    return {"tensor_count": 0, "parameter_count": 0, "sum": 0.0,
            "sum_sq": 0.0, "sum_abs": 0.0, "zero_count": 0,
            "min": None, "max": None}


def _add(stats: dict[str, Any], tensor: torch.Tensor) -> None:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not values.numel():
        return
    stats["tensor_count"] += 1
    stats["parameter_count"] += values.numel()
    stats["sum"] += values.sum().item()
    stats["sum_sq"] += values.square().sum().item()
    stats["sum_abs"] += values.abs().sum().item()
    stats["zero_count"] += (values == 0).sum().item()
    low, high = values.min().item(), values.max().item()
    stats["min"] = low if stats["min"] is None else min(stats["min"], low)
    stats["max"] = high if stats["max"] is None else max(stats["max"], high)


def _finish(stats: dict[str, Any]) -> dict[str, Any]:
    n = stats["parameter_count"]
    if not n:
        return {"tensor_count": 0, "parameter_count": 0}
    mean = stats["sum"] / n
    return {"tensor_count": stats["tensor_count"], "parameter_count": n,
            "mean": mean, "std": max(stats["sum_sq"] / n - mean * mean, 0.0) ** .5,
            "mean_abs": stats["sum_abs"] / n, "l2_norm": stats["sum_sq"] ** .5,
            "min": stats["min"], "max": stats["max"],
            "zero_fraction": stats["zero_count"] / n}


def _identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), tensor.storage_offset(), tensor.numel(), str(tensor.dtype)


def attention_type(config: dict[str, Any], layer: int) -> str:
    every = int(config.get("full_attention_every_n", 0) or 0)
    global_layer = every > 0 and (layer + 1) % every == 0
    if global_layer and int(config.get("compressed_attn_stride", 0) or 0) > 0:
        return "compressed_global"
    if not global_layer and every and int(config.get("swa_window_size", 0) or 0) > 0:
        return "sliding_window"
    return "full"


def _structure(key: str, config: dict[str, Any]) -> str:
    match = _LAYER_KEY.match(key)
    if not match:
        if key.startswith("token_emb."):
            return "token_embedding"
        if key.startswith("lm_head."):
            return "output_head"
        if key.startswith("norm."):
            return "final_norm"
        return "other"
    layer, suffix = int(match.group(1)), match.group(2)
    if suffix.startswith(("attn_norm.", "ff_norm.")):
        return "normalization"
    if suffix.startswith("attn."):
        return f"attention.{attention_type(config, layer)}"
    if suffix.startswith("ff.gate.") or suffix.startswith("ff.expert_bias"):
        return "moe_router"
    if suffix.startswith("ff.experts."):
        return "moe_experts"
    if suffix.startswith("ff."):
        return "feed_forward"
    return "other"


def _component(key: str) -> str:
    suffix = _LAYER_KEY.match(key).group(2)
    if suffix.startswith("attn_norm."):
        return "attention_norm"
    if suffix.startswith("attn."):
        return "attention"
    if suffix.startswith("ff_norm."):
        return "ffn_norm"
    if suffix.startswith("ff.gate.") or suffix.startswith("ff.expert_bias"):
        return "moe_router"
    if suffix.startswith("ff.experts."):
        return "moe_experts"
    return "feed_forward" if suffix.startswith("ff.") else "other"


def analyze_state_dict(state_dict: dict[str, torch.Tensor], config: Any, *, include_tensors: bool = False) -> dict[str, Any]:
    """Return unique-parameter statistics grouped by layer and structure.

    Tied state-dict entries, including the token embedding and LM head, are
    de-duplicated in aggregate totals but retained as aliases in tensor output.
    """
    cfg = checkpoint_config_to_dict(config)
    overall, structures = _empty(), defaultdict(_empty)
    layers = defaultdict(lambda: defaultdict(_empty))
    seen, aliases, tensors = set(), {}, []
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        identity = _identity(tensor)
        alias = identity in seen
        if not alias:
            seen.add(identity)
            _add(overall, tensor)
            _add(structures[_structure(key, cfg)], tensor)
            match = _LAYER_KEY.match(key)
            if match:
                _add(layers[int(match.group(1))][_component(key)], tensor)
        if include_tensors:
            one = _empty(); _add(one, tensor)
            tensors.append({"name": key, "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype).removeprefix("torch."),
                            "structure": _structure(key, cfg), "is_shared_alias": alias,
                            "shared_with": aliases.get(identity), "statistics": _finish(one)})
        aliases.setdefault(identity, key)

    layer_rows = []
    for index, components in sorted(layers.items()):
        total = _empty()
        for stats in components.values():
            for field in ("tensor_count", "parameter_count", "sum", "sum_sq", "sum_abs", "zero_count"):
                total[field] += stats[field]
            if stats["min"] is not None:
                total["min"] = stats["min"] if total["min"] is None else min(total["min"], stats["min"])
                total["max"] = stats["max"] if total["max"] is None else max(total["max"], stats["max"])
        layer_rows.append({"index": index, "attention_type": attention_type(cfg, index),
                           "ffn_type": "moe" if int(cfg.get("n_experts", 1) or 1) > 1 else "dense",
                           "components": {k: _finish(v) for k, v in sorted(components.items())},
                           "total": _finish(total)})
    result = {"architecture": cfg, "overall": _finish(overall),
              "state_dict_tensor_count": sum(isinstance(x, torch.Tensor) for x in state_dict.values()),
              "unique_tensor_count": len(seen),
              "structures": {k: _finish(v) for k, v in sorted(structures.items())},
              "layers": layer_rows}
    if include_tensors:
        result["tensors"] = tensors
    return result


def format_layer_summary(analysis: dict[str, Any]) -> str:
    lines = ["Layer  Attention          Params       Attn       FFN/Experts  Router",
             "-----  -----------------  -----------  ---------  -----------  --------"]
    for row in analysis["layers"]:
        c = row["components"]
        n = lambda name: c.get(name, {}).get("parameter_count", 0)
        lines.append(f"{row['index']:>5}  {row['attention_type']:<17}  {row['total']['parameter_count']:>11,}  "
                     f"{n('attention'):>9,}  {n('feed_forward') + n('moe_experts'):>11,}  {n('moe_router'):>8,}")
    return "\n".join(lines)

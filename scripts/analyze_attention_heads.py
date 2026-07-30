#!/usr/bin/env python3
"""Export per-attention-head weight statistics from an MMC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model.checkpoints import architecture_label, checkpoint_config_to_dict, load_checkpoint
from src.model.weight_analysis import _add, _empty, _finish, attention_type


def tensor_stats(tensor):
    stats = _empty()
    _add(stats, tensor)
    return _finish(stats)


def analyze_attention_heads(checkpoint: dict) -> dict:
    """Return per-head Q/K/V/output statistics for an MMC checkpoint."""
    config = checkpoint_config_to_dict(checkpoint["config"])
    state = checkpoint["model_state_dict"]
    n_heads, d_model = config["n_heads"], config["d_model"]
    d_head = d_model // n_heads
    rows = []

    for layer in range(config["n_layers"]):
        prefix = f"blocks.{layer}.attn."
        if prefix + "qkv_proj.weight" in state:
            qkv = state[prefix + "qkv_proj.weight"].chunk(3, dim=0)
            projections = dict(zip(("q", "k", "v"), qkv))
        else:
            projections = {name: state[prefix + f"{name}_proj.weight"] for name in ("q", "k", "v")}
        projections["output"] = state[prefix + "out_proj.weight"]

        for projection, weight in projections.items():
            for head in range(n_heads):
                if projection == "output":
                    # Columns are the concatenated per-head attention outputs.
                    part = weight[:, head * d_head:(head + 1) * d_head]
                else:
                    # Rows are ordered as contiguous Q/K/V head blocks.
                    part = weight[head * d_head:(head + 1) * d_head, :]
                rows.append({
                    "layer": layer,
                    "attention_type": attention_type(config, layer),
                    "projection": projection,
                    "head": head,
                    "shape": list(part.shape),
                    "statistics": tensor_stats(part),
                })

    report = {
        "architecture_label": architecture_label(checkpoint["config"]),
        "step": checkpoint.get("step"),
        "val_loss": checkpoint.get("val_loss"),
        "n_layers": config["n_layers"],
        "n_heads": n_heads,
        "d_head": d_head,
        "head_projections": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Q/K/V/output-projection statistics for every attention head.")
    parser.add_argument("checkpoint", help="Path to a .pt checkpoint")
    parser.add_argument("--json-out", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args()
    report = analyze_attention_heads(load_checkpoint(args.checkpoint, map_location="cpu"))
    report["checkpoint"] = str(args.checkpoint)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(report['head_projections']):,} head-projection records to {args.json_out}")


if __name__ == "__main__":
    main()

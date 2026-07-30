#!/usr/bin/env python3
"""Analyze an MMC checkpoint's weights by layer and model structure."""

import argparse
import json
from pathlib import Path

from src.model.checkpoints import architecture_label, load_checkpoint
from scripts.analyze_attention_heads import analyze_attention_heads
from src.model.weight_analysis import analyze_state_dict, format_layer_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize checkpoint weights by layer and structure.")
    parser.add_argument("checkpoint", help="Path to a .pt checkpoint")
    parser.add_argument("--json-out", type=Path, help="Write the full analysis as JSON")
    parser.add_argument("--include-tensors", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary-only", action="store_true",
                        help="Omit per-tensor and per-attention-head records from the JSON.")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'config' and 'model_state_dict'.")
    report = analyze_state_dict(checkpoint["model_state_dict"], checkpoint["config"],
                                include_tensors=not args.summary_only)
    if not args.summary_only:
        heads = analyze_attention_heads(checkpoint)
        report["attention_heads"] = {
            "n_heads": heads["n_heads"], "d_head": heads["d_head"],
            "head_projections": heads["head_projections"],
        }
        report["schema_version"] = "mmc-weight-report/v2"
    report.update({"checkpoint": str(args.checkpoint), "step": checkpoint.get("step"),
                   "val_loss": checkpoint.get("val_loss"),
                   "architecture_label": architecture_label(checkpoint["config"])})
    overall = report["overall"]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {report['architecture_label']}")
    print(f"Unique parameters: {overall['parameter_count']:,}")
    print(f"Global mean/std: {overall['mean']:.6g} / {overall['std']:.6g}\n\nStructures:")
    for name, stats in report["structures"].items():
        if stats["parameter_count"]:
            print(f"  {name:<28} {stats['parameter_count']:>12,} params  mean={stats['mean']:.5g} "
                  f"std={stats['std']:.5g} l2={stats['l2_norm']:.5g}")
    print("\n" + format_layer_summary(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote unified JSON analysis to {args.json_out}")


if __name__ == "__main__":
    main()

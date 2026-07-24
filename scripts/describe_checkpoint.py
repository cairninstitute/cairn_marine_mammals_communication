#!/usr/bin/env python3
"""Print architecture and training metadata from an MMC checkpoint."""

import argparse
import json

from src.model.checkpoints import architecture_label, checkpoint_config_to_dict, load_checkpoint, summarize_config


def main():
    parser = argparse.ArgumentParser(description="Describe an MMC checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = checkpoint_config_to_dict(ckpt["config"])
    summary = summarize_config(ckpt["config"])
    payload = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "has_optimizer_state": "optimizer_state_dict" in ckpt,
        "has_muon_state": "muon_state_dict" in ckpt,
        "architecture": architecture_label(ckpt["config"]),
        "summary": summary,
        "config": cfg,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"Checkpoint: {payload['checkpoint']}")
    print(f"Architecture: {payload['architecture']}")
    print(f"Step: {payload['step']}")
    print(f"Val loss: {payload['val_loss']}")
    print(f"Has optimizer state: {payload['has_optimizer_state']}")
    print(f"Has muon state: {payload['has_muon_state']}")
    print(f"Summary: {payload['summary']}")


if __name__ == '__main__':
    main()

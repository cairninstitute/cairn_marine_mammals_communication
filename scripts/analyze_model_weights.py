#!/usr/bin/env python3
"""Write a portable, architecture-agnostic tensor-statistics report."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.model.weight_analysis import _add, _empty, _finish, _identity


def find_state_dict(value: Any) -> dict[str, torch.Tensor]:
    """Accept a plain state dict or common checkpoint wrapper."""
    if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
        return value
    if isinstance(value, dict):
        for key in ("model_state_dict", "state_dict", "model", "weights"):
            candidate = value.get(key)
            if isinstance(candidate, dict) and candidate and all(isinstance(v, torch.Tensor) for v in candidate.values()):
                return candidate
    raise ValueError("No tensor state dict found; expected a plain state dict or a model_state_dict/state_dict/model/weights wrapper.")


def group_name(name: str, depth: int) -> str:
    parts = name.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else name


def analyze(state: dict[str, torch.Tensor], grouping_depth: int) -> dict[str, Any]:
    overall, groups, seen, aliases, rows = _empty(), defaultdict(_empty), set(), {}, []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        identity, alias = _identity(tensor), _identity(tensor) in seen
        if not alias:
            seen.add(identity)
            _add(overall, tensor)
            _add(groups[group_name(name, grouping_depth)], tensor)
        one = _empty(); _add(one, tensor)
        rows.append({"name": name, "path": name.split("."), "shape": list(tensor.shape),
                     "dtype": str(tensor.dtype).removeprefix("torch."),
                     "group": group_name(name, grouping_depth), "is_shared_alias": alias,
                     "shared_with": aliases.get(identity), "statistics": _finish(one)})
        aliases.setdefault(identity, name)
    return {"schema_version": "weight-report/v1", "report_kind": "generic_tensor_statistics",
            "grouping_depth": grouping_depth, "overall": _finish(overall),
            "unique_tensor_count": len(seen), "state_dict_tensor_count": len(rows),
            "groups": {name: _finish(stats) for name, stats in sorted(groups.items())}, "tensors": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable architecture-agnostic weight report.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--grouping-depth", type=int, default=2)
    args = parser.parse_args()
    if args.grouping_depth < 1:
        parser.error("--grouping-depth must be positive")
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = analyze(find_state_dict(loaded), args.grouping_depth)
    report["checkpoint"] = str(args.checkpoint)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {report['state_dict_tensor_count']:,} tensor records ({report['unique_tensor_count']:,} unique) to {args.json_out}")


if __name__ == "__main__":
    main()

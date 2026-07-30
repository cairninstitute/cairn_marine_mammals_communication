# Reusable Weight-Analysis Tooling

Analysis artifacts are durable JSON files: they can be committed, archived with
a checkpoint, compared by another program, or rendered later in a browser. They
contain no tensor values, only shapes, names, alias information, and aggregate
distribution statistics.

## MMC reports

`scripts/analyze_checkpoint_weights.py` understands this repository's
transformer and writes one self-contained `mmc-weight-report/v2` JSON: semantic
layer/MoE statistics, per-tensor records, and per-attention-head records.
`analysis/weight_diagram.html` consumes that one report and is intentionally
MMC-specific. `scripts/analyze_attention_heads.py` remains available when only
a standalone head report is needed.

## Generic report format

For another PyTorch model, use the architecture-agnostic exporter:

```bash
PYTHONPATH=. .venv/bin/python scripts/analyze_model_weights.py model.pt \
  --json-out analysis/model_weight_report.json --grouping-depth 2
```

It accepts a plain state dict or a checkpoint holding one under
`model_state_dict`, `state_dict`, `model`, or `weights`. Output declares
`schema_version: "weight-report/v1"` and contains:

```text
overall                 aggregate unique-tensor statistics
groups                  name-prefix aggregates at the chosen path depth
tensors[]               one row per saved tensor, including path/shape/stats
```

Each row has parameter count, mean, standard deviation, mean-absolute value,
L2 norm, min/max, and zero fraction. Tied weights are aliases and are
excluded from aggregate totals. The exporter preserves tensor names rather
than guessing an architecture, so it works with arbitrary naming schemes and
shapes. A small architecture adapter can add semantic records (for example,
attention heads or MoE experts) alongside this stable base report.

## Saving visual analyses

Keep the checkpoint identifier or hash, JSON report(s), exporter command and
commit, architecture-adapter reports, and viewer revision together. The
current MMC viewer is shareable when its two JSON inputs remain beside it and
it is served via local HTTP. A generic viewer can consume the v1 base schema
while optional adapters enable architecture-aware drill-downs.

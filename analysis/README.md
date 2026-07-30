# Checkpoint weight explorer

This directory is a static, browser-based explorer for the released MMC checkpoint analyses. It has no JavaScript build step or third-party browser dependencies.

## View the included reports

Serve the repository root, or this directory, with any static file host. For local viewing:

```bash
python3 -m http.server 8000 --directory analysis
```

Then open:

- `http://localhost:8000/weight_diagram.html` — 10K SWA + MoE report
- `http://localhost:8000/weight_diagram.html?report=reports/128k_weight_analysis.json` — 128K NSA + MoE report

The explorer uses `fetch`, so browsers will not load a report from a `file://` URL. For public hosting, publish the complete `analysis/` directory unchanged (for example through GitHub Pages or any ordinary static host).

## Included data

`reports/` contains sanitized, precomputed `mmc-weight-report/v2` files. They include aggregate tensor statistics, architecture-aware layer/component/expert records, attention-head statistics, and—only for the 10K report—forward-pass measurements for the documented prompt. They contain no model tensor values, checkpoints, token arrays, or local filesystem paths.

## Analyze another MMC checkpoint

Install the repository dependencies, download a compatible checkpoint, then run:

```bash
PYTHONPATH=. python3 scripts/analyze_checkpoint_weights.py /path/to/checkpoint.pt \
  --json-out analysis/reports/model_weight_analysis.json
```

Open the explorer with `?report=reports/model_weight_analysis.json`. The script infers the MMC architecture from the checkpoint configuration. Add `--summary-only` for a compact report without per-tensor or per-head records.

For forward-pass activation measurements, use `scripts/measure_forward_branches.py`; its output can be embedded in a unified report as described in `docs/weight_analysis_tooling.md`.

## Scope

The viewer is architecture-aware for the MMC transformer families in this repository (SWA/Full attention + MoE and NSA + MoE). `scripts/analyze_model_weights.py` is the architecture-agnostic alternative for other PyTorch state dictionaries; it emits `weight-report/v1`, which this MMC-specific viewer does not render.

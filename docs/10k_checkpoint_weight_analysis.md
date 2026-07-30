# 10K Checkpoint Weight Analysis

## Scope and inputs

This note records a weight-only analysis of the released 10K-context model for future comparison and investigation.

- **Checkpoint:** `runs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k_v2/best_model.pt`
- **Architecture:** 16-layer SWA + MoE, 16 attention heads, eight experts per layer, 1,024-dimensional model width.
- **Analysis artifact:** `analysis/10k_weight_analysis.json` (a unified tensor and attention-head report).

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/analyze_checkpoint_weights.py \
  /path/to/best_model.pt \
  --json-out analysis/10k_weight_analysis.json
```

The unified exporter runs on CPU and reports parameter counts, distribution moments, ranges, L2 norms, and zero fractions. It includes per-tensor records and per-attention-head records in one JSON file; it does not execute the model.

## Model-wide composition

| Structure | Parameters | Share of trainable weights |
|---|---:|---:|
| MoE expert matrices | 402,653,184 | 84.0% |
| Attention projections | 67,108,864 | 14.0% |
| Token embedding | 9,440,256 | 2.0% |
| Routers and RMSNorm vectors | 163,840 | <0.1% |

The trainable-weight total is **479,367,168**. A separate 32-value `rope.inv_freq` buffer is saved in the state dict but is not trainable.

## Observed layer patterns

### MoE experts

- Combined MoE L2 norm grows from **72.2** in layer 0 to approximately **90.6** in layer 15, concentrating model capacity in later layers.
- Expert magnitudes vary most in layer 3 (coefficient of variation **11.4%**; expert L2 range **19.76–26.98**).
- Layers 9 and 11 are unusually balanced across experts (coefficient of variation **1.5%** and **1.4%**, respectively).
- The largest individual expert is layer 10, expert 3 (combined L2 **33.99**); the smallest is layer 3, expert 0 (**19.76**).

### Attention

- Local sliding-window layers have stronger Q/K than V/output weights on average: mean per-head L2 is **4.72** for Q, **4.96** for K, **2.17** for V, and **1.99** for output.
- Full-attention refresh layers 4, 9, and 14 reverse that pattern: mean per-head L2 is **3.28** for Q, **3.50** for K, **5.35** for V, and **5.15** for output.
- QKV magnitude peaks in middle local layers 5–8, with layer 6 reaching QKV L2 **33.32**.
- Global layers have very large output projections: layer 4 **17.72**, layer 9 **22.00**, and layer 14 **22.72** L2.

### Routing and normalization

- Router L2 decreases from **2.78** in layer 0 to **0.74** in layer 10 while expert-matrix magnitude grows with depth.
- Attention RMSNorm means spike in full-attention layers: layer 4 **0.98**, layer 9 **1.30**, and layer 14 **1.16**.
- FFN RMSNorm mean rises through the middle/deeper stack and peaks at layer 10 (**1.25**).
- The token embedding has standard deviation **0.0113** and L2 **34.82**; the final RMSNorm has mean **1.845** and L2 **60.91**, indicating substantial late-stage rescaling.

## Interpretation and limits

The weight geometry is consistent with a division of labor: early and middle local layers emphasize Q/K matching, full-attention refresh layers emphasize V/output transport and integration, and deeper layers use larger, more uniform expert transformations.

- Weight magnitude alone does not establish functional importance, routing frequency, or head activity.
- Routing conclusions require logits and expert selections on representative audio-token inputs.
- Attention conclusions require activation, attention-map, and ablation analysis; this document records hypotheses, not causal proof.

### Layer-0 routing and expert anomaly

Layer 0 has a distinctive early-routing geometry. Its MoE gate has L2 norm
**2.775**, compared with a mean of **1.211** across layers 1--15 (a **2.29x**
increase). Its standard deviation (**0.03066**) and mean absolute value
(**0.02163**) are also about 2.3x higher, so the difference is broad weight
scaling rather than an isolated coefficient.

Within layer 0, expert 1 is the largest expert: its combined W1/W2/W3 L2 norm
is **29.53**, compared with a layer mean of **25.46** (a 2.05-standard-
deviation outlier). Every matrix is elevated: W1 **19.44**, W2 **15.63**, and
W3 **15.80**. The weight-only evidence is consistent with an early
specialization in which a stronger first router separates relatively raw local
audio features and this expert applies a stronger transform to one routed
feature regime. It does *not* establish what that regime is, how often the
expert is selected, or its activation/output magnitude.

To test that hypothesis, capture a forward pass on representative audio and
measure layer-0 router entropy and margins, per-expert selection frequency,
and each selected expert's output norm relative to the residual stream.

## Activation-grounded lessons learned

A forward pass was measured on the first **3,105 interleaved DAC tokens**
(**4.01 seconds**) of the real `sanctsound_hi01_01_015542.npy` sample. The
unified report records its per-layer residual updates, routing statistics,
per-head/per-expert activations, and final-output statistics.

- **Global refresh layers are functionally dominant.** Full-attention layers
  4, 9, and 14 have attention updates of **1.00x**, **0.89x**, and **0.98x**
  relative to incoming residual RMS; typical middle SWA layers are **0.03x--0.15x**.
- **Layer 15 is also a large writer.** Its local-attention update is **0.92x**,
  consistent with a final local rewrite/readout after the last global refresh.
- **Weight norm is not activation importance.** Excluding layer 0, attention
  weight L2 versus update ratio has Pearson correlation **0.25**; for MoE
  expert L2 versus MoE update it is **0.53**.
- **Layer 0 bootstraps the residual stream.** Its attention update is **16.94x**
  because the incoming embedding-scale residual is small, while its MoE update
  is **0.24x**, comparable to many later layers.
- **Routing is broad but can be sharper.** Largest expert selection is about
  **15%** (uniform baseline: 12.5%). Layer-3 and layer-15 entropy are **1.27**
  and **1.46 nats**, below ln(8) ≈ **2.08**, without a routing monopoly.
- **The prompt boundary is much more certain.** Mean top-token confidence over
  all prompt positions is **14.0%**; the next token has **71.5%** confidence
  and **0.79 nats** entropy. Predictable nine-codebook cadence likely contributes.

These values describe one prompt, not a dataset-wide average; repeat across
acoustic contexts before treating them as general model behavior.

## Generated data

- `analysis/10k_weight_analysis.json` is the self-contained report: per-tensor statistics plus 1,024 Q/K/V/output head records (16 layers × 16 heads × four projections).

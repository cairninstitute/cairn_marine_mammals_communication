# Model architecture diagrams

`scripts/draw_model_diagram.py` renders a publication-ready SVG directly from an MMC training YAML configuration. It resolves the named model preset, applies YAML overrides, and mirrors `TransformerBlock` attention selection.

It visualizes the decoder flow, exact per-layer attention classes, local/full/compressed attention parameters, and MoE expert count, top-k routing, routing policy, expert width, and checkpointing.

```bash
PYTHONPATH=. python3 scripts/draw_model_diagram.py \
  configs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k.yaml \
  --output blog/assets/generated/128k_architecture.svg
```

Regenerate every published configuration:

```bash
for config in configs/*.yaml; do
  name=$(basename "$config" .yaml)
  PYTHONPATH=. python3 scripts/draw_model_diagram.py "$config" \
    --output "blog/assets/generated/${name}.svg"
done
```

The standalone SVG requires no browser build tooling or external renderer. It is intentionally model-aware: arbitrary PyTorch state dictionaries need an adapter because they do not reliably retain attention or routing semantics.

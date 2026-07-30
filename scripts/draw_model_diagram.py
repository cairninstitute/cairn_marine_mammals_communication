#!/usr/bin/env python3
"""Generate a standalone SVG architecture diagram from an MMC YAML config."""
from __future__ import annotations
import argparse
import html
import math
from pathlib import Path
import yaml
from src.model.config import get_config

INK = "#15283b"
MUTED = "#486174"

def esc(value):
    return html.escape(str(value), quote=True)

def load(path):
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    values = dict(raw["model"])
    preset = values.pop("preset")
    return raw, get_config(preset, **values), preset

def attention(layer, cfg):
    global_layer = cfg.full_attention_every_n > 0 and layer % cfg.full_attention_every_n == 0
    if global_layer and cfg.compressed_attn_stride > 0:
        return "compressed global"
    if cfg.swa_window_size > 0 and cfg.full_attention_every_n > 0 and not global_layer:
        return "local SWA"
    return "full causal"

def txt(x, y, value, css="body", anchor=""):
    anchor = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x}" y="{y}" class="{css}"{anchor}>{esc(value)}</text>'

def multi(x, y, values, css="small", anchor="middle", leading=16):
    spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else leading}">{esc(value)}</tspan>' for i, value in enumerate(values))
    return f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">{spans}</text>'

def box(x, y, w, h, fill, stroke, radius=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'

def arrow(x1, y1, x2, y2):
    return f'<path d="M {x1} {y1} L {x2} {y2}" stroke="#7d98aa" stroke-width="2.5" marker-end="url(#arrow)"/>'

def mini_arrow(x1, y1, x2, y2):
    return f'<path d="M {x1} {y1} L {x2} {y2}" stroke="#7d98aa" stroke-width="1.5" marker-end="url(#mini-arrow)"/>'

def compressed_attention_diagram(x, y, cfg):
    """Draw Q and compressed K/V paths inside a compressed-attention block."""
    return [
        txt(x + 12, y + 17, "COMPRESSED GLOBAL ATTENTION", "detail"),
        box(x + 12, y + 32, 58, 34, "#fff", "#376fa7", 6),
        multi(x + 41, y + 46, ["Q", "full res"], "detail", leading=11),
        box(x + 12, y + 74, 58, 24, "#fff", "#376fa7", 6),
        txt(x + 41, y + 90, "K / V", "detail", "middle"),
        box(x + 89, y + 69, 72, 32, "#dbeaf8", "#376fa7", 6),
        multi(x + 125, y + 81, ["stride", str(cfg.compressed_attn_stride)], "detail", leading=10),
        box(x + 183, y + 45, 80, 43, "#fff", "#376fa7", 6),
        multi(x + 223, y + 61, ["causal", "SDPA"], "detail", leading=11),
        mini_arrow(x + 72, y + 49, x + 181, y + 58),
        mini_arrow(x + 72, y + 86, x + 87, y + 86),
        mini_arrow(x + 163, y + 84, x + 181, y + 75),
    ]

def moe_diagram(x, y, cfg):
    """Draw router, selected experts, and weighted merge inside an MoE block."""
    selected = min(cfg.moe_top_k, 2)
    parts = [
        txt(x + 12, y + 17, f"MoE · TOP-{cfg.moe_top_k} OF {cfg.n_experts}", "detail"),
        box(x + 12, y + 43, 62, 43, "#fff", "#7460a8", 6),
        multi(x + 43, y + 58, ["router", "scores"], "detail", leading=11),
        box(x + 102, y + 31, 61, 29, "#fff", "#7460a8", 6),
        txt(x + 132, y + 50, "expert 1", "detail", "middle"),
        box(x + 102, y + 70, 61, 29, "#fff" if selected == 2 else "#f7f5fb", "#7460a8" if selected == 2 else "#b8acd5", 6),
        txt(x + 132, y + 89, "expert 2" if selected == 2 else "not selected", "detail", "middle"),
        box(x + 190, y + 43, 75, 43, "#fff", "#7460a8", 6),
        multi(x + 227, y + 58, ["weighted", "merge"], "detail", leading=11),
        mini_arrow(x + 76, y + 58, x + 100, y + 45),
        mini_arrow(x + 165, y + 45, x + 188, y + 57),
    ]
    if selected == 2:
        parts += [mini_arrow(x + 76, y + 72, x + 100, y + 85), mini_arrow(x + 165, y + 85, x + 188, y + 73)]
    return parts

def layer_cutaway(x, y, title, count, attention_lines, style, cfg, ffn):
    """Render one continuous transformer-layer path without a skip overlay."""
    fill, stroke = style
    norm_fill, norm_stroke = "#f7fafc", "#9ab1c1"
    parts = [
        box(x, y, 1316, 210, "#fff", "#d7e3ec", 14),
        txt(x + 28, y + 32, f"{title} · {count} layer{'s' if count != 1 else ''}", "section"),
        box(x + 54, y + 86, 160, 78, norm_fill, norm_stroke, 10),
        multi(x + 134, y + 117, ["RMSNorm"], "body"),
        arrow(x + 216, y + 125, x + 272, y + 125),
        box(x + 274, y + 70, 278, 110, fill, stroke, 10),
        *(compressed_attention_diagram(x + 274, y + 70, cfg) if attention_lines[0] == "Compressed global attention" else [multi(x + 413, y + 108, attention_lines, "body", leading=19)]),
        arrow(x + 554, y + 125, x + 610, y + 125),
        box(x + 612, y + 86, 160, 78, norm_fill, norm_stroke, 10),
        multi(x + 692, y + 117, ["RMSNorm"], "body"),
        arrow(x + 774, y + 125, x + 830, y + 125),
        box(x + 832, y + 70, 280, 110, "#f5f2fc", "#7460a8", 10),
        *moe_diagram(x + 832, y + 70, cfg),
        arrow(x + 1114, y + 125, x + 1190, y + 125),
        txt(x + 1230, y + 116, "OUTPUT", "section", "middle"),
        txt(x + 134, y + 192, "pre-attention norm", "detail", "middle"),
        txt(x + 692, y + 192, "pre-FFN norm", "detail", "middle"),
    ]
    return parts

def render(path, raw, cfg, preset):
    name = path.stem.replace("_", " ")
    codebooks = raw.get("data", {}).get("interleave_codebooks")
    context = f"{cfg.max_seq_len:,} tokens" + (f" · {codebooks}-codebook interleaved audio" if codebooks else "")
    schedule = [attention(layer, cfg) for layer in range(1, cfg.n_layers + 1)]
    nsa = cfg.compressed_attn_stride > 0
    routing = "bias-routed" if cfg.use_bias_routing else "aux-loss balanced"
    ffn = f"{routing} MoE · top {cfg.moe_top_k} of {cfg.n_experts} experts" if cfg.n_experts > 1 else "dense SwiGLU feed-forward"
    context_title = f"{cfg.max_seq_len // 1024}K" if cfg.max_seq_len % 1024 == 0 else f"{cfg.max_seq_len:,}-token"
    model_size = "Medium" if preset.startswith("medium") else "Large"
    architecture_title = f"{context_title} Context Hybrid SWA/NSA MoE Architecture" if nsa else f"{model_size} {context_title} Context Hybrid SWA MoE Architecture"
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 1160" role="img" aria-labelledby="title desc">',
        f'<title id="title">MMC causal decoder · {esc(name)}</title>',
        f'<desc id="desc">{esc(f"{cfg.n_layers}-layer causal transformer. {context}. {ffn}.")}</desc>',
        f'<style>text{{font-family:Inter,Arial,sans-serif;fill:{INK}}}.title{{font-size:34px;font-weight:700}}.subtitle{{font-size:17px;fill:{MUTED}}}.section{{font-size:14px;font-weight:700;letter-spacing:1px;fill:{MUTED}}}.body{{font-size:16px;font-weight:650}}.small{{font-size:13px;fill:{MUTED}}}.tile{{font-size:13px;font-weight:700}}.detail{{font-size:11px;fill:{MUTED}}}</style>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#7d98aa"/></marker><marker id="mini-arrow" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#7d98aa"/></marker></defs>', '<rect width="1440" height="1160" fill="#fbfdff"/>', box(22, 22, 1396, 1116, "#fff", "#d7e3ec", 18),
        txt(62, 78, architecture_title, "title"),
        txt(62, 108, f"{name} · preset: {preset} · {context}", "subtitle"),
        txt(62, 154, "CAUSAL DECODER FLOW", "section"),
    ]
    flow = [(62, 170, ["Input token IDs"], ["(B, T)"]), (270, 228, ["Token embedding"], [f"vocab {cfg.vocab_size:,} → d={cfg.d_model:,}"]), (536, 228, ["RoPE + dropout"], [f"{cfg.n_heads} heads · d_head={cfg.d_head}"]), (802, 278, [f"Transformer block × {cfg.n_layers}"], ["RMSNorm → attention → residual", "RMSNorm → FFN → residual"]), (1124, 250, ["RMSNorm +", "tied LM head"], [f"logits over {cfg.vocab_size:,} tokens"])]
    for index, (x, width, title, detail) in enumerate(flow):
        if index:
            previous_x, previous_width = flow[index - 1][0], flow[index - 1][1]
            svg.append(arrow(previous_x + previous_width + 4, 230, x - 4, 230))
        svg += [box(x, 178, width, 104, "#f3f8fb", "#9ab1c1"), multi(x + width / 2, 211, title, "body", leading=18), multi(x + width / 2, 247, detail, "small", leading=15)]
    cadence = f"every {cfg.full_attention_every_n}th layer" if cfg.full_attention_every_n else "no periodic global-layer schedule"
    svg += [txt(62, 314, "ATTENTION SCHEDULE — BOTTOM TO TOP", "section"), txt(62, 340, f"Layer selection mirrors TransformerBlock: {cadence} is global; remaining layers are local when SWA is enabled.", "small")]
    for index, kind in enumerate(schedule):
        row, col = divmod(index, 8)
        x, y = 62 + col * 172, 370 + row * 102
        if kind == "local SWA":
            fill, stroke, line1, line2 = "#e8f6f1", "#58b89a", "local SWA", f"window {cfg.swa_window_size:,}"
        elif kind == "compressed global":
            fill, stroke, line1, line2 = "#eaf1fb", "#376fa7", "compressed global", f"stride {cfg.compressed_attn_stride} K/V"
        else:
            fill, stroke, line1, line2 = "#fff0e9", "#dc6b55", "full causal", "complete history"
        svg += [box(x, y, 158, 88, fill, stroke, 10), txt(x + 79, y + 28, f"Layer {index + 1}", "tile", "middle"), txt(x + 79, y + 53, line1, "detail", "middle"), txt(x + 79, y + 71, line2, "detail", "middle")]
    lower_y = 370 + math.ceil(cfg.n_layers / 8) * 102 + 36
    local_count = schedule.count("local SWA")
    global_count = len(schedule) - local_count
    context_label = f"{cfg.max_seq_len // 1024}K tokens" if cfg.max_seq_len % 1024 == 0 else f"{cfg.max_seq_len:,} tokens"
    global_lines = ["Compressed global attention", "full-resolution queries", f"K/V anchors every {cfg.compressed_attn_stride} tokens · ≈{math.ceil(cfg.max_seq_len / cfg.compressed_attn_stride):,} anchors", f"query chunks: {cfg.compressed_attn_chunk:,} tokens"] if nsa else ["Full attention", context_label]
    svg += [txt(62, lower_y - 18, "WHAT IS INSIDE EACH TYPE OF TRANSFORMER LAYER", "section")]
    svg += layer_cutaway(62, lower_y, "LOCAL TRANSFORMER LAYER", local_count, ["Sliding window attention (SWA)", f"{cfg.swa_window_size:,} tokens"], ("#e8f6f1", "#58b89a"), cfg, ffn)
    svg += layer_cutaway(62, lower_y + 238, "GLOBAL TRANSFORMER LAYER", global_count, global_lines, ("#eaf1fb", "#376fa7") if nsa else ("#fff0e9", "#dc6b55"), cfg, ffn)
    svg += [txt(62, lower_y + 480, f"All layers use the same {ffn}; attention is the only per-layer variation. Gradient checkpointing: {'on' if cfg.use_gradient_checkpointing else 'off'}.", "small"), "</svg>"]
    return "\n".join(svg)

def main():
    parser = argparse.ArgumentParser(description="Render an MMC architecture SVG from a YAML training config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, cfg, preset = load(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.config, raw, cfg, preset), encoding="utf-8")
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()

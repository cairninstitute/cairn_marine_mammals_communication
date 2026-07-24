#!/usr/bin/env python3
"""Unified prompted audio generation for released MMC DAC 9-codebook checkpoints.

Supports released 10k, 64k, and 128k checkpoints by reconstructing architecture
from the checkpoint itself.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.model.checkpoints import architecture_label, load_model_from_checkpoint, summarize_config
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
SEP_GAP_TOKEN = 9217
TOKENS_PER_SEC = 86.133
INTERLEAVED_PER_SEC = TOKENS_PER_SEC * N_CB


def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    n_cb, _ = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)


def pick_best_prompts(
    token_dir: Path,
    scores_csv: Path,
    n: int = 5,
    min_detector: float = 0.8,
) -> list[dict]:
    rows = []
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            det = float(row["detector_score"]) if row["detector_score"] else 0.0
            rows.append({
                "npy_file": row["npy_file"],
                "flac_name": row.get("flac_name", ""),
                "detector_score": det,
                "whale_cv": float(row["whale_cv"]),
                "energy_ratio": float(row["energy_ratio"]),
                "path": token_dir / row["npy_file"],
                "selection": "ranked",
            })

    rows.sort(key=lambda r: r["detector_score"] * 0.7 + r["whale_cv"] * 0.3, reverse=True)

    selected, seen = [], set()
    for r in rows:
        if r["detector_score"] < min_detector:
            continue
        prefix = "_".join(r["npy_file"].split("_")[:3])
        if prefix not in seen:
            selected.append(r)
            seen.add(prefix)
        if len(selected) >= n:
            break

    if len(selected) < n:
        for r in rows:
            if r not in selected:
                prefix = "_".join(r["npy_file"].split("_")[:3])
                if prefix not in seen:
                    selected.append(r)
                    seen.add(prefix)
            if len(selected) >= n:
                break

    return selected[:n]


def explicit_prompts(paths: list[str], token_dir: Path) -> list[dict]:
    prompts = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = token_dir / raw
        prompts.append({
            "npy_file": path.name,
            "flac_name": "",
            "detector_score": float("nan"),
            "whale_cv": float("nan"),
            "energy_ratio": float("nan"),
            "path": path,
            "selection": "explicit",
        })
    return prompts


def infer_mode(config) -> str:
    if config.max_seq_len >= 131072:
        return "long"
    if config.compressed_attn_stride > 0:
        return "nsa"
    return "standard"


def generate_standard(model, prompt_t, actual_new, temperature, top_k):
    with torch.no_grad():
        return model.generate(
            prompt_t,
            max_new_tokens=actual_new,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            eos_token_id=-1,
        )


def generate_with_sep_stopping(model, prompt, max_new_tokens, temperature=0.85, top_k=80, top_p=0.0):
    model.eval()
    prompt = prompt[:, :model.config.max_seq_len]
    with torch.no_grad():
        result = model.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

    generated = []
    for step in range(max_new_tokens):
        with torch.no_grad():
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled[indices_to_remove] = -float("inf")
                probs = scaled.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        token_val = next_token.item()
        generated.append(next_token)
        if token_val in (SEP_TOKEN, SEP_GAP_TOKEN):
            print(f"     -> stopped at step {step + 1} (token {token_val})")
            break

        with torch.no_grad():
            result = model.forward(next_token, past_kv=past_kv)
            past_kv = result.get("past_kv", None)
            logits = result["logits"][:, -1, :]

    if generated:
        return torch.cat([prompt] + generated, dim=1)
    return prompt


def format_metric(value: float) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def prompt_tokens_from_args(args, max_seq_len: int) -> int:
    if args.prompt_token_length is not None:
        return min(args.prompt_token_length, max_seq_len - 1)
    prompt_tokens = int(round(args.prompt_seconds * INTERLEAVED_PER_SEC / N_CB)) * N_CB
    return min(prompt_tokens, max_seq_len // 2)


def max_new_tokens_from_args(args, max_seq_len: int, prompt_tokens: int, run_mode: str) -> int:
    remaining = max_seq_len - prompt_tokens
    if remaining <= 0:
        return 0
    if args.max_new_tokens is not None:
        return min(args.max_new_tokens, remaining)
    if args.max_new_seconds is not None:
        requested = int(round(args.max_new_seconds * INTERLEAVED_PER_SEC / N_CB)) * N_CB
        return min(requested, remaining)
    if run_mode == "sep-stop":
        return min(int(10 * INTERLEAVED_PER_SEC), remaining)
    return remaining


def main():
    parser = argparse.ArgumentParser(description="Generate prompted audio from released MMC DAC checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--token-dir", default="data/tokenized/sanctsound_humpback_dac")
    parser.add_argument("--scores-csv", default=None, help="chunk_scores.csv path (default: token-dir/chunk_scores.csv)")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: <checkpoint-dir>/prompted_unified)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--prompt-seconds", type=float, default=4.0, help="Prompt duration in seconds when prompt length is not set explicitly")
    parser.add_argument("--prompt-token-offset", type=int, default=0, help="Starting token offset inside each selected .npy prompt file")
    parser.add_argument("--prompt-token-length", type=int, default=None, help="Exact prompt token count to use instead of --prompt-seconds")
    parser.add_argument("--prompt-file", action="append", default=None, help="Explicit .npy prompt file to use. Repeatable. Relative paths resolve under --token-dir")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--min-detector", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum number of new tokens to generate")
    parser.add_argument("--max-new-seconds", type=float, default=None, help="Maximum generated continuation duration in seconds")
    parser.add_argument("--decode-audio", choices=["full", "generated", "both"], default="both", help="Whether to save prompt+continuation audio, continuation-only audio, or both")
    parser.add_argument(
        "--mode",
        choices=["auto", "standard", "sep-stop"],
        default="auto",
        help="Generation behavior. auto uses sep-stop for 128k-style long-context checkpoints and standard otherwise.",
    )
    args = parser.parse_args()

    if args.prompt_token_offset < 0:
        parser.error("--prompt-token-offset must be >= 0")
    if args.prompt_token_length is not None and args.prompt_token_length <= 0:
        parser.error("--prompt-token-length must be > 0")
    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be >= 0")
    if args.max_new_seconds is not None and args.max_new_seconds < 0:
        parser.error("--max-new-seconds must be >= 0")
    if args.max_new_tokens is not None and args.max_new_seconds is not None:
        parser.error("--max-new-tokens and --max-new-seconds are mutually exclusive")

    token_dir = Path(args.token_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else token_dir / "chunk_scores.csv"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "prompted_unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt_path}...")
    ckpt, config, model = load_model_from_checkpoint(ckpt_path, device=args.device)
    max_seq_len = config.max_seq_len
    arch = architecture_label(config)
    print(
        f"  {sum(p.numel() for p in model.parameters()):,} params  "
        f"arch={arch}  "
        f"max_seq_len={max_seq_len}  "
        f"val_loss={ckpt.get('val_loss', '?'):.4f}  step={ckpt.get('step', '?')}"
    )
    print(f"  config={summarize_config(config)}")

    inferred = infer_mode(config)
    if args.mode == "auto":
        run_mode = "sep-stop" if inferred == "long" else "standard"
    else:
        run_mode = args.mode
    print(f"Generation mode: {run_mode} (inferred architecture class: {inferred})")

    print("Loading DAC tokenizer...")
    tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)
    print(f"  DAC {tokenizer.sample_rate} Hz, {tokenizer.hop_length} hop, ~{tokenizer.tokens_per_second:.1f} tokens/sec per codebook")

    prompt_tokens = prompt_tokens_from_args(args, max_seq_len)
    prompt_sec_actual = prompt_tokens / INTERLEAVED_PER_SEC
    max_new = max_new_tokens_from_args(args, max_seq_len, prompt_tokens, run_mode)

    if run_mode == "sep-stop":
        print(
            f"\nPrompt: {prompt_tokens} tokens ({prompt_sec_actual:.1f}s)  "
            f"Max generation: {max_new} tokens (~{max_new / INTERLEAVED_PER_SEC:.1f}s max)  "
            f"(stops at SEP tokens {SEP_GAP_TOKEN}/{SEP_TOKEN})"
        )
    else:
        print(
            f"\nPrompt: {prompt_tokens} tokens ({prompt_sec_actual:.1f}s)  "
            f"Generate up to: {max_new} new tokens (~{max_new / INTERLEAVED_PER_SEC:.1f}s)"
        )

    if args.prompt_file:
        prompts = explicit_prompts(args.prompt_file, token_dir)
        print(f"\nUsing {len(prompts)} explicit prompt file(s)...")
    else:
        print(f"\nSelecting top {args.n_samples} prompts from {scores_csv}...")
        prompts = pick_best_prompts(token_dir, scores_csv, n=args.n_samples, min_detector=args.min_detector)

    if not prompts:
        print("No suitable prompts found")
        return

    for p in prompts:
        print(
            f"  {p['npy_file']:50s} sel={p['selection']:8s} "
            f"det={format_metric(p['detector_score'])}  cv={format_metric(p['whale_cv'])}  er={format_metric(p['energy_ratio'])}"
        )

    print(f"\nGenerating -> {out_dir}/")
    for i, seg in enumerate(prompts):
        npy_path = seg["path"]
        if not npy_path.exists():
            print(f"[{i}] SKIP {npy_path.name} (file not found)")
            continue

        codes_2d = np.load(str(npy_path))
        tokens_1d = interleave_2d(codes_2d)
        available = len(tokens_1d) - args.prompt_token_offset
        if available <= N_CB:
            print(f"[{i}] SKIP {npy_path.name} (offset {args.prompt_token_offset} leaves insufficient tokens)")
            continue

        actual_prompt_tokens = min(prompt_tokens, available)
        prompt_end = args.prompt_token_offset + actual_prompt_tokens
        prompt = tokens_1d[args.prompt_token_offset:prompt_end]
        actual_new = min(max_new, max_seq_len - len(prompt))

        print(f"\n[{i}] {npy_path.name}")
        print(f"     chunk has {len(tokens_1d)} tokens ({len(tokens_1d) / INTERLEAVED_PER_SEC:.1f}s)")
        print(f"     prompt_offset={args.prompt_token_offset} prompt={len(prompt)} tokens, generating up to {actual_new} new tokens...")

        prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
        if run_mode == "sep-stop":
            generated = generate_with_sep_stopping(
                model,
                prompt_t,
                max_new_tokens=actual_new,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else 80,
                top_p=args.top_p,
            )
        else:
            generated = generate_standard(model, prompt_t, actual_new, args.temperature, args.top_k)

        full_tokens = generated[0].cpu().numpy()
        gen_tokens = full_tokens[len(prompt):]
        print(f"     generated {len(gen_tokens)} tokens")

        full_audio = None
        gen_audio = None
        if args.decode_audio in {"full", "both"}:
            print("     decoding full audio (prompt + continuation)...")
            full_audio = tokenizer.decode_tokens_to_audio(full_tokens, n_codebooks=N_CB, sep_token=SEP_TOKEN)
        if args.decode_audio in {"generated", "both"}:
            print("     decoding generated-only audio...")
            gen_audio = tokenizer.decode_tokens_to_audio(gen_tokens, n_codebooks=N_CB, sep_token=SEP_TOKEN) if len(gen_tokens) else np.array([], dtype=np.float32)

        sr = tokenizer.sample_rate
        stem = npy_path.stem

        if full_audio is not None:
            full_dur = len(full_audio) / sr
            prompt_dur = len(tokenizer.decode_tokens_to_audio(prompt, n_codebooks=N_CB, sep_token=SEP_TOKEN)) / sr
            gen_dur = max(full_dur - prompt_dur, 0.0)
            full_path = out_dir / f"full_{i:02d}_{stem}.wav"
            sf.write(str(full_path), full_audio, sr)
            print(f"     full   ({full_dur:.1f}s = {prompt_dur:.1f}s prompt + {gen_dur:.1f}s gen) -> {full_path.name}")

        if gen_audio is not None:
            gen_only_dur = len(gen_audio) / sr if len(gen_audio) else 0.0
            gen_path = out_dir / f"generated_{i:02d}_{stem}.wav"
            sf.write(str(gen_path), gen_audio, sr)
            print(f"     output ({gen_only_dur:.1f}s generated only) -> {gen_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()

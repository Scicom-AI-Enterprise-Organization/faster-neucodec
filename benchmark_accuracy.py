"""
Accuracy benchmark: neucodec==0.0.4 (before) vs current release fp32/fp16 (after).
Quality is scored with UTMOSv2 (predicted MOS). MSE and SNR vs original are also reported.

neucodec==0.0.4 is auto-installed to /tmp on the first run — no manual setup needed
for the baseline. UTMOSv2 must be installed separately (see below).

Run
---
python3 benchmark_accuracy.py \
    --audio_dir Elise_audio --num_files 100 --num_repetitions 20 \
    --out results.csv
"""

import csv
import gc
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T

# ── original neucodec version to use as baseline ─────────────────────────────
ORIG_VERSION = "0.0.4"
ORIG_TARGET  = f"/tmp/neucodec_{ORIG_VERSION.replace('.', '_')}"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


# ── neucodec loader (swaps between v0.0.4 and current codebase) ──────────────

def _ensure_orig_installed():
    if not os.path.isdir(ORIG_TARGET):
        print(f"[setup] pip install neucodec=={ORIG_VERSION} --target {ORIG_TARGET}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            f"neucodec=={ORIG_VERSION}", "--target", ORIG_TARGET, "--no-deps",
        ])


def _purge_neucodec():
    for key in [k for k in sys.modules if k == "neucodec" or k.startswith("neucodec.")]:
        del sys.modules[key]


def _load_orig_classes():
    """Import NeuCodec / DistillNeuCodec from neucodec==0.0.4."""
    _ensure_orig_installed()
    _purge_neucodec()
    sys.path.insert(0, ORIG_TARGET)
    from neucodec import NeuCodec, DistillNeuCodec
    return NeuCodec, DistillNeuCodec


def _load_curr_classes():
    """Import NeuCodec / DistillNeuCodec from the current (optimised) codebase."""
    _purge_neucodec()
    if ORIG_TARGET in sys.path:
        sys.path.remove(ORIG_TARGET)
    from neucodec import NeuCodec, DistillNeuCodec
    return NeuCodec, DistillNeuCodec


# ── UTMOSv2 ──────────────────────────────────────────────────────────────────

def load_utmos(upstream_path: str = None):
    """Return a scorer callable: (wav_16k [1, T]) -> float MOS.
    Tries the system-installed utmosv2 first; falls back to upstream_path if given."""
    if upstream_path:
        sys.path.insert(0, upstream_path)
    try:
        import utmosv2
        _model = utmosv2.create_model(pretrained=True)
        _model.eval()

        def _score(wav_16k: torch.Tensor) -> float:
            data = wav_16k.squeeze(0).float().cpu()   # [T]
            out  = _model.predict(data=data, sr=16_000)
            return float(out[0]) if hasattr(out, "__len__") else float(out)

        return _score
    except Exception as exc:
        print(f"[warn] UTMOSv2 failed: {exc}")
        print("       pip3 install git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2")
        return None


# ── audio helpers ─────────────────────────────────────────────────────────────

def find_audio(audio_dir: str, num_files: int, seed: int) -> list[Path]:
    paths = [p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    random.Random(seed).shuffle(paths)
    return paths[:num_files]


def load_16k_mono(path: Path) -> torch.Tensor:
    """Return [1, 1, T] tensor at 16 kHz."""
    y, sr = torchaudio.load(str(path))
    if y.shape[0] > 1:
        y = y.mean(0, keepdim=True)
    if sr != 16_000:
        y = T.Resample(sr, 16_000)(y)
    return y[None, :]   # [1, 1, T]


# ── metrics ───────────────────────────────────────────────────────────────────

def snr_db(ref: torch.Tensor, deg: torch.Tensor) -> float:
    noise = deg - ref
    sp = (ref ** 2).mean()
    np_ = (noise ** 2).mean()
    return float("inf") if np_ == 0 else 10.0 * math.log10((sp / np_).item())


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, n: int, device: str) -> float:
    """Mean latency in ms over n calls, always under no_grad."""
    ts = []
    for _ in range(n):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        sync(device)
        ts.append((time.perf_counter() - t0) * 1_000)
    return sum(ts) / len(ts)


# ── per-variant benchmark ─────────────────────────────────────────────────────

def run_variant(model, audio_16k: torch.Tensor, n_reps: int,
                device: str, out_sr: int, score_fn) -> dict:
    """
    Encode + decode one audio clip, return timing and quality metrics.
    audio_16k: [1, 1, T] at 16 kHz on CPU (float32).
    Models move audio to their device and dtype internally.
    """
    audio_in = audio_16k.float().cpu()

    with torch.no_grad():
        codes = model.encode_code(audio_in)   # warmup
        recon = model.decode_code(codes)      # warmup

    enc_ms = timed(lambda: model.encode_code(audio_in), n_reps, device)
    with torch.no_grad():
        codes = model.encode_code(audio_in)
    dec_ms = timed(lambda: model.decode_code(codes), n_reps, device)
    with torch.no_grad():
        recon = model.decode_code(codes)

    # quality vs original
    recon_16k = T.Resample(out_sr, 16_000)(recon.float().cpu())   # [1, 1, T]
    n = min(audio_16k.shape[-1], recon_16k.shape[-1])
    ref = audio_16k[0, 0, :n].float()
    deg = recon_16k[0, 0, :n].float()
    mse  = F.mse_loss(ref, deg).item()
    snr  = snr_db(ref, deg)

    utmos_recon = float("nan")
    if score_fn is not None:
        try:
            utmos_recon = score_fn(recon_16k[0])   # [1, T]
        except Exception as exc:
            print(f"\n  [warn] UTMOS recon: {exc}")

    return dict(enc_ms=enc_ms, dec_ms=dec_ms,
                mse=mse, snr_db=snr, utmos_recon=utmos_recon)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU.")
        device = "cpu"

    score_fn = load_utmos(args.use_upstream)  # always try; upstream path is optional
    paths    = find_audio(args.audio_dir, args.num_files, args.seed)
    if not paths:
        raise RuntimeError(f"No audio files found in {args.audio_dir!r}")
    print(f"[info] {len(paths)} audio files | {args.num_repetitions} reps | device={device}\n")

    # Load all audio up front (16 kHz mono)
    audios: list[tuple[str, torch.Tensor, float]] = []
    for p in paths:
        try:
            y = load_16k_mono(p)
            utmos_orig = score_fn(y[0]) if score_fn else float("nan")
            audios.append((p.name, y, utmos_orig))
        except Exception as exc:
            print(f"[skip] {p.name}: {exc}")
    print(f"[info] loaded {len(audios)} files\n")

    # Variants: (label, use_half, load_fn)
    variants = [
        (f"v{ORIG_VERSION}_fp32", False, _load_orig_classes),
        ("current_fp32",          False, _load_curr_classes),
        ("current_fp16",          True,  _load_curr_classes),
    ]

    # CSV
    fieldnames = ["file", "variant",
                  "utmos_orig", "utmos_recon", "utmos_delta",
                  "mse_vs_orig", "snr_db_vs_orig",
                  "encode_ms", "decode_ms", "total_ms"]
    out_fh = open(args.out, "w", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
    writer.writeheader()

    prev_load_fn = None

    for label, use_half, load_fn in variants:
        print(f"\n── variant: {label} {'─'*40}")

        # Only reload modules when switching between orig and current
        if load_fn is not prev_load_fn:
            NeuCodec, DistillNeuCodec = load_fn()
            prev_load_fn = load_fn

        cls   = DistillNeuCodec if "distill" in args.model else NeuCodec
        model = cls.from_pretrained(args.model).eval().to(device)
        if use_half:
            model = model.half()
        out_sr = model.sample_rate

        for idx, (fname, audio_16k, utmos_orig) in enumerate(audios):
            try:
                m = run_variant(model, audio_16k, args.num_repetitions, device, out_sr, score_fn)
            except Exception as exc:
                print(f"  [{idx+1}/{len(audios)}] {fname}  ERROR: {exc}")
                continue

            utmos_delta = (
                m["utmos_recon"] - utmos_orig
                if not (math.isnan(utmos_orig) or math.isnan(m["utmos_recon"]))
                else float("nan")
            )
            writer.writerow({
                "file":            fname,
                "variant":         label,
                "utmos_orig":      f"{utmos_orig:.4f}",
                "utmos_recon":     f"{m['utmos_recon']:.4f}",
                "utmos_delta":     f"{utmos_delta:.4f}",
                "mse_vs_orig":     f"{m['mse']:.8f}",
                "snr_db_vs_orig":  f"{m['snr_db']:.3f}",
                "encode_ms":       f"{m['enc_ms']:.2f}",
                "decode_ms":       f"{m['dec_ms']:.2f}",
                "total_ms":        f"{m['enc_ms'] + m['dec_ms']:.2f}",
            })
            out_fh.flush()

            print(f"  [{idx+1:3d}/{len(audios)}] {fname:<40s}  "
                  f"utmos={m['utmos_recon']:.3f} (Δ{utmos_delta:+.3f})  "
                  f"snr={m['snr_db']:.1f}dB  "
                  f"enc={m['enc_ms']:.0f}ms dec={m['dec_ms']:.0f}ms")

        del model
        gc.collect()

    out_fh.close()
    print(f"\n[done] → {args.out}")

    try:
        import pandas as pd
        df = pd.read_csv(args.out)
        cols = ["utmos_orig", "utmos_recon", "utmos_delta",
                "mse_vs_orig", "snr_db_vs_orig", "encode_ms", "decode_ms", "total_ms"]
        cols = [c for c in cols if c in df.columns]
        print("\n── mean per variant ─────────────────────────────────────────")
        print(df.groupby("variant")[cols].mean().to_string(float_format="%.4f"))
    except ImportError:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--audio_dir",       required=True)
    parser.add_argument("--num_files",       type=int, default=100)
    parser.add_argument("--num_repetitions", type=int, default=20,
                        help="encode/decode calls averaged per file for timing")
    parser.add_argument("--out",             default="results.csv")
    parser.add_argument("--use_upstream",    default=None,
                        help="path to UTMOSv2 pip --target install dir")
    parser.add_argument("--model",           default="neuphonic/neucodec",
                        choices=["neuphonic/neucodec", "neuphonic/distill-neucodec"])
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

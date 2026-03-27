"""
Speed benchmark: neucodec==0.0.4 (before) vs current release fp32/fp16 (after).

neucodec==0.0.4 is auto-installed to /tmp on the first run.

Run
---
python3 benchmark_speed.py --audio_dir Elise_audio --num_files 50 --num_repetitions 20
python3 benchmark_speed.py  # synthetic audio, quick smoke-test
"""

import gc
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchaudio
from torchaudio import transforms as T

ORIG_VERSION = "0.0.4"
ORIG_TARGET  = f"/tmp/neucodec_{ORIG_VERSION.replace('.', '_')}"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


# ── neucodec loader ───────────────────────────────────────────────────────────

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
    _ensure_orig_installed()
    _purge_neucodec()
    sys.path.insert(0, ORIG_TARGET)
    from neucodec import NeuCodec, DistillNeuCodec
    return NeuCodec, DistillNeuCodec


def _load_curr_classes():
    _purge_neucodec()
    if ORIG_TARGET in sys.path:
        sys.path.remove(ORIG_TARGET)
    from neucodec import NeuCodec, DistillNeuCodec
    return NeuCodec, DistillNeuCodec


# ── audio helpers ─────────────────────────────────────────────────────────────

def find_audio(audio_dir: str, num_files: int, seed: int) -> list[Path]:
    paths = [p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    random.Random(seed).shuffle(paths)
    return paths[:num_files]


def load_16k_mono(path: Path) -> torch.Tensor:
    y, sr = torchaudio.load(str(path))
    if y.shape[0] > 1:
        y = y.mean(0, keepdim=True)
    if sr != 16_000:
        y = T.Resample(sr, 16_000)(y)
    return y[None, :]   # [1, 1, T]


def synthetic_audio(duration_s: float) -> torch.Tensor:
    return torch.zeros(1, 1, int(duration_s * 16_000))


# ── timing ────────────────────────────────────────────────────────────────────

def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, n: int, device: str) -> tuple[float, float]:
    """(mean_ms, std_ms) over n calls, always under no_grad."""
    ts = []
    for _ in range(n):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        sync(device)
        ts.append((time.perf_counter() - t0) * 1_000)
    mean = sum(ts) / len(ts)
    std  = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5
    return mean, std


def benchmark_file(model, audio: torch.Tensor, n: int, device: str) -> dict:
    # Always pass float32 CPU tensor — models move to their device internally.
    # This also keeps v0.0.4 working (it can't accept CUDA tensors for numpy ops).
    audio_cpu = audio.float().cpu()
    # Wrap in no_grad so v0.0.4 (which lacks @inference_mode) doesn't OOM on
    # longer clips by accumulating activations for backprop.
    enc_fn = lambda: model.encode_code(audio_cpu)
    dec_fn = lambda codes=None: model.decode_code(codes)

    with torch.no_grad():
        codes = enc_fn()   # warmup
        dec_fn(codes)

    enc_ms, enc_std = timed(enc_fn, n, device)
    with torch.no_grad():
        codes = enc_fn()
    dec_ms, dec_std = timed(lambda: dec_fn(codes), n, device)
    dur_ms = audio.shape[-1] / 16.0   # audio duration (16 kHz → ms)
    return dict(enc_ms=enc_ms, enc_std=enc_std,
                dec_ms=dec_ms, dec_std=dec_std,
                total_ms=enc_ms + dec_ms,
                rtf=dur_ms / (enc_ms + dec_ms))


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU.")
        device = "cpu"

    # Audio clips
    if args.audio_dir:
        paths  = find_audio(args.audio_dir, args.num_files, args.seed)
        audios = []
        for p in paths:
            try:
                audios.append((p.name, load_16k_mono(p)))
            except Exception as exc:
                print(f"[skip] {p.name}: {exc}")
        print(f"[info] {len(audios)} audio files from {args.audio_dir}")
    else:
        audios = [(f"synthetic_{args.duration}s", synthetic_audio(args.duration))]
        print(f"[info] synthetic {args.duration}s audio")

    print(f"[info] device={device}  reps={args.num_repetitions}\n")

    variants = [
        (f"v{ORIG_VERSION}_fp32", False, _load_orig_classes),
        ("current_fp32",          False, _load_curr_classes),
        ("current_fp16",          True,  _load_curr_classes),
    ]

    # Accumulate per-variant results
    from collections import defaultdict
    results: dict[str, list] = defaultdict(list)  # label → list of dicts

    prev_load_fn = None
    for label, use_half, load_fn in variants:
        print(f"── variant: {label} {'─'*42}")

        if load_fn is not prev_load_fn:
            NeuCodec, DistillNeuCodec = load_fn()
            prev_load_fn = load_fn

        cls   = DistillNeuCodec if "distill" in args.model else NeuCodec
        model = cls.from_pretrained(args.model).eval().to(device)
        if use_half:
            model = model.half()

        for fname, audio in audios:
            try:
                r = benchmark_file(model, audio, args.num_repetitions, device)
            except Exception as exc:
                print(f"  {fname}  ERROR: {exc}")
                continue
            results[label].append(r)
            print(f"  {fname:<45s}  enc={r['enc_ms']:.0f}ms  "
                  f"dec={r['dec_ms']:.0f}ms  RTF={r['rtf']:.2f}x")

        del model
        gc.collect()
        print()

    # Summary table
    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    col_w = max(len(lb) for lb, *_ in variants) + 2
    print(f"\n{'─'*72}")
    print(f"  {'Variant':<{col_w}}  {'Enc ms':>8}  {'Dec ms':>8}  "
          f"{'Total ms':>10}  {'RTF':>6}  {'Files':>5}")
    print(f"{'─'*72}")

    means = {}
    for label, *_ in variants:
        rs = results[label]
        if not rs:
            continue
        m = {k: mean([r[k] for r in rs]) for k in ("enc_ms", "dec_ms", "total_ms", "rtf")}
        means[label] = m
        print(f"  {label:<{col_w}}  {m['enc_ms']:>8.1f}  {m['dec_ms']:>8.1f}  "
              f"{m['total_ms']:>10.1f}  {m['rtf']:>6.2f}x  {len(rs):>5}")

    # Speedup rows
    baseline_label = f"v{ORIG_VERSION}_fp32"
    if baseline_label in means:
        base = means[baseline_label]
        print(f"{'─'*72}")
        for label in ("current_fp32", "current_fp16"):
            if label not in means:
                continue
            m = means[label]
            enc_x   = f"x{base['enc_ms']   / m['enc_ms']:.2f}"
            dec_x   = f"x{base['dec_ms']   / m['dec_ms']:.2f}"
            total_x = f"x{base['total_ms'] / m['total_ms']:.2f}"
            tag     = f"speedup vs {baseline_label} -> {label}"
            print(f"  {tag:<{col_w}}  {enc_x:>8}  {dec_x:>8}  {total_x:>10}")

    print(f"{'─'*72}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--audio_dir",       default=None,
                        help="audio directory; omit to use synthetic silence")
    parser.add_argument("--num_files",       type=int, default=50)
    parser.add_argument("--num_repetitions", type=int, default=20)
    parser.add_argument("--duration",        type=float, default=5.0,
                        help="synthetic audio duration in seconds")
    parser.add_argument("--model",           default="neuphonic/neucodec",
                        choices=["neuphonic/neucodec", "neuphonic/distill-neucodec"])
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

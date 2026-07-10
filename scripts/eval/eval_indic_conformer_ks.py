#!/usr/bin/env python3
"""
eval_whisper_small_ks.py — Baseline WER evaluation of muneebharoon/whisper-small-ks
on IndicVoices-R Kashmiri test split.

Loads parquet files directly (bypasses datasets library + torchcodec requirement).

Usage:
    python scripts/eval/eval_indic_conformer_ks.py            # 200 samples
    python scripts/eval/eval_indic_conformer_ks.py --n 50     # quick sanity check
    python scripts/eval/eval_indic_conformer_ks.py --n 0      # full test split
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output so Nastaliq characters don't crash on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from jiwer import wer as compute_wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID  = "muneebharoon/whisper-small-ks"
MODEL_CACHE = "E:/VANI/datasets/hf_cache/hub"

PARQUET_DIR = Path(
    r"E:\VANI\datasets\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r"
    r"\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri"
)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n",       type=int,   default=200,  help="Samples to eval (0=all)")
parser.add_argument("--min_dur", type=float, default=2.0,  help="Min duration seconds")
parser.add_argument("--max_dur", type=float, default=20.0, help="Max duration seconds")
args = parser.parse_args()

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
processor = WhisperProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"  Model on {device.upper()}  ({time.time()-t0:.1f}s)", flush=True)
print("  No forced language token (Kashmiri not in Whisper vocab)", flush=True)

# ── load parquet test files ────────────────────────────────────────────────────
print("Loading IndicVoices-R Kashmiri test split from parquet ...", flush=True)
test_files = sorted(PARQUET_DIR.glob("test-*.parquet"))
print(f"  Found {len(test_files)} test parquet file(s)", flush=True)

tables = [pq.read_table(f, memory_map=True) for f in test_files]
import pyarrow as pa
table = pa.concat_tables(tables)
print(f"  Total rows: {len(table)}", flush=True)

# Filter by duration (uses pre-computed column — no audio decode needed)
d = table.to_pydict()
indices = [
    i for i, dur in enumerate(d["duration"])
    if args.min_dur <= dur <= args.max_dur
]
print(f"  After {args.min_dur}–{args.max_dur}s filter: {len(indices)} samples", flush=True)

if args.n > 0:
    indices = indices[: args.n]
print(f"  Evaluating {len(indices)} samples\n", flush=True)

# ── evaluate ──────────────────────────────────────────────────────────────────
references, hypotheses = [], []
errors = skipped = 0
t_start = time.time()

for step, i in enumerate(indices):
    ref = (d["normalized"][i] or d["verbatim"][i] or "").strip()
    if not ref:
        skipped += 1
        continue

    audio_bytes = d["audio"][i]["bytes"]
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
    except Exception as e:
        print(f"  [WARN] sample {i} audio decode failed: {e}", flush=True)
        errors += 1
        continue

    if sr != 16000:
        import torchaudio.functional as AF
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio   = AF.resample(audio_t, sr, 16000).squeeze(0).numpy()

    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"  [WARN] sample {i} inference failed: {e}", flush=True)
        errors += 1
        continue

    references.append(ref)
    hypotheses.append(hyp)

    n_done = step + 1
    if n_done % 20 == 0:
        interim_wer = compute_wer(references, hypotheses) * 100
        elapsed = time.time() - t_start
        spd = elapsed / n_done
        eta  = (len(indices) - n_done) * spd
        print(f"  [{n_done}/{len(indices)}]  WER={interim_wer:.2f}%  "
              f"{spd:.1f}s/sample  ETA={eta/60:.1f}min", flush=True)

# ── results ───────────────────────────────────────────────────────────────────
total_time = time.time() - t_start
final_wer  = compute_wer(references, hypotheses) * 100

print("\n" + "=" * 60)
print(f"KASHMIRI BASELINE EVAL — {MODEL_ID}")
print("=" * 60)
print(f"Samples eval'd : {len(references)}")
print(f"Skipped        : {skipped}  (empty transcript)")
print(f"Errors         : {errors}")
print(f"Final WER      : {final_wer:.2f}%")
print(f"Total time     : {total_time/60:.1f} min  ({total_time/max(len(references),1):.1f}s/sample)")
print("=" * 60)

print("\nSample predictions (first 10):")
for r, h in zip(references[:10], hypotheses[:10]):
    print(f"  REF: {r}")
    print(f"  HYP: {h}")
    print()

out = {
    "model": MODEL_ID,
    "language": "Kashmiri",
    "n_eval": len(references),
    "wer": round(final_wer, 2),
    "errors": errors,
    "skipped": skipped,
    "total_time_min": round(total_time / 60, 1),
}
out_path = ROOT / "logs" / "eval_whisper_small_ks.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Results saved -> {out_path}")

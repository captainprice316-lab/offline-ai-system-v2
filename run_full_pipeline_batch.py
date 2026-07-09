#!/usr/bin/env python3
"""
run_full_pipeline_batch.py
==========================
Runs 30 FLEURS samples each for English (en_us) and Mandarin (cmn_hans_cn)
through the complete 10-stage VANI pipeline.

Pre-loads ASR model once to avoid per-file reload overhead.
Results saved to output/fleurs_en_pipeline/ and output/fleurs_zh_pipeline/.

Run:
    python3 run_full_pipeline_batch.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Offline mode — datasets already cached
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

from utils import load_config, get_logger
from asr_module import ASRModule
from pipeline import run_pipeline

import soundfile as sf
from datasets import load_dataset

N_SAMPLES = 30
DUR_MIN   = 2.0
DUR_MAX   = 20.0

SUBSETS = [
    ("en_us",       "en", "English",  ROOT / "output" / "fleurs_en_pipeline"),
    ("cmn_hans_cn", "zh", "Mandarin", ROOT / "output" / "fleurs_zh_pipeline"),
]

config = load_config()
logger = get_logger("vani.batch")

# ── Pre-load Whisper ASR once ─────────────────────────────────────────────────
whisper_path = ROOT / config["paths"]["whisper_model"]
print(f"Pre-loading ASR model from {whisper_path} ...", end=" ", flush=True)
asr = ASRModule(
    model_path=str(whisper_path),
    device="cpu",
    cfg=config.get("asr", {}),
)
print("ok")
preloaded_models = {"asr": asr}


def save_fleurs_sample_to_wav(sample: dict, wav_path: str) -> float:
    """Write FLEURS audio to WAV; return duration in seconds."""
    audio_data = sample["audio"]
    if isinstance(audio_data, dict) and "array" in audio_data:
        arr = np.asarray(audio_data["array"], dtype=np.float32)
        sr  = int(audio_data.get("sampling_rate", 16000))
    else:
        s   = audio_data.get_all_samples()
        arr = s.data.numpy()[0].astype(np.float32)
        sr  = int(s.sample_rate)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr  = 16000
    sf.write(wav_path, arr, sr)
    return len(arr) / sr


# ── Run each subset ────────────────────────────────────────────────────────────
all_summaries = {}

for subset, lang_code, lang_name, out_dir in SUBSETS:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'─'*68}")
    print(f"  {lang_name} ({subset})  →  {out_dir}")
    print(f"{'─'*68}")

    try:
        ds = load_dataset("google/fleurs", subset, split="test")
    except Exception as e:
        print(f"  Cannot load {subset}: {e}")
        continue

    records = []
    count   = 0

    for idx in range(min(len(ds), 80)):
        if count >= N_SAMPLES:
            break

        sample = ds[idx]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as f:
            wav_path = f.name

        try:
            dur = save_fleurs_sample_to_wav(sample, wav_path)
            if not (DUR_MIN <= dur <= DUR_MAX):
                print(f"  [{idx+1:3d}] skip  dur={dur:.1f}s (out of range)")
                os.unlink(wav_path)
                continue

            t0  = time.perf_counter()
            result = run_pipeline(
                audio_file=Path(wav_path),
                config=config,
                logger=logger,
                progress_cb=None,
                models=preloaded_models,
            )
            elapsed = time.perf_counter() - t0

            # Save individual result JSON
            stem = f"fleurs_{subset}_{idx+1:03d}"
            result["audio_file"] = stem
            out_json = out_dir / f"{stem}_result.json"
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Extract key metrics
            final_lang  = result.get("detected_language", "unk")
            correct     = (final_lang == lang_code)
            segs        = result.get("segments", [])
            confs       = [s.get("confidence", 0.0) for s in segs]
            mean_conf   = float(np.mean(confs)) if confs else 0.0
            speech_dur  = result.get("total_speech_sec", dur)
            proc_time   = result.get("processing_time_s", elapsed)
            rtf         = proc_time / speech_dur if speech_dur > 0 else 0.0
            trans_done  = bool(result.get("translation", {}).get("translated_text", "").strip())
            n_segs      = len(segs)

            records.append({
                "idx":        idx,
                "label":      stem,
                "lang_code":  lang_code,
                "detected":   final_lang,
                "correct":    correct,
                "dur_s":      round(dur, 2),
                "proc_s":     round(proc_time, 2),
                "rtf":        round(rtf, 3),
                "n_segs":     n_segs,
                "mean_conf":  round(mean_conf, 3),
                "translated": trans_done,
            })
            count += 1

            mark = "✓" if correct else "✗"
            print(f"  [{idx+1:3d}] {mark}  lang={final_lang:4s}  "
                  f"conf={mean_conf:.3f}  RTF={rtf:.2f}  "
                  f"dur={dur:.1f}s  segs={n_segs}  trans={'Y' if trans_done else 'N'}")

        except Exception as e:
            print(f"  [{idx+1:3d}] ERROR: {e}")
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    if not records:
        print(f"  No usable results for {subset}")
        continue

    # ── Per-language summary ─────────────────────────────────────────────────
    n          = len(records)
    n_correct  = sum(1 for r in records if r["correct"])
    n_segs_ok  = sum(1 for r in records if r["n_segs"] > 0)
    n_trans    = sum(1 for r in records if r["translated"])
    mean_conf  = float(np.mean([r["mean_conf"] for r in records]))
    mean_rtf   = float(np.mean([r["rtf"]       for r in records]))
    acc        = n_correct / n * 100

    summary = {
        "lang_code":    lang_code,
        "subset":       subset,
        "n":            n,
        "n_correct":    n_correct,
        "accuracy_pct": round(acc, 1),
        "n_segs_ok":    n_segs_ok,
        "trans_str":    f"{n_segs_ok}/{n}",
        "n_translated": n_trans,
        "trans_rate":   f"{n_trans}/{n}",
        "mean_conf":    round(mean_conf, 3),
        "mean_rtf":     round(mean_rtf, 2),
        "records":      records,
    }
    all_summaries[lang_code] = summary

    print(f"\n  ── {lang_name} SUMMARY ──")
    print(f"     N = {n}  |  Accuracy = {n_correct}/{n} ({acc:.1f}%)")
    print(f"     Transcribed = {n_segs_ok}/{n}  |  Translated = {n_trans}/{n}")
    print(f"     Mean Conf = {mean_conf:.3f}  |  Mean RTF = {mean_rtf:.2f}×")

# ── Save combined summary ──────────────────────────────────────────────────────
summary_path = ROOT / "output" / "pipeline_batch_summary.json"
with open(summary_path, "w") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nSaved summary → {summary_path}")

# ── Print Table II rows ────────────────────────────────────────────────────────
print("\n── Updated Table II rows ────────────────────────────────────────────────")
order = ["en", "zh"]
full_names = {"en": "English (en)", "zh": "Mandarin (zh)"}
for lc in order:
    if lc not in all_summaries:
        continue
    s = all_summaries[lc]
    print(f"{full_names[lc]:<18} & {s['n']:<3} & {s['trans_str']:<6} "
          f"& {s['mean_conf']:.3f} & {s['mean_rtf']:.2f}$\\times$ \\\\")
    print(f"  LangID: {s['n_correct']}/{s['n']} ({s['accuracy_pct']}%)  "
          f"Translation: {s['trans_rate']}")

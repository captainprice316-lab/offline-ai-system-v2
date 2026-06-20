#!/usr/bin/env python3
"""
compute_asr_stats_additional.py
================================
Runs 30 FLEURS samples each for Mandarin (cmn_hans_cn) and English (en_us)
through Whisper ASR to compute avg_logprob-based confidence and RTF.
Outputs results to output/asr_stats_additional.json.
"""

import json, os, sys, time, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

WHISPER_PATH = ROOT / "models/whisper-large-v3-turbo-ct2"
N_SAMPLES    = 30

from faster_whisper import WhisperModel

print("Loading Whisper ...", end=" ", flush=True)
whisper = WhisperModel(str(WHISPER_PATH), device="cpu", compute_type="int8",
                       cpu_threads=min(os.cpu_count() or 4, 8))
print("ok")

from datasets import load_dataset
import soundfile as sf


def audio_to_wav(sample: dict, tmp_path: str) -> float:
    """Write sample to wav and return speech duration in seconds."""
    audio_data = sample.get("audio", sample)
    if isinstance(audio_data, dict) and "array" in audio_data:
        arr = np.asarray(audio_data["array"], dtype=np.float32)
        sr  = int(audio_data.get("sampling_rate", 16000))
    else:
        # HF AudioDecoder (torchcodec) object
        s   = audio_data.get_all_samples()
        arr = s.data.numpy()[0].astype(np.float32)
        sr  = int(s.sample_rate)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr  = 16000
    sf.write(tmp_path, arr, sr)
    return len(arr) / sr   # duration in seconds


def run_asr(wav_path: str, speech_dur: float) -> dict:
    t0 = time.perf_counter()
    segs_iter, info = whisper.transcribe(
        wav_path, beam_size=2, best_of=1, temperature=0.0,
        condition_on_previous_text=False, word_timestamps=False, vad_filter=False,
    )
    segments = list(segs_iter)          # materialise the generator
    elapsed = time.perf_counter() - t0

    transcribed = [s for s in segments if s.no_speech_prob < 0.70]
    confs = []
    for s in transcribed:
        c = min(1.0, max(0.0, 1 + s.avg_logprob / 4))
        confs.append(c)

    mean_conf = float(np.mean(confs)) if confs else 0.0
    rtf       = elapsed / speech_dur if speech_dur > 0 else 0.0

    return {
        "n_segs_total":       len(segments),
        "n_segs_transcribed": len(transcribed),
        "mean_conf":          round(mean_conf, 3),
        "rtf":                round(rtf, 3),
        "speech_dur_s":       round(speech_dur, 2),
        "proc_time_s":        round(elapsed, 2),
        "whisper_lang":       info.language,
        "whisper_lang_prob":  round(float(info.language_probability), 3),
    }


# ── Subsets to evaluate ───────────────────────────────────────────────────────
SUBSETS = [
    ("cmn_hans_cn", "zh", "Mandarin"),
    ("en_us",       "en", "English"),
]

all_results = {}

for subset, lang, name in SUBSETS:
    print(f"\n── {name} ({subset}) ─────────────────────────────────────────────")
    try:
        ds = load_dataset("google/fleurs", subset, split="test")
    except Exception as e:
        print(f"  Cannot load {subset}: {e}")
        continue

    records = []
    count = 0
    for idx in range(min(len(ds), 60)):   # scan up to 60 to get 30 usable
        if count >= N_SAMPLES:
            break
        sample = ds[idx]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            dur = audio_to_wav(sample, tmp_path)
            # Skip very short/very long clips (match ablation 2–20s filter)
            if not (2.0 <= dur <= 20.0):
                print(f"  [{idx+1}] skip (dur={dur:.1f}s)")
                os.unlink(tmp_path)
                continue
            r = run_asr(tmp_path, dur)
            r["idx"] = idx
            records.append(r)
            count += 1
            status = "ok" if r["n_segs_transcribed"] > 0 else "silent"
            print(f"  [{idx+1}] {status}  conf={r['mean_conf']:.3f}  "
                  f"RTF={r['rtf']:.2f}  dur={r['speech_dur_s']:.1f}s  "
                  f"wlang={r['whisper_lang']} {r['whisper_lang_prob']:.3f}")
        except Exception as e:
            print(f"  [{idx+1}] error: {e}")
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass

    if not records:
        print(f"  No usable samples found for {subset}")
        continue

    n         = len(records)
    trans_n   = sum(1 for r in records if r["n_segs_transcribed"] > 0)
    mean_conf = float(np.mean([r["mean_conf"] for r in records]))
    mean_rtf  = float(np.mean([r["rtf"]       for r in records]))

    summary = {
        "lang":      lang,
        "subset":    subset,
        "n":         n,
        "trans":     f"{trans_n}/{n}",
        "mean_conf": round(mean_conf, 3),
        "mean_rtf":  round(mean_rtf,  2),
        "records":   records,
    }
    all_results[lang] = summary
    print(f"\n  SUMMARY: N={n}  Trans={trans_n}/{n}  "
          f"MeanConf={mean_conf:.3f}  MeanRTF={mean_rtf:.2f}x")

# ── Save ──────────────────────────────────────────────────────────────────────
out = ROOT / "output" / "asr_stats_additional.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {out}")

# ── Print Table III rows ──────────────────────────────────────────────────────
print("\n── Table III rows (paste into LaTeX) ───────────────────────────────────")
for lang, s in all_results.items():
    full = {"zh": "Mandarin (zh)", "en": "English (en)"}.get(lang, lang)
    print(f"{full:<18} & {s['n']:<3} & {s['trans']:<6} & {s['mean_conf']:.3f} & {s['mean_rtf']:.2f}$\\times$ \\\\")

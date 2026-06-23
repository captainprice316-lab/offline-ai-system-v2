"""
evaluate_datasets.py – VANI evaluation on open-source Punjabi and Hindi datasets.
Runs N samples per language, computes WER/CER, LangID accuracy, RTF, translation
success rate, and saves results to eval_results.json for paper reporting.
"""

import os, sys, json, time, tempfile, re, gc
import numpy as np

# Force offline for HF model loading, but allow dataset streaming
os.environ["HF_HUB_OFFLINE"]       = "0"   # allow dataset download
os.environ["TRANSFORMERS_OFFLINE"]  = "1"   # but keep models local
os.environ["HF_DATASETS_OFFLINE"]   = "0"
os.environ["HF_TOKEN"] = "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"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import soundfile as sf
from datasets import load_dataset
import jiwer

from utils import load_config, get_logger, ensure_dir, free_memory
from pipeline import run_pipeline
from pathlib import Path

log    = get_logger("vani.eval")
cfg    = load_config()
paths  = cfg["paths"]

N_SAMPLES   = 30          # samples per language
MIN_DUR_SEC = 2.0         # skip clips shorter than this
MAX_DUR_SEC = 20.0        # skip very long clips (too slow on CPU)
EVAL_DIR    = ensure_dir(Path(ROOT) / "eval_audio")
RESULTS_FILE = Path(ROOT) / "eval_results_4lang.json"

# ── Text normalisation ─────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Strip punctuation, collapse whitespace, lowercase ASCII."""
    text = re.sub(r'[।॥,.!?;:\"\'\(\)\[\]\-_]', ' ', text or '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ── Pre-load cached models once ────────────────────────────────────────────────

print("\n[INIT] Loading ASR and LangID models (once)...")
from asr_module       import ASRModule
from language_module  import FastTextLangDetector

_asr = ASRModule(
    model_path = str(Path(ROOT) / paths["whisper_model"]),
    device     = "cpu",
    cfg        = cfg.get("asr", {}),
)
_ft  = FastTextLangDetector(str(Path(ROOT) / paths["fasttext_model"]))

try:
    from mms_module import MMSLangDetector
    _mms = MMSLangDetector(
        model_path = str(Path(ROOT) / paths.get("mms_lid_model", "models/mms-lid-256")),
        device     = "cpu",
    )
    print("[INIT] MMS-LID loaded")
except Exception as e:
    _mms = None
    print(f"[INIT] MMS-LID unavailable: {e}")

CACHED_MODELS = {"asr": _asr, "fasttext": _ft}
if _mms:
    CACHED_MODELS["mms"] = _mms

print("[INIT] Models ready.\n")

# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate_language(dataset_name, split, audio_key, transcript_key,
                      true_lang, lang_label, n=N_SAMPLES):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {lang_label} ({true_lang}) — {dataset_name}")
    print(f"Target: {n} samples | {MIN_DUR_SEC}–{MAX_DUR_SEC}s duration")
    print('='*60)

    from datasets import load_dataset, Audio
    ds = load_dataset(dataset_name, split=split, streaming=True,
                      token=os.environ["HF_TOKEN"])
    # Force decode audio to numpy arrays at 16kHz
    ds = ds.cast_column(audio_key, Audio(sampling_rate=16000, decode=True))

    results = []
    skipped = 0
    processed = 0

    for i, sample in enumerate(ds):
        if processed >= n:
            break

        audio_info   = sample.get(audio_key, {})
        reference    = sample.get(transcript_key, "").strip()

        # Handle torchcodec AudioDecoder (newer datasets lib) or plain dict
        try:
            if hasattr(audio_info, "get_all_samples"):
                # torchcodec AudioDecoder
                samples = audio_info.get_all_samples()
                audio_array = samples.data.numpy().squeeze()  # (channels, samples) → (samples,)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=0)   # stereo → mono
                sr = samples.sample_rate
            elif isinstance(audio_info, dict):
                audio_array = audio_info.get("array")
                sr          = audio_info.get("sampling_rate", 16000)
            else:
                skipped += 1
                continue
        except Exception as e:
            skipped += 1
            continue

        if audio_array is None or not reference:
            skipped += 1
            continue

        duration = len(audio_array) / sr
        if duration < MIN_DUR_SEC or duration > MAX_DUR_SEC:
            skipped += 1
            continue

        # Save temp WAV
        wav_path = EVAL_DIR / f"{lang_label}_{processed:04d}.wav"
        arr_16k = audio_array
        if sr != 16000:
            import librosa
            arr_16k = librosa.resample(np.array(audio_array, dtype=np.float32),
                                       orig_sr=sr, target_sr=16000)
        sf.write(str(wav_path), arr_16k, 16000)

        print(f"  [{processed+1}/{n}] {wav_path.name} ({duration:.1f}s) | ref: {reference[:50]}")

        t0 = time.time()
        try:
            result = run_pipeline(
                wav_path, cfg, log,
                progress_cb=None,
                models=CACHED_MODELS,
            )
        except Exception as e:
            print(f"    PIPELINE ERROR: {e}")
            skipped += 1
            continue

        elapsed = round(time.time() - t0, 2)

        if not result:
            print(f"    NO SPEECH DETECTED — skipping")
            skipped += 1
            continue

        hypothesis   = result.get("transcript", "")
        detected_lang = result.get("final_language", "?")
        lang_correct  = detected_lang == true_lang
        seg_confs     = [s["confidence"] for s in result.get("segments", [])]
        mean_conf     = round(sum(seg_confs) / len(seg_confs), 3) if seg_confs else 0.0
        speech_sec    = result.get("total_speech_sec", duration)
        rtf           = round(elapsed / max(speech_sec, 0.1), 3)
        translation   = result.get("translation", {})
        trans_success = translation.get("success", False)
        trans_text    = translation.get("translated_text", "")

        # WER / CER
        ref_norm = normalise(reference)
        hyp_norm = normalise(hypothesis)
        try:
            wer = round(jiwer.wer(ref_norm, hyp_norm) * 100, 2) if ref_norm and hyp_norm else None
            cer = round(jiwer.cer(ref_norm, hyp_norm) * 100, 2) if ref_norm and hyp_norm else None
        except Exception:
            wer = cer = None

        row = {
            "sample_id":       processed,
            "file":            wav_path.name,
            "language":        lang_label,
            "true_lang":       true_lang,
            "duration_sec":    round(duration, 2),
            "reference":       reference,
            "hypothesis":      hypothesis,
            "detected_lang":   detected_lang,
            "lang_correct":    lang_correct,
            "mean_seg_conf":   mean_conf,
            "wer":             wer,
            "cer":             cer,
            "rtf":             rtf,
            "elapsed_sec":     elapsed,
            "trans_success":   trans_success,
            "translation":     trans_text[:120],
            "vote_note":       result.get("vote_note", ""),
            "whisper_lang":    result.get("whisper_language", ""),
            "whisper_prob":    result.get("whisper_language_probability", 0),
            "fasttext_lang":   result.get("fasttext_language", ""),
            "fasttext_conf":   result.get("fasttext_confidence", 0),
            "mms_lang":        result.get("mms_language"),
            "mms_conf":        result.get("mms_confidence"),
            "mem_peak_mb":     result.get("mem_peak_mb"),
            "threat_level":    result.get("threat_level", ""),
        }
        results.append(row)
        processed += 1

        lid_mark = "✓" if lang_correct else "✗"
        wer_str  = f"{wer:.1f}%" if wer is not None else "n/a"
        print(f"    LangID: {detected_lang} {lid_mark} | conf: {mean_conf} | WER: {wer_str} | RTF: {rtf}x | {elapsed}s")
        if trans_success and trans_text:
            print(f"    Trans: {trans_text[:80]}")

        # Free memory between runs
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    if not results:
        print(f"  No results collected for {lang_label}")
        return []

    wers   = [r["wer"] for r in results if r["wer"] is not None]
    cers   = [r["cer"] for r in results if r["cer"] is not None]
    confs  = [r["mean_seg_conf"] for r in results]
    rtfs   = [r["rtf"] for r in results]
    lid_ok = sum(1 for r in results if r["lang_correct"])
    trans_ok = sum(1 for r in results if r["trans_success"])

    print(f"\n  ── {lang_label} SUMMARY ({len(results)} samples) ──")
    print(f"  LangID Accuracy : {lid_ok}/{len(results)} ({100*lid_ok/len(results):.1f}%)")
    print(f"  Mean WER        : {sum(wers)/len(wers):.1f}%" if wers else "  Mean WER: n/a")
    print(f"  Mean CER        : {sum(cers)/len(cers):.1f}%" if cers else "  Mean CER: n/a")
    print(f"  Mean Seg Conf   : {sum(confs)/len(confs):.3f}")
    print(f"  Mean RTF        : {sum(rtfs)/len(rtfs):.2f}x")
    print(f"  Trans Success   : {trans_ok}/{len(results)}")
    print(f"  Skipped         : {skipped}")

    return results


# ── Run evaluations ────────────────────────────────────────────────────────────

all_results = []

# 1. Punjabi
pa_results = evaluate_language(
    dataset_name    = "shunyalabs/punjabi-speech-dataset",
    split           = "train",
    audio_key       = "audio",
    transcript_key  = "transcript",
    true_lang       = "pa",
    lang_label      = "Punjabi",
    n               = N_SAMPLES,
)
all_results.extend(pa_results)

# 2. Hindi
hi_results = evaluate_language(
    dataset_name    = "MatrixSpeechAI/All_Hindi_ASR_v1.2",
    split           = "train",
    audio_key       = "audio",
    transcript_key  = "transcription",
    true_lang       = "hi",
    lang_label      = "Hindi",
    n               = N_SAMPLES,
)
all_results.extend(hi_results)

# 3. Urdu
ur_results = evaluate_language(
    dataset_name    = "m-aliabbas/common_voice_urdu",
    split           = "train",
    audio_key       = "audio",
    transcript_key  = "transcription",
    true_lang       = "ur",
    lang_label      = "Urdu",
    n               = N_SAMPLES,
)
all_results.extend(ur_results)

# 4. Nepali
ne_results = evaluate_language(
    dataset_name    = "iamTangsang/OpenSLR54-Nepali-ASR",
    split           = "train",
    audio_key       = "utterance",
    transcript_key  = "transcription",
    true_lang       = "ne",
    lang_label      = "Nepali",
    n               = N_SAMPLES,
)
all_results.extend(ne_results)

# ── Save all results ───────────────────────────────────────────────────────────

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n\n{'='*60}")
print(f"EVALUATION COMPLETE — {len(all_results)} samples total")
print(f"Results saved to: {RESULTS_FILE}")

# ── Final aggregate table ──────────────────────────────────────────────────────

for lang in ["Punjabi", "Hindi", "Urdu", "Nepali"]:
    rows = [r for r in all_results if r["language"] == lang]
    if not rows: continue
    wers  = [r["wer"] for r in rows if r["wer"] is not None]
    cers  = [r["cer"] for r in rows if r["cer"] is not None]
    confs = [r["mean_seg_conf"] for r in rows]
    rtfs  = [r["rtf"] for r in rows]
    lid   = sum(1 for r in rows if r["lang_correct"])
    trok  = sum(1 for r in rows if r["trans_success"])
    print(f"\n{lang} ({len(rows)} samples):")
    print(f"  WER  : {sum(wers)/len(wers):.1f}% (min {min(wers):.1f} / max {max(wers):.1f})" if wers else "  WER: n/a")
    print(f"  CER  : {sum(cers)/len(cers):.1f}%"  if cers else "  CER: n/a")
    print(f"  Conf : {sum(confs)/len(confs):.3f}")
    print(f"  RTF  : {sum(rtfs)/len(rtfs):.2f}x")
    print(f"  LangID: {lid}/{len(rows)} ({100*lid/len(rows):.1f}%)")
    print(f"  Trans : {trok}/{len(rows)} ({100*trok/len(rows):.1f}%)")

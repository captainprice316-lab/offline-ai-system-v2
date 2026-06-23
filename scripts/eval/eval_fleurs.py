#!/usr/bin/env python3
"""
eval_fleurs.py — FLEURS automated LangID evaluation (IEEE SLT 2026)

Evaluates 4 ablation configurations on 14 FLEURS language subsets (up to 295 samples each,
filtered to 2--20 s duration; default --n 295 is the Urdu full-test-split ceiling):
  1. Whisper-only
  2. Whisper + FastText   (2-way vote)
  3. Whisper + FastText + MMS-LID   (3-way vote, no Script-Cascade)
  4. Full System   (3-way vote + Script-Cascade)

Languages: hi, pa, ne, bn (Indic) | ur, ps, sd (Arabic-script) |
           zh, my (East/SE Asian) | fa, ar (West Asian) | tg, uz, kk (Central Asian)

Usage:
    python eval_fleurs.py                   # all 14 subsets
    python eval_fleurs.py --subset bn_in    # single subset
    python eval_fleurs.py --n 50            # fewer samples (faster sanity check)
    python eval_fleurs.py --no-mms          # skip MMS-LID (configs 3,4 use 2-way)
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Force offline mode — all FLEURS subsets are pre-cached locally
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--n", type=int, default=295, help="Samples per language subset (295 = Urdu full test split)")
parser.add_argument("--no-mms", action="store_true", help="Disable MMS-LID (faster)")
_SUBSET_CHOICES = [
    "hi_in", "pa_in", "ur_pk", "ne_np", "cmn_hans_cn",
    "bn_in", "ps_af", "fa_ir", "ar_eg", "my_mm",
    "sd_in", "tg_tj", "uz_uz", "kk_kz",
    "both", "all",
]
parser.add_argument("--subset", choices=_SUBSET_CHOICES, default="both")
args, _ = parser.parse_known_args()

N_SAMPLES  = args.n
USE_MMS    = not args.no_mms

_ALL_SUBSETS = {
    # South Asian — Indic scripts
    "hi_in":      "hi",   # Hindi (Devanagari)
    "pa_in":      "pa",   # Punjabi (Gurmukhi)
    "ne_np":      "ne",   # Nepali (Devanagari)
    "bn_in":      "bn",   # Bengali
    # South Asian — Arabic/Nastaliq script
    "ur_pk":      "ur",   # Urdu
    "ps_af":      "ps",   # Pashto
    "sd_in":      "sd",   # Sindhi
    # East / Southeast Asian
    "cmn_hans_cn": "zh",  # Mandarin Chinese
    "my_mm":      "my",   # Burmese
    # West Asian / Persian
    "fa_ir":      "fa",   # Persian / Farsi
    "ar_eg":      "ar",   # Arabic (Egyptian)
    # Central Asian
    "tg_tj":      "tg",   # Tajik
    "uz_uz":      "uz",   # Uzbek
    "kk_kz":      "kk",   # Kazakh
}
RUN_SUBSETS = (
    _ALL_SUBSETS
    if args.subset in ("both", "all")
    else {args.subset: _ALL_SUBSETS[args.subset]}
)

# ── Model paths ───────────────────────────────────────────────────────────────
WHISPER_PATH  = ROOT / "models/whisper-large-v3-turbo-ct2"
FASTTEXT_PATH = ROOT / "models/langid/lid.176.bin"
MMS_PATH      = ROOT / "models/mms-lid-256"
OUT_PATH      = ROOT / "output/fleurs_eval_results.json"

# ── Config registry ───────────────────────────────────────────────────────────
CONFIGS = ["whisper_only", "whisper_ft", "whisper_ft_mms", "full_system"]
CONFIG_LABELS = {
    "whisper_only":    "Whisper-only",
    "whisper_ft":      "Whisper + FastText",
    "whisper_ft_mms":  "Whisper + FastText + MMS-LID",
    "full_system":     "Full System (+ Script-Cascade)",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    from faster_whisper import WhisperModel
    from language_module import FastTextLangDetector, DialectDetector, LanguageRouter
    from mms_module import MMSLangDetector

    print("Loading Whisper ...", end=" ", flush=True)
    whisper = WhisperModel(
        str(WHISPER_PATH), device="cpu", compute_type="int8",
        cpu_threads=min(os.cpu_count() or 4, 8),
    )
    print("ok")

    print("Loading FastText ...", end=" ", flush=True)
    ft = FastTextLangDetector(model_path=str(FASTTEXT_PATH))
    print("ok")

    mms = None
    if USE_MMS:
        print("Loading MMS-LID ...", end=" ", flush=True)
        try:
            mms = MMSLangDetector(model_path=str(MMS_PATH), device="cpu")
            print("ok")
        except Exception as e:
            print(f"FAILED ({e}) — running without MMS-LID")

    dd     = DialectDetector()
    router = LanguageRouter()
    return whisper, ft, mms, dd, router


# ── Audio helpers ─────────────────────────────────────────────────────────────

def audio_to_wav(sample: dict, tmp_path: str) -> None:
    """Write FLEURS sample audio to a temp WAV file."""
    import soundfile as sf

    audio_data = sample.get("audio", sample)

    # Standard HuggingFace datasets format
    if isinstance(audio_data, dict) and "array" in audio_data:
        arr = np.asarray(audio_data["array"], dtype=np.float32)
        sr  = int(audio_data.get("sampling_rate", 16000))
    else:
        # torchcodec AudioDecoder (alternative datasets backend)
        s   = audio_data.get_all_samples()
        arr = s.data.numpy()[0].astype(np.float32)
        sr  = 16000

    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr  = 16000

    sf.write(tmp_path, arr, sr)


# ── Per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    whisper,
    ft,
    mms,
    dd,
    router,
    audio_path: str,
    true_lang: str,
) -> Dict:
    from language_module import WHISPER_TO_ISO

    # ── Whisper ───────────────────────────────────────────────────────────────
    segs_iter, info = whisper.transcribe(
        audio_path,
        beam_size=2,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
        vad_filter=False,   # keep full audio for LangID accuracy
    )
    transcript = " ".join(s.text.strip() for s in segs_iter if s.text.strip())
    wl  = WHISPER_TO_ISO.get(info.language.lower(), info.language.lower())
    wlp = float(info.language_probability)

    # ── FastText ──────────────────────────────────────────────────────────────
    if transcript.strip():
        ft_res = ft.detect(transcript)
        fl, fc = ft_res["language"], ft_res["confidence"]
    else:
        fl, fc = "unknown", 0.0

    # ── MMS-LID ───────────────────────────────────────────────────────────────
    ml, mc = "unknown", 0.0
    if mms is not None:
        try:
            mms_res = mms.detect(audio_path)
            ml, mc  = mms_res["language"], float(mms_res["confidence"])
        except Exception:
            pass

    # ── DialectDetector (script heuristic) ────────────────────────────────────
    if transcript.strip():
        dd_res  = dd.detect_code_mix(transcript)
        dialect = dd_res.get("dialect", "unknown")
    else:
        dialect = "unknown"

    # ── Config 1: Whisper-only ────────────────────────────────────────────────
    pred_c1 = wl

    # ── Config 2: Whisper + FastText ──────────────────────────────────────────
    r2 = router.detect_family(
        whisper_lang=wl, transcript=transcript,
        fasttext_lang=fl, fasttext_conf=fc,
        whisper_lang_prob=wlp,
        dialect=None,
        mms_lang=None, mms_conf=0.0,
    )
    pred_c2 = r2["final_language"]

    # ── Config 3: Whisper + FastText + MMS-LID (no Script-Cascade) ───────────
    r3 = router.detect_family(
        whisper_lang=wl, transcript=transcript,
        fasttext_lang=fl, fasttext_conf=fc,
        whisper_lang_prob=wlp,
        dialect=None,
        mms_lang=ml, mms_conf=mc,
    )
    pred_c3 = r3["final_language"]

    # ── Config 4: Full System (+ Script-Cascade) ──────────────────────────────
    r4 = router.detect_family(
        whisper_lang=wl, transcript=transcript,
        fasttext_lang=fl, fasttext_conf=fc,
        whisper_lang_prob=wlp,
        dialect=dialect,
        mms_lang=ml, mms_conf=mc,
    )
    pred_c4 = r4["final_language"]

    predictions = {
        "whisper_only":   pred_c1,
        "whisper_ft":     pred_c2,
        "whisper_ft_mms": pred_c3,
        "full_system":    pred_c4,
    }
    correct = {cfg: (predictions[cfg] == true_lang) for cfg in CONFIGS}

    return {
        "true_lang":    true_lang,
        "transcript":   transcript[:200],
        "whisper_lang": wl,
        "whisper_prob": round(wlp, 3),
        "ft_lang":      fl,
        "ft_conf":      round(fc, 3),
        "mms_lang":     ml,
        "mms_conf":     round(mc, 3),
        "dialect":      dialect,
        "predictions":  predictions,
        "correct":      correct,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== FLEURS LangID Evaluation — VANI ===")
    print(f"    Samples per subset : {N_SAMPLES}")
    print(f"    MMS-LID            : {'enabled' if USE_MMS else 'disabled'}")
    print(f"    Subsets            : {list(RUN_SUBSETS)}\n")

    t_start = time.time()
    whisper, ft, mms, dd, router = load_models()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: 'datasets' package not installed.")

    all_results: Dict[str, List[Dict]] = {}
    accuracy: Dict[str, Dict[str, float]] = {cfg: {} for cfg in CONFIGS}

    for subset, true_lang in RUN_SUBSETS.items():
        print(f"\n── {subset}  (ground-truth lang: {true_lang}) ─────────────", flush=True)

        try:
            ds = load_dataset("google/fleurs", subset, split="test")
        except Exception as e:
            print(f"  Cannot load dataset: {e}")
            continue

        subset_results: List[Dict] = []
        correct_n = {cfg: 0 for cfg in CONFIGS}
        n_done = 0
        n_errors = 0

        # Iterate all samples; keep only those within 2--20 s (matching paper criterion)
        for idx in range(len(ds)):
            if n_done >= N_SAMPLES:
                break

            sample = ds[idx]

            # Duration filter: 2--20 s (handles both dict and AudioDecoder formats)
            audio = sample.get("audio", {})
            try:
                if isinstance(audio, dict) and "array" in audio:
                    arr_len = len(audio["array"])
                    sr = audio.get("sampling_rate", 16000) or 16000
                else:
                    # torchcodec AudioDecoder
                    _s = audio.get_all_samples()
                    arr_len = _s.data.shape[-1]
                    sr = int(_s.sample_rate) if hasattr(_s, "sample_rate") else 16000
                duration = arr_len / sr if arr_len > 0 else 0.0
            except Exception:
                duration = 5.0  # assume valid if we can't check
            if not (2.0 <= duration <= 20.0):
                continue

            reference = (sample.get("transcription") or "").strip()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            try:
                audio_to_wav(sample, tmp_path)
                result = evaluate_sample(
                    whisper, ft, mms, dd, router, tmp_path, true_lang
                )
                result["reference"] = reference
                subset_results.append(result)
                for cfg in CONFIGS:
                    if result["correct"][cfg]:
                        correct_n[cfg] += 1
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  [warn] sample {n_done}: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            n_done += 1

            if n_done % 20 == 0 or n_done >= N_SAMPLES:
                parts = []
                for cfg in CONFIGS:
                    acc = correct_n[cfg] / n_done * 100
                    parts.append(f"{cfg[:8]}={acc:.0f}%")
                print(f"  [{n_done:>3}/{N_SAMPLES}]  " + "  ".join(parts), flush=True)

        all_results[subset] = subset_results
        for cfg in CONFIGS:
            accuracy[cfg][subset] = correct_n[cfg] / n_done if n_done else 0.0

        # ── WER + confidence stats for this subset ────────────────────────────
        try:
            import jiwer
            wer_transform = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ])
            hyps = [r["transcript"].lower().strip() for r in subset_results
                    if r.get("reference") and r.get("transcript")]
            refs = [r["reference"].lower().strip()   for r in subset_results
                    if r.get("reference") and r.get("transcript")]
            if hyps and refs:
                wer_val = jiwer.wer(refs, hyps,
                                    hypothesis_transform=wer_transform,
                                    reference_transform=wer_transform)
                accuracy.setdefault("_wer", {})[subset] = round(wer_val * 100, 1)
            else:
                accuracy.setdefault("_wer", {})[subset] = None
        except Exception:
            accuracy.setdefault("_wer", {})[subset] = None

        confs = [r.get("whisper_prob", 0) for r in subset_results]
        accuracy.setdefault("_conf", {})[subset] = round(
            sum(confs) / len(confs), 3) if confs else 0.0

        print(f"\n  Final ({subset}, n={n_done}):")
        for cfg in CONFIGS:
            acc = accuracy[cfg][subset] * 100
            print(f"    {CONFIG_LABELS[cfg]:<44}  {acc:5.1f}%")
        wer_str = f"{accuracy['_wer'][subset]:.1f}%" if accuracy['_wer'][subset] is not None else "N/A"
        print(f"    Mean confidence: {accuracy['_conf'][subset]:.3f}  |  WER (script-naive): {wer_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    subsets = list(RUN_SUBSETS)

    print(f"\n{'='*72}")
    print("  LANGID ACCURACY SUMMARY  (% correct)")
    print(f"{'='*72}")
    col_w = 10
    hdr = f"  {'Configuration':<44}" + "".join(f"{s:>{col_w}}" for s in subsets)
    if len(subsets) > 1:
        hdr += f"{'Avg':>{col_w}}"
    print(hdr)
    print("  " + "-" * (44 + col_w * (len(subsets) + (1 if len(subsets) > 1 else 0))))

    for cfg in CONFIGS:
        accs = [accuracy[cfg].get(s, 0) * 100 for s in subsets]
        row  = f"  {CONFIG_LABELS[cfg]:<44}" + "".join(f"{a:>{col_w-1}.1f}%" for a in accs)
        if len(subsets) > 1:
            row += f"{sum(accs)/len(accs):>{col_w-1}.1f}%"
        print(row)

    print(f"{'='*72}")
    print(f"\n  Elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy":    accuracy,
        "n_samples":   N_SAMPLES,
        "subsets":     RUN_SUBSETS,
        "configs":     CONFIG_LABELS,
        "results":     all_results,
        "elapsed_sec": round(elapsed, 1),
        "use_mms":     USE_MMS,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"  Results saved → {OUT_PATH}")

    # ── Also write ablation_results.csv (format expected by update_paper_ablation.py) ──
    _SUBSET_TO_LANG = _ALL_SUBSETS
    _CFG_TO_COL = {
        "whisper_only":   "c1_ok",
        "whisper_ft":     "c2_ok",
        "whisper_ft_mms": "c3_ok",
        "full_system":    "c4_ok",
    }
    _PRED_COL = {
        "whisper_only":   "c1_whisper",
        "whisper_ft":     "c2_w_ft",
        "whisper_ft_mms": "c3_3way",
        "full_system":    "c4_vani",
    }

    csv_path = ROOT / "ablation_results.csv"
    import csv as _csv
    fieldnames = ["lang","true","duration","whisper_lang","whisper_prob",
                  "transcript","c1_whisper","c2_w_ft","c3_3way","c4_vani",
                  "c1_ok","c2_ok","c3_ok","c4_ok"]

    # If only running a subset, append new rows rather than overwriting existing data
    new_langs  = {_SUBSET_TO_LANG.get(s, s) for s in all_results}
    append_mode = (len(RUN_SUBSETS) < len(_ALL_SUBSETS)) and csv_path.exists()
    if append_mode:
        # Load existing rows, drop any that match langs we just re-evaluated
        existing_rows = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                if row.get("lang") not in new_langs:
                    existing_rows.append(row)
    else:
        existing_rows = []

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for subset, samples in all_results.items():
            lang = _SUBSET_TO_LANG.get(subset, subset)
            for s in samples:
                preds = s.get("predictions", {})
                corr  = s.get("correct", {})
                writer.writerow({
                    "lang":       lang,
                    "true":       lang,
                    "duration":   "",
                    "whisper_lang": s.get("whisper_lang", ""),
                    "whisper_prob": s.get("whisper_prob", ""),
                    "transcript": s.get("transcript", "")[:300],
                    "c1_whisper": preds.get("whisper_only", ""),
                    "c2_w_ft":    preds.get("whisper_ft", ""),
                    "c3_3way":    preds.get("whisper_ft_mms", ""),
                    "c4_vani":    preds.get("full_system", ""),
                    "c1_ok": int(corr.get("whisper_only", False)),
                    "c2_ok": int(corr.get("whisper_ft", False)),
                    "c3_ok": int(corr.get("whisper_ft_mms", False)),
                    "c4_ok": int(corr.get("full_system", False)),
                })
    mode_str = "(appended to existing data)" if append_mode else ""
    print(f"  CSV saved      → {csv_path} {mode_str}")
    print("\nRun:  python update_paper_ablation.py   to patch VANI_Paper.tex")


if __name__ == "__main__":
    main()

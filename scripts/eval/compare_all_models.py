"""
compare_all_models.py  --  Cross-model WER / chrF evaluation for VANI
======================================================================
Compares three systems across all 7 fine-tuned languages:

  A) Whisper large-v3 baseline  (openai/whisper-large-v3, no fine-tuning)
     NOTE: until 2026-07-10 this silently loaded whisper-large-v3-TURBO-ct2 while every
     docstring, report and paper called it "large-v3". Turbo defaults to the translate
     task, which is the whole reason the published zh baseline read 100.03% WER. Build
     the real baseline with: python scripts/build_baseline_ct2.py
  B) Whisper fine-tuned CT2     (VANI deployed models)
  C) SeamlessM4T v2 large       (Meta, single-model ASR + translation)

  Translation quality (Whisper+NLLB vs SeamlessM4T S2TT) measured with chrF
  against FLEURS English reference sentences.

Datasets:
  pa/ps/ur/ne/zh/hi  ->  FLEURS test split  (ASR WER + translation chrF)
  ks                 ->  IndicVoices val split (ASR WER only, no EN refs)

Metrics:
  WER   = Word Error Rate  (jiwer, lower is better)
  chrF  = chrF score       (sacrebleu, higher is better, 0-100)

Usage:
  # Full evaluation (all 7 languages, all models) — takes several hours
  python scripts/eval/compare_all_models.py

  # Single language, faster
  python scripts/eval/compare_all_models.py --lang ur

  # Skip SeamlessM4T (if not downloaded yet)
  python scripts/eval/compare_all_models.py --skip-seamless

  # Limit samples per language (quick test)
  python scripts/eval/compare_all_models.py --samples 20

  # Results saved to:
  #   docs/model_comparison_results.json
  #   docs/model_comparison_report.md
"""

import argparse
import json
import pathlib
import sys
import time
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
SEAMLESS_DIR = MODELS_DIR / "seamless-m4t-v2-large"
OUT_JSON     = ROOT / "docs" / "model_comparison_results.json"
OUT_HYPS     = ROOT / "eval_data" / "model_comparison_hyps.jsonl"
OUT_MD       = ROOT / "docs" / "model_comparison_report.md"

# ── Language config ────────────────────────────────────────────────────────────
LANG_CFG = {
    "pa": {
        "fleurs":       "pa_in",
        "nllb_src":     "pan_Guru",
        "seamless_src": "pan",
        "name":         "Punjabi",
        "script":       "Gurmukhi",
        "whisper_lang": "pa",
    },
    "ps": {
        "fleurs":       "ps_af",
        "nllb_src":     "pbt_Arab",
        "seamless_src": "pbt",
        "name":         "Pashto",
        "script":       "Nastaliq",
        "whisper_lang": "ps",
    },
    "ur": {
        "fleurs":       "ur_pk",
        "nllb_src":     "urd_Arab",
        "seamless_src": "urd",
        "name":         "Urdu",
        "script":       "Nastaliq",
        "whisper_lang": "ur",
    },
    "ne": {
        "fleurs":       "ne_np",
        "nllb_src":     "npi_Deva",
        "seamless_src": "npi",
        "name":         "Nepali",
        "script":       "Devanagari",
        "whisper_lang": "ne",
    },
    "zh": {
        "fleurs":       "cmn_hans_cn",
        "nllb_src":     "zho_Hans",
        "seamless_src": "cmn",
        "name":         "Mandarin",
        "script":       "Simplified Han",
        "whisper_lang": "zh",
    },
    "hi": {
        "fleurs":       "hi_in",
        "nllb_src":     "hin_Deva",
        "seamless_src": "hin",
        "name":         "Hindi",
        "script":       "Devanagari",
        "whisper_lang": "hi",
    },
    "ks": {
        "fleurs":       None,   # no FLEURS config for Kashmiri
        "custom_ds":    "humair025/KashmiriSpeech-IndicVoices",
        "nllb_src":     "kas_Arab",
        "seamless_src": None,   # SeamlessM4T v2 does not support Kashmiri (kas)
        "name":         "Kashmiri",
        "script":       "Nastaliq",
        "whisper_lang": "ur",   # ur proxy token in fine-tuned model
    },
}

# ── Text normalisation ─────────────────────────────────────────────────────────

# Normalisation and WER/CER live in text_norm.py — one definition for every evaluator.
# Three divergent copies of normalise() are why zh was scored wrong for months.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from text_norm import normalise, compute_wer, compute_cer  # noqa: E402


# ── Dataset loader ─────────────────────────────────────────────────────────────

def decode_audio_bytes(audio_dict: dict, target_sr: int = 16000):
    """Decode raw audio bytes (from Audio(decode=False)) using soundfile."""
    import io
    import numpy as np
    import soundfile as sf
    raw = audio_dict.get("bytes")
    path = audio_dict.get("path")
    if raw:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    elif path:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        return np.zeros(16000, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype("float32")


def load_fleurs_test(fleurs_config: str, n_samples: int, token: str = None):
    from datasets import load_dataset, Audio
    ds = load_dataset(
        "google/fleurs", fleurs_config,
        split="test", token=token,
        cache_dir=str(DATA_DIR / "fleurs"),
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=42).select(range(n_samples))
    return ds


def load_fleurs_en_refs_for_samples(samples: list, token: str = None) -> list:
    """Given already-loaded samples (with 'id' field), return aligned English refs."""
    from datasets import load_dataset, Audio
    en = load_dataset(
        "google/fleurs", "en_us",
        split="test", token=token,
        cache_dir=str(DATA_DIR / "fleurs"),
    ).cast_column("audio", Audio(decode=False))
    en_map = {row["id"]: normalise(row["transcription"], "en") for row in en}
    return [en_map.get(s["id"], "") for s in samples]


def load_indicvoices_test(n_samples: int, token: str = None):
    from datasets import load_dataset, Audio
    # Reuse training cache (validation.arrow already downloaded there); avoids re-download
    ks_cache = ROOT / "finetune_runs" / "ks" / "data" / "custom"
    fallback_cache = DATA_DIR / "custom"
    cache_dir = str(ks_cache) if ks_cache.exists() else str(fallback_cache)
    ds = load_dataset(
        "humair025/KashmiriSpeech-IndicVoices",
        split="validation", token=token,
        cache_dir=cache_dir,
    )
    ds = ds.cast_column("audio_filepath", Audio(decode=False))
    ds = ds.filter(lambda x: x["duration"] is not None and 2.0 <= x["duration"] <= 20.0)
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=42).select(range(n_samples))
    return ds


# ── Model runners ──────────────────────────────────────────────────────────────

def _get_audio_array(sample: dict) -> "np.ndarray":
    """Extract and decode audio from a sample regardless of column name."""
    raw = sample.get("audio") or sample.get("audio_filepath")
    return decode_audio_bytes(raw)


# The TRUE large-v3, built by scripts/build_baseline_ct2.py. Previously this pointed at
# whisper-large-v3-turbo-ct2 while every docstring and the paper called it "large-v3".
# Turbo defaults to the translate task, which is why the zh baseline read 100.03%.
BASELINE_NAME = "whisper-large-v3-ct2"


def run_whisper_baseline(samples, lang_cfg: dict, device: str):
    """Whisper large-v3 (openai/whisper-large-v3), NO fine-tuning. Returns RAW text."""
    from faster_whisper import WhisperModel
    base_path = MODELS_DIR / BASELINE_NAME
    if not base_path.exists():
        print(f"  [WARN] Baseline model not found at {base_path}. "
              f"Build it with: python scripts/build_baseline_ct2.py")
        return [None] * len(samples)
    wm = WhisperModel(str(base_path), device=device, compute_type="int8")
    preds = []
    for i, sample in enumerate(samples):
        arr  = _get_audio_array(sample)
        segs, _ = wm.transcribe(arr, language=lang_cfg["whisper_lang"], task="transcribe")
        preds.append(" ".join(s.text for s in segs).strip())
        if (i + 1) % 10 == 0:
            print(f"    whisper-baseline: {i+1}/{len(samples)}")
    del wm
    return preds


def run_whisper_finetuned(samples, lang: str, lang_cfg: dict, device: str):
    """Fine-tuned Whisper CT2 (VANI deployed model). Returns RAW text."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    model_key = f"whisper_model_{lang}"
    model_path = ROOT / cfg["paths"].get(model_key, cfg["paths"]["whisper_model"])
    if not model_path.exists():
        print(f"  [WARN] Fine-tuned model not found: {model_path}, skipping")
        return [None] * len(samples)
    from faster_whisper import WhisperModel
    wm = WhisperModel(str(model_path), device=device, compute_type="int8")
    preds = []
    for i, sample in enumerate(samples):
        arr  = _get_audio_array(sample)
        segs, _ = wm.transcribe(arr, language=lang_cfg["whisper_lang"], task="transcribe")
        preds.append(" ".join(s.text for s in segs).strip())
        if (i + 1) % 10 == 0:
            print(f"    whisper-finetuned: {i+1}/{len(samples)}")
    del wm
    return preds


def run_nllb_translation(whisper_preds: list, lang_cfg: dict, device: str):
    """NLLB-200 translation: takes Whisper ASR output -> English."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    nllb_path = MODELS_DIR / "nllb-200-distilled-600M"
    if not nllb_path.exists():
        print(f"  [WARN] NLLB model not found: {nllb_path}, skipping")
        return [None] * len(whisper_preds)
    tok = AutoTokenizer.from_pretrained(str(nllb_path), src_lang=lang_cfg["nllb_src"])
    model = AutoModelForSeq2SeqLM.from_pretrained(str(nllb_path)).to(device)
    model.eval()
    forced_id = tok.convert_tokens_to_ids("eng_Latn")
    preds = []
    for i, text in enumerate(whisper_preds):
        if text is None:
            preds.append(None); continue
        enc = tok(text, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=forced_id, max_new_tokens=256)
        preds.append(normalise(tok.decode(out[0], skip_special_tokens=True), "en"))
        if (i + 1) % 10 == 0:
            print(f"    nllb: {i+1}/{len(whisper_preds)}")
    del model, tok
    return preds


def run_seamless_asr(samples, lang_cfg: dict, device: str):
    """SeamlessM4T v2: speech -> source-language text. Returns RAW text.

    Note it used to normalise with `sm_lang` ("cmn"), while references were normalised
    with the dataset code ("zh") -- so the two sides never went through the same
    normaliser. Raw text now, normalised once in compute_wer against the dataset code.
    """
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    if not SEAMLESS_DIR.exists():
        print(f"  [WARN] SeamlessM4T not found at {SEAMLESS_DIR}, skipping")
        return [None] * len(samples), [None] * len(samples)
    proc  = AutoProcessor.from_pretrained(str(SEAMLESS_DIR))
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(SEAMLESS_DIR), torch_dtype=dtype
    ).to(device)
    model.eval()
    sm_lang = lang_cfg["seamless_src"]
    asr_preds = []
    s2tt_preds = []
    for i, sample in enumerate(samples):
        arr = _get_audio_array(sample)
        inputs = proc(audio=arr, return_tensors="pt",
                      sampling_rate=16000, src_lang=sm_lang).to(device)
        with torch.no_grad():
            # ASR
            toks_asr = model.generate(**inputs, tgt_lang=sm_lang)
            asr_text = proc.decode(toks_asr[0], skip_special_tokens=True)
            # S2TT -> English
            toks_en  = model.generate(**inputs, tgt_lang="eng")
            en_text  = proc.decode(toks_en[0], skip_special_tokens=True)
        asr_preds.append(asr_text.strip())
        s2tt_preds.append(en_text.strip())
        if (i + 1) % 10 == 0:
            print(f"    seamless: {i+1}/{len(samples)}")
    del model, proc
    return asr_preds, s2tt_preds


# ── Metrics ────────────────────────────────────────────────────────────────────

# compute_wer / compute_cer come from text_norm; do not redefine them here.


def compute_chrf(preds: list, refs: list) -> float:
    import sacrebleu
    valid = [(p, r) for p, r in zip(preds, refs) if p is not None and r]
    if not valid:
        return None
    p_list, r_list = zip(*valid)
    score = sacrebleu.corpus_chrf(list(p_list), [list(r_list)])
    return round(score.score, 2)


# ── Per-language evaluation ────────────────────────────────────────────────────

def evaluate_language(lang: str, n_samples: int, skip_seamless: bool,
                      skip_baseline: bool, device: str, token: str) -> dict:
    cfg = LANG_CFG[lang]
    print(f"\n{'='*60}")
    print(f"  Evaluating: {cfg['name']} ({lang.upper()})  —  {n_samples} samples")
    print(f"{'='*60}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("  Loading test data ...")
    # References stay RAW. Normalisation happens once, inside compute_wer/compute_cer,
    # so both sides always go through the identical normaliser.
    if cfg.get("fleurs"):
        samples  = list(load_fleurs_test(cfg["fleurs"], n_samples, token))
        src_refs = [s["transcription"] for s in samples]
        en_refs  = load_fleurs_en_refs_for_samples(samples, token)
    else:
        raw      = load_indicvoices_test(n_samples, token)
        samples  = list(raw)
        src_refs = [s["normalized"] for s in samples]
        en_refs  = None   # no English references for Kashmiri IndicVoices

    print(f"  Loaded {len(samples)} samples")

    results = {"lang": lang, "name": cfg["name"], "n": len(samples)}
    hyps = []   # raw per-utterance output, so this never has to be re-run to re-score

    def record(system, preds, model_name):
        for i, p in enumerate(preds):
            if p is None:
                continue
            hyps.append({"lang": lang, "system": system, "model": model_name,
                         "idx": i, "ref": src_refs[i], "hyp": p})

    # ── A: Whisper baseline ───────────────────────────────────────────────────
    if skip_baseline:
        results["whisper_baseline_wer"]  = None
        results["whisper_baseline_time"] = 0
        print("\n  [A] Whisper baseline: skipped")
    else:
        print(f"\n  [A] Whisper large-v3 baseline ({BASELINE_NAME}) ...")
        t0 = time.time()
        base_preds = run_whisper_baseline(samples, cfg, device)
        results["whisper_baseline_wer"]  = compute_wer(base_preds, src_refs, lang)
        results["whisper_baseline_cer"]  = compute_cer(base_preds, src_refs, lang)
        results["whisper_baseline_time"] = round(time.time() - t0, 1)
        record("whisper_baseline", base_preds, BASELINE_NAME)
        print(f"    WER: {results['whisper_baseline_wer']}%  "
              f"CER: {results['whisper_baseline_cer']}%  ({results['whisper_baseline_time']}s)")

    # ── B: Whisper fine-tuned ─────────────────────────────────────────────────
    print("\n  [B] Whisper fine-tuned (CT2 int8) ...")
    t0 = time.time()
    ft_preds = run_whisper_finetuned(samples, lang, cfg, device)
    results["whisper_ft_wer"]  = compute_wer(ft_preds, src_refs, lang)
    results["whisper_ft_cer"]  = compute_cer(ft_preds, src_refs, lang)
    results["whisper_ft_time"] = round(time.time() - t0, 1)
    record("whisper_ft", ft_preds, f"whisper_model_{lang}")
    print(f"    WER: {results['whisper_ft_wer']}%  CER: {results['whisper_ft_cer']}%  "
          f"({results['whisper_ft_time']}s)")

    # ── B2: Whisper fine-tuned + NLLB translation ─────────────────────────────
    if en_refs:
        print("\n  [B2] Whisper fine-tuned + NLLB-200 -> English ...")
        t0 = time.time()
        # NLLB previously received normalised ASR output; keep that so chrF stays comparable.
        nllb_preds = run_nllb_translation([normalise(p, lang) if p else p for p in ft_preds],
                                          cfg, device)
        results["whisper_nllb_chrf"]  = compute_chrf(nllb_preds, en_refs)
        results["whisper_nllb_time"]  = round(time.time() - t0, 1)
        print(f"    chrF: {results['whisper_nllb_chrf']}  ({results['whisper_nllb_time']}s)")
    else:
        results["whisper_nllb_chrf"] = None
        print("\n  [B2] Whisper+NLLB: skipped (no English references)")

    # ── C: SeamlessM4T ───────────────────────────────────────────────────────
    if not skip_seamless and cfg.get("seamless_src") is not None:
        print("\n  [C] SeamlessM4T v2 large ...")
        t0 = time.time()
        sm_asr, sm_s2tt = run_seamless_asr(samples, cfg, device)
        results["seamless_asr_wer"]   = compute_wer(sm_asr, src_refs, lang)
        results["seamless_asr_cer"]   = compute_cer(sm_asr, src_refs, lang)
        results["seamless_s2tt_chrf"] = compute_chrf(sm_s2tt, en_refs) if en_refs else None
        results["seamless_time"]      = round(time.time() - t0, 1)
        record("seamless_zs", sm_asr, "seamless-m4t-v2-large")
        print(f"    ASR WER: {results['seamless_asr_wer']}%  "
              f"CER: {results['seamless_asr_cer']}%  "
              f"S2TT chrF: {results['seamless_s2tt_chrf']}  "
              f"({results['seamless_time']}s)")
    else:
        results["seamless_asr_wer"]   = None
        results["seamless_s2tt_chrf"] = None
        if skip_seamless:
            print("\n  [C] SeamlessM4T: skipped (--skip-seamless)")
        else:
            # ks has seamless_src=None: SeamlessM4T v2 has no Kashmiri. Confirmed
            # 2026-07-10 -- the urd-proxy also fails (WER 109%, CER 69%).
            print("\n  [C] SeamlessM4T: skipped (language not supported by model)")

    return results, hyps


# ── Report generation ──────────────────────────────────────────────────────────

def fmt(val, suffix=""):
    if val is None:
        return "—"
    return f"{val}{suffix}"


def generate_report(all_results: list):
    lines = [
        "# VANI — Cross-Model Evaluation Report",
        "**Whisper baseline vs Fine-tuned Whisper vs SeamlessM4T v2**",
        f"**Date:** {time.strftime('%d %B %Y')}  ·  **Hardware:** RTX 5060 8 GB CUDA",
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | What it measures | Range |",
        "|--------|-----------------|-------|",
        "| **WER** | Word Error Rate — ASR accuracy in source language (lower = better) | 0–100% |",
        "| **chrF** | Character F-score — translation quality to English (higher = better) | 0–100 |",
        "",
        "---",
        "",
        "## ASR Word Error Rate (Source Language Transcription)",
        "",
        "| Language | Script | Whisper Baseline | Whisper Fine-Tuned | SeamlessM4T v2 | Improvement |",
        "|----------|--------|-----------------|-------------------|----------------|-------------|",
    ]

    for r in all_results:
        base = r.get("whisper_baseline_wer")
        ft   = r.get("whisper_ft_wer")
        sm   = r.get("seamless_asr_wer")
        imp  = f"+{round(base-ft,1)}pp" if base and ft else "—"
        lines.append(
            f"| {r['name']} ({r['lang'].upper()}) | {LANG_CFG[r['lang']]['script']} "
            f"| {fmt(base,'%')} | {fmt(ft,'%')} | {fmt(sm,'%')} | {imp} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Translation Quality — chrF Score (→ English)",
        "",
        "| Language | Whisper+NLLB chrF | SeamlessM4T S2TT chrF | Winner |",
        "|----------|------------------|----------------------|--------|",
    ]

    for r in all_results:
        nllb = r.get("whisper_nllb_chrf")
        sm   = r.get("seamless_s2tt_chrf")
        if nllb is None and sm is None:
            winner = "—"
        elif nllb is None:
            winner = "SeamlessM4T"
        elif sm is None:
            winner = "Whisper+NLLB"
        else:
            winner = "SeamlessM4T" if sm > nllb else "Whisper+NLLB"
        lines.append(
            f"| {r['name']} ({r['lang'].upper()}) | {fmt(nllb)} | {fmt(sm)} | {winner} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    # Auto-generate findings
    ft_wins_wer   = [r for r in all_results if r.get("whisper_ft_wer") and r.get("seamless_asr_wer")
                     and r["whisper_ft_wer"] < r["seamless_asr_wer"]]
    sm_wins_wer   = [r for r in all_results if r.get("whisper_ft_wer") and r.get("seamless_asr_wer")
                     and r["seamless_asr_wer"] < r["whisper_ft_wer"]]
    sm_wins_chrf  = [r for r in all_results if r.get("seamless_s2tt_chrf") and r.get("whisper_nllb_chrf")
                     and r["seamless_s2tt_chrf"] > r["whisper_nllb_chrf"]]
    nllb_wins_chrf = [r for r in all_results if r.get("seamless_s2tt_chrf") and r.get("whisper_nllb_chrf")
                      and r["whisper_nllb_chrf"] >= r["seamless_s2tt_chrf"]]

    if ft_wins_wer:
        langs = ", ".join(r["name"] for r in ft_wins_wer)
        lines.append(f"1. **Fine-tuned Whisper beats SeamlessM4T on ASR WER** for: {langs}")
    if sm_wins_wer:
        langs = ", ".join(r["name"] for r in sm_wins_wer)
        lines.append(f"2. **SeamlessM4T beats fine-tuned Whisper on ASR WER** for: {langs}")
    if sm_wins_chrf:
        langs = ", ".join(r["name"] for r in sm_wins_chrf)
        lines.append(f"3. **SeamlessM4T S2TT beats Whisper+NLLB on translation** for: {langs}")
    if nllb_wins_chrf:
        langs = ", ".join(r["name"] for r in nllb_wins_chrf)
        lines.append(f"4. **Whisper+NLLB beats SeamlessM4T on translation** for: {langs}")

    lines += [
        "",
        "---",
        "",
        f"*Generated: {time.strftime('%d %B %Y %H:%M')}  ·  VANI v2  ·  RTX 5060 8 GB*",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",          type=str, default=None,
                        help="Single language to evaluate (pa/ps/ur/ne/zh/hi/ks). Default: all.")
    parser.add_argument("--samples",       type=int, default=100,
                        help="Max test samples per language (default 100)")
    parser.add_argument("--skip-seamless", action="store_true",
                        help="Skip SeamlessM4T evaluation")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip Whisper baseline (no fine-tuning)")
    parser.add_argument("--device",        type=str, default=None,
                        help="cpu / cuda (auto-detect if not set)")
    args = parser.parse_args()

    import torch
    import os
    token  = os.environ.get("HF_TOKEN")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    langs  = [args.lang] if args.lang else list(LANG_CFG.keys())

    print(f"\nVANI Cross-Model Evaluation")
    print(f"Languages : {', '.join(langs)}")
    print(f"Samples   : {args.samples} per language")
    print(f"Device    : {device}")
    print(f"SeamlessM4T: {'SKIP' if args.skip_seamless else 'YES'}")

    # When filtering to a single language, merge with existing JSON so prior results are not lost
    all_results = []
    if args.lang and OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            all_results = [r for r in existing if r.get("lang") != args.lang]
        except Exception:
            pass
    t_total = time.time()
    all_hyps = []

    for lang in langs:
        try:
            r, hyps = evaluate_language(
                lang, args.samples,
                skip_seamless=args.skip_seamless,
                skip_baseline=args.skip_baseline,
                device=device, token=token,
            )
            all_results.append(r)
            all_hyps.extend(hyps)
        except Exception as e:
            print(f"\n[ERROR] {lang}: {e}")
            import traceback; traceback.print_exc()

    # ── Save results ──────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Results JSON -> {OUT_JSON}")

    # Raw hypotheses, so a normalisation fix never costs another GPU run again.
    # (This file not existing is why the zh whitespace bug required a full re-run.)
    if all_hyps:
        mode = "a" if (args.lang and OUT_HYPS.exists()) else "w"
        with OUT_HYPS.open(mode, encoding="utf-8") as fh:
            for h in all_hyps:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(f"[OK] Raw hypotheses ({len(all_hyps)} rows) -> {OUT_HYPS}")

    report = generate_report(all_results)
    OUT_MD.write_text(report, encoding="utf-8")
    print(f"[OK] Report MD   -> {OUT_MD}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY — Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"{'='*80}")
    print(f"  {'Language':<14} {'Base WER':>10} {'FT WER':>10} {'SM WER':>10} "
          f"{'W+NLLB chrF':>13} {'SM S2TT chrF':>13}")
    print(f"  {'-'*73}")
    for r in all_results:
        print(f"  {r['name']:<14} "
              f"{fmt(r.get('whisper_baseline_wer'),'%'):>10} "
              f"{fmt(r.get('whisper_ft_wer'),'%'):>10} "
              f"{fmt(r.get('seamless_asr_wer'),'%'):>10} "
              f"{fmt(r.get('whisper_nllb_chrf')):>13} "
              f"{fmt(r.get('seamless_s2tt_chrf')):>13}")
    print()


if __name__ == "__main__":
    main()

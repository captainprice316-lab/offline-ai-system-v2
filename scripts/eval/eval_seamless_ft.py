"""
eval_seamless_ft.py — Evaluate fine-tuned SeamlessM4T v2 LoRA adapters
=======================================================================
Loads each fine-tuned adapter from finetune_runs_seamless/<lang>/adapter/,
merges LoRA weights into the base model, then runs ASR + S2TT on FLEURS
test samples. Results saved to docs/seamless_ft_results.json.

Usage:
    python scripts/eval/eval_seamless_ft.py              # all 6 languages
    python scripts/eval/eval_seamless_ft.py --lang ur    # single language
    python scripts/eval/eval_seamless_ft.py --samples 50
"""

import argparse
import io
import json
import os
import pathlib
import re
import sys
import time
import unicodedata

import numpy as np

ROOT         = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
SEAMLESS_DIR = MODELS_DIR / "seamless-m4t-v2-large"
RUNS_DIR     = ROOT / "finetune_runs_seamless"
OUT_JSON     = ROOT / "docs" / "seamless_ft_results.json"
OUT_HYPS     = ROOT / "eval_data" / "seamless_ft_hyps.jsonl"

LANG_CFG = {
    "pa": {"fleurs": "pa_in",       "sm_lang": "pan", "name": "Punjabi"},
    "ps": {"fleurs": "ps_af",       "sm_lang": "pbt", "name": "Pashto"},
    "ur": {"fleurs": "ur_pk",       "sm_lang": "urd", "name": "Urdu"},
    "ne": {"fleurs": "ne_np",       "sm_lang": "npi", "name": "Nepali"},
    "zh": {"fleurs": "cmn_hans_cn", "sm_lang": "cmn", "name": "Mandarin"},
    "hi": {"fleurs": "hi_in",       "sm_lang": "hin", "name": "Hindi"},
}


# ── Normalisation ──────────────────────────────────────────────────────────────
# One definition, in text_norm.py. This file used to carry its own copy with no CJK
# segmentation, which is why zh sm_ft_asr_wer was published as 60.53 -- a whitespace
# artefact, not a model result.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from text_norm import normalise, compute_wer, compute_cer  # noqa: E402


# ── Audio decoding ─────────────────────────────────────────────────────────────

def decode_audio(audio_dict: dict, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    arr = audio_dict.get("array")
    if arr is not None:
        arr = np.array(arr, dtype=np.float32)
        sr  = audio_dict.get("sampling_rate", target_sr)
    else:
        raw  = audio_dict.get("bytes")
        path = audio_dict.get("path")
        if raw:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        elif path:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
        else:
            return np.zeros(target_sr, dtype="float32")

    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype("float32")


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_test_samples(fleurs_cfg: str, n: int):
    from datasets import load_dataset, Audio
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("google/fleurs", fleurs_cfg, split="test",
                      token=token, cache_dir=str(DATA_DIR / "fleurs"))
    ds = ds.cast_column("audio", Audio(decode=False))
    if len(ds) > n:
        ds = ds.shuffle(seed=42).select(range(n))
    return list(ds)


def load_en_refs(samples: list) -> list:
    from datasets import load_dataset, Audio
    token = os.environ.get("HF_TOKEN")
    en = load_dataset("google/fleurs", "en_us", split="test",
                      token=token, cache_dir=str(DATA_DIR / "fleurs"))
    en = en.cast_column("audio", Audio(decode=False))
    en_map = {r["id"]: normalise(r["transcription"], "en") for r in en}
    return [en_map.get(s["id"], "") for s in samples]


# ── Metrics ────────────────────────────────────────────────────────────────────

# compute_wer / compute_cer come from text_norm; do not redefine them here.


def compute_chrf(preds, refs):
    import sacrebleu
    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    if not valid:
        return None
    p_list, r_list = zip(*valid)
    return round(sacrebleu.corpus_chrf(list(p_list), [list(r_list)]).score, 2)


# ── Inference with fine-tuned SM4T ─────────────────────────────────────────────

def run_ft_seamless(samples, sm_lang: str, adapter_dir: pathlib.Path, device: str):
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel

    print(f"  Loading base model ...")
    processor  = AutoProcessor.from_pretrained(str(SEAMLESS_DIR))
    dtype      = torch.float16 if device != "cpu" else torch.float32
    base_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(SEAMLESS_DIR), torch_dtype=dtype
    )

    print(f"  Loading LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = model.merge_and_unload()   # merge LoRA into base weights
    model = model.to(device)
    model.eval()

    asr_preds  = []
    s2tt_preds = []

    for i, sample in enumerate(samples):
        arr    = decode_audio(sample["audio"])
        inputs = processor(audio=arr, return_tensors="pt",
                           sampling_rate=16000, src_lang=sm_lang).to(device)
        with torch.no_grad():
            toks_asr = model.generate(**inputs, tgt_lang=sm_lang)
            asr_text = processor.decode(toks_asr[0], skip_special_tokens=True)
            toks_en  = model.generate(**inputs, tgt_lang="eng")
            en_text  = processor.decode(toks_en[0],  skip_special_tokens=True)

        # RAW. Normalisation happens once in compute_wer, against the DATASET code.
        # This used to normalise with sm_lang ("cmn"), while refs used "zh" -- so the
        # two sides never went through the same normaliser.
        asr_preds.append(asr_text.strip())
        s2tt_preds.append(en_text.strip())

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(samples)}")

    del model, base_model, processor
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    return asr_preds, s2tt_preds


# ── Per-language evaluation ────────────────────────────────────────────────────

def evaluate_language(lang: str, n_samples: int, device: str) -> dict:
    cfg         = LANG_CFG[lang]
    sm_lang     = cfg["sm_lang"]
    adapter_dir = RUNS_DIR / lang / "adapter"

    print(f"\n{'='*60}")
    print(f"  Fine-tuned SeamlessM4T: {cfg['name']} ({lang})  —  {n_samples} samples")
    print(f"{'='*60}")

    if not adapter_dir.exists():
        print(f"  [SKIP] Adapter not found: {adapter_dir}")
        return None

    print("  Loading test data ...")
    samples = load_test_samples(cfg["fleurs"], n_samples)
    src_refs = [s["transcription"] for s in samples]   # RAW; normalised in compute_wer
    en_refs  = load_en_refs(samples)
    print(f"  Loaded {len(samples)} samples")

    t0 = time.time()
    asr_preds, s2tt_preds = run_ft_seamless(samples, sm_lang, adapter_dir, device)
    elapsed = round(time.time() - t0, 1)

    asr_wer  = compute_wer(asr_preds, src_refs, lang)
    asr_cer  = compute_cer(asr_preds, src_refs, lang)
    s2tt_chrf = compute_chrf(s2tt_preds, en_refs)

    # Raw hypotheses, so a normalisation fix never costs another GPU run.
    with OUT_HYPS.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(asr_preds):
            fh.write(json.dumps({"lang": lang, "system": "seamless_ft",
                                 "model": str(adapter_dir.name), "idx": i,
                                 "ref": src_refs[i], "hyp": p}, ensure_ascii=False) + "\n")

    print(f"\n  ASR WER : {asr_wer}%   CER: {asr_cer}%")
    print(f"  S2TT chrF: {s2tt_chrf}")
    print(f"  Time     : {elapsed}s")

    return {
        "lang":            lang,
        "name":            cfg["name"],
        "n":               len(samples),
        "sm_ft_asr_wer":   asr_wer,
        "sm_ft_asr_cer":   asr_cer,
        "sm_ft_s2tt_chrf": s2tt_chrf,
        "time":            elapsed,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",    type=str, default=None,
                        help="Single language (pa/ps/ur/ne/zh/hi). Default: all.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device",  type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    langs  = [args.lang] if args.lang else list(LANG_CFG.keys())

    print(f"\nSeamlessM4T Fine-Tuned Evaluation")
    print(f"Languages : {', '.join(langs)}")
    print(f"Samples   : {args.samples}")
    print(f"Device    : {device}")

    # Merge with existing results if single lang
    all_results = []
    if args.lang and OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            all_results = [r for r in existing if r.get("lang") != args.lang]
        except Exception:
            pass

    t_total = time.time()
    for lang in langs:
        try:
            r = evaluate_language(lang, args.samples, device)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {lang}: {e}")
            import traceback; traceback.print_exc()

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[OK] Results -> {OUT_JSON}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({(time.time()-t_total)/60:.1f} min total)")
    print(f"{'='*60}")
    print(f"  {'Language':<14} {'FT SM4T WER':>12} {'FT SM4T chrF':>13}")
    print(f"  {'-'*42}")
    for r in all_results:
        print(f"  {r['name']:<14} {str(r.get('sm_ft_asr_wer','—'))+'%':>12} "
              f"{str(r.get('sm_ft_s2tt_chrf','—')):>13}")


if __name__ == "__main__":
    main()

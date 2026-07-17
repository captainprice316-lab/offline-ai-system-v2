"""Evaluate the Kashmiri SeamlessM4T LoRA adapter (custom __kas__ token).

Head-to-head vs the deployed Whisper-ks model:
  - IndicVoices-R test split (same 372-sample set as training eval; Whisper: 74.02% WER)
  - robustness_cache/ks 30-clip set   (Whisper deployed, clean: 81.46% WER / 47.95% CER)

Scores WER + CER twice: raw jiwer (same ruler as the 74.02 figure) and NFC-normalised
both sides (the ks ruler is known-suspect: a Unicode normalisation mismatch inflated the
96.87% Whisper baseline). Raw per-utterance hypotheses are persisted to JSONL so any
future normaliser fix re-scores without a GPU re-run.

Usage:  python scripts/eval/eval_ks_seamless.py [--limit N] [--skip-robustness]
"""
import argparse
import io
import json
import pathlib
import sys
import unicodedata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT        = pathlib.Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "finetune_runs_seamless" / "ks" / "adapter"
BASE_DIR    = ROOT / "models" / "seamless-m4t-v2-large"
ROBUST_DIR  = ROOT / "robustness_cache" / "ks"
OUT_JSONL   = ROOT / "eval_data" / "ks_seamless_hyps.jsonl"
OUT_JSON    = ROOT / "docs" / "ks_seamless_results.json"

sys.path.insert(0, str(ROOT))
from finetune_seamless import LANG_CFG, load_indicvoices_ks, decode_audio  # noqa: E402


def score(refs, hyps):
    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    def nfc(s): return unicodedata.normalize("NFC", s)
    out = {
        "wer_raw": round(100 * jiwer_wer(refs, hyps), 2),
        "cer_raw": round(100 * jiwer_cer(refs, hyps), 2),
        "wer_nfc": round(100 * jiwer_wer([nfc(r) for r in refs], [nfc(h) for h in hyps]), 2),
        "cer_nfc": round(100 * jiwer_cer([nfc(r) for r in refs], [nfc(h) for h in hyps]), 2),
        "n": len(refs),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap IndicVoices test samples")
    ap.add_argument("--skip-robustness", action="store_true")
    # Decode fixes from the 2026-07-17 probe: the 129.29 WER was dominated by
    # early-EOS under-generation; duration-scaled min_new_tokens + no-repeat-
    # ngram recovered 128.28 -> 94.31 on a 50-sample subset (beams alone: worse).
    ap.add_argument("--min-tok-per-sec", type=float, default=0.0,
                    help="duration-scaled min_new_tokens (0 = off; probe used 2.5)")
    ap.add_argument("--min-tok-cap", type=int, default=180)
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="no_repeat_ngram_size (0 = off; probe used 3)")
    ap.add_argument("--adapter-dir", type=str, default=None,
                    help="override adapter dir (default: finetune_runs_seamless/ks/adapter)")
    args = ap.parse_args()

    global ADAPTER_DIR, OUT_JSONL, OUT_JSON
    if args.adapter_dir:
        ADAPTER_DIR = pathlib.Path(args.adapter_dir)
        tag = ADAPTER_DIR.parent.name          # e.g. ks_r16
        OUT_JSONL = ROOT / "eval_data" / f"{tag}_seamless_hyps.jsonl"
        OUT_JSON  = ROOT / "docs" / f"{tag}_seamless_results.json"

    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading adapter from {ADAPTER_DIR} on {device} ...")
    processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR))  # tokenizer has __kas__
    tok = processor.tokenizer
    kas_id = tok.convert_tokens_to_ids("__kas__")
    assert kas_id != tok.unk_token_id, "__kas__ missing from adapter tokenizer"

    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if base.get_input_embeddings().weight.shape[0] < len(tok):
        base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to(device).eval()

    # generation-config kas mapping is not persisted with the adapter — re-apply
    gc = model.generation_config
    lang_map = getattr(gc, "text_decoder_lang_to_code_id", None)
    if lang_map is not None:
        try:
            lang_map["kas"] = kas_id
        except TypeError:
            pass

    def transcribe(arr):
        gen_kwargs = {}
        if args.min_tok_per_sec > 0:
            dur = len(arr) / 16000.0
            gen_kwargs["min_new_tokens"] = min(
                args.min_tok_cap, max(5, int(dur * args.min_tok_per_sec)))
        if args.no_repeat_ngram > 0:
            gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram
        feat = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            out = model.generate(**feat, tgt_lang="kas", num_beams=1, max_new_tokens=200,
                                 **gen_kwargs)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    OUT_JSONL.parent.mkdir(exist_ok=True)
    results = {}
    jl = open(OUT_JSONL, "w", encoding="utf-8")

    # ── 1. IndicVoices-R test split (Whisper-ks ruler: 74.02 WER) ────────────
    _, val_ds = load_indicvoices_ks(LANG_CFG["ks"])
    refs, hyps = [], []
    n = len(val_ds) if args.limit is None else min(args.limit, len(val_ds))
    print(f"[eval] IndicVoices-R test split: {n} samples")
    for i in range(n):
        s   = val_ds[i]
        hyp = transcribe(decode_audio(s["audio"]))
        refs.append(s["transcription"]); hyps.append(hyp)
        jl.write(json.dumps({"set": "indicvoices_test", "idx": i,
                             "ref": s["transcription"], "hyp": hyp}, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"       {i+1}/{n}  interim {score(refs, hyps)}")
    results["indicvoices_test"] = score(refs, hyps)
    print(f"[done] indicvoices_test: {results['indicvoices_test']}")

    # ── 2. robustness_cache/ks 30 clips (Whisper deployed: 81.46/47.95 clean) ─
    if not args.skip_robustness and (ROBUST_DIR / "refs.jsonl").exists():
        import soundfile as sf
        rob_refs = {json.loads(l)["idx"]: json.loads(l)["ref"]
                    for l in open(ROBUST_DIR / "refs.jsonl", encoding="utf-8")}
        refs, hyps = [], []
        wavs = sorted(ROBUST_DIR.glob("*.wav"))
        print(f"[eval] robustness_cache/ks: {len(wavs)} clips")
        for w in wavs:
            idx = int(w.stem)
            if idx not in rob_refs:
                continue
            arr, sr = sf.read(str(w), dtype="float32", always_2d=False)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            hyp = transcribe(arr)
            refs.append(rob_refs[idx]); hyps.append(hyp)
            jl.write(json.dumps({"set": "robustness_clean", "idx": idx,
                                 "ref": rob_refs[idx], "hyp": hyp}, ensure_ascii=False) + "\n")
        results["robustness_clean"] = score(refs, hyps)
        print(f"[done] robustness_clean: {results['robustness_clean']}")

    jl.close()
    results["_decode_settings"] = {
        "min_tok_per_sec": args.min_tok_per_sec,
        "min_tok_cap": args.min_tok_cap,
        "no_repeat_ngram": args.no_repeat_ngram,
    }
    results["whisper_ks_reference"] = {
        "indicvoices_test_wer_raw": 74.02,
        "robustness_clean_wer": 81.46, "robustness_clean_cer": 47.95,
        "note": "Whisper-ks: training-eval 74.02 (raw jiwer, same split); robustness "
                "figures from eval_data/wer_robustness_results.csv (clean).",
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT_JSON}\n[saved] {OUT_JSONL}")


if __name__ == "__main__":
    main()

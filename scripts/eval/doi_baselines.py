# -*- coding: utf-8 -*-
"""doi_baselines.py — zero-shot baselines for Dogri, so doi_iv has something to beat.

A fine-tuned number means nothing without the alternatives it replaced, and Dogri
never had any. This measures, on the SAME 390-clip IndicVoices-R Dogri test split
and the same normalisation ladder as eval_doi_seamless.py:

  1. whisper_auto  - un-fine-tuned whisper-large-v3 with language auto-detection.
                     This is what VANI actually does for Dogri today: no doi model
                     exists, so the pipeline falls back to default Whisper.
  2. whisper_hi    - the same model forced to Hindi, the closest supported language.
                     Whisper has no <|doi|>, so some proxy is unavoidable.
  3. sm4t_hin      - zero-shot SeamlessM4T with tgt_lang=hin (SCRIPT match:
                     Dogri is written in Devanagari).
  4. sm4t_pan      - zero-shot SeamlessM4T with tgt_lang=pan (GENETIC match:
                     Dogri is far closer to Punjabi, but SeamlessM4T knows
                     Punjabi only in Gurmukhi).

3 vs 4 is not just a baseline: it is a free, training-cost-zero probe of which
prior actually suits Dogri, which is the question behind initialising the custom
__doi__ token from __hin__ rather than __pan__. If pan beats hin here despite the
script mismatch, that is evidence the genetic tie matters more than assumed.

Usage:
    python scripts/eval/doi_baselines.py [--limit N] [--systems whisper_auto sm4t_hin ...]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

BASE_SM4T = ROOT / "models" / "seamless-m4t-v2-large"
BASE_WHIS = ROOT / "models" / "whisper-large-v3-ct2"
DOI_TEST = pathlib.Path(
    r"E:\VANI\datasets\hf_cache\hub\datasets--ai4bharat--indicvoices_r"
    r"\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Dogri")
OUT_JSON = ROOT / "docs" / "doi_baselines.json"

from ks_ruler_study import norm  # noqa: E402


def score(pairs, level=0):
    from jiwer import wer as jwer, cer as jcer
    refs = [norm(r, level) for r, _ in pairs]
    hyps = [norm(h, level) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    refs, hyps = [r for r, _ in keep], [h for _, h in keep]
    return {"wer": round(100 * jwer(refs, hyps), 2),
            "cer": round(100 * jcer(refs, hyps), 2), "n": len(refs)}


def load_clips(test_dir, limit=None):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(str(pathlib.Path(test_dir) / "test-*.parquet"))):
        t = pq.read_table(f, columns=["audio", "normalized"]).to_pydict()
        for a, txt in zip(t["audio"], t["normalized"]):
            if txt and txt.strip():
                rows.append((a, txt))
        del t
    return rows[:limit] if limit else rows


def decode_audio(audio):
    import soundfile as sf
    arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr


def run_whisper(rows, language, device):
    from faster_whisper import WhisperModel
    wm = WhisperModel(str(BASE_WHIS), device=device, compute_type="int8")
    out, detected = [], {}
    for i, (audio, ref) in enumerate(rows):
        segs, info = wm.transcribe(decode_audio(audio), language=language,
                                   task="transcribe")
        out.append((ref, " ".join(s.text for s in segs).strip()))
        if language is None:
            detected[info.language] = detected.get(info.language, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"     {i+1}/{len(rows)}", flush=True)
    del wm
    if detected:
        top = sorted(detected.items(), key=lambda x: -x[1])[:5]
        print(f"     auto-detected languages: {top}")
    return out, detected


def run_sm4t(rows, tgt_lang, device):
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    proc = AutoProcessor.from_pretrained(str(BASE_SM4T))
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(BASE_SM4T),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    out = []
    for i, (audio, ref) in enumerate(rows):
        arr = decode_audio(audio)
        feat = proc.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            gen = model.generate(**feat, tgt_lang=tgt_lang, num_beams=1, max_new_tokens=200)
        out.append((ref, proc.decode(gen[0], skip_special_tokens=True).strip()))
        if (i + 1) % 50 == 0:
            print(f"     {i+1}/{len(rows)}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--test-dir", default=str(DOI_TEST))
    ap.add_argument("--systems", nargs="+",
                    default=["whisper_auto", "whisper_hi", "sm4t_hin", "sm4t_pan"])
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_clips(args.test_dir, args.limit)
    print(f"[data] Dogri test: {len(rows)} clips, device={device}\n")

    results, hyps = {}, {}
    for sysname in args.systems:
        print(f"[run] {sysname}")
        if sysname == "whisper_auto":
            pairs, det = run_whisper(rows, None, device)
            results["_whisper_auto_detected"] = det
        elif sysname == "whisper_hi":
            pairs, _ = run_whisper(rows, "hi", device)
        elif sysname.startswith("sm4t_"):
            pairs = run_sm4t(rows, sysname.split("_", 1)[1], device)
        else:
            raise SystemExit(f"unknown system {sysname}")
        results[sysname] = {f"L{l}": score(pairs, l) for l in (0, 2)}
        hyps[sysname] = pairs
        print(f"     -> L0 WER {results[sysname]['L0']['wer']}  "
              f"L2 WER {results[sysname]['L2']['wer']}\n", flush=True)

    print("=" * 62)
    print(f"{'system':16}{'L0 WER':>10}{'L0 CER':>10}{'L2 WER':>10}{'L2 CER':>10}")
    for s in args.systems:
        r = results[s]
        print(f"{s:16}{r['L0']['wer']:>10}{r['L0']['cer']:>10}"
              f"{r['L2']['wer']:>10}{r['L2']['cer']:>10}")
    if "sm4t_hin" in results and "sm4t_pan" in results:
        h, p = results["sm4t_hin"]["L0"]["wer"], results["sm4t_pan"]["L0"]["wer"]
        print(f"\nzero-shot proxy: {'__pan__ (Punjabi, genetic)' if p < h else '__hin__ (Hindi, script)'} "
              f"transcribes Dogri better  (pan {p} vs hin {h})")

    # MERGE, never overwrite. A --systems run used to clobber the whole file,
    # which is how the whisper_auto/whisper_hi numbers quoted in the report were
    # lost: they were produced first, then a later sm4t-only invocation replaced
    # the file and left them unsourced. Same for the hypothesis file.
    prior = {}
    if OUT_JSON.exists():
        prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    prior.update(results)
    OUT_JSON.write_text(json.dumps(prior, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    hyp_path = ROOT / "eval_data" / "doi_baselines_hyps.jsonl"
    kept = []
    if hyp_path.exists():
        for line in hyp_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("system") not in hyps:
                kept.append(rec)
    with open(hyp_path, "w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for s, pairs in hyps.items():
            for i, (r, h) in enumerate(pairs):
                fh.write(json.dumps({"system": s, "idx": i, "ref": r, "hyp": h},
                                    ensure_ascii=False) + "\n")
    print(f"\n[saved] {OUT_JSON}  (systems now on file: "
          f"{', '.join(k for k in prior if not k.startswith('_'))})")


if __name__ == "__main__":
    main()

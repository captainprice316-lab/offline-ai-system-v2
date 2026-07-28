# -*- coding: utf-8 -*-
"""eval_doi_seamless.py — WER/CER for the Dogri SeamlessM4T adapter.

Dogri is the 8th border language in VANI's problem statement and had never been
fine-tuned or evaluated (report 4.5): SeamlessM4T ships no __doi__ token, and no
Dogri audio existed locally. doi_iv is therefore the FIRST Dogri model this
project has produced, and this is the first Dogri number — there is no prior
result to beat.

Test set = the IndicVoices-R Dogri test split (the same corpus family and the
same held-out-split discipline used for Kashmiri). Decode settings mirror the
production path in src/seamless_asr.py, minus the Kashmiri-specific
min_new_tokens fix: Dogri showed no early-EOS pathology, so nothing justifies it.

Scoring reuses ks_ruler_study.norm so Dogri is measured on the same
normalisation ladder as every other language in this project. Devanagari is
unaffected by the Perso-Arabic diacritic levels, so L0 and L2 should be close —
unlike Kashmiri, where they differ by ~14 pp.

Usage:
    python scripts/eval/eval_doi_seamless.py --adapter-dir finetune_runs_seamless/doi_iv/adapter
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import sys

# NOTE: do NOT rewrap sys.stdout here — ks_ruler_study already does it on import,
# and double-wrapping lets the first TextIOWrapper be collected, which closes the
# shared buffer ("I/O operation on closed file") as soon as anything prints.

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

BASE_DIR = ROOT / "models" / "seamless-m4t-v2-large"
DOI_TEST = pathlib.Path(
    r"E:\VANI\datasets\hf_cache\hub\datasets--ai4bharat--indicvoices_r"
    r"\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Dogri")

from ks_ruler_study import norm  # noqa: E402


def score(pairs, level):
    from jiwer import wer as jwer, cer as jcer
    refs = [norm(r, level) for r, _ in pairs]
    hyps = [norm(h, level) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    refs, hyps = [r for r, _ in keep], [h for _, h in keep]
    return {"wer": round(100 * jwer(refs, hyps), 2),
            "cer": round(100 * jcer(refs, hyps), 2), "n": len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir",
                    default=str(ROOT / "finetune_runs_seamless" / "doi_iv" / "adapter"))
    ap.add_argument("--test-dir", default=str(DOI_TEST))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--beams", type=int, default=1, help="production decodes greedily")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel

    adapter = pathlib.Path(args.adapter_dir)
    tag = adapter.parent.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {adapter} on {device}")

    processor = AutoProcessor.from_pretrained(str(adapter))
    tok = processor.tokenizer
    doi_id = tok.convert_tokens_to_ids("__doi__")
    assert doi_id != tok.unk_token_id, "__doi__ missing from adapter tokenizer"

    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    if base.get_input_embeddings().weight.shape[0] < len(tok):
        base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()

    # the generation config's language map is not persisted with the adapter
    lang_map = getattr(model.generation_config, "text_decoder_lang_to_code_id", None)
    if lang_map is not None:
        try:
            lang_map["doi"] = doi_id
        except TypeError:
            pass

    rows = []
    for f in sorted(glob.glob(str(pathlib.Path(args.test_dir) / "test-*.parquet"))):
        t = pq.read_table(f, columns=["audio", "normalized", "duration"]).to_pydict()
        for a, txt, dur in zip(t["audio"], t["normalized"], t["duration"]):
            if txt and txt.strip():
                rows.append((a, txt))
        del t
    if args.limit:
        rows = rows[:args.limit]
    print(f"[data] Dogri test: {len(rows)} clips")

    def transcribe(audio):
        raw = audio.get("bytes")
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        feat = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            out = model.generate(**feat, tgt_lang="doi", num_beams=args.beams,
                                 max_new_tokens=200)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    out_jsonl = ROOT / "eval_data" / f"{tag}_seamless_hyps.jsonl"
    out_json = ROOT / "docs" / f"{tag}_seamless_results.json"
    out_jsonl.parent.mkdir(exist_ok=True)
    pairs = []
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for i, (audio, ref) in enumerate(rows):
            hyp = transcribe(audio)
            pairs.append((ref, hyp))
            fh.write(json.dumps({"set": "indicvoices_doi_test", "idx": i,
                                 "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"       {i+1}/{len(rows)}  interim L0 {score(pairs,0)}", flush=True)

    results = {f"L{lvl}": score(pairs, lvl) for lvl in (0, 1, 2, 3, 4)}
    print("\n=== DOGRI (first evaluation in this project) ===")
    for lvl, r in results.items():
        print(f"  {lvl}: WER {r['wer']:6}  CER {r['cer']:6}  (n={r['n']})")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_json}\n[saved] {out_jsonl}")


if __name__ == "__main__":
    main()

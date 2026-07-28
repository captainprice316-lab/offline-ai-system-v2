# -*- coding: utf-8 -*-
"""eval_ks_production_path.py — measure the artefact VANI actually ships.

Every Kashmiri number in this project (74.02 -> ... -> 50.26) came from
eval_ks_seamless.py, which builds the model with PeftModel.from_pretrained and
therefore gets the adapter exactly as trained. Production does something
different: src/seamless_asr.py BAKES the trainable-token deltas into the base
embedding and then loads only the LoRA matrices, because PEFT cannot stack a
trainable_tokens_delta alongside other named adapters.

Those two paths were equivalent for ks_max, which had a single trainable token.
They were NOT equivalent for ks_cloud3, which has 21 — the baking code applied
only delta[0], leaving the 20 repaired characters at neighbour-init values. This
script exists so that the deployed path is measured directly rather than assumed
to match, which is the same lesson the 74.02-vs-79.29 correction taught.

Run it against the SAME 372-clip IndicVoices-R test split and ruler as
eval_ks_seamless.py so the numbers are directly comparable.

Usage:
    python scripts/eval/eval_ks_production_path.py [--limit N]
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="ks_production")
    args = ap.parse_args()

    import numpy as np
    import yaml
    from eval_ks_seamless import load_indicvoices_ks, LANG_CFG, decode_audio
    from seamless_asr import SeamlessASR

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    asr_cfg = cfg["asr"]
    model_path = ROOT / "models" / "seamless-m4t-v2-large"
    print(f"[production] SeamlessASR with adapters: {asr_cfg.get('seamless_adapters')}")

    asr = SeamlessASR(str(model_path), device="cuda", cfg=asr_cfg)

    _, val_ds = load_indicvoices_ks(LANG_CFG["ks"])
    n = len(val_ds) if args.limit is None else min(args.limit, len(val_ds))
    print(f"[eval] {n} clips through the DEPLOYED code path\n")

    pairs = []
    out_jsonl = ROOT / "eval_data" / f"{args.tag}_hyps.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for i in range(n):
            s = val_ds[i]
            arr = decode_audio(s["audio"])
            # _generate is the exact call the pipeline makes per chunk
            hyp = asr._generate(np.asarray(arr, dtype="float32"), "kas")
            pairs.append((s["transcription"], hyp))
            fh.write(json.dumps({"set": "indicvoices_test", "idx": i,
                                 "ref": s["transcription"], "hyp": hyp},
                                ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"     {i+1}/{n}  interim L2 {score(pairs,2)}", flush=True)

    results = {f"L{l}": score(pairs, l) for l in (0, 1, 2, 3, 4)}
    print("\n=== KASHMIRI THROUGH THE PRODUCTION PATH ===")
    for lvl, r in results.items():
        print(f"  {lvl}: WER {r['wer']:6}  CER {r['cer']:6}  (n={r['n']})")
    print("\n  compare: eval-path ks_cloud3 L2 WER 50.26 / CER 23.34")

    out = ROOT / "docs" / f"{args.tag}_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}\n[saved] {out_jsonl}")


if __name__ == "__main__":
    main()

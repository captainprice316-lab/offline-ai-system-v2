# -*- coding: utf-8 -*-
"""ks_dtype_probe.py — is the eval/production gap just the input-feature dtype?

The deployed path scores 2.07 pp worse than every published Kashmiri number
(52.33 vs 50.26 L2 WER) even though the model weights, the trainable-token
deltas and the extracted features are all verified identical. Only 5.1% of
hypotheses match, and production's are systematically shorter.

The one remaining difference: eval_ks_seamless.py casts input_features to fp16
before an fp16 model, while SeamlessASR._generate passes fp32 straight through.
This probe runs BOTH dtypes through the SAME loaded model on the same clips and
scores each against the eval-path hypotheses, so the question is settled in a
few minutes rather than by two full 35-minute evaluations.

Usage:
    python scripts/eval/ks_dtype_probe.py [--limit 60]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ks_ruler_study import norm  # noqa: E402


def wer(pairs, level=2):
    from jiwer import wer as jwer
    refs = [norm(r, level) for r, _ in pairs]
    hyps = [norm(h, level) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    return round(100 * jwer([r for r, _ in keep], [h for _, h in keep]), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    from eval_ks_seamless import load_indicvoices_ks, LANG_CFG, decode_audio
    from seamless_asr import SeamlessASR, _gen_kwargs

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))["asr"]
    asr = SeamlessASR(str(ROOT / "models" / "seamless-m4t-v2-large"),
                      device="cuda", cfg=cfg)
    model, proc, dev = asr.model, asr.processor, asr.device

    def gen(arr, half: bool):
        """Exactly SeamlessASR._generate, with the fp16 cast as the only variable."""
        gk = _gen_kwargs("kas", len(arr) / 16000.0)
        inputs = proc(audio=arr, return_tensors="pt", sampling_rate=16000)
        if half:
            inputs = {k: (v.half() if v.dtype == torch.float32 else v).to(dev)
                      for k, v in inputs.items()}
        else:
            inputs = inputs.to(dev)
        with torch.no_grad():
            model.set_adapter(asr._adapters["kas"])
            toks = model.generate(**inputs, tgt_lang="kas", **gk)
        return proc.decode(toks[0], skip_special_tokens=True).strip()

    # eval-path hypotheses for the same clips, for a direct string comparison
    ev = {}
    p = ROOT / "eval_data" / "ks_cloud3_seamless_hyps.jsonl"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("set") == "indicvoices_test":
            ev[r["idx"]] = r["hyp"]

    _, val = load_indicvoices_ks(LANG_CFG["ks"])
    n = min(args.limit, len(val))
    fp32_pairs, fp16_pairs, eval_pairs = [], [], []
    match32 = match16 = 0
    for i in range(n):
        s = val[i]
        arr = np.asarray(decode_audio(s["audio"]), dtype="float32")
        ref = s["transcription"]
        h32, h16 = gen(arr, half=False), gen(arr, half=True)
        fp32_pairs.append((ref, h32))
        fp16_pairs.append((ref, h16))
        if i in ev:
            eval_pairs.append((ref, ev[i]))
            match32 += (h32 == ev[i])
            match16 += (h16 == ev[i])
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{n}", flush=True)

    print(f"\n=== {n} clips, same model, only the input dtype differs ===")
    print(f"  production as-is (fp32 features) : L2 WER {wer(fp32_pairs):6}   "
          f"matches eval hyp on {match32}/{len(eval_pairs)}")
    print(f"  with .half() cast (eval-style)   : L2 WER {wer(fp16_pairs):6}   "
          f"matches eval hyp on {match16}/{len(eval_pairs)}")
    print(f"  eval path (reference)            : L2 WER {wer(eval_pairs):6}")
    verdict = ("dtype EXPLAINS the gap" if match16 > match32 * 3 or
               abs(wer(fp16_pairs) - wer(eval_pairs)) < 0.6
               else "dtype does NOT explain it - keep looking")
    print(f"\n  => {verdict}")


if __name__ == "__main__":
    main()

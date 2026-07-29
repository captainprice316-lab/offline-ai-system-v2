# -*- coding: utf-8 -*-
"""ks_multiadapter_probe.py — is the eval/production gap multi-adapter interference?

The deployed path scores ~2 pp worse than the path every published Kashmiri
number came from, and the model itself has been exonerated: LoRA weights match
480/480, the trainable-token deltas are byte-identical, extracted features are
identical, and max_new_tokens, input dtype and base_layer substitution have each
been ruled out by direct measurement.

The one structural difference left is how the model is ASSEMBLED. The eval path
builds a PeftModel with a SINGLE adapter, always active. Production builds one
carrying four (hi, ne, ps, ks) and switches with set_adapter(). If set_adapter
does not fully isolate them — or if stacking changes which modules are wrapped —
Kashmiri would be decoded through a model that is not the one it was evaluated on.

This runs the production class twice on the same clips: once with the deployed
four-adapter config, once with ks alone. If ks-only closes the gap, the defect is
interference and it affects every deployed language, not just Kashmiri.

Usage:
    python scripts/eval/ks_multiadapter_probe.py [--limit 60]
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
    import yaml
    from eval_ks_seamless import load_indicvoices_ks, LANG_CFG, decode_audio
    from seamless_asr import SeamlessASR

    full_cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))["asr"]
    ks_only = dict(full_cfg)
    ks_only["seamless_adapters"] = {
        "ks": full_cfg["seamless_adapters"]["ks"]}

    # eval-path hypotheses for the same clips
    ev = {}
    for line in open(ROOT / "eval_data" / "ks_cloud3_seamless_hyps.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("set") == "indicvoices_test":
            ev[r["idx"]] = r["hyp"]

    _, val = load_indicvoices_ks(LANG_CFG["ks"])
    n = min(args.limit, len(val))
    clips = []
    for i in range(n):
        s = val[i]
        clips.append((i, np.asarray(decode_audio(s["audio"]), dtype="float32"),
                      s["transcription"]))

    results = {}
    for label, cfg in (("all 4 adapters (deployed)", full_cfg), ("ks only", ks_only)):
        print(f"\n[run] {label}: {list(cfg['seamless_adapters'])}")
        asr = SeamlessASR(str(ROOT / "models" / "seamless-m4t-v2-large"),
                          device="cuda", cfg=cfg)
        pairs, match = [], 0
        for i, arr, ref in clips:
            hyp = asr._generate(arr, "kas")
            pairs.append((ref, hyp))
            match += (i in ev and hyp == ev[i])
        results[label] = (wer(pairs), match)
        print(f"     L2 WER {wer(pairs)}   identical to eval hyp: {match}/{n}")
        del asr
        import torch
        torch.cuda.empty_cache()

    eval_pairs = [(ref, ev[i]) for i, _, ref in clips if i in ev]
    print(f"\n=== {n} clips ===")
    for k, (w, m) in results.items():
        print(f"  {k:26} L2 WER {w:6}   matches eval {m}/{n}")
    print(f"  {'eval path (reference)':26} L2 WER {wer(eval_pairs):6}")
    a = results["all 4 adapters (deployed)"][0]
    b = results["ks only"][0]
    print(f"\n  => {'multi-adapter interference EXPLAINS the gap' if a - b > 0.8 else 'interference does NOT explain it'}"
          f"  (4-adapter {a} vs ks-only {b})")


if __name__ == "__main__":
    main()

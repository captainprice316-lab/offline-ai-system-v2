# -*- coding: utf-8 -*-
"""rescore_ks.py — LM rescoring of the Kashmiri n-best list.

Step 3. Combines the acoustic score with the Kneser-Ney trigram LM:

    score(h) = am(h) + lambda * lm(h) + beta * |h|

where am() is the raw sum log-prob from beam search (natural log), lm() is the
LM log-probability converted to natural log, and beta is a per-word insertion
bonus that offsets the LM's bias toward short hypotheses.

HONEST TUNING. lambda/beta must not be tuned on the clips we then report, or the
result is fitted noise. This uses 2-FOLD CROSS-VALIDATION: tune on fold A and
decode fold B, tune on B and decode A, then report the pooled WER over all 372
clips — every reported clip was decoded with weights chosen without seeing it.
The oracle and 1-best rows bound what rescoring could possibly achieve.

Usage:
    python scripts/lm/rescore_ks.py
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(HERE))

LN10 = math.log(10.0)


def wer(pairs):
    from jiwer import wer as jwer
    refs = [r for r, h in pairs if r.strip()]
    hyps = [h for r, h in pairs if r.strip()]
    return 100 * jwer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", default=str(ROOT / "eval_data" / "ks_cloud3_nbest.jsonl"))
    ap.add_argument("--lm", default=str(ROOT / "models" / "ks_lm" / "ks_kn3.pkl"))
    ap.add_argument("--level", type=int, default=2, help="normalisation level for scoring")
    ap.add_argument("--out", default=str(ROOT / "docs" / "ks_lm_rescore.json"))
    args = ap.parse_args()

    from ks_ruler_study import norm
    from ks_lm import KNTrigramLM

    rows = [json.loads(l) for l in open(args.nbest, encoding="utf-8") if l.strip()]
    print(f"[data] {len(rows)} clips, {sum(len(r['cands']) for r in rows)} candidates")
    lm = KNTrigramLM.load(pathlib.Path(args.lm))
    print(f"[lm]   vocab {len(lm.vocab):,}")

    # Pre-compute normalised text and LM scores once.
    for r in rows:
        r["nref"] = norm(r["ref"], args.level)
        for c in r["cands"]:
            c["ntext"] = norm(c["text"], args.level)
            w = c["ntext"].split()
            c["lm"] = lm.logprob(w) * LN10          # log10 -> natural log
            c["nw"] = len(w)

    def decode(subset, lam, beta):
        out = []
        for r in subset:
            best = max(r["cands"], key=lambda c: c["am"] + lam * c["lm"] + beta * c["nw"])
            out.append((r["nref"], best["ntext"]))
        return out

    # ── baselines ─────────────────────────────────────────────────────────────
    onebest = [(r["nref"], r["cands"][0]["ntext"]) for r in rows]
    print(f"\nbeam 1-best (no LM)      : {wer(onebest):.2f}%")

    oracle = []
    for r in rows:
        from jiwer import wer as jwer
        best = min(r["cands"], key=lambda c: jwer(r["nref"], c["ntext"]) if r["nref"].strip() else 0)
        oracle.append((r["nref"], best["ntext"]))
    orc = wer(oracle)
    print(f"oracle n-best (ceiling)  : {orc:.2f}%   <- best achievable by any rescorer")

    # ── 2-fold CV ─────────────────────────────────────────────────────────────
    LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]
    BETAS   = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    folds = [[r for i, r in enumerate(rows) if i % 2 == 0],
             [r for i, r in enumerate(rows) if i % 2 == 1]]

    pooled, chosen = [], []
    for tune_i, test_i in ((0, 1), (1, 0)):
        tune, test = folds[tune_i], folds[test_i]
        best = (1e9, 0.0, 0.0)
        for lam in LAMBDAS:
            for beta in BETAS:
                w = wer(decode(tune, lam, beta))
                if w < best[0]:
                    best = (w, lam, beta)
        _, lam, beta = best
        chosen.append({"tuned_on_fold": tune_i, "lambda": lam, "beta": beta,
                       "tune_wer": round(best[0], 2)})
        print(f"fold {tune_i} -> lambda={lam}, beta={beta} (tune WER {best[0]:.2f}%)")
        pooled += decode(test, lam, beta)

    final = wer(pooled)
    base = wer(onebest)
    print(f"\n=== LM RESCORING (2-fold CV, honest) ===")
    print(f"  beam 1-best  : {base:.2f}%")
    print(f"  + LM rescore : {final:.2f}%   ({final-base:+.2f} pp)")
    print(f"  oracle       : {orc:.2f}%")
    if base > orc:
        print(f"  captured {100*(base-final)/(base-orc):.1f}% of the available headroom")

    res = {"beam_1best_wer": round(base, 2), "rescored_wer": round(final, 2),
           "oracle_wer": round(orc, 2), "delta_pp": round(final - base, 2),
           "folds": chosen, "n_clips": len(rows),
           "n_cands": sum(len(r["cands"]) for r in rows), "level": args.level}
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

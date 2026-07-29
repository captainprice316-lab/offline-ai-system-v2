# -*- coding: utf-8 -*-
"""significance.py — bootstrap confidence intervals for every claimed WER difference.

Several deployment decisions in this project rest on sub-1 pp margins: ps_cloud
was deployed over ps_aug on 0.75 pp, ks_cloud4 was rejected on 0.43 pp. Those are
point estimates on 372-425 clips with no uncertainty attached, which is the first
thing a referee will ask about.

This does a paired bootstrap over CLIPS (the unit of independence), which is the
standard test for ASR: resample clips with replacement, recompute both systems'
corpus WER on the same resample, and report the distribution of the difference.
Paired, because both systems saw identical audio — that removes clip-difficulty
variance and is far more sensitive than comparing two independent intervals.

WER is a ratio of corpus totals, not a mean of per-clip rates, so each resample
re-aggregates errors and reference length rather than averaging percentages.

Reads the committed eval_data/*.jsonl hypothesis files: no GPU, no adapters.

Usage:
    python scripts/eval/significance.py [--boot 10000]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from ks_ruler_study import norm  # noqa: E402

ED = ROOT / "eval_data"

# (label, file, set-filter, normalisation level)
SYSTEMS = {
    "ks_cloud3":  ("ks_cloud3_seamless_hyps.jsonl", "indicvoices_test", 2),
    "ks_cloud2":  ("ks_cloud2_seamless_hyps.jsonl", "indicvoices_test", 2),
    "ks_cloud":   ("ks_cloud_seamless_hyps.jsonl",  "indicvoices_test", 2),
    "ks_cloud4":  ("ks_cloud4_seamless_hyps.jsonl", "indicvoices_test", 2),
    "ks_max2":    ("ks_max2_seamless_hyps.jsonl",   "indicvoices_test", 2),
    "ks_whisper": ("ks_whisper_test_hyps.jsonl",    None,               2),
    "doi_iv":     ("doi_iv_seamless_hyps.jsonl",    "indicvoices_doi_test", 2),
    "doi_iv2":    ("doi_iv2_seamless_hyps.jsonl",   "indicvoices_doi_test", 2),
}

# comparisons the paper actually makes
PAIRS = [
    ("ks_cloud3", "ks_cloud2", "vocabulary repair (deployed decision)"),
    ("ks_cloud2", "ks_cloud",  "training to convergence"),
    ("ks_cloud4", "ks_cloud3", "warm start - REJECTED on 0.43 pp"),
    ("ks_cloud3", "ks_max2",   "r=128 + vocabulary vs r=32"),
    ("ks_cloud3", "ks_whisper", "deployed vs the Whisper it replaced"),
    ("doi_iv2",   "doi_iv",    "Dogri training to convergence"),
]


def load(label):
    fn, subset, lvl = SYSTEMS[label]
    p = ED / fn
    if not p.exists():
        return None
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if subset and r.get("set") != subset:
            continue
        out[r["idx"]] = (norm(r["ref"], lvl), norm(r["hyp"], lvl))
    return out


def counts(ref, hyp):
    """(edit distance, reference length) in words — the two WER aggregates."""
    import jiwer
    o = jiwer.process_words([ref], [hyp])
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def corpus_wer(pairs_counts, idxs):
    e = sum(pairs_counts[i][0] for i in idxs)
    n = sum(pairs_counts[i][1] for i in idxs)
    return 100.0 * e / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    cache = {}
    print(f"paired bootstrap over clips, {args.boot:,} resamples\n")
    print(f"{'comparison':44}{'A':>8}{'B':>8}{'diff':>8}{'95% CI':>18}{'p':>8}")
    print("-" * 94)
    results = {}

    for a, b, why in PAIRS:
        da, db = load(a), load(b)
        if da is None or db is None:
            print(f"{a} vs {b}: missing hypotheses, skipped")
            continue
        common = sorted(set(da) & set(db))
        if not common:
            print(f"{a} vs {b}: no shared clips, skipped")
            continue
        for lbl, d in ((a, da), (b, db)):
            if lbl not in cache:
                cache[lbl] = {}
            for i in common:
                if i not in cache[lbl]:
                    cache[lbl][i] = counts(*d[i])
        ca, cb = cache[a], cache[b]
        wa, wb = corpus_wer(ca, common), corpus_wer(cb, common)
        obs = wa - wb

        diffs = []
        n = len(common)
        for _ in range(args.boot):
            samp = [common[random.randrange(n)] for _ in range(n)]
            diffs.append(corpus_wer(ca, samp) - corpus_wer(cb, samp))
        diffs.sort()
        lo, hi = diffs[int(0.025 * args.boot)], diffs[int(0.975 * args.boot)]
        # two-sided p: how often the resampled difference crosses zero
        side = sum(1 for d in diffs if (d >= 0) != (obs >= 0)) / args.boot
        p = min(1.0, 2 * side)
        sig = "*" if p < 0.05 else " "
        print(f"{a+' vs '+b:44}{wa:8.2f}{wb:8.2f}{obs:+8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{p:8.3f}{sig}")
        results[f"{a}_vs_{b}"] = {"why": why, "wer_a": round(wa, 2), "wer_b": round(wb, 2),
                                  "diff": round(obs, 2), "ci95": [round(lo, 2), round(hi, 2)],
                                  "p": round(p, 4), "n_clips": n,
                                  "significant_at_0.05": bool(p < 0.05)}

    print("\n* = significant at p < 0.05 (paired bootstrap). Negative diff means A is better.")
    for k, v in results.items():
        if not v["significant_at_0.05"]:
            print(f"  NOT SIGNIFICANT: {k} ({v['why']}) — diff {v['diff']:+.2f}, "
                  f"CI {v['ci95']}, p={v['p']}")
    out = ROOT / "docs" / "significance.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

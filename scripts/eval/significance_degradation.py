# -*- coding: utf-8 -*-
"""significance_degradation.py — can a 30-clip sweep support a deployment decision?

Every adapter in this project was gated on a 5-condition radio-degradation sweep
of 30 clips per condition, and the verdicts were reported as win counts ("4/5",
"5/5"). After the clean-WER bootstrap showed that even 100 clips cannot resolve
sub-2 pp differences, those win counts need the same scrutiny: a "4/5" built from
five individually-insignificant comparisons is not evidence of anything.

This matters most for Pashto. ps_cloud's 0.75 pp clean-WER advantage over ps_aug
is not significant (p = 0.32), so its deployment now rests entirely on the sweep.
If the sweep cannot resolve its margins either, that deployment has no
statistically significant support on any measure — which the report must say.

Paired bootstrap over clips within each condition, same method as
significance.py. Also reports a sign test over the five conditions, which is the
correct test for a "wins k of 5" claim.

Usage:
    python scripts/eval/significance_degradation.py
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from text_norm import normalise  # noqa: E402

CONDS = ["clean", "bandpass", "awgn_10", "awgn_0", "codec_mp3"]
BOOT = 10000
random.seed(42)


def load_robustness(lang, system):
    out = defaultdict(dict)
    for line in open(ROOT / "eval_data" / "wer_robustness_hyps.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("lang") == lang and r.get("system") == system:
            out[r["condition"]][r.get("idx", len(out[r["condition"]]))] = (r["ref"], r["hyp"])
    return out


def counts(pair, lang):
    import jiwer
    ref, hyp = normalise(pair[0], lang), normalise(pair[1], lang)
    if not ref.strip():
        return None
    o = jiwer.process_words([ref], [hyp])
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def wer(c, idxs):
    e = sum(c[i][0] for i in idxs)
    n = sum(c[i][1] for i in idxs)
    return 100.0 * e / max(n, 1)


def compare(a_data, b_data, lang, a_name, b_name):
    print(f"\n{a_name} vs {b_name}  (paired bootstrap, {BOOT:,} resamples, n=30/condition)")
    print(f"  {'condition':12}{'A':>8}{'B':>8}{'diff':>8}{'95% CI':>18}{'p':>8}")
    wins = sig_wins = 0
    rows = {}
    for cond in CONDS:
        da, db = a_data.get(cond, {}), b_data.get(cond, {})
        common = sorted(set(da) & set(db))
        ca = {i: counts(da[i], lang) for i in common}
        cb = {i: counts(db[i], lang) for i in common}
        common = [i for i in common if ca[i] and cb[i]]
        if not common:
            continue
        wa, wb = wer(ca, common), wer(cb, common)
        obs = wa - wb
        diffs = []
        n = len(common)
        for _ in range(BOOT):
            s = [common[random.randrange(n)] for _ in range(n)]
            diffs.append(wer(ca, s) - wer(cb, s))
        diffs.sort()
        lo, hi = diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT)]
        side = sum(1 for d in diffs if (d >= 0) != (obs >= 0)) / BOOT
        p = min(1.0, 2 * side)
        wins += obs < 0
        sig_wins += (obs < 0 and p < 0.05)
        rows[cond] = {"a": round(wa, 2), "b": round(wb, 2), "diff": round(obs, 2),
                      "ci95": [round(lo, 2), round(hi, 2)], "p": round(p, 4),
                      "significant": bool(p < 0.05)}
        print(f"  {cond:12}{wa:8.2f}{wb:8.2f}{obs:+8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
              f"{p:8.3f}{'*' if p < 0.05 else ' '}")
    # sign test over 5 conditions: P(>= wins) under a fair coin
    from math import comb
    k, n5 = wins, len(rows)
    tail = sum(comb(n5, j) for j in range(k, n5 + 1)) / (2 ** n5)
    print(f"  -> raw win count {k}/{n5}; sign test one-sided p = {tail:.3f}; "
          f"conditions with a SIGNIFICANT win: {sig_wins}/{n5}")
    return rows, wins, sig_wins, tail


def main():
    out = {}
    ps_cloud = load_robustness("ps", "seamless_ft_ps_cloud")
    ps_aug = load_robustness("ps", "seamless_ft_ps_aug")
    ps_bal2 = load_robustness("ps", "seamless_ft_ps_bal2")
    if ps_cloud and ps_aug:
        r, w, sw, t = compare(ps_cloud, ps_aug, "ps", "ps_cloud", "ps_aug")
        out["ps_cloud_vs_ps_aug"] = {"conditions": r, "wins": w,
                                     "significant_wins": sw, "sign_test_p": round(t, 4)}
    if ps_aug and ps_bal2:
        r, w, sw, t = compare(ps_aug, ps_bal2, "ps", "ps_aug", "ps_bal2")
        out["ps_aug_vs_ps_bal2"] = {"conditions": r, "wins": w,
                                    "significant_wins": sw, "sign_test_p": round(t, 4)}
    p = ROOT / "docs" / "significance_degradation.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()

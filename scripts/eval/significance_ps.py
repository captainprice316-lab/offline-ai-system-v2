# -*- coding: utf-8 -*-
"""Paired bootstrap for the Pashto deployment decisions (n=100 FLEURS clips).

ps_cloud was deployed over ps_aug on a 0.75 pp margin, and ps_aug2 was rejected
on 0.55 pp. Both are small enough that they may not be distinguishable from noise
at n=100, which is a much smaller test set than the Kashmiri/Dogri 372-425.
"""
import json, pathlib, random, sys, collections
ROOT = pathlib.Path(r"C:\Users\vis15\offline_ai_system_v2")
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from text_norm import normalise                      # noqa: E402
import jiwer                                          # noqa: E402

random.seed(42)
rows = collections.defaultdict(dict)
for line in open(ROOT / "eval_data" / "seamless_ft_hyps.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows[r["lang"]][r["idx"]] = (r["ref"], r["hyp"])

def counts(pair):
    ref, hyp = normalise(pair[0], "ps"), normalise(pair[1], "ps")
    if not ref.strip():
        return None
    o = jiwer.process_words([ref], [hyp])
    return o.substitutions + o.deletions + o.insertions, len(ref.split())

def wer(c, idxs):
    e = sum(c[i][0] for i in idxs); n = sum(c[i][1] for i in idxs)
    return 100.0 * e / max(n, 1)

PAIRS = [("ps_cloud", "ps_aug",  "DEPLOYED ps_cloud over ps_aug"),
         ("ps_aug2",  "ps_aug",  "REJECTED ps_aug2"),
         ("ps_cloud", "ps_bal2", "r=128 vs r=32 noise-aug"),
         ("ps_aug",   "ps_cv",   "balanced data vs CV-dominated")]

print(f"{'comparison':38}{'A':>8}{'B':>8}{'diff':>8}{'95% CI':>18}{'p':>8}")
print("-" * 88)
out = {}
for a, b, why in PAIRS:
    if a not in rows or b not in rows:
        print(f"{a} vs {b}: missing"); continue
    common = sorted(set(rows[a]) & set(rows[b]))
    ca = {i: counts(rows[a][i]) for i in common}
    cb = {i: counts(rows[b][i]) for i in common}
    common = [i for i in common if ca[i] and cb[i]]
    wa, wb = wer(ca, common), wer(cb, common)
    obs = wa - wb
    diffs = []
    n = len(common)
    for _ in range(10000):
        s = [common[random.randrange(n)] for _ in range(n)]
        diffs.append(wer(ca, s) - wer(cb, s))
    diffs.sort()
    lo, hi = diffs[250], diffs[9750]
    side = sum(1 for d in diffs if (d >= 0) != (obs >= 0)) / 10000
    p = min(1.0, 2 * side)
    print(f"{a+' vs '+b:38}{wa:8.2f}{wb:8.2f}{obs:+8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}{p:8.3f}"
          f"{'*' if p < 0.05 else ' '}")
    out[f"{a}_vs_{b}"] = {"why": why, "wer_a": round(wa,2), "wer_b": round(wb,2),
                          "diff": round(obs,2), "ci95": [round(lo,2), round(hi,2)],
                          "p": round(p,4), "n_clips": n,
                          "significant_at_0.05": bool(p < 0.05)}
(ROOT / "docs" / "significance_ps.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\n[saved] docs/significance_ps.json")

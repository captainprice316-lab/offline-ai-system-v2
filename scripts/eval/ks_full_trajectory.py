# -*- coding: utf-8 -*-
"""Score EVERY Kashmiri system ever built on one ruler, so the campaign
trajectory can be plotted without mixing rulers.

docs/ks_cloud_ruler_compare.json already covers ks_max onwards. The two
earliest SeamlessM4T attempts -- `ks` (custom __kas__ token FROZEN) and
`ks_r16` (same, rank 16) -- were only ever reported at L0, which is why the
"129%"/"92%" figures quoted in early notes are not comparable with the L2
figures the project decides on. This rescores their stored hypotheses through
the identical norm()/score() and writes the full trajectory.

CPU-only. Reads eval_data/ks_*_seamless_hyps.jsonl, writes
docs/ks_trajectory.json.
"""
import json, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from ks_ruler_study import norm, score, load_jsonl, LEVELS  # noqa: E402,F401

ED = ROOT / "eval_data"

# chronological order of the Kashmiri campaign
SRC = [
    ("whisper",   ED / "ks_whisper_test_hyps.jsonl", True),
    ("ks",        ED / "ks_seamless_hyps.jsonl", False),
    ("ks_r16",    ED / "ks_r16_seamless_hyps.jsonl", False),
    ("ks_max",    ED / "ks_max_seamless_hyps.jsonl", False),
    ("ks_max2",   ED / "ks_max2_seamless_hyps.jsonl", False),
    ("ks_cloud",  ED / "ks_cloud_seamless_hyps.jsonl", False),
    ("ks_cloud2", ED / "ks_cloud2_seamless_hyps.jsonl", False),
    ("ks_cloud3", ED / "ks_cloud3_seamless_hyps.jsonl", False),
    ("ks_cloud4", ED / "ks_cloud4_seamless_hyps.jsonl", False),
]


def pairs_for(path, is_whisper):
    if not path.exists():
        return None
    rows = load_jsonl(path)
    if not is_whisper:
        rows = [r for r in rows if r.get("set") == "indicvoices_test"]
    rows = sorted(rows, key=lambda r: r["idx"])
    return [(r["ref"], r["hyp"]) for r in rows]


out = {}
print(f"{'system':11}{'n':>6}{'L0 WER':>10}{'L2 WER':>10}{'L2 CER':>10}")
for name, path, is_w in SRC:
    p = pairs_for(path, is_w)
    if not p:
        print(f"{name:11}  -- missing {path.name}")
        continue
    l0 = score(p, 0)
    l2 = score(p, 2)
    out[name] = {"n": len(p), "L0_wer": l0["wer"], "L2_wer": l2["wer"],
                 "L2_cer": l2["cer"]}
    print(f"{name:11}{len(p):>6}{l0['wer']:>10}{l2['wer']:>10}{l2['cer']:>10}")

(ROOT / "docs" / "ks_trajectory.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n[saved] docs/ks_trajectory.json")

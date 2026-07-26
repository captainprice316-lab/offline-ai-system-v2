# -*- coding: utf-8 -*-
"""Score ks_max2 on the SAME diacritic-normalisation ladder as the ks_max ruler
study, head-to-head vs ks_max and deployed Whisper-ks on the 372-clip IndicVoices
test split. Reuses norm()/score() from ks_ruler_study so the ruler is identical.
CPU-only. Reads:
  eval_data/ks_max2_seamless_hyps.jsonl   (this campaign, set=indicvoices_test)
  eval_data/ks_max_seamless_hyps.jsonl    (deployed adapter)
  eval_data/ks_whisper_test_hyps.jsonl    (deployed Whisper-ks; optional)
Writes docs/ks_max2_ruler_compare.json + prints the table.
"""
import json, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from ks_ruler_study import norm, score, load_jsonl, LEVELS  # noqa: E402

ED = ROOT / "eval_data"
SRC = {
    "ks_max2":  ED / "ks_max2_seamless_hyps.jsonl",
    "ks_max":   ED / "ks_max_seamless_hyps.jsonl",
    "whisper":  ED / "ks_whisper_test_hyps.jsonl",
}

def pairs_for(path, is_whisper=False):
    if not path.exists():
        return None
    rows = load_jsonl(path)
    if not is_whisper:
        rows = [r for r in rows if r.get("set") == "indicvoices_test"]
    rows = sorted(rows, key=lambda r: r["idx"])
    return [(r["ref"], r["hyp"]) for r in rows]

systems = {}
for name, path in SRC.items():
    p = pairs_for(path, is_whisper=(name == "whisper"))
    if p:
        systems[name] = p
        print(f"[load] {name}: {len(p)} pairs")
    else:
        print(f"[skip] {name}: {path.name} missing")

order = [s for s in ("ks_max2", "ks_max", "whisper") if s in systems]
results = {}
print(f"\n{'level':22} " + "".join(f"{s+' WER':>13}{s+' CER':>13}" for s in order))
for lvl in sorted(LEVELS):
    row = {}
    line = f"{LEVELS[lvl]:22} "
    for s in order:
        sc = score(systems[s], lvl)
        row[s] = sc
        line += f"{sc['wer']:>13}{sc['cer']:>13}"
    results[f"L{lvl}"] = row
    print(line)

# boundary-free CER (word-segmentation-agnostic) at L2
print("\nboundary-free CER (L2, no spaces):")
bf = {}
for s in order:
    sc = score(systems[s], 2, boundary_free=True)
    bf[s] = sc
    print(f"  {s:10} CER={sc['cer']}")
results["L2_boundary_free_cer"] = bf

# verdict at L2 (the decisive diacritic-normalised level)
if "ks_max2" in systems:
    print("\n=== VERDICT (L2 = diacritic-normalised, the deciding ruler) ===")
    km2 = results["L2"]["ks_max2"]
    for opp in ("ks_max", "whisper"):
        if opp in results["L2"]:
            o = results["L2"][opp]
            dw = round(km2["wer"] - o["wer"], 2)
            dc = round(km2["cer"] - o["cer"], 2)
            print(f"  ks_max2 vs {opp:8}: WER {km2['wer']} vs {o['wer']} ({'+' if dw>0 else ''}{dw})"
                  f"  | CER {km2['cer']} vs {o['cer']} ({'+' if dc>0 else ''}{dc})  "
                  f"({'ks_max2 WINS' if dw<0 else 'ks_max2 loses' if dw>0 else 'tie'} on WER)")

(ROOT / "docs" / "ks_max2_ruler_compare.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[saved] docs/ks_max2_ruler_compare.json")

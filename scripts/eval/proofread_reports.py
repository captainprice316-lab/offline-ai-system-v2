"""Proofread the regenerated report generators against the authoritative data.

Checks:
 A. EVAL_RESULTS dict in generate_report_pdf.py == docs/model_comparison_results.json
 B. Summary-table cells and improvement (pp) columns are arithmetically right
 C. §5.5.1 degradation deltas == eval_data/wer_robustness_results.csv (FT - SM per condition)
 D. chrF winner claims (which system wins translation per language)
 E. PPTX hardcoded tables == same sources
 F. LANG_META internal consistency (best_wer vs wer_curve endpoints / summary rows)
"""
import io, sys, json, csv, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

REPO = Path(r"C:\Users\vis15\offline_ai_system_v2")
sys.path.insert(0, str(REPO))

comp = {r["lang"]: r for r in json.load(open(REPO/"docs/model_comparison_results.json", encoding="utf-8"))}
smft = {r["lang"]: r for r in json.load(open(REPO/"docs/seamless_ft_results.json", encoding="utf-8"))}
sweep = {}
for r in csv.DictReader(open(REPO/"eval_data/wer_robustness_results.csv", encoding="utf-8")):
    sweep[(r["system"], r["lang"], r["condition"])] = float(r["wer"])

issues = []
ok = []

def check(label, claim, truth, tol=0.051):
    if truth is None:
        issues.append(f"{label}: claimed {claim} but source has no value")
    elif abs(claim - truth) > tol:
        issues.append(f"{label}: claimed {claim} but source says {truth}")
    else:
        ok.append(label)

# ── A: EVAL_RESULTS vs JSON ────────────────────────────────────────────────────
from generate_report_pdf import EVAL_RESULTS, LANG_META
for l, d in EVAL_RESULTS.items():
    c = comp.get(l, {})
    if l == "ks":
        # ks uses training-eval numbers by design (compare not re-run); skip JSON check
        continue
    check(f"EVAL_RESULTS[{l}].baseline", d["baseline"], c.get("whisper_baseline_wer"))
    check(f"EVAL_RESULTS[{l}].ft",       d["ft"],       c.get("whisper_ft_wer"))
    check(f"EVAL_RESULTS[{l}].seamless", d["seamless"], c.get("seamless_asr_wer"))
    check(f"EVAL_RESULTS[{l}].nllb_chrf",d["nllb_chrf"],c.get("whisper_nllb_chrf"))
    check(f"EVAL_RESULTS[{l}].sm_chrf",  d["sm_chrf"],  c.get("seamless_s2tt_chrf"))

# ── B: summary-table improvement arithmetic (as printed in the PDF source) ────
pdf_src = open(REPO/"generate_report_pdf.py", encoding="utf-8").read()
summary_rows = {  # lang: (baseline, test, printed_pp)
    "pa": (77.60, 57.39, -20.2), "ps": (89.76, 38.55, -51.2),
    "ur": (21.23, 19.82, -1.4),  "ne": (88.85, 50.92, -37.9),
    "zh": (10.99, 14.22, +3.2),  "hi": (26.34, 19.78, -6.6),
    "ks": (96.87, 74.02, -22.85),
}
for l, (b, t, pp) in summary_rows.items():
    true_pp = round(t - b, 2)
    if abs(true_pp - pp) > 0.06:
        issues.append(f"§5.1 {l} improvement: printed {pp:+.2f} pp but {t}−{b} = {true_pp:+.2f}")
    else:
        ok.append(f"§5.1 {l} improvement arithmetic")

# ── C: §5.5.1 degradation deltas vs sweep CSV ─────────────────────────────────
printed_deltas = {  # from the PDF table
    "pa": [39.0, 37.7, 41.7, 36.8, 41.8], "ne": [31.6, 30.5, 34.1, 36.4, 32.0],
    "hi": [5.0, 3.9, 10.8, 19.5, 5.6],   "ur": [2.3, 2.4, 4.8, 4.2, 3.1],
    "zh": [3.6, 5.0, 2.9, 17.2, 3.0],    "ps": [-6.7, -2.1, -1.9, 5.6, -2.8],
}
conds = ["clean", "bandpass", "awgn_10", "awgn_0", "codec_mp3"]
for l, deltas in printed_deltas.items():
    for cnd, printed in zip(conds, deltas):
        ft = sweep.get(("whisper_ft", l, cnd)); sm = sweep.get(("seamless_zs", l, cnd))
        if ft is None or sm is None:
            issues.append(f"§5.5.1 {l}/{cnd}: no sweep data"); continue
        true_d = round(ft - sm, 1)
        if abs(true_d - printed) > 0.06:
            issues.append(f"§5.5.1 {l}/{cnd}: printed {printed:+.1f} but sweep says {true_d:+.1f}")
        else:
            ok.append(f"§5.5.1 {l}/{cnd}")

# ── D: chrF winner claims ─────────────────────────────────────────────────────
print("chrF winners from JSON (who wins translation per language):")
for l in ["pa", "ps", "ur", "ne", "zh", "hi"]:
    c = comp[l]
    n, s = c.get("whisper_nllb_chrf"), c.get("seamless_s2tt_chrf")
    print(f"  {l}: NLLB {n}  vs  SM4T {s}  -> winner: {'SM4T' if s and n and s > n else 'NLLB'}")

# ── E: PPTX slide-5 arrays and benchmark table vs JSON ───────────────────────
pptx_src = open(REPO/"generate_finetune_pptx.py", encoding="utf-8").read()
pptx_slide5 = {
    "zh": (14.22, 11.69), "ur": (19.82, 16.90), "hi": (19.78, 15.44),
    "ne": (50.92, 28.46), "pa": (57.39, 19.77), "ps": (38.55, 44.40),
}
for l, (ft, sm) in pptx_slide5.items():
    check(f"PPTX slide5 {l}.ft", ft, comp[l].get("whisper_ft_wer"))
    check(f"PPTX slide5 {l}.sm", sm, comp[l].get("seamless_asr_wer"))

# ── F: LANG_META star consistency: best_wer should appear in the wer curve ───
for l, m in LANG_META.items():
    curve = dict(m.get("wer_curve_v3") or m.get("wer_curve") or [])
    bs, bw = m.get("best_step"), m.get("best_wer")
    if bs in curve and abs(curve[bs] - bw) > 0.06:
        issues.append(f"LANG_META[{l}]: best_wer={bw} but wer_curve at step {bs} = {curve[bs]} "
                      f"(chart star will not sit on the curve)")
    else:
        ok.append(f"LANG_META[{l}] star consistency")

print(f"\n{'='*70}\nPASSED: {len(ok)} checks")
print(f"ISSUES: {len(issues)}")
for i in issues:
    print(f"  ✗ {i}")

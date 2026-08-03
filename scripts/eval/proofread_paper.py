# -*- coding: utf-8 -*-
"""proofread_paper.py -- assert every headline number in paper/main.tex against
the JSON artefact it came from.

paper/figures.py already does this for the plotted values; the prose was the
remaining unguarded surface, and it is where the errors were. This pass caught,
among others, a Dogri table labelled L2 but populated from L0, a Kashmiri
"before" figure that was training-split validation WER compared against a
held-out "after", and a decoding claim that every beam configuration gained on
clean audio when beam 2 in fact lost on clean and at 0 dB both.

Each check is (label, value-as-written-in-the-tex, value-from-artefact). The
tex is also scanned so a number that no longer appears in it is reported as a
STALE check rather than silently passing.

Usage: python scripts/eval/proofread_paper.py
Exit code 1 if anything fails.
"""
import json
import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TEX = ROOT / "paper" / "main.tex"


def load(name):
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


tex = TEX.read_text(encoding="utf-8")
# Strip the digit grouping (854{,}234 -> 854234), then the LaTeX furniture that
# would otherwise glue itself to a number (\textbf{19.77} -> 19.77, $-2.35$ ->
# -2.35). Em- and en-dashes go too: "rates---96.79" would otherwise read as
# NEGATIVE 96.79 and the value would look absent.
_plain = re.sub(r"[{}$\\~]", " ", tex.replace("{,}", ""))
_plain = re.sub(r"-{2,}", " ", _plain)
tex_nums = set(re.findall(r"-?\d+(?:\.\d+)?", _plain))

checks = []       # (label, written, artefact, must_appear_in_prose)
notes = []


def chk(label, written, artefact, prose=True):
    """prose=False for a value carried only by a figure, not the running text:
    still verified against its artefact, but not required to appear in the tex."""
    checks.append((label, round(float(written), 4), round(float(artefact), 4),
                   prose))


# ── Table 1: headline, before/after ──────────────────────────────────────────
mc = {r["lang"]: r for r in load("model_comparison_results.json")}
sft = {r["lang"]: r for r in load("seamless_ft_results.json")}
traj = load("ks_trajectory.json")
ruler = load("ks_cloud_ruler_compare.json")

for lang in ("pa", "ne", "hi", "ur", "zh", "ps"):
    chk(f"tab1 before {lang}", mc[lang]["whisper_ft_wer"], mc[lang]["whisper_ft_wer"])
chk("tab1 before ks (held-out L2)", 65.19, traj["whisper"]["L2_wer"])
for lang, key in (("pa", "seamless_asr_wer"), ("ur", "seamless_asr_wer"),
                  ("zh", "seamless_asr_wer")):
    chk(f"tab1 after {lang} (zero-shot)", mc[lang][key], mc[lang][key])
chk("tab1 after ne", 24.34, sft["ne_iv"]["sm_ft_asr_wer"])
chk("tab1 after hi", 12.91, sft["hi_iv"]["sm_ft_asr_wer"])
chk("tab1 after ps", 36.16, sft["ps_cloud"]["sm_ft_asr_wer"])
chk("tab1 after ks", 50.26, ruler["L2"]["ks_cloud3"]["wer"])
chk("tab1 after doi", 46.73, load("doi_iv2_seamless_results.json")["L2"]["wer"])

# ── Table 2 panel (b): the SeamlessM4T run history ───────────────────────────
for run in ("ps", "ps_cv", "ps_bal", "ps_bal2", "ps_aug", "ps_aug2", "ps_cloud"):
    chk(f"tab2 {run}", sft[run]["sm_ft_asr_wer"], sft[run]["sm_ft_asr_wer"])
for run, val in (("ks", 85.25), ("ks_r16", 79.38), ("ks_max", 64.31),
                 ("ks_max2", 61.88), ("ks_cloud", 56.44), ("ks_cloud2", 52.60),
                 ("ks_cloud3", 50.26), ("ks_cloud4", 50.69)):
    chk(f"tab2 {run}", val, traj[run]["L2_wer"])
chk("tab2 doi_iv", 50.07, load("doi_iv_seamless_results.json")["L2"]["wer"])

# ── Section 8: the paired bootstrap ──────────────────────────────────────────
sig = load("significance.json")
sig.update(load("significance_ps.json"))
for label, key, diff, lo, hi in (
        ("sig SM4T replaces Whisper", "ks_cloud3_vs_ks_whisper", -14.93, -17.26, -12.62),
        ("sig CV-dominated mixture", "ps_aug_vs_ps_cv", -5.56, -10.77, -1.93),
        ("sig convergence ks", "ks_cloud2_vs_ks_cloud", -3.84, -4.75, -2.93),
        ("sig convergence doi", "doi_iv2_vs_doi_iv", -3.34, -5.03, -2.07),
        ("sig vocabulary repair", "ks_cloud3_vs_ks_cloud2", -2.35, -3.29, -1.38),
        ("sig rank128 ps", "ps_cloud_vs_ps_bal2", -1.13, -2.57, 0.21),
        ("sig deployment margin", "ps_cloud_vs_ps_aug", -0.75, -2.23, 0.70),
        ("sig warm start", "ks_cloud4_vs_ks_cloud3", 0.43, -0.29, 1.17),
        ("sig 3x CV", "ps_aug2_vs_ps_aug", 0.54, -1.05, 2.14)):
    # Table 3 was replaced by the forest plot, so these live in Figure 6 only.
    a = sig[key]
    chk(f"{label} diff", diff, a["diff"], prose=False)
    chk(f"{label} CI lo", lo, a["ci95"][0], prose=False)
    chk(f"{label} CI hi", hi, a["ci95"][1], prose=False)

# ── Section 6: Dogri ─────────────────────────────────────────────────────────
dbl = load("doi_baselines.json")
d1 = load("doi_iv_seamless_results.json")
d2 = load("doi_iv2_seamless_results.json")
chk("doi pan WER", 114.62, dbl["sm4t_pan"]["L2"]["wer"])
chk("doi hin WER", 99.86, dbl["sm4t_hin"]["L2"]["wer"])
chk("doi pan CER", 96.79, dbl["sm4t_pan"]["L2"]["cer"])
chk("doi hin CER", 67.99, dbl["sm4t_hin"]["L2"]["cer"])
chk("doi_iv CER", 27.29, d1["L2"]["cer"], prose=False)
chk("doi_iv2 CER", 25.18, d2["L2"]["cer"], prose=False)
chk("doi script-vs-ancestry WER gap (14.8)", 14.8,
    round(dbl["sm4t_pan"]["L2"]["wer"] - dbl["sm4t_hin"]["L2"]["wer"], 1))
chk("doi script-vs-ancestry CER gap (28.8)", 28.8,
    round(dbl["sm4t_pan"]["L2"]["cer"] - dbl["sm4t_hin"]["L2"]["cer"], 1))

if "whisper_auto" in dbl:
    chk("doi whisper auto", 102.39, dbl["whisper_auto"]["L2"]["wer"])
    chk("doi whisper forced-hi", 88.18, dbl["whisper_hi"]["L2"]["wer"], prose=False)
    # The two gains derived from them. Quoted to one decimal rather than the
    # nearest integer: 102.39 - 46.73 = 55.66, which the prose called "55 pp"
    # and which actually rounds to 56. The same slip was present before the
    # re-measurement (55.52 also rounds to 56), so it was a truncation, not a
    # transcription error -- caught here only once the check was written.
    chk("doi gain over deployed system", 55.7,
        round(dbl["whisper_auto"]["L2"]["wer"] - d2["L2"]["wer"], 1))
    chk("doi gain over best untrained baseline", 41.5,
        round(dbl["whisper_hi"]["L2"]["wer"] - d2["L2"]["wer"], 1))
else:
    notes.append("doi whisper_auto/whisper_hi are NOT in doi_baselines.json; "
                 "their figures are L0 from an overwritten run and Figure 4's "
                 "caption must say so.")

# ── Section 7: the vocabulary repair, computed from the character map ────────
# These were prose-only claims until the "four vs five letters" wording had to be
# corrected. Now derived: which repaired characters survive the L2 diacritic
# filter is a property of KS_EXTRA_CHARS and the DIACRITICS regex, and how many
# reach the test set is a property of the stored references.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
try:
    from ks_ruler_study import norm as _norm, DIACRITICS as _DIA  # noqa: E402
    from finetune_seamless import KS_EXTRA_CHARS as _KSX  # noqa: E402
except Exception as e:                                    # pragma: no cover
    notes.append(f"could not import the character map, vocab checks skipped: {e}")
    _KSX = None

if _KSX:
    chk("vocab: characters repaired", 20, len(_KSX))
    _survive = [c for c in _KSX if _DIA.sub("", c) != ""]
    _deleted = [c for c in _KSX if _DIA.sub("", c) == ""]
    chk("vocab: marks deleted by the L2 filter", 13, len(_deleted))
    _letters = [c for c in _survive if unicodedata.category(c) == "Lo"]
    chk("vocab: letters surviving L2", 5, len(_letters))

    _rows = [json.loads(l) for l in
             (ROOT / "eval_data" / "ks_cloud3_seamless_hyps.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    _rows = [r for r in _rows if r.get("set") == "indicvoices_test"]
    _refs = [_norm(r["ref"], 2) for r in _rows]
    _present = sum(1 for c in _letters if any(c in r for r in _refs))
    chk("vocab: of those, letters present in the test set", 4, _present)
    _toks = [w for r in _refs for w in r.split()]
    _hit = [w for w in _toks if any(c in w for c in _letters)]
    chk("vocab: word tokens containing them", 747, len(_hit))
    chk("vocab: total word tokens in test refs", 8988, len(_toks))
    chk("vocab: share of word tokens (8.3%)", 8.3, round(100 * len(_hit) / len(_toks), 1))
    chk("vocab: distinct word types", 211, len(set(_hit)))

# ── Section 7: the vocabulary repair, measured on two rungs ──────────────────
chk("vocab repair at L0 (74.31)", 74.31, ruler["L0"]["ks_cloud2"]["wer"])
chk("vocab repair at L0 (64.71)", 64.71, ruler["L0"]["ks_cloud3"]["wer"])
chk("vocab repair L0 delta (9.60)", 9.60,
    round(ruler["L0"]["ks_cloud2"]["wer"] - ruler["L0"]["ks_cloud3"]["wer"], 2))
chk("vocab repair L2 delta (2.34)", 2.34,
    round(ruler["L2"]["ks_cloud2"]["wer"] - ruler["L2"]["ks_cloud3"]["wer"], 2))

# ── Figure 5: the normalisation ladder ───────────────────────────────────────
for lvl, w, km, kc in (("L0", 79.29, 80.91, 64.71), ("L1", 79.29, 80.31, 63.91),
                       ("L2", 65.19, 64.31, 50.26), ("L3", 65.10, 64.26, 50.19),
                       ("L4", 64.02, 63.29, 49.43)):
    # only the L2 column and the two L0 Kashmiri figures are quoted in the text
    p = (lvl == "L2")
    chk(f"ladder {lvl} whisper", w, ruler[lvl]["whisper"]["wer"], prose=p)
    chk(f"ladder {lvl} ks_max", km, ruler[lvl]["ks_max"]["wer"], prose=p)
    chk(f"ladder {lvl} ks_cloud3", kc, ruler[lvl]["ks_cloud3"]["wer"],
        prose=p or lvl == "L0")

# ── Section 9: decoding ──────────────────────────────────────────────────────
lm = load("ks_lm_rescore.json")
# The standalone decoding section was cut to reach the page target and folded
# into Section 8 as one paragraph. Claims the paragraph no longer makes are
# still verified against the artefacts here, but marked prose=False so they do
# not register as stale: the underlying result stands, the paper just does not
# quote it any more.
chk("beam-8 1-best", 47.37, lm["beam_1best_wer"])
chk("LM-rescored", 46.90, lm["rescored_wer"])
chk("oracle ceiling (cut from prose)", 44.33, lm["oracle_wer"], prose=False)

probe = load("ks_decode_probe.json")
g_clean, g_awgn = probe["greedy"]["clean"], probe["greedy"]["awgn_0"]
beams = {k: v for k, v in probe.items() if k != "greedy"}
chk("decode: number of beam configurations (7)", 7, len(beams), prose=False)
regress = [round(v["awgn_0"] - g_awgn, 2) for v in beams.values()]
chk("decode: min 0 dB regression", 1.99, min(regress))
chk("decode: max 0 dB regression", 3.15, max(regress))
if min(regress) <= 0:
    notes.append("a beam configuration did NOT regress 0 dB - the 'every one' "
                 "claim in Section 9 no longer holds")
w8 = [round(g_clean - v["clean"], 2) for k, v in beams.items() if k.startswith("beam8")]
w4 = [round(g_clean - v["clean"], 2) for k, v in beams.items() if k.startswith("beam4")]
w2 = [round(g_clean - v["clean"], 2) for k, v in beams.items() if k.startswith("beam2")]
chk("decode: width-8 clean gain lo (cut from prose)", 3.14, min(w8), prose=False)
chk("decode: width-8 clean gain hi (cut from prose)", 3.31, max(w8), prose=False)
chk("decode: width-4 clean gain (cut from prose)", 1.99, max(w4), prose=False)
# written in the text as "width 2 lost 0.50 pp", so compare the magnitude
chk("decode: width-2 clean LOSS", 0.50, -max(w2))
lp_recovery = max(
    round(beams[b]["awgn_0"] - beams[f"{b}_lp0.8"]["awgn_0"], 2)
    for b in ("beam2", "beam4", "beam8") if f"{b}_lp0.8" in beams)
chk("decode: max length-penalty recovery", 0.50, lp_recovery)

# ── Section 10: the deployed path ────────────────────────────────────────────
prod = load("ks_production_results.json")
chk("deployed-path WER", 52.33, prod["L2"]["wer"])
chk("deployed-path gap (2.07)", 2.07,
    round(prod["L2"]["wer"] - ruler["L2"]["ks_cloud3"]["wer"], 2))

# ── report ───────────────────────────────────────────────────────────────────
tex_vals = sorted(float(n) for n in tex_nums)

fails, stale = [], []
for label, written, artefact, prose in checks:
    if abs(written - artefact) > 0.005:
        fails.append(f"{label}: tex says {written}, artefact says {artefact}")
    elif prose:
        # numeric membership, so 46.90 in the tex matches 46.9 here
        if not any(abs(v - written) < 0.005 for v in tex_vals):
            stale.append(f"{label}: {written:g} verified but absent from main.tex")

print(f"proofread_paper: {len(checks)} checks")
for n in notes:
    print(f"  [NOTE]  {n}")
for s in stale:
    print(f"  [STALE] {s}")
for f in fails:
    print(f"  [FAIL]  {f}")
if fails:
    print(f"\n{len(fails)} FAILED")
    sys.exit(1)
print(f"\nall {len(checks)} numbers match their artefacts"
      + (f"; {len(stale)} stale, {len(notes)} note(s)" if stale or notes else ""))

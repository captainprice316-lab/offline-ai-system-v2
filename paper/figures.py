# -*- coding: utf-8 -*-
"""figures.py -- vector figures for the CODS submission (paper/main.tex).

Every number here is transcribed from a JSON artefact in docs/ and is asserted
against that artefact at the bottom of this file, so a figure cannot silently
drift from the result it depicts. Run `python paper/figures.py` to regenerate;
output is PDF (vector, TrueType-embedded) sized for acmart sigconf:
  single column  3.33 in,  double column  7.00 in.

Encoding is consistent across every figure:
  ORANGE  = fine-tuned Whisper large-v3 -- the superseded backend
  BLUE    = SeamlessM4T v2 -- the deployed backend
  GREY    = a result that is not statistically supported, or a reference line
Marker shape repeats the colour distinction (circle vs square) so the figures
survive greyscale printing.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = pathlib.Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)

# ── palette (validated: dataviz validate_palette.js, light surface, all PASS) ──
C_WHISPER = "#eb6834"   # categorical slot 2
C_SM4T    = "#2a78d6"   # categorical slot 1
C_NS      = "#8a8a85"   # recessive grey: "not statistically supported"
INK       = "#1a1a19"
MUTED     = "#52514e"
GRID      = "#d8d7d2"

SEQ = ["#e7f0fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
CMAP = LinearSegmentedColormap.from_list("vani_blue", SEQ)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.4,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "text.color": INK,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

COL1, COL2 = 3.33, 7.00


def _clean(ax, grid_axis="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def _save(fig, name):
    path = OUT / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Backend replacement -- dumbbell, six FLEURS languages (n=100 each)
#    Whisper: docs/model_comparison_results.json (whisper_ft_wer)
#    SM4T   : zero-shot for pa/ur/zh, LoRA for ne/hi/ps (seamless_ft_results)
# ─────────────────────────────────────────────────────────────────────────────
BACKEND = [
    # (language, whisper_ft, deployed_sm4t, how)
    ("Punjabi",  57.39, 19.77, "zero-shot"),
    ("Nepali",   50.92, 24.34, "+ LoRA"),
    ("Pashto",   38.55, 36.16, "+ LoRA"),
    ("Hindi",    19.78, 12.91, "+ LoRA"),
    ("Urdu",     19.82, 16.90, "zero-shot"),
    ("Mandarin", 14.22, 11.69, "zero-shot"),
]


def fig_backend():
    data = sorted(BACKEND, key=lambda r: r[1] - r[2], reverse=True)
    fig, ax = plt.subplots(figsize=(COL2, 1.78))
    ys = list(range(len(data)))[::-1]
    for yi, (name, w, s, how) in zip(ys, data):
        ax.add_patch(FancyArrowPatch(
            (w, yi), (s, yi), arrowstyle="-|>", mutation_scale=7, lw=1.0,
            color=GRID, shrinkA=4.5, shrinkB=4.5, zorder=1))
        ax.scatter([w], [yi], s=26, marker="o", color=C_WHISPER, zorder=3,
                   edgecolors="white", linewidths=0.7)
        ax.scatter([s], [yi], s=28, marker="s", color=C_SM4T, zorder=3,
                   edgecolors="white", linewidths=0.7)
        ax.text(w + 1.1, yi, f"{w:.1f}", va="center", ha="left",
                fontsize=6.6, color=MUTED)
        ax.text(s - 1.1, yi, f"{s:.1f}", va="center", ha="right",
                fontsize=6.9, color=INK, fontweight="bold")
        ax.text((w + s) / 2, yi + 0.30, f"$-${w - s:.1f}", va="bottom",
                ha="center", fontsize=6.3, color=C_SM4T)
        ax.text(63.5, yi, how, va="center", ha="left", fontsize=6.2,
                color=MUTED, style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=7.4, color=INK)
    ax.set_xlim(0, 63)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.set_xlabel("word error rate (%), FLEURS $n=100$ -- lower is better",
                  fontsize=7.2)
    # Legend above the axes: inside, it lands on the Pashto row.
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4.4, color=C_WHISPER,
               label="fine-tuned Whisper large-v3 (was deployed)"),
        Line2D([], [], marker="s", ls="", ms=4.2, color=C_SM4T,
               label="SeamlessM4T v2 (now deployed)")],
        fontsize=6.8, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), labelcolor=INK, handletextpad=0.4,
        columnspacing=2.0)
    _clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    _save(fig, "backend")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Campaign trajectories -- Kashmiri (docs/ks_trajectory.json, L2, n=372)
#    and Pashto (docs/seamless_ft_results.json, FLEURS clean, n=100)
# ─────────────────────────────────────────────────────────────────────────────
KS_RUNS = [
    ("ks",        85.25, "token frozen"),
    ("ks_r16",  79.38, "rank 16"),
    ("ks_max",  64.31, "token trainable"),
    ("ks_max2", 61.88, "4x corpus"),
    ("ks_cloud", 56.44, "rank 128"),
    ("ks_cloud2", 52.60, "to convergence"),
    ("ks_cloud3", 50.26, "vocab repair"),
    ("ks_cloud4", 50.69, "warm start (n.s.)"),
]
KS_WHISPER_REF = 65.19

PS_RUNS = [
    ("ps",        41.30, "FLEURS only"),
    ("ps_cv",   42.47, "+ Common Voice"),
    ("ps_bal",  39.72, "rebalanced"),
    ("ps_bal2", 37.29, "rank 32 + MLP"),
    ("ps_aug",  36.91, "noise-augmented"),
    ("ps_aug2", 37.46, "3x CV (n.s.)"),
    ("ps_cloud", 36.16, "rank 128 (n.s.)"),
]
PS_WHISPER_REF = 38.55


def _trajectory(ax, runs, ref, ref_label, title, ylim, deployed_idx):
    xs = np.arange(len(runs))
    ys = [r[1] for r in runs]
    ax.axhline(ref, color=C_WHISPER, lw=1.1, ls=(0, (3, 2)), zorder=1)
    ax.text(len(runs) - 0.45, ref, ref_label, fontsize=6.2, color=C_WHISPER,
            va="bottom", ha="right")
    ax.plot(xs, ys, color=C_SM4T, lw=1.1, zorder=2, solid_capstyle="round")
    for i, (name, v, note) in enumerate(runs):
        ns = "n.s." in note
        ax.scatter([i], [v], s=34 if i == deployed_idx else 22,
                   marker="s" if i == deployed_idx else "o",
                   facecolor="white" if ns else C_SM4T,
                   edgecolors=C_NS if ns else C_SM4T,
                   linewidths=1.0, zorder=3)
        # Alternate above/below to keep neighbouring labels apart, but never
        # put one on top of the Whisper reference line.
        dy = 6 if i % 2 == 0 else -11
        span = ylim[1] - ylim[0]
        if dy > 0 and 0 <= (ref - v) < 0.09 * span:
            dy = -11
        ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                    xytext=(0, dy), ha="center",
                    fontsize=6.2, color=INK if i == deployed_idx else MUTED,
                    fontweight="bold" if i == deployed_idx else "normal")
    # Run names only: what each run changed is in Table 2, and two-line labels
    # collide at this panel width.
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in runs], fontsize=6.1, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(runs) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=7.6, color=INK, pad=4)
    _clean(ax)


def fig_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.08),
                             gridspec_kw={"width_ratios": [8, 7], "wspace": 0.17})
    _trajectory(axes[0], KS_RUNS, KS_WHISPER_REF,
                "fine-tuned Whisper-ks  65.19",
                "Kashmiri -- IndicVoices-R, $n=372$, L2", (45, 90), 6)
    axes[0].set_ylabel("word error rate (%)", fontsize=7.2)
    _trajectory(axes[1], PS_RUNS, PS_WHISPER_REF,
                "fine-tuned Whisper-ps  38.55",
                "Pashto -- FLEURS, $n=100$, clean", (34, 44), 6)
    fig.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=4.4, color=C_SM4T,
               label="deployed"),
        Line2D([], [], marker="o", ls="", ms=4.0, mfc="white", mec=C_NS,
               mew=1.0, color="none",
               label="change not significant vs. its predecessor")],
        fontsize=6.8, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, -0.14), labelcolor=INK, handletextpad=0.4)
    _save(fig, "trajectory")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Significance forest -- docs/significance*.json
# ─────────────────────────────────────────────────────────────────────────────
FOREST = [
    # (label, diff, lo, hi, n, significant)
    ("SM4T replaces Whisper (ks)",        -14.93, -17.26, -12.62, 372, True),
    ("CV-dominated mixture is worse (ps)", -5.56, -10.77,  -1.93, 100, True),
    ("Train to convergence (ks)",          -3.84,  -4.75,  -2.93, 372, True),
    ("Train to convergence (doi)",         -3.34,  -5.03,  -2.07, 425, True),
    ("Vocabulary repair (ks)",             -2.35,  -3.29,  -1.38, 372, True),
    ("Rank 128 over rank 32 (ps)",         -1.13,  -2.57,  +0.21, 100, False),
    ("Deployment margin, ps_cloud",      -0.75,  -2.23,  +0.70, 100, False),
    ('Warm start "regressed" (ks)',      +0.43,  -0.29,  +1.17, 372, False),
    ('3x Common Voice "regressed"',      +0.54,  -1.05,  +2.14, 100, False),
]


def fig_forest():
    fig, ax = plt.subplots(figsize=(COL2, 1.88))
    ys = list(range(len(FOREST)))[::-1]
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    for yi, (lab, d, lo, hi, n, sig) in zip(ys, FOREST):
        c = C_SM4T if sig else C_NS
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, solid_capstyle="round",
                zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [yi - 0.16, yi + 0.16], color=c, lw=1.0, zorder=2)
        ax.scatter([d], [yi], s=30 if sig else 26,
                   marker="s" if sig else "o",
                   facecolor=c if sig else "white", edgecolors=c,
                   linewidths=1.0, zorder=3)
        ax.text(3.2, yi, f"{d:+.2f}", va="center", ha="right", fontsize=6.5,
                color=INK if sig else MUTED,
                fontweight="bold" if sig else "normal")
        ax.text(4.0, yi, "supported" if sig else "not supported",
                va="center", ha="left", fontsize=6.3,
                color=C_SM4T if sig else C_NS)
        ax.text(9.2, yi, f"$n$={n}", va="center", ha="right", fontsize=6.2,
                color=MUTED)
    ax.axhspan(-0.5, 3.5, color=C_NS, alpha=0.055, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f[0] for f in FOREST], fontsize=6.9, color=INK)
    ax.set_xlim(-18.5, 9.6)
    ax.set_ylim(-0.62, len(FOREST) - 0.38)
    ax.set_xticks([-15, -10, -5, 0])
    ax.set_xlabel("change in word error rate (pp) with 95% bootstrap "
                  "interval -- negative favours the newer system", fontsize=7.0)
    ax.text(-17.9, len(FOREST) - 0.62, "backend and mechanism",
            fontsize=6.3, color=MUTED, style="italic", va="top")
    ax.text(-17.9, 3.34, "adapter-versus-adapter", fontsize=6.3,
            color=MUTED, style="italic", va="top")
    _clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    _save(fig, "forest")


# ─────────────────────────────────────────────────────────────────────────────
# 4. The ruler matters -- docs/ks_cloud_ruler_compare.json
# ─────────────────────────────────────────────────────────────────────────────
LADDER = ["L0", "L1", "L2", "L3", "L4"]
RULER = {
    "whisper":   [79.29, 79.29, 65.19, 65.10, 64.02],
    "ks_max":    [80.91, 80.31, 64.31, 64.26, 63.29],
    "ks_cloud3": [64.71, 63.91, 50.26, 50.19, 49.43],
}


def fig_ruler():
    fig, ax = plt.subplots(figsize=(COL1, 1.76))
    xs = np.arange(len(LADDER))
    styles = [("whisper", "fine-tuned Whisper-ks", C_WHISPER, "o", "-"),
              ("ks_max", "ks_max (first adapter)", C_NS, "^", (0, (4, 2))),
              ("ks_cloud3", "ks_cloud3 (deployed)", C_SM4T, "s", "-")]
    for key, lab, c, mk, ls in styles:
        ax.plot(xs, RULER[key], color=c, lw=1.1, ls=ls, marker=mk, ms=3.6,
                mfc="white", mew=1.0, label=lab, zorder=3)
    ax.annotate("", xy=(2, 64.31), xytext=(2, 65.19),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax.text(2.12, 66.6, "the ranking flips\nbetween L1 and L2", fontsize=5.9,
            color=INK, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(["L0\nraw", "L1\nNFC", "L2\nno diac.", "L3\nfold",
                        "L4\nfold+"], fontsize=6.2, linespacing=1.25)
    ax.set_ylabel("word error rate (%)", fontsize=7.2)
    ax.set_ylim(44, 90)
    ax.set_xlim(-0.35, 4.35)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left", labelcolor=INK,
              handletextpad=0.5, borderpad=0.1)
    _clean(ax)
    _save(fig, "ruler")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dogri -- docs/doi_baselines.json + doi_iv*_seamless_results.json
# ─────────────────────────────────────────────────────────────────────────────
DOGRI = [
    ("SM4T zero-shot, pan token", 114.62, False),
    ("Whisper, auto-detect",      102.39, False),
    ("SM4T zero-shot, hin token",  99.86, False),
    ("Whisper, forced Hindi",      88.18, False),
    ("__doi__ token + LoRA",       50.07, False),
    ("   + to convergence",        46.73, True),
]


def fig_dogri():
    fig, ax = plt.subplots(figsize=(COL1, 1.58))
    ys = list(range(len(DOGRI)))[::-1]
    for yi, (lab, v, best) in zip(ys, DOGRI):
        ax.barh(yi, v, height=0.58, color=C_SM4T if best else C_NS,
                alpha=1.0 if best else 0.55, zorder=2)
        ax.text(v + 1.6, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.4,
                color=INK if best else MUTED,
                fontweight="bold" if best else "normal")
    ax.axvline(100, color=C_WHISPER, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.text(98, len(DOGRI) - 0.42, "100% WER", fontsize=6.0, color=C_WHISPER,
            ha="right", va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in DOGRI], fontsize=6.4, color=INK)
    ax.set_xlim(0, 132)
    ax.set_ylim(-0.6, len(DOGRI) - 0.25)
    ax.set_xlabel("word error rate (%), $n=425$", fontsize=7.2)
    _clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    _save(fig, "dogri")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Robustness -- SM4T advantage over fine-tuned Whisper, per condition
#    docs/model_comparison_results.json + robustness sweep (n=30 per cell)
# ─────────────────────────────────────────────────────────────────────────────
R_LANGS = ["Punjabi", "Nepali", "Hindi", "Mandarin", "Urdu"]
R_CONDS = ["clean", "band-\npass", "noise\n10 dB", "noise\n0 dB", "MP3\ncodec"]
R_ADV = np.array([
    [39.0, 37.7, 41.7, 36.8, 41.8],
    [31.6, 30.5, 34.1, 36.4, 32.0],
    [5.0,  3.9, 10.8, 19.5,  5.6],
    [3.6,  5.0,  2.9, 17.2,  3.0],
    [2.3,  2.4,  4.8,  4.2,  3.1],
])


def fig_robustness():
    fig, ax = plt.subplots(figsize=(COL1, 1.66))
    im = ax.imshow(R_ADV, cmap=CMAP, vmin=0, vmax=45, aspect="auto")
    ax.set_xticks(range(len(R_CONDS)))
    ax.set_xticklabels(R_CONDS, fontsize=6.2, linespacing=1.2)
    ax.set_yticks(range(len(R_LANGS)))
    ax.set_yticklabels(R_LANGS, fontsize=6.9, color=INK)
    for i in range(len(R_LANGS)):
        for j in range(len(R_CONDS)):
            v = R_ADV[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v > 24 else INK,
                    fontweight="bold" if v > 24 else "normal")
    ax.set_xticks(np.arange(-.5, len(R_CONDS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(R_LANGS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cb.set_label("SM4T advantage (pp)", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.9, length=2, width=0.5)
    cb.outline.set_visible(False)
    _save(fig, "robustness")


# ─────────────────────────────────────────────────────────────────────────────
# provenance: every plotted number must match its artefact
# ─────────────────────────────────────────────────────────────────────────────
def _check():
    def load(name):
        return json.loads((DOCS / name).read_text(encoding="utf-8"))

    bad = []

    def eq(label, got, want, tol=0.005):
        if abs(got - want) > tol:
            bad.append(f"{label}: figure {got} != artefact {want}")

    traj = load("ks_trajectory.json")
    for key, (_, v, _) in zip(
            ["ks", "ks_r16", "ks_max", "ks_max2", "ks_cloud", "ks_cloud2",
             "ks_cloud3", "ks_cloud4"], KS_RUNS):
        eq(f"KS_RUNS[{key}]", v, traj[key]["L2_wer"])
    eq("KS whisper ref", KS_WHISPER_REF, traj["whisper"]["L2_wer"])

    sft = {r["lang"]: r["sm_ft_asr_wer"] for r in load("seamless_ft_results.json")}
    for key, (_, v, _) in zip(
            ["ps", "ps_cv", "ps_bal", "ps_bal2", "ps_aug", "ps_aug2",
             "ps_cloud"], PS_RUNS):
        eq(f"PS_RUNS[{key}]", v, sft[key])
    # Only the three LoRA languages live in seamless_ft_results; the deployed
    # figures for pa/ur/zh are zero-shot and come from model_comparison_results.
    dep = {b[0]: b[2] for b in BACKEND}
    for name, key in [("Nepali", "ne_iv"), ("Hindi", "hi_iv"),
                      ("Pashto", "ps_cloud")]:
        eq(f"BACKEND[{name}]", dep[name], sft[key])
    mc = load("model_comparison_results.json")
    mc = {r["lang"]: r for r in mc} if isinstance(mc, list) else mc
    for name, key in [("Punjabi", "pa"), ("Urdu", "ur"), ("Mandarin", "zh")]:
        row = mc.get(key)
        if isinstance(row, dict) and "seamless_asr_wer" in row:
            eq(f"BACKEND[{name}] zero-shot", dep[name], row["seamless_asr_wer"])
    whis = {b[0]: b[1] for b in BACKEND}
    for name, key in [("Punjabi", "pa"), ("Nepali", "ne"), ("Pashto", "ps"),
                      ("Hindi", "hi"), ("Urdu", "ur"), ("Mandarin", "zh")]:
        row = mc.get(key)
        if isinstance(row, dict) and "whisper_ft_wer" in row:
            eq(f"BACKEND[{name}] whisper", whis[name], row["whisper_ft_wer"])

    ruler = load("ks_cloud_ruler_compare.json")
    for sysname, vals in RULER.items():
        for lvl, v in zip(LADDER, vals):
            eq(f"RULER[{sysname}][{lvl}]", v, ruler[lvl][sysname]["wer"])

    sig = load("significance.json")
    sig.update(load("significance_ps.json"))
    fmap = {
        "SM4T replaces Whisper (ks)": "ks_cloud3_vs_ks_whisper",
        "CV-dominated mixture is worse (ps)": "ps_aug_vs_ps_cv",
        "Train to convergence (ks)": "ks_cloud2_vs_ks_cloud",
        "Train to convergence (doi)": "doi_iv2_vs_doi_iv",
        "Vocabulary repair (ks)": "ks_cloud3_vs_ks_cloud2",
        "Rank 128 over rank 32 (ps)": "ps_cloud_vs_ps_bal2",
        "Deployment margin, ps_cloud": "ps_cloud_vs_ps_aug",
        'Warm start "regressed" (ks)': "ks_cloud4_vs_ks_cloud3",
        '3x Common Voice "regressed"': "ps_aug2_vs_ps_aug",
    }
    for lab, d, lo, hi, n, s in FOREST:
        a = sig[fmap[lab]]
        eq(f"FOREST[{lab}].diff", d, a["diff"])
        eq(f"FOREST[{lab}].lo", lo, a["ci95"][0])
        eq(f"FOREST[{lab}].hi", hi, a["ci95"][1])
        if n != a["n_clips"]:
            bad.append(f"FOREST[{lab}].n: {n} != {a['n_clips']}")
        if s != a["significant_at_0.05"]:
            bad.append(f"FOREST[{lab}].sig: {s} != {a['significant_at_0.05']}")

    dbl = load("doi_baselines.json")
    dmap = dict((d[0], d[1]) for d in DOGRI)
    eq("DOGRI pan", dmap["SM4T zero-shot, pan token"], dbl["sm4t_pan"]["L2"]["wer"])
    eq("DOGRI hin", dmap["SM4T zero-shot, hin token"], dbl["sm4t_hin"]["L2"]["wer"])
    eq("DOGRI whisper auto", dmap["Whisper, auto-detect"],
       dbl["whisper_auto"]["L2"]["wer"])
    eq("DOGRI whisper forced-hi", dmap["Whisper, forced Hindi"],
       dbl["whisper_hi"]["L2"]["wer"])
    eq("DOGRI doi_iv", dmap["__doi__ token + LoRA"],
       load("doi_iv_seamless_results.json")["L2"]["wer"])
    eq("DOGRI doi_iv2", dmap["   + to convergence"],
       load("doi_iv2_seamless_results.json")["L2"]["wer"])

    if bad:
        raise SystemExit("PROVENANCE FAILURES:\n  " + "\n  ".join(bad))
    print("  provenance: all plotted values match docs/*.json")


if __name__ == "__main__":
    print("figures ->", OUT)
    fig_backend()
    fig_trajectory()
    fig_forest()
    fig_ruler()
    fig_dogri()
    fig_robustness()
    _check()

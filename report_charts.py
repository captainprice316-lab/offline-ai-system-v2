"""report_charts.py — shared matplotlib figures for the VANI report PDF + PPTX.

One source of truth so both documents render byte-identical charts. Each
function returns a BytesIO PNG buffer. Numbers are cited from the source
JSON/CSV artefacts (see comments); the proofreader
(scripts/eval/proofread_docs_deep.py) asserts them against those sources.

Design (per the dataviz method): consistent encoding across every figure —
GREY = fine-tuned Whisper (the superseded / rollback backend),
TEAL = deployed SeamlessM4T (the production backend). GREEN = improvement.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# ── shared palette ──────────────────────────────────────────────────────────
C_WHISPER = "#9E9E9E"   # neutral grey — superseded backend
C_SM4T    = "#0B6E75"   # deep teal — deployed winner
C_CONN    = "#CFCFCF"
C_GOOD    = "#2E7D32"
C_BAD     = "#B71C1C"
INK       = "#37474F"
MUTED     = "#78909C"


def _save(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── 1. HERO: deployed SM4T vs FT-Whisper WER (dumbbell) ──────────────────────
# n=100 FLEURS held-out, same scorer. ft_whisper: model_comparison_results.json
# whisper_ft_wer. deployed: zero-shot = seamless_asr_wer (pa/ur/zh); LoRA =
# seamless_ft_results.json <lang>_iv / ps_aug (hi/ne/ps). Kashmiri is excluded
# on purpose — different corpus + ruler; it has its own chart (#3).
HERO_DATA = [
    # (name, ft_whisper, deployed_sm4t, backend_tag)
    ("Punjabi (pa)",  57.39, 19.77, "zero-shot"),
    ("Nepali (ne)",   50.92, 24.34, "+ LoRA"),
    ("Hindi (hi)",    19.78, 12.91, "+ LoRA"),
    ("Urdu (ur)",     19.82, 16.90, "zero-shot"),
    ("Mandarin (zh)", 14.22, 11.69, "zero-shot"),
    ("Pashto (ps)",   38.55, 36.91, "+ LoRA"),
]


def hero_backend_dumbbell():
    data = sorted(HERO_DATA, key=lambda r: r[1] - r[2], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ys = list(range(len(data)))[::-1]
    for yi, (name, w, s, tag) in zip(ys, data):
        ax.add_patch(FancyArrowPatch((w, yi), (s, yi), arrowstyle="-|>",
                     mutation_scale=14, lw=2.2, color=C_CONN,
                     shrinkA=7, shrinkB=7, zorder=1))
        ax.scatter([w], [yi], s=150, color=C_WHISPER, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.scatter([s], [yi], s=170, color=C_SM4T, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(w + 1.3, yi, f"{w:.1f}", va="center", ha="left",
                fontsize=9, color=MUTED)
        ax.text(s - 1.3, yi, f"{s:.1f}", va="center", ha="right",
                fontsize=9.5, color=INK, fontweight="bold")
        ax.text((w + s) / 2, yi + 0.28, f"-{w - s:.1f} pp", va="bottom",
                ha="center", fontsize=8.5, color=C_GOOD, fontweight="bold")
        ax.text(-2.2, yi - 0.30, tag, va="center", ha="right",
                fontsize=7.5, color=MUTED, style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=10)
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlabel("Word Error Rate (%)  —  lower is better", fontsize=10.5)
    ax.set_title("Deployed SeamlessM4T backend beats fine-tuned Whisper on WER",
                 fontsize=13.5, fontweight="bold", pad=14)
    ax.text(0.0, 1.015, "n = 100 FLEURS held-out test, same scorer",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_WHISPER,
               markersize=11, label="Fine-tuned Whisper"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SM4T,
               markersize=11, label="Deployed SeamlessM4T")],
        loc="lower right", fontsize=9.5, frameon=False)
    fig.tight_layout()
    return _save(fig)


# ── 2. Robustness heatmap: SM4T advantage (pp) per condition ─────────────────
# FT WER - SM4T WER, positive = SM4T better. FINETUNE_REPORT 5.5.1 (zero-shot
# comparison; the deployed hi/ne LoRA adapters improve on these further).
ROBUST_LANGS = ["Punjabi", "Nepali", "Hindi", "Urdu", "Mandarin"]
ROBUST_CONDS = ["Clean", "Bandpass", "Noise\n10 dB", "Noise\n0 dB", "MP3\ncodec"]
ROBUST_ADV = np.array([
    [39.0, 37.7, 41.7, 36.8, 41.8],
    [31.6, 30.5, 34.1, 36.4, 32.0],
    [ 5.0,  3.9, 10.8, 19.5,  5.6],
    [ 2.3,  2.4,  4.8,  4.2,  3.1],
    [ 3.6,  5.0,  2.9, 17.2,  3.0],
])


def robustness_heatmap():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    im = ax.imshow(ROBUST_ADV, cmap="Greens", vmin=0, vmax=45, aspect="auto")
    ax.set_xticks(range(len(ROBUST_CONDS)))
    ax.set_xticklabels(ROBUST_CONDS, fontsize=9.5)
    ax.set_yticks(range(len(ROBUST_LANGS)))
    ax.set_yticklabels(ROBUST_LANGS, fontsize=10.5)
    for i in range(len(ROBUST_LANGS)):
        for j in range(len(ROBUST_CONDS)):
            v = ROBUST_ADV[i, j]
            ax.text(j, i, f"+{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 22 else "#1B5E20", fontweight="bold")
    ax.set_title("SeamlessM4T's WER advantage over fine-tuned Whisper (pp)\n"
                 "holds in every condition and widens under noise",
                 fontsize=12, fontweight="bold", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("SM4T advantage (pp)", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(ROBUST_CONDS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ROBUST_LANGS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return _save(fig)


# ── 3. Kashmiri ruler correction: the verdict flips ──────────────────────────
# ks_ruler_study.json indicvoices_test. Whisper = deployed CT2; ks_max = adapter.
KS_GROUPS  = ["Raw WER", "Diacritic-\nnormalised WER", "CER (raw)"]
KS_WHISPER = [79.29, 65.19, 44.23]
KS_KSMAX   = [80.91, 64.31, 39.33]


def ks_ruler_bars():
    x = np.arange(len(KS_GROUPS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    b1 = ax.bar(x - w/2, KS_WHISPER, w, label="Whisper-ks (deployed CT2)",
                color=C_WHISPER, edgecolor="white")
    b2 = ax.bar(x + w/2, KS_KSMAX, w, label="ks_max (SeamlessM4T adapter)",
                color=C_SM4T, edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=INK)
    ax.annotate("Whisper\nappears ahead", xy=(0, 80), xytext=(0, 92),
                ha="center", fontsize=8, color=C_BAD, fontweight="bold")
    for gi in (1, 2):
        ax.annotate("SM4T wins", xy=(gi, KS_KSMAX[gi] + 1), xytext=(gi, KS_KSMAX[gi] + 14),
                    ha="center", fontsize=8.5, color=C_GOOD, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_GOOD))
    ax.set_xticks(x)
    ax.set_xticklabels(KS_GROUPS, fontsize=10)
    ax.set_ylabel("Error rate (%)  —  lower is better", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Kashmiri: correcting the Perso-Arabic scoring ruler reverses the verdict\n"
                 "(same 372 IndicVoices clips, same scorer)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path("scratch_charts")
    out.mkdir(exist_ok=True)
    for name, fn in [("hero", hero_backend_dumbbell),
                     ("robustness", robustness_heatmap),
                     ("ks_ruler", ks_ruler_bars)]:
        (out / f"{name}_final.png").write_bytes(fn().read())
        print(f"wrote {name}_final.png")

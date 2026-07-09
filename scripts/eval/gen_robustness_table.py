"""Quick script to regenerate robustness_table.tex from robustness_results.csv."""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from utils import ROOT as VANI_ROOT

CONDITIONS = [
    "clean", "bandpass", "awgn_20", "awgn_10", "awgn_5", "awgn_0", "ptt_clip", "codec_mp3",
]
LANG_NAMES = {
    "pa": "Punjabi", "hi": "Hindi", "ur": "Urdu", "ne": "Nepali",
    "zh": "Mandarin", "ps": "Pashto", "ks": "Kashmiri",
}

csv_path = VANI_ROOT / "eval_data" / "robustness_results.csv"
rows = []
with open(csv_path, newline="") as f:
    for r in csv.DictReader(f):
        r["c1_ok"] = int(r["c1_ok"])
        r["c2_ok"] = int(r["c2_ok"])
        r["c3_ok"] = int(r["c3_ok"])
        r["c4_ok"] = int(r["c4_ok"])
        rows.append(r)

# Determine which languages and conditions are present in the data
present_langs  = sorted({r["lang"] for r in rows}, key=lambda l: list(LANG_NAMES.keys()).index(l) if l in LANG_NAMES else 99)
present_conds  = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]

print(f"Languages: {present_langs}")
print(f"Conditions: {present_conds}")

# Print ASCII summary table
print("\nFull VANI LangID Accuracy (%) — c4_ok")
header = f"{'Condition':<12}" + "".join(f"{LANG_NAMES.get(l, l):>10}" for l in present_langs) + f"{'Ovrl':>7}"
print(header)
print("-" * len(header))
for cond in present_conds:
    crows = [r for r in rows if r["condition"] == cond]
    parts = []
    for lang in present_langs:
        lr = [r["c4_ok"] for r in crows if r["lang"] == lang]
        parts.append(sum(lr)/len(lr)*100 if lr else float('nan'))
    valid = [p for p in parts if not (isinstance(p, float) and p != p)]
    ovrl = sum(valid)/len(valid) if valid else 0
    row_str = f"{cond:<12}" + "".join(f"{p:>9.1f}%" for p in parts) + f"{ovrl:>6.1f}%"
    print(row_str)

# Build LaTeX table
cond_labels = {
    "clean":    "Clean",    "bandpass": "Bandpass",
    "awgn_20":  "AWGN 20dB","awgn_10": "AWGN 10dB",
    "awgn_5":   "AWGN 5dB", "awgn_0":  "AWGN 0dB",
    "ptt_clip": "PTT clip", "codec_mp3":"MP3 16kbps",
}
cfg_label = {"c1_ok":"Whisper","c2_ok":"+FastText","c3_ok":"+MMS","c4_ok":"Full VANI"}

col_langs = " & ".join(f"\\textbf{{{LANG_NAMES.get(l,l)}}}" for l in present_langs)
tex = (
    "\\begin{table}[htbp]\n"
    "\\caption{LangID Accuracy (\\%) under Radio-Channel Degradations}\n"
    "\\label{tab:robustness}\n"
    "\\begin{center}\n"
    f"\\begin{{tabular}}{{ll{'c'*len(present_langs)}r}}\n"
    "\\toprule\n"
    f"\\textbf{{Condition}} & \\textbf{{Config.}} & {col_langs} & \\textbf{{Ovrl.}} \\\\\n"
    "\\midrule\n"
)

for cond in present_conds:
    cond_rows = [r for r in rows if r["condition"] == cond]
    for i, cfg in enumerate(["c1_ok","c2_ok","c3_ok","c4_ok"]):
        cond_label = cond_labels.get(cond, cond) if i == 0 else ""
        cells, accs = [], []
        for lang in present_langs:
            lr = [r[cfg] for r in cond_rows if r["lang"] == lang]
            acc = sum(lr)/len(lr)*100 if lr else float('nan')
            accs.append(acc)
            cells.append(f"{acc:.1f}" if lr else "--")
        valid = [a for a in accs if a == a]
        ovr = sum(valid)/len(valid) if valid else 0
        if cfg == "c4_ok":
            bold_cells = " & ".join(f"\\textbf{{{c}}}" for c in cells)
            tex += f"{cond_label} & {cfg_label[cfg]} & {bold_cells} & \\textbf{{{ovr:.1f}}} \\\\\n"
        else:
            tex += f"{cond_label} & {cfg_label[cfg]} & " + " & ".join(cells) + f" & {ovr:.1f} \\\\\n"
    tex += "\\midrule\n"

tex += (
    "\\multicolumn{" + str(len(present_langs)+3) + "}{l}{"
    "\\footnotesize Degradations applied offline; ks tested with standard turbo model "
    "(ks-specific model not used for LangID eval).} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{center}\n"
    "\\end{table}"
)

tex_path = VANI_ROOT / "eval_data" / "robustness_table.tex"
tex_path.write_text(tex, encoding="utf-8")
print(f"\nLaTeX table saved to {tex_path}")

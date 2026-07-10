"""
score_wer_robustness.py – score the hypotheses produced by wer_robustness_eval.py
================================================================================
Reads eval_data/wer_robustness_hyps.jsonl and emits WER + CER per
(system x language x condition), plus the table that answers the actual question:

    "for each language, which ASR backend should config.yaml route to, given the
     audio VANI really sees?"

Scoring is deliberately separate from inference. Re-running this after fixing a
normaliser costs seconds; re-running the sweep costs hours on an 8 GB card.

Both WER and CER are reported. CER is the honest headline for Mandarin (jiwer
tokenises on whitespace, so WER on Han text only means anything because Whisper
happens to emit spaced output) and for Kashmiri (Perso-Arabic, where a single
normalisation slip inflates WER by tens of points but barely moves CER).

Usage
-----
    python scripts/eval/score_wer_robustness.py
    python scripts/eval/score_wer_robustness.py --metric cer
    python scripts/eval/score_wer_robustness.py --csv eval_data/wer_robustness_results.csv
"""

import sys, json, csv, io, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

IN_JSONL = ROOT / "eval_data" / "wer_robustness_hyps.jsonl"
OUT_CSV  = ROOT / "eval_data" / "wer_robustness_results.csv"

CONDITION_ORDER = ["clean", "bandpass", "awgn_10", "awgn_0", "codec_mp3"]


# One normaliser for every evaluator. See text_norm.py for why.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import normalise, compute_wer, compute_cer, NO_WORD_BOUNDARY  # noqa: E402


def score(refs, hyps, lang):
    """Return (wer, cer) as percentages, or (None, None) if nothing scorable."""
    wer = compute_wer(hyps, refs, lang)
    cer = compute_cer(hyps, refs, lang)
    if wer is None:
        return None, None
    return round(min(wer, 999.9), 2), round(min(cer, 999.9), 2)


def load(path):
    if not path.exists():
        sys.exit(f"no hypotheses at {path} — run wer_robustness_eval.py first")
    buckets = defaultdict(lambda: {"ref": [], "hyp": [], "model": set()})
    torn = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            b = buckets[(r["system"], r["lang"], r["condition"])]
            b["ref"].append(r["ref"])
            b["hyp"].append(r["hyp"])
            b["model"].add(r.get("model", "?"))
    if torn:
        print(f"[warn] skipped {torn} unparseable line(s) — likely a killed run\n")
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",  type=Path, default=IN_JSONL)
    ap.add_argument("--csv",    type=Path, default=OUT_CSV)
    ap.add_argument("--metric", choices=["wer", "cer"], default="wer",
                    help="metric shown in the comparison table (both go to CSV)")
    args = ap.parse_args()

    buckets = load(args.jsonl)

    rows = []
    for (system, lang, cond), b in buckets.items():
        wer, cer = score(b["ref"], b["hyp"], lang)
        rows.append({"system": system, "lang": lang, "condition": cond,
                     "n": len(b["ref"]), "wer": wer, "cer": cer,
                     "model": "|".join(sorted(b["model"]))})

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["system", "lang", "condition", "n",
                                           "wer", "cer", "model"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["system"], r["lang"], r["condition"])):
            w.writerow(r)

    systems    = sorted({r["system"] for r in rows})
    langs      = sorted({r["lang"] for r in rows})
    conds      = [c for c in CONDITION_ORDER if c in {r["condition"] for r in rows}]
    conds     += sorted({r["condition"] for r in rows} - set(conds))
    M          = args.metric
    get        = lambda s, l, c: next((r[M] for r in rows
                                       if r["system"] == s and r["lang"] == l
                                       and r["condition"] == c), None)
    fmt        = lambda v: "  -  " if v is None else f"{v:5.1f}"

    for system in systems:
        print(f"\n{system}  —  {M.upper()} %  (lower is better)")
        print("lang  " + "".join(f"{c:>11}" for c in conds))
        print("-" * (6 + 11 * len(conds)))
        for l in langs:
            cells = "".join(f"{fmt(get(system, l, c)):>11}" for c in conds)
            print(f"{l:5} {cells}")

    # The point of the whole exercise: does the clean-speech routing survive noise?
    if len(systems) >= 2:
        a, b = "whisper_ft", "seamless_zs"
        if a in systems and b in systems:
            print(f"\n\nROUTING CHECK — {M.upper()}, {b} minus {a}")
            print("negative = seamless wins = route this language to seamless\n")
            print("lang  " + "".join(f"{c:>11}" for c in conds) + "     verdict")
            print("-" * (6 + 11 * len(conds) + 22))
            for l in langs:
                deltas, cells = [], ""
                for c in conds:
                    va, vb = get(a, l, c), get(b, l, c)
                    if va is None or vb is None:
                        cells += f"{'  -  ':>11}"
                    else:
                        d = vb - va
                        deltas.append(d)
                        cells += f"{d:+10.1f} "
                if not deltas:
                    verdict = "no seamless support"
                elif all(d < 0 for d in deltas):
                    verdict = "SEAMLESS (all conds)"
                elif all(d > 0 for d in deltas):
                    verdict = "WHISPER (all conds)"
                else:
                    verdict = "MIXED - inspect"
                print(f"{l:5} {cells}    {verdict}")
            print("\n'MIXED' means the clean-speech ranking does not hold under "
                  "degradation. Route on the noisy conditions, not on clean.")

    if any(r["lang"] in NO_WORD_BOUNDARY for r in rows):
        print("\n[caveat] zh is char-segmented before scoring (see NO_WORD_BOUNDARY). Numbers are"
              "\n         still NOT normalised: Seamless writes 二零一一年, FLEURS writes 2011年."
              "\n         That inflates Seamless's zh error. Treat zh numbers as an open item.")

    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

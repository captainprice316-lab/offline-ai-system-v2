"""ks_ruler_study.py — does the Kashmiri WER gap survive honest normalisation?
================================================================================
ks_max (SeamlessM4T, trainable __kas__) lost the WER gate (80.91 vs Whisper-ks
74.02 on the IndicVoices test split) while WINNING CER on the same-clip
robustness set (38.96 vs 47.95) — and this project's own docs call CER the
honest metric for Perso-Arabic Kashmiri because WER over-penalises orthographic
and word-segmentation variation. This study applies a normalisation ladder to
BOTH systems' raw hypotheses and reports WER/CER at each level.

Phase 1 (--gen-whisper, GPU ~15 min): transcribe the same 372 IndicVoices test
clips with the DEPLOYED whisper-large-v3-ks-ct2 through the production ASR
path, persisting hypotheses to eval_data/ks_whisper_test_hyps.jsonl. (The
published 74.02 was the training-time eval of the merged fp16 model — its raw
hypotheses were never saved, and the deployable artefact is the CT2 model, so
this is the fairer ruler anyway.)

Phase 2 (default, CPU): score ks_max vs Whisper on both sets at each
normalisation level -> docs/ks_ruler_study.json + printed table.

Levels:
  L0 raw                    (reproduces the published numbers)
  L1 NFC + strip zero-width + collapse whitespace
  L2 L1 + strip Perso-Arabic combining diacritics
  L3 L2 + conservative folding (yeh/kaf/alef-maqsura variants, digits,
        punctuation stripped)
  L4 L3 + aggressive folding (alef/hamza carriers, heh variants)
  +  boundary-free CER at L3 (spaces removed: pure content accuracy)
"""
import argparse
import io
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

KSMAX_HYPS   = ROOT / "eval_data" / "ks_max_seamless_hyps.jsonl"
WHISPER_ROB  = ROOT / "eval_data" / "wer_robustness_hyps.jsonl"      # system=whisper_ft lang=ks clean
WHISPER_TEST = ROOT / "eval_data" / "ks_whisper_test_hyps.jsonl"     # phase 1 output
OUT_JSON     = ROOT / "docs" / "ks_ruler_study.json"

# ── Normalisation ladder ──────────────────────────────────────────────────────

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏﻿"), None)
# Perso-Arabic combining marks (harakat, superscript alef, Quranic marks)
DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]")
PUNCT = re.compile(r"[،؛؟٪-٭۔.,;:!?\"'()\[\]{}«»\-—_/\\]")

FOLD_CONSERVATIVE = str.maketrans({
    "ي": "ی",  # ARABIC YEH -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ك": "ک",  # ARABIC KAF -> KEHEH
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})
FOLD_AGGRESSIVE = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ؤ": "و",  # WAW WITH HAMZA -> WAW
    "ئ": "ی",  # YEH WITH HAMZA -> FARSI YEH
    "ة": "ه",  # TEH MARBUTA -> HEH
    "ہ": "ه", "ھ": "ه",  # HEH GOAL / DOACHASHMEE -> HEH
    "ے": "ی",  # YEH BARREE -> FARSI YEH
})


def norm(text: str, level: int) -> str:
    t = text
    if level >= 1:
        t = unicodedata.normalize("NFC", t).translate(ZERO_WIDTH)
        t = re.sub(r"\s+", " ", t).strip()
    if level >= 2:
        t = DIACRITICS.sub("", t)
    if level >= 3:
        t = t.translate(FOLD_CONSERVATIVE)
        t = PUNCT.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
    if level >= 4:
        t = t.translate(FOLD_AGGRESSIVE)
    return t


def score(pairs, level, boundary_free=False):
    from jiwer import wer as jwer, cer as jcer
    refs = [norm(r, level) for r, _ in pairs]
    hyps = [norm(h, level) for _, h in pairs]
    if boundary_free:
        refs = [r.replace(" ", "") for r in refs]
        hyps = [h.replace(" ", "") for h in hyps]
        return {"cer": round(100 * jcer(refs, hyps), 2), "n": len(refs)}
    keep = [(r, h) for r, h in zip(refs, hyps) if r]
    refs, hyps = [r for r, _ in keep], [h for _, h in keep]
    return {"wer": round(100 * jwer(refs, hyps), 2),
            "cer": round(100 * jcer(refs, hyps), 2), "n": len(refs)}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def collect_pairs():
    """Returns {set_name: {system: [(ref, hyp), ...]}} with idx-aligned clips."""
    out = defaultdict(dict)

    ksmax = load_jsonl(KSMAX_HYPS)
    for set_name in ("indicvoices_test", "robustness_clean"):
        rows = sorted((r for r in ksmax if r["set"] == set_name), key=lambda r: r["idx"])
        out[set_name]["ks_max"] = [(r["ref"], r["hyp"]) for r in rows]

    rob = [r for r in load_jsonl(WHISPER_ROB)
           if r["system"] == "whisper_ft" and r["lang"] == "ks" and r["condition"] == "clean"]
    out["robustness_clean"]["whisper"] = [(r["ref"], r["hyp"])
                                          for r in sorted(rob, key=lambda r: r["idx"])]

    if WHISPER_TEST.exists():
        wt = sorted(load_jsonl(WHISPER_TEST), key=lambda r: r["idx"])
        out["indicvoices_test"]["whisper"] = [(r["ref"], r["hyp"]) for r in wt]
    else:
        print(f"[warn] {WHISPER_TEST.name} missing — run --gen-whisper first "
              "for the n=372 comparison; proceeding with robustness set only")
    return out


# ── Phase 1: Whisper CT2 hypotheses on the test split ─────────────────────────

def gen_whisper(device="cuda"):
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import soundfile as sf
    import tempfile, os
    from finetune_seamless import LANG_CFG, load_indicvoices_ks, decode_audio
    from wer_robustness_eval import make_whisper, load_config

    cfg = load_config()
    asr, model_name, is_ft = make_whisper(cfg, "ks", device)
    print(f"[gen] {model_name} (fine-tuned: {is_ft})")

    _, val_ds = load_indicvoices_ks(LANG_CFG["ks"])
    print(f"[gen] {len(val_ds)} test clips (same loader/order as every ks eval)")
    with open(WHISPER_TEST, "w", encoding="utf-8") as fh:
        for i in range(len(val_ds)):
            s = val_ds[i]
            arr = decode_audio(s["audio"])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, arr, 16000)
                p = tmp.name
            try:
                asr.reset_language_cache()
                hyp = asr.transcribe(p, language_hint="ks").get("transcript", "") or ""
            finally:
                os.unlink(p)
            fh.write(json.dumps({"idx": i, "ref": s["transcription"], "hyp": hyp.strip()},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"[gen] {i+1}/{len(val_ds)}")
    print(f"[gen] saved -> {WHISPER_TEST}")


# ── Phase 2: the ladder ───────────────────────────────────────────────────────

LEVELS = {0: "L0 raw", 1: "L1 NFC+zw+ws", 2: "L2 +no diacritics",
          3: "L3 +fold cons.+no punct", 4: "L4 +fold aggressive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-whisper", action="store_true",
                    help="phase 1: transcribe the 372 test clips with deployed Whisper-ks CT2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.gen_whisper:
        gen_whisper(args.device)
        return

    pairs = collect_pairs()
    results = {}
    for set_name, systems in pairs.items():
        results[set_name] = {}
        print(f"\n{'='*74}\n{set_name}  ({', '.join(f'{s}: n={len(p)}' for s, p in systems.items())})\n{'='*74}")
        hdr = f"{'level':28}" + "".join(f"{s+' WER':>13}{s+' CER':>13}" for s in systems)
        print(hdr + f"{'WER gap':>10}")
        for lvl, name in LEVELS.items():
            row = {}
            line = f"{name:28}"
            for s, p in systems.items():
                r = score(p, lvl)
                row[s] = r
                line += f"{r['wer']:>13.2f}{r['cer']:>13.2f}"
            if len(systems) == 2:
                gap = row["ks_max"]["wer"] - row["whisper"]["wer"]
                line += f"{gap:>+10.2f}"
                row["wer_gap_ksmax_minus_whisper"] = round(gap, 2)
            results[set_name][name] = row
            print(line)
        # boundary-free CER at L3
        row = {}
        line = f"{'L3 boundary-free CER':28}"
        for s, p in systems.items():
            r = score(p, 3, boundary_free=True)
            row[s] = r
            line += f"{'—':>13}{r['cer']:>13.2f}"
        results[set_name]["L3 boundary-free CER"] = row
        print(line)

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()

"""
ablation_eval.py – VANI LangID Ablation Evaluation
====================================================
Reproduces the 120-sample evaluation from the paper AND runs four
ablation configurations so we can fill in the ablation table:

  Config 1 – Whisper-only          (no FastText, no MMS-LID)
  Config 2 – Whisper + FastText    (2-way vote, no MMS-LID)
  Config 3 – Whisper + FastText + MMS-LID  (3-way vote, no script override)
  Config 4 – Full VANI             (3-way + Unicode script override)

Usage:
    python ablation_eval.py              # run all 4 languages
    python ablation_eval.py hi pa        # specific languages only
    python ablation_eval.py --samples 30 # samples per language (default 30)

Output:
    ablation_results.csv   – per-sample results for all configs
    ablation_summary.txt   – accuracy table ready to paste into LaTeX
"""

import os, sys, gc, time, csv, argparse, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]      = "0"   # need internet to download test sets
os.environ["TRANSFORMERS_OFFLINE"] = "0"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, ROOT as VANI_ROOT

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "pa": {
        "hf_id":      "google/fleurs",
        "config":     "pa_in",
        "split":      "test",
        "audio_col":  "audio",
        "text_col":   "transcription",
        "label":      "pa",
        "name":       "Punjabi",
    },
    "hi": {
        "hf_id":      "google/fleurs",
        "config":     "hi_in",
        "split":      "test",
        "audio_col":  "audio",
        "text_col":   "transcription",
        "label":      "hi",
        "name":       "Hindi",
    },
    "ur": {
        "hf_id":      "google/fleurs",
        "config":     "ur_pk",
        "split":      "test",
        "audio_col":  "audio",
        "text_col":   "transcription",
        "label":      "ur",
        "name":       "Urdu",
    },
    "ne": {
        "hf_id":      "google/fleurs",
        "config":     "ne_np",
        "split":      "test",
        "audio_col":  "audio",
        "text_col":   "transcription",
        "label":      "ne",
        "name":       "Nepali",
    },
}

# Duration filter: 2–20 seconds (matches paper)
MIN_DUR, MAX_DUR = 2.0, 20.0


# ── Four ablation configurations ──────────────────────────────────────────────

def langid_whisper_only(whisper_lang, whisper_prob, transcript, audio_path):
    """Config 1: accept Whisper's language prediction as-is."""
    return whisper_lang or "unknown"


def langid_whisper_fasttext(whisper_lang, whisper_prob, transcript, audio_path, ft_det):
    """Config 2: 2-way confidence-weighted vote."""
    if not transcript:
        return whisper_lang or "unknown"
    ft = ft_det.detect(transcript)
    ft_lang, ft_conf = ft["language"], ft["confidence"]

    scores = {}
    if whisper_lang:
        scores[whisper_lang] = scores.get(whisper_lang, 0) + whisper_prob
    if ft_lang and ft_lang != "unknown":
        scores[ft_lang] = scores.get(ft_lang, 0) + ft_conf
    return max(scores, key=scores.get) if scores else "unknown"


def langid_three_way(whisper_lang, whisper_prob, transcript, audio_path, ft_det, mms_det):
    """Config 3: 3-way vote (no script override)."""
    if not transcript:
        return whisper_lang or "unknown"
    ft  = ft_det.detect(transcript)
    mms = mms_det.detect(audio_path)
    ft_lang, ft_conf   = ft["language"],  ft["confidence"]
    mms_lang, mms_conf = mms["language"], mms["confidence"]

    votes = {}
    if whisper_lang:
        votes[whisper_lang] = votes.get(whisper_lang, 0) + whisper_prob
    if ft_lang and ft_lang != "unknown":
        votes[ft_lang] = votes.get(ft_lang, 0) + ft_conf
    if mms_lang:
        votes[mms_lang] = votes.get(mms_lang, 0) + mms_conf
    return max(votes, key=votes.get) if votes else "unknown"


def langid_full_vani(whisper_lang, whisper_prob, transcript, audio_path, router,
                     ft_det, mms_det):
    """Config 4: full VANI ensemble with script override."""
    from language_module import DialectDetector
    if not transcript:
        return whisper_lang or "unknown"
    ft    = ft_det.detect(transcript)
    mms   = mms_det.detect(audio_path)
    dial  = DialectDetector()
    dd    = dial.detect_code_mix(transcript)
    result = router.detect_family(
        whisper_lang=      whisper_lang,
        transcript=        transcript,
        fasttext_lang=     ft["language"],
        fasttext_conf=     ft["confidence"],
        whisper_lang_prob= whisper_prob,
        dialect=           dd["dialect"],
        mms_lang=          mms["language"],
        mms_conf=          mms["confidence"],
    )
    return result["final_language"]


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(languages, n_samples):
    config   = load_config()
    paths    = config["paths"]
    device   = config.get("device", "cpu")

    # Load models once
    print("\nLoading ASR model...")
    from asr_module import ASRModule
    asr = ASRModule(
        model_path=str(VANI_ROOT / paths["whisper_model"]),
        device=device,
        cfg=config.get("asr", {}),
    )

    print("Loading FastText LangID...")
    from language_module import FastTextLangDetector, LanguageRouter
    ft_det = FastTextLangDetector(model_path=str(VANI_ROOT / paths["fasttext_model"]))
    router = LanguageRouter(confidence_threshold=0.60)

    print("Loading MMS-LID...")
    from mms_module import MMSLangDetector
    mms_det = MMSLangDetector(
        model_path=str(VANI_ROOT / paths.get("mms_lid_model", "models/mms-lid-256")),
        device=device,
    )

    from datasets import load_dataset
    import soundfile as sf
    import numpy as np

    all_rows   = []
    CONFIGS    = ["whisper_only", "w_ft_2way", "w_ft_mms_3way", "vani_full"]
    cfg_names  = {
        "whisper_only":  "Whisper-only",
        "w_ft_2way":     "Whisper + FastText",
        "w_ft_mms_3way": "3-way (no override)",
        "vani_full":     "VANI Full",
    }

    for lang in languages:
        info = DATASETS[lang]
        print(f"\n{'='*55}", flush=True)
        print(f"  Language: {info['name']} ({lang.upper()})  —  {n_samples} samples", flush=True)
        print(f"  Dataset : {info['hf_id']}", flush=True)
        print(f"{'='*55}", flush=True)

        # Load dataset
        kwargs = {"split": info["split"]}
        if info.get("config"):
            kwargs["name"] = info["config"]
        try:
            ds = load_dataset(info["hf_id"], **kwargs, streaming=True)
        except Exception as e:
            print(f"  [ERR] Could not load dataset: {e}")
            continue

        collected, skipped = 0, 0
        for sample in ds:
            if collected >= n_samples:
                break

            # Get audio array — handle both datasets v3 AudioDecoder and old dict format
            audio_data = sample.get(info["audio_col"])
            if audio_data is None:
                skipped += 1; continue
            try:
                if hasattr(audio_data, "get_all_samples"):
                    # datasets v3: torchcodec AudioDecoder
                    samples = audio_data.get_all_samples()
                    arr = np.array(samples.data[0], dtype=np.float32)
                    sr  = int(samples.sample_rate)
                elif isinstance(audio_data, dict):
                    arr = np.array(audio_data["array"], dtype=np.float32)
                    sr  = int(audio_data["sampling_rate"])
                else:
                    skipped += 1; continue
            except Exception:
                skipped += 1; continue

            dur = len(arr) / sr
            if dur < MIN_DUR or dur > MAX_DUR:
                skipped += 1; continue

            ref_text = sample.get(info["text_col"], "")

            # Write to temp file for MMS-LID (needs a path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, arr, sr)
                audio_path = tmp.name

            try:
                # ASR
                asr.reset_language_cache()
                asr_result = asr.transcribe(audio_path, language_hint=None)
                transcript   = asr_result.get("transcript", "")
                whisper_lang = asr_result.get("language", "") or ""
                whisper_prob = asr_result.get("language_probability", 0.0) or 0.0

                # Four ablation decisions
                c1 = langid_whisper_only(whisper_lang, whisper_prob, transcript, audio_path)
                c2 = langid_whisper_fasttext(whisper_lang, whisper_prob, transcript, audio_path, ft_det)
                c3 = langid_three_way(whisper_lang, whisper_prob, transcript, audio_path, ft_det, mms_det)
                c4 = langid_full_vani(whisper_lang, whisper_prob, transcript, audio_path, router, ft_det, mms_det)

                true_label = info["label"]
                row = {
                    "lang":         lang,
                    "true":         true_label,
                    "duration":     round(dur, 2),
                    "whisper_lang": whisper_lang,
                    "whisper_prob": round(whisper_prob, 3),
                    "transcript":   transcript[:80],
                    "c1_whisper":   c1,
                    "c2_w_ft":      c2,
                    "c3_3way":      c3,
                    "c4_vani":      c4,
                    "c1_ok": int(c1 == true_label),
                    "c2_ok": int(c2 == true_label),
                    "c3_ok": int(c3 == true_label),
                    "c4_ok": int(c4 == true_label),
                }
                all_rows.append(row)
                collected += 1
                print(f"  [{collected:2d}/{n_samples}]  true={true_label}  "
                      f"c1={c1} c2={c2} c3={c3} c4={c4}  "
                      f"{'✓' if c4==true_label else '✗'}", flush=True)

            except Exception as e:
                print(f"  [ERR] Sample skipped: {e}")
                skipped += 1
            finally:
                os.unlink(audio_path)

        print(f"  Collected {collected}, skipped {skipped}", flush=True)

    mms_det.unload()
    del asr, mms_det, ft_det, router; gc.collect()
    return all_rows, CONFIGS, cfg_names


def print_summary(rows, configs, cfg_names):
    from collections import defaultdict
    langs = sorted({r["lang"] for r in rows})
    lang_names = {"pa": "Punjabi", "hi": "Hindi", "ur": "Urdu", "ne": "Nepali"}

    print("\n" + "="*70)
    print("ABLATION STUDY — LangID Accuracy")
    print("="*70)

    header = f"{'Configuration':<28}" + "".join(f"{lang_names.get(l,l):>12}" for l in langs) + f"{'Overall':>10}"
    print(header)
    print("-"*70)

    for cfg in configs:
        ok_key = f"{cfg.replace('w_ft_2way','c2').replace('w_ft_mms_3way','c3').replace('vani_full','c4').replace('whisper_only','c1')}_ok"
        # Map cfg to ok key
        ok_map = {
            "whisper_only":  "c1_ok",
            "w_ft_2way":     "c2_ok",
            "w_ft_mms_3way": "c3_ok",
            "vani_full":     "c4_ok",
        }
        key = ok_map[cfg]
        accs = []
        line = f"{cfg_names[cfg]:<28}"
        for lang in langs:
            lang_rows = [r for r in rows if r["lang"] == lang]
            if not lang_rows:
                line += f"{'—':>12}"; continue
            acc = sum(r[key] for r in lang_rows) / len(lang_rows) * 100
            accs.append(acc)
            line += f"{acc:>11.1f}%"
        overall = sum(accs) / len(accs) if accs else 0
        line += f"{overall:>9.1f}%"
        print(line)

    print("="*70)

    # LaTeX table snippet
    print("\n--- LaTeX ablation table (paste into paper) ---\n")
    print(r"\begin{table}[t]")
    print(r"\caption{LangID Accuracy Ablation — Component Contribution}")
    print(r"\label{tab:ablation}")
    print(r"\begin{center}")
    print(r"\begin{tabular}{lccccr}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{pa} & \textbf{hi} & \textbf{ur} & \textbf{ne} & \textbf{Overall} \\")
    print(r"\midrule")

    ok_map = {
        "whisper_only":  "c1_ok",
        "w_ft_2way":     "c2_ok",
        "w_ft_mms_3way": "c3_ok",
        "vani_full":     "c4_ok",
    }
    for cfg in configs:
        key  = ok_map[cfg]
        vals = []
        for lang in ["pa","hi","ur","ne"]:
            lang_rows = [r for r in rows if r["lang"] == lang]
            if not lang_rows:
                vals.append("—")
            else:
                acc = sum(r[key] for r in lang_rows) / len(lang_rows) * 100
                vals.append(f"{acc:.1f}\\%")
        all_rows_lang = [r for r in rows]
        overall = sum(r[key] for r in all_rows_lang) / len(all_rows_lang) * 100 if all_rows_lang else 0
        vals.append(f"\\textbf{{{overall:.1f}\\%}}")
        name = cfg_names[cfg].replace("+", "\\texttt{+}")
        print(f"{name} & " + " & ".join(vals) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{center}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("languages", nargs="*", default=["pa","hi","ur","ne"])
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    bad = [l for l in args.languages if l not in DATASETS]
    if bad:
        print(f"Unknown language(s): {bad}. Available: {list(DATASETS)}")
        sys.exit(1)

    rows, configs, cfg_names = evaluate(args.languages, args.samples)

    if not rows:
        print("No results collected.")
        sys.exit(1)

    # Save CSV
    csv_path = ROOT / "ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    # Save summary
    import io as _io
    old_stdout = sys.stdout
    sys.stdout = buf = _io.StringIO()
    print_summary(rows, configs, cfg_names)
    sys.stdout = old_stdout
    summary = buf.getvalue()
    print(summary)
    (ROOT / "ablation_summary.txt").write_text(summary)
    print(f"Summary saved to ablation_summary.txt")


if __name__ == "__main__":
    main()

"""
robustness_eval.py – VANI Radio-Channel Robustness Evaluation
=============================================================
Evaluates all four VANI ablation configurations under realistic radio-channel
degradations to assess operational viability for SIGINT use cases.

Degradation conditions:
  clean      – unmodified FLEURS audio
  bandpass   – 300–3400 Hz telephony bandpass (HF/VHF radio channel)
  awgn_20    – additive white Gaussian noise, SNR = 20 dB
  awgn_10    – SNR = 10 dB
  awgn_5     – SNR = 5 dB
  awgn_0     – SNR = 0 dB
  ptt_clip   – PTT-style hard clip: 150 ms silence at start + random 60 ms click
  codec_gsm  – GSM 06.10 codec (AMR narrowband, via ffmpeg)

Phase 1: Download and cache N samples per language from FLEURS to disk.
Phase 2: For each condition, apply degradation and run the 4-config LangID eval.
Phase 3: Print accuracy table and write robustness_results.csv.

Usage:
    python robustness_eval.py                  # all 4 langs, 30 samples each
    python robustness_eval.py hi pa --n 50     # specific langs, 50 samples
    python robustness_eval.py --phase1-only    # just download/cache
"""

import os, sys, gc, csv, argparse, warnings, tempfile, shutil, subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]      = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
CACHE_DIR = ROOT / "robustness_cache"

from utils import load_config, ROOT as VANI_ROOT

DATASETS = {
    "pa": {"hf_id": "google/fleurs", "config": "pa_in",  "split": "test",
           "audio_col": "audio", "label": "pa", "name": "Punjabi"},
    "hi": {"hf_id": "google/fleurs", "config": "hi_in",  "split": "test",
           "audio_col": "audio", "label": "hi", "name": "Hindi"},
    "ur": {"hf_id": "google/fleurs", "config": "ur_pk",  "split": "test",
           "audio_col": "audio", "label": "ur", "name": "Urdu"},
    "ne": {"hf_id": "google/fleurs", "config": "ne_np",  "split": "test",
           "audio_col": "audio", "label": "ne", "name": "Nepali"},
}

MIN_DUR, MAX_DUR = 2.0, 20.0

CONDITIONS = [
    "clean",
    "bandpass",
    "awgn_20",
    "awgn_10",
    "awgn_5",
    "awgn_0",
    "ptt_clip",
    "codec_mp3",
]

# ── Degradation functions ──────────────────────────────────────────────────────

def apply_bandpass(arr, sr, low=300, high=3400, order=6):
    """Telephony bandpass filter (300–3400 Hz), 6th-order Butterworth."""
    nyq = sr / 2
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, arr).astype(np.float32)


def apply_awgn(arr, snr_db):
    """Additive white Gaussian noise at target SNR."""
    rng = np.random.default_rng(seed=42)
    sig_power = np.mean(arr ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), arr.shape).astype(np.float32)
    return np.clip(arr + noise, -1.0, 1.0)


def apply_ptt_clip(arr, sr):
    """PTT-style degradation: 150 ms silence at start, 60 ms click transient."""
    rng = np.random.default_rng(seed=42)
    silence_samples = int(0.15 * sr)
    click_samples   = int(0.06 * sr)
    click = (rng.random(click_samples).astype(np.float32) * 2 - 1) * 0.4
    out = np.concatenate([
        np.zeros(silence_samples, dtype=np.float32),
        click,
        arr[silence_samples + click_samples:],
    ])
    return out[:len(arr)]


def apply_codec_mp3(arr, sr):
    """MP3 narrowband codec via ffmpeg round-trip (16 kbit/s, ~GSM quality)."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.wav")
        mp3 = os.path.join(td, "compressed.mp3")
        out = os.path.join(td, "out.wav")
        sf.write(src, arr, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", src, "-codec:a", "libmp3lame", "-b:a", "16k", mp3],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", mp3, "-ar", str(sr), out],
            check=True,
        )
        result, _ = sf.read(out, dtype="float32")
    if len(result) < len(arr):
        result = np.pad(result, (0, len(arr) - len(result)))
    return result[:len(arr)]


def degrade(arr, sr, condition):
    """Apply the requested condition to a float32 numpy array."""
    if condition == "clean":
        return arr
    if condition == "bandpass":
        return apply_bandpass(arr, sr)
    if condition.startswith("awgn_"):
        snr = int(condition.split("_")[1])
        return apply_awgn(arr, snr)
    if condition == "ptt_clip":
        return apply_ptt_clip(arr, sr)
    if condition == "codec_mp3":
        return apply_codec_mp3(arr, sr)
    raise ValueError(f"Unknown condition: {condition}")


# ── Phase 1: download and cache ────────────────────────────────────────────────

def download_and_cache(languages, n_samples):
    """Download n_samples per language from FLEURS and save as WAV to cache."""
    from datasets import load_dataset

    CACHE_DIR.mkdir(exist_ok=True)
    cached = {}

    for lang in languages:
        info = DATASETS[lang]
        lang_dir = CACHE_DIR / lang
        lang_dir.mkdir(exist_ok=True)

        existing = sorted(lang_dir.glob("*.wav"))
        if len(existing) >= n_samples:
            print(f"  {info['name']}: {len(existing)} cached — skipping download", flush=True)
            cached[lang] = existing[:n_samples]
            continue

        print(f"\n  Downloading {info['name']} from {info['hf_id']} ...", flush=True)
        ds = load_dataset(info["hf_id"], name=info["config"],
                          split=info["split"], streaming=True)
        saved, skipped = 0, 0
        for sample in ds:
            if saved >= n_samples:
                break
            audio_data = sample.get(info["audio_col"])
            if audio_data is None:
                skipped += 1; continue
            try:
                if hasattr(audio_data, "get_all_samples"):
                    s = audio_data.get_all_samples()
                    arr = np.array(s.data[0], dtype=np.float32)
                    sr  = int(s.sample_rate)
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

            wav_path = lang_dir / f"{saved:04d}.wav"
            sf.write(str(wav_path), arr, sr)
            saved += 1
            print(f"    saved {saved}/{n_samples}", end="\r", flush=True)

        print(f"  {info['name']}: saved {saved}, skipped {skipped}", flush=True)
        cached[lang] = sorted(lang_dir.glob("*.wav"))[:n_samples]

    return cached


# ── Phase 2: evaluate one condition ───────────────────────────────────────────

def evaluate_condition(condition, cached_files, asr, ft_det, mms_det, router):
    """Run all 4 ablation configs under `condition`. Returns list of row dicts."""
    from language_module import DialectDetector

    rows = []
    for lang, wav_paths in cached_files.items():
        true_label = DATASETS[lang]["label"]
        for wav_path in wav_paths:
            arr, sr = sf.read(str(wav_path), dtype="float32")
            deg_arr = degrade(arr, sr, condition)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, deg_arr, sr)
                audio_path = tmp.name

            try:
                asr.reset_language_cache()
                asr_result   = asr.transcribe(audio_path, language_hint=None)
                transcript   = asr_result.get("transcript", "")
                whisper_lang = asr_result.get("language", "") or ""
                whisper_prob = asr_result.get("language_probability", 0.0) or 0.0

                # Config 1
                c1 = whisper_lang or "unknown"

                # Config 2
                ft = ft_det.detect(transcript)
                scores = {}
                if whisper_lang:
                    scores[whisper_lang] = scores.get(whisper_lang, 0) + whisper_prob
                if ft["language"] not in ("", "unknown"):
                    scores[ft["language"]] = scores.get(ft["language"], 0) + ft["confidence"]
                c2 = max(scores, key=scores.get) if scores else "unknown"

                # Config 3 (3-way, no override)
                mms = mms_det.detect(audio_path)
                votes = {}
                for (l, c) in [(whisper_lang, whisper_prob),
                               (ft["language"], ft["confidence"]),
                               (mms["language"], mms["confidence"])]:
                    if l and l not in ("", "unknown"):
                        votes[l] = votes.get(l, 0) + c
                c3 = max(votes, key=votes.get) if votes else "unknown"

                # Config 4 (full VANI with script override)
                dial = DialectDetector()
                dd   = dial.detect_code_mix(transcript)
                r4   = router.detect_family(
                    whisper_lang=      whisper_lang,
                    transcript=        transcript,
                    fasttext_lang=     ft["language"],
                    fasttext_conf=     ft["confidence"],
                    whisper_lang_prob= whisper_prob,
                    dialect=           dd["dialect"],
                    mms_lang=          mms["language"],
                    mms_conf=          mms["confidence"],
                )
                c4 = r4["final_language"]

                rows.append({
                    "condition": condition,
                    "lang":      lang,
                    "c1_ok":     int(c1 == true_label),
                    "c2_ok":     int(c2 == true_label),
                    "c3_ok":     int(c3 == true_label),
                    "c4_ok":     int(c4 == true_label),
                })

            except Exception as e:
                print(f"  [ERR] {lang} {wav_path.name}: {e}", flush=True)
            finally:
                os.unlink(audio_path)

    return rows


# ── Phase 3: print summary table ──────────────────────────────────────────────

CFG_NAMES = {
    "c1_ok": "Whisper-only",
    "c2_ok": "+FastText",
    "c3_ok": "+MMS-LID",
    "c4_ok": "Full VANI",
}

def print_table(all_rows, languages):
    lang_names = {l: DATASETS[l]["name"] for l in languages}

    print("\n" + "="*90)
    print("ROBUSTNESS EVALUATION — LangID Accuracy (%) by Condition")
    print("="*90)

    # Header
    hdr = f"{'Condition':<14}" + f"{'Configuration':<22}"
    for l in languages:
        hdr += f"{lang_names[l]:>10}"
    hdr += f"{'Ovrl':>8}"
    print(hdr)
    print("-"*90)

    for cond in CONDITIONS:
        cond_rows = [r for r in all_rows if r["condition"] == cond]
        if not cond_rows:
            continue
        for i, cfg in enumerate(["c1_ok", "c2_ok", "c3_ok", "c4_ok"]):
            row_rows = cond_rows
            cond_label = cond if i == 0 else ""
            line = f"{cond_label:<14}{CFG_NAMES[cfg]:<22}"
            accs = []
            for l in languages:
                lr = [r[cfg] for r in row_rows if r["lang"] == l]
                acc = sum(lr) / len(lr) * 100 if lr else 0
                accs.append(acc)
                line += f"{acc:>9.1f}%"
            overall = sum(accs) / len(accs)
            line += f"{overall:>7.1f}%"
            print(line)
        print()

    print("="*90)


def build_latex_table(all_rows, languages):
    """Build a LaTeX table for the paper."""
    col_langs = " & ".join(f"\\textbf{{{DATASETS[l]['name']}}}" for l in languages)
    tex = (
        "\\begin{table}[htbp]\n"
        "\\caption{LangID Accuracy (\\%) under Radio-Channel Degradations}\n"
        "\\label{tab:robustness}\n"
        "\\begin{center}\n"
        f"\\begin{{tabular}}{{ll{'c'*len(languages)}r}}\n"
        "\\toprule\n"
        f"\\textbf{{Condition}} & \\textbf{{Config.}} & {col_langs} & \\textbf{{Ovrl.}} \\\\\n"
        "\\midrule\n"
    )

    cond_labels = {
        "clean":    "Clean",
        "bandpass": "Bandpass",
        "awgn_20":  "AWGN 20 dB",
        "awgn_10":  "AWGN 10 dB",
        "awgn_5":   "AWGN 5 dB",
        "awgn_0":   "AWGN 0 dB",
        "ptt_clip": "PTT clip",
        "codec_mp3":"MP3 16kbps",
    }

    for cond in CONDITIONS:
        cond_rows = [r for r in all_rows if r["condition"] == cond]
        if not cond_rows:
            continue
        for i, cfg in enumerate(["c1_ok", "c2_ok", "c3_ok", "c4_ok"]):
            cond_label = cond_labels.get(cond, cond) if i == 0 else ""
            cells = []
            accs = []
            for l in languages:
                lr = [r[cfg] for r in cond_rows if r["lang"] == l]
                acc = sum(lr) / len(lr) * 100 if lr else 0
                accs.append(acc)
                cells.append(f"{acc:.1f}")
            ovr = sum(accs) / len(accs)
            cfg_name = {"c1_ok":"Whisper","c2_ok":"+FastText",
                        "c3_ok":"+MMS","c4_ok":"Full VANI"}[cfg]
            if cfg == "c4_ok":
                row_cells = " & ".join(f"\\textbf{{{c}}}" for c in cells)
                tex += (f"{cond_label} & {cfg_name} & {row_cells} & "
                        f"\\textbf{{{ovr:.1f}}} \\\\\n")
            else:
                tex += f"{cond_label} & {cfg_name} & " + " & ".join(cells) + f" & {ovr:.1f} \\\\\n"
        tex += "\\midrule\n"

    tex += (
        "\\multicolumn{6}{l}{\\footnotesize Degradations applied offline to FLEURS test samples before ASR.} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{center}\n"
        "\\end{table}"
    )
    return tex


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("languages", nargs="*", default=["pa","hi","ur","ne"])
    parser.add_argument("--n", type=int, default=30, help="Samples per language")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS,
                        help="Subset of conditions to run")
    args = parser.parse_args()

    bad = [l for l in args.languages if l not in DATASETS]
    if bad:
        sys.exit(f"Unknown language(s): {bad}")

    print(f"\nRobustness eval: {args.languages}, {args.n} samples/lang, "
          f"conditions: {args.conditions}", flush=True)

    # Phase 1: cache audio
    print("\n[Phase 1] Downloading / checking cache ...", flush=True)
    cached = download_and_cache(args.languages, args.n)

    if args.phase1_only:
        print("\nPhase 1 complete. Run without --phase1-only to evaluate.")
        return

    # Load models
    config = load_config()
    paths  = config["paths"]
    device = config.get("device", "cpu")

    print("\n[Phase 2] Loading models ...", flush=True)
    from asr_module import ASRModule
    from language_module import FastTextLangDetector, LanguageRouter
    from mms_module import MMSLangDetector

    asr = ASRModule(
        model_path=str(VANI_ROOT / paths["whisper_model"]),
        device=device, cfg=config.get("asr", {}),
    )
    ft_det  = FastTextLangDetector(model_path=str(VANI_ROOT / paths["fasttext_model"]))
    router  = LanguageRouter(confidence_threshold=0.60)
    mms_det = MMSLangDetector(
        model_path=str(VANI_ROOT / paths.get("mms_lid_model", "models/mms-lid-256")),
        device=device,
    )

    all_rows = []

    for cond in args.conditions:
        print(f"\n[{cond}] evaluating ...", flush=True)
        rows = evaluate_condition(cond, cached, asr, ft_det, mms_det, router)
        all_rows.extend(rows)

        # Quick per-condition summary
        for cfg in ["c4_ok"]:
            accs = {}
            for l in args.languages:
                lr = [r[cfg] for r in rows if r["lang"] == l]
                accs[l] = sum(lr)/len(lr)*100 if lr else 0
            ovr = sum(accs.values()) / len(accs)
            print(f"  Full VANI: " +
                  " | ".join(f"{l}={accs[l]:.0f}%" for l in args.languages) +
                  f" | ovrl={ovr:.1f}%", flush=True)

    mms_det.unload()
    del asr, mms_det, ft_det, router; gc.collect()

    # Phase 3: output
    print_table(all_rows, args.languages)

    csv_path = ROOT / "robustness_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_path}", flush=True)

    tex = build_latex_table(all_rows, args.languages)
    print("\n--- LaTeX table ---\n")
    print(tex)
    (ROOT / "robustness_table.tex").write_text(tex)
    print("\nLatex table saved to robustness_table.tex")


if __name__ == "__main__":
    main()

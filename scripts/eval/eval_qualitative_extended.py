#!/usr/bin/env python3
"""
eval_qualitative_extended.py
============================
Runs the 3-source LangID pipeline on:
  - The 5 original qualitative audio files (en + pa)
  - 2 representative samples per language from each cached FLEURS subset
    (hi_in, pa_in, ur_pk, ne_np, cmn_hans_cn)

Outputs:
  - qual_extended_results.json  — full per-sample results
  - Prints the LaTeX Table I rows for copy-paste into the paper

Run:
    python3 eval_qualitative_extended.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

WHISPER_PATH  = ROOT / "models/whisper-large-v3-turbo-ct2"
FASTTEXT_PATH = ROOT / "models/langid/lid.176.bin"
MMS_PATH      = ROOT / "models/mms-lid-256"

# ── Cached FLEURS subsets to sample from (2 samples each) ───────────────────
FLEURS_SUBSETS = {
    "hi_in":       "hi",
    "pa_in":       "pa",
    "ur_pk":       "ur",
    "ne_np":       "ne",
    "cmn_hans_cn": "zh",
}
# pa and hi have real files already — pull 1 extra from FLEURS for those,
# 2 each for ur/ne/zh (no local files)
N_PER_LANG_DEFAULT = 2
N_PER_LANG_EXTRA   = 1   # for langs that already have real audio files above

# ── Existing qualitative audio files ────────────────────────────────────────
AUDIO_DIR = ROOT / "input_audio"
ORIG_FILES = [
    ("harvard.wav",                       "en"),
    ("LJ001-0004.wav",                    "en"),
    ("Speaker26_000.wav",                 "en"),
    ("sent_1.wav",                        "pa"),
    ("sent_10.wav",                       "pa"),
    ("common_voice_pa-IN_21717904.mp3",   "pa"),
    ("Hindi_0009.wav",                    "hi"),
]

# ── Load models ──────────────────────────────────────────────────────────────
from faster_whisper import WhisperModel
from language_module import FastTextLangDetector, DialectDetector, LanguageRouter, WHISPER_TO_ISO
from mms_module import MMSLangDetector

print("Loading Whisper ...", end=" ", flush=True)
whisper = WhisperModel(str(WHISPER_PATH), device="cpu", compute_type="int8",
                       cpu_threads=min(os.cpu_count() or 4, 8))
print("ok")

print("Loading FastText ...", end=" ", flush=True)
ft = FastTextLangDetector(model_path=str(FASTTEXT_PATH))
print("ok")

print("Loading MMS-LID ...", end=" ", flush=True)
mms = MMSLangDetector(model_path=str(MMS_PATH), device="cpu")
print("ok")

dd     = DialectDetector()
router = LanguageRouter()


def run_sample(audio_path: str, true_lang: str, label: str) -> dict:
    """Run 3-source LangID on one audio file. Returns result dict."""
    segs_iter, info = whisper.transcribe(
        audio_path, beam_size=2, best_of=1, temperature=0.0,
        condition_on_previous_text=False, word_timestamps=False, vad_filter=False,
    )
    transcript = " ".join(s.text.strip() for s in segs_iter if s.text.strip())
    wl  = WHISPER_TO_ISO.get(info.language.lower(), info.language.lower())
    wlp = float(info.language_probability)

    ft_res = ft.detect(transcript) if transcript.strip() else {"language": "unk", "confidence": 0.0}
    fl, fc = ft_res["language"], ft_res["confidence"]

    try:
        mms_res = mms.detect(audio_path)
        ml, mc  = mms_res["language"], float(mms_res["confidence"])
    except Exception:
        ml, mc = "unk", 0.0

    dd_res  = dd.detect_code_mix(transcript) if transcript.strip() else {}
    dialect = dd_res.get("dialect", "unknown")

    r4 = router.detect_family(
        whisper_lang=wl, transcript=transcript,
        fasttext_lang=fl, fasttext_conf=fc,
        whisper_lang_prob=wlp,
        dialect=dialect,
        mms_lang=ml, mms_conf=mc,
    )
    final = r4["final_language"]

    return {
        "label":       label,
        "true_lang":   true_lang,
        "whisper":     wl,
        "whisper_prob": round(wlp, 3),
        "ft":          fl,
        "ft_conf":     round(fc, 3),
        "mms":         ml,
        "mms_conf":    round(mc, 3),
        "final":       final,
        "correct":     (final == true_lang),
    }


def audio_to_wav(sample: dict, tmp_path: str) -> None:
    import soundfile as sf
    audio_data = sample.get("audio", sample)
    if isinstance(audio_data, dict) and "array" in audio_data:
        arr = np.asarray(audio_data["array"], dtype=np.float32)
        sr  = int(audio_data.get("sampling_rate", 16000))
    else:
        s   = audio_data.get_all_samples()
        arr = s.data.numpy()[0].astype(np.float32)
        sr  = 16000
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    sf.write(tmp_path, arr, 16000)


# ── Run original files ────────────────────────────────────────────────────────
results = []
print("\n─── Original qualitative files ─────────────────────────────────────────")
for fname, true_lang in ORIG_FILES:
    fpath = AUDIO_DIR / fname
    if not fpath.exists():
        print(f"  [skip] {fname} not found at {fpath}")
        continue
    print(f"  {fname} ({true_lang}) ...", end=" ", flush=True)
    r = run_sample(str(fpath), true_lang, fname.replace(".wav", ""))
    results.append(r)
    mark = "✓" if r["correct"] else "✗"
    print(f"{mark}  Whisper={r['whisper']} {r['whisper_prob']:.2f}  "
          f"FT={r['ft']} {r['ft_conf']:.2f}  MMS={r['mms']} {r['mms_conf']:.2f}  "
          f"→ {r['final']}")

# ── Run FLEURS samples ────────────────────────────────────────────────────────
from datasets import load_dataset

print("\n─── FLEURS cached samples ───────────────────────────────────────────────")
# langs with real audio already: use fewer FLEURS extras
langs_with_real_files = {"pa", "hi", "en"}

lang_order = ["hi_in", "pa_in", "ur_pk", "ne_np", "cmn_hans_cn"]
for subset in lang_order:
    true_lang = FLEURS_SUBSETS[subset]
    n_want = N_PER_LANG_EXTRA if true_lang in langs_with_real_files else N_PER_LANG_DEFAULT
    print(f"\n  {subset} (ground truth: {true_lang}, pulling {n_want} FLEURS samples)")
    try:
        ds = load_dataset("google/fleurs", subset, split="test")
    except Exception as e:
        print(f"    Cannot load: {e}")
        continue

    count = 0
    for idx in range(min(len(ds), 20)):   # scan up to 20 to find n_want usable ones
        if count >= n_want:
            break
        sample = ds[idx]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            audio_to_wav(sample, tmp_path)
            label = f"fleurs_{subset}_{idx+1:03d}"
            r = run_sample(tmp_path, true_lang, label)
            results.append(r)
            mark = "✓" if r["correct"] else "✗"
            print(f"    [{idx+1}] {mark}  "
                  f"Whisper={r['whisper']} {r['whisper_prob']:.2f}  "
                  f"FT={r['ft']} {r['ft_conf']:.2f}  "
                  f"MMS={r['mms']} {r['mms_conf']:.2f}  → {r['final']}")
            count += 1
        except Exception as e:
            print(f"    [{idx+1}] error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_path = ROOT / "output" / "qual_extended_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")

# ── Print LaTeX Table I rows ──────────────────────────────────────────────────
print("\n" + "="*70)
print("LaTeX Table I rows (Three-Source LangID Results)")
print("="*70)

LANG_FULL = {
    "en": "en", "pa": "pa", "hi": "hi", "ur": "ur", "ne": "ne", "zh": "zh",
}

def fmt_cell(lang, prob, true_lang):
    check = r"$\checkmark$" if lang == true_lang else r"$\times$"
    return f"{lang}~{prob:.3f}{check}"

def fmt_final(lang, true_lang):
    check = r"$\checkmark$" if lang == true_lang else r"$\times$"
    bold  = lang == true_lang
    inner = f"\\textbf{{{lang}}}{check}" if bold else f"{lang}{check}"
    return inner

print(r"\midrule")
prev_true = None
for r in results:
    if r["true_lang"] != prev_true and prev_true is not None:
        print(r"\midrule")
    prev_true = r["true_lang"]

    wok   = r["whisper"] == r["true_lang"]
    wcell = f"{r['whisper']}~{r['whisper_prob']:.3f}" + (r"$\checkmark$" if wok else r"$\times$")

    fok   = r["ft"] == r["true_lang"]
    fcell = f"{r['ft']}~{r['ft_conf']:.3f}" + (r"$\checkmark$" if fok else r"$\times$")

    mok   = r["mms"] == r["true_lang"]
    mcell = f"{r['mms']}~{r['mms_conf']:.3f}" + (r"$\checkmark$" if mok else r"$\times$")

    final_cell = fmt_final(r["final"], r["true_lang"])

    label = r["label"][:16]
    print(f"{label:<18} & {r['true_lang']} & {wcell} & {fcell} & {mcell} & {final_cell} \\\\")

print(r"\bottomrule")

# ── Accuracy summary ──────────────────────────────────────────────────────────
total   = len(results)
correct = sum(1 for r in results if r["correct"])
print(f"\nOverall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
by_lang = {}
for r in results:
    l = r["true_lang"]
    by_lang.setdefault(l, [0, 0])
    by_lang[l][1] += 1
    if r["correct"]:
        by_lang[l][0] += 1
for l, (c, n) in sorted(by_lang.items()):
    print(f"  {l}: {c}/{n} ({c/n*100:.0f}%)")

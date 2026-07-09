"""
Build demo_clips/ — N representative clips per language for the VANI demo.

Sources real 16 kHz mono clips from robustness_cache/<lang>/, picking the N clips
whose durations are closest to a target (~10 s, min 3 s), and copies them with
clear numbered language-named filenames plus a manifest.json.

Dogri (doi) has no audio anywhere in the repo and no fine-tuned ASR model, so it
is intentionally omitted (listed as 'missing' in the manifest).

Run:  venv\Scripts\python.exe scripts\data\build_demo_clips.py
"""
import json
import shutil
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "robustness_cache"
DST  = ROOT / "demo_clips"
TARGET_SEC   = 10.0
MIN_SEC      = 3.0
N_PER_LANG   = 5

# VANI language code -> (display name, Sanket LID class for the demo)
LANGS = {
    "pa": ("Punjabi",  "punjabi"),
    "hi": ("Hindi",    None),        # not in Sanket's 8-class set
    "ur": ("Urdu",     "urdu"),
    "ne": ("Nepali",   None),        # not in Sanket's 8-class set
    "zh": ("Mandarin", "mandarin"),
    "ps": ("Pashto",   "pashto"),
    "ks": ("Kashmiri", "kashmiri"),
    "doi": ("Dogri",   "dogri"),     # no audio available
}


def pick_n_closest(lang_dir: Path, target: float, n: int, min_sec: float):
    """Return up to n (path, duration), the clips with duration closest to target
    (and >= min_sec), sorted by ascending duration for stable naming."""
    cands = []
    for w in sorted(lang_dir.glob("*.wav")):
        try:
            d = sf.info(str(w)).duration
        except Exception:
            continue
        if d >= min_sec:
            cands.append((w, d))
    cands.sort(key=lambda pd: abs(pd[1] - target))   # closest to target first
    chosen = cands[:n]
    chosen.sort(key=lambda pd: pd[1])                 # then by duration for naming
    return chosen


def main():
    DST.mkdir(parents=True, exist_ok=True)
    # Clear stale wavs so re-runs don't leave old single-clip files behind.
    for old in DST.glob("*.wav"):
        old.unlink()

    manifest = []
    total_ok = 0
    for code, (name, sanket) in LANGS.items():
        lang_dir = SRC / code
        picks = (pick_n_closest(lang_dir, TARGET_SEC, N_PER_LANG, MIN_SEC)
                 if lang_dir.exists() else [])
        if not picks:
            print(f"  [MISSING] {name} ({code}) — no source audio, skipped")
            manifest.append({"lang": code, "name": name, "clips": [],
                             "status": "missing", "sanket_class": sanket})
            continue
        entry = {"lang": code, "name": name, "status": "ok",
                 "sanket_class": sanket, "clips": []}
        for i, (src, dur) in enumerate(picks, 1):
            out_name = f"{code}_{name.lower()}_{i}.wav"
            shutil.copyfile(src, DST / out_name)
            entry["clips"].append({"file": out_name, "duration_sec": round(dur, 1),
                                   "source": str(src.relative_to(ROOT))})
        total_ok += 1
        _durs = ", ".join(f"{c['duration_sec']}s" for c in entry["clips"])
        print(f"  [OK] {name:9s} ({code}) -> {len(picks)} clips  ({_durs})")
        manifest.append(entry)

    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_clips = sum(len(m.get("clips", [])) for m in manifest)
    print(f"\n  {total_ok}/{len(LANGS)} languages · {n_clips} clips -> {DST}")
    print(f"  manifest: {DST / 'manifest.json'}")


if __name__ == "__main__":
    main()

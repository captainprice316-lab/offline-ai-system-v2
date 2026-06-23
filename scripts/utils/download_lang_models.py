"""
download_lang_models.py – Download and convert language-specific Whisper ASR models
=====================================================================================
Downloads fine-tuned Whisper models from HuggingFace and converts them to
CTranslate2 (CT2) format for use with faster-whisper in VANI.

Models selected:
  zh — BELLE-2/Belle-whisper-large-v3-turbo-zh
       3.07% CER on AISHELL-1 (vs ~26% for base turbo)
       Trained on AISHELL-1/2 + WenetSpeech + HKUST (~telephone/meeting speech)

  ps — Nasimbahar/pashto-ghag-whisper-medium-asr
       14.63% WER on Common Voice 17 Pashto (54h training data)
       Only Pashto model with sub-20% WER on natural speech

Usage:
    python download_lang_models.py          # download both zh and ps
    python download_lang_models.py zh       # Mandarin only
    python download_lang_models.py ps       # Pashto only
    python download_lang_models.py --no-cleanup  # keep raw HF files after conversion
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
HF_CACHE   = MODELS_DIR / "_hf_cache"   # temp dir — deleted after conversion by default
MODELS_DIR.mkdir(exist_ok=True)

LANG_MODELS = {
    "zh": {
        "hf_repo":    "BELLE-2/Belle-whisper-large-v3-turbo-zh",
        "ct2_name":   "belle-whisper-large-v3-turbo-zh-ct2",
        "base_arch":  "whisper-large-v3-turbo",
        "dl_size":    "~3.0 GB",
        "ct2_size":   "~0.9 GB after int8",
        "purpose":    "Mandarin Chinese — 3.07% CER vs ~26% base turbo",
        "config_key": "whisper_model_zh",
    },
    "ps": {
        "hf_repo":    "Nasimbahar/pashto-ghag-whisper-medium-asr",
        "ct2_name":   "whisper-medium-pashto-ct2",
        "base_arch":  "whisper-medium",
        "dl_size":    "~1.5 GB",
        "ct2_size":   "~0.5 GB after int8",
        "purpose":    "Pashto — 14.63% WER on CV17 (54h training data)",
        "config_key": "whisper_model_ps",
    },
    "pa": {
        "hf_repo":    "openai/whisper-large-v3",
        "ct2_name":   "whisper-large-v3-pa-ct2",
        "base_arch":  "whisper-large-v3",
        "dl_size":    "~3.1 GB",
        "ct2_size":   "~1.5 GB after int8",
        "purpose":    "Punjabi (Gurmukhi) — full large-v3 (better multilingual than turbo, dedicated for pa)",
        "config_key": "whisper_model_pa",
    },
}

# Ignore non-PyTorch weights to save bandwidth
_IGNORE = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"]


def _ct2_converter() -> str:
    """Prefer venv's converter binary; fall back to PATH. Handles Windows (Scripts/) and Unix (bin/)."""
    for subdir in ("Scripts", "bin"):
        for name in ("ct2-transformers-converter.exe", "ct2-transformers-converter"):
            candidate = ROOT / "venv" / subdir / name
            if candidate.exists():
                return str(candidate)
    return "ct2-transformers-converter"


def _hf_file_list(repo_id: str, ignore: list) -> list:
    """Return list of (filename, size) for all files in an HF repo."""
    from huggingface_hub import list_repo_files, hf_hub_url
    import fnmatch
    files = []
    for f in list_repo_files(repo_id):
        if any(fnmatch.fnmatch(f, pat) for pat in ignore):
            continue
        files.append(f)
    return files


def _download_file(url: str, dest: Path, chunk_mb: int = 8, max_retries: int = 10) -> bool:
    """Stream-download url to dest with resume, retries, and progress bar."""
    import urllib.request, time
    existing = dest.stat().st_size if dest.exists() else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    for attempt in range(1, max_retries + 1):
        try:
            req  = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=60)
            total = existing + int(resp.headers.get("Content-Length", 0) or
                                   resp.headers.get("content-length", 0))
            mode = "ab" if existing else "wb"
            done = existing
            chunk = chunk_mb * 1024 * 1024
            with open(dest, mode) as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total:
                        pct  = done * 100 // total
                        bar  = ("#" * (pct // 5)).ljust(20)
                        mb_done  = done  / 1048576
                        mb_total = total / 1048576
                        print(f"\r    [{bar}] {pct:3d}%  {mb_done:7.1f}/{mb_total:.1f} MB", end="", flush=True)
            print()
            return True
        except Exception as e:
            print(f"\n    [retry {attempt}/{max_retries}] {e}")
            time.sleep(min(5 * attempt, 30))
            existing = dest.stat().st_size if dest.exists() else 0
            headers  = {"Range": f"bytes={existing}-"} if existing else {}
    return False


def download_and_convert(lang: str, info: dict, cleanup: bool = True) -> bool:
    ct2_dir = MODELS_DIR / info["ct2_name"]
    hf_dir  = HF_CACHE / lang

    print(f"\n{'='*60}")
    print(f"  Lang     : {lang.upper()}  ({info['base_arch']})")
    print(f"  Repo     : {info['hf_repo']}")
    print(f"  Purpose  : {info['purpose']}")
    print(f"  Download : {info['dl_size']}  ->  CT2: {info['ct2_size']}")
    print(f"{'='*60}")

    if ct2_dir.exists() and any(ct2_dir.iterdir()):
        print(f"  [SKIP] Already exists: {ct2_dir}")
        print(f"         Delete to re-download:  rmdir /s /q \"{ct2_dir}\"")
        return True

    # ── Step 1: Download raw HF model ─────────────────────────────────────────
    print(f"\n  [1/2] Downloading from HuggingFace -> {hf_dir}")
    HF_CACHE.mkdir(exist_ok=True)
    hf_dir.mkdir(exist_ok=True)

    try:
        files = _hf_file_list(info["hf_repo"], _IGNORE)
    except Exception as e:
        print(f"  [ERR] Could not list repo files: {e}")
        return False

    base_url = f"https://huggingface.co/{info['hf_repo']}/resolve/main"
    all_ok = True
    for i, fname in enumerate(files, 1):
        dest = hf_dir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        size_mb = dest.stat().st_size / 1048576 if dest.exists() else 0
        print(f"\n  [{i}/{len(files)}] {fname}")
        if dest.exists() and dest.stat().st_size > 0:
            # Skip small files already present; re-verify large ones by size
            if dest.stat().st_size < 10 * 1024 * 1024:
                print(f"    [done] {size_mb:.1f} MB")
                continue
        url = f"{base_url}/{fname}"
        ok = _download_file(url, dest)
        if not ok:
            print(f"    [ERR] Failed after retries: {fname}")
            all_ok = False

    if not all_ok:
        print("\n  [ERR] Some files failed to download.")
        return False
    print(f"\n  [OK]  Downloaded to {hf_dir}")

    # ── Step 2: Convert to CT2 int8 ───────────────────────────────────────────
    print(f"\n  [2/2] Converting to CT2 (int8) -> {ct2_dir}")
    cmd = [
        _ct2_converter(),
        "--model",        str(hf_dir),
        "--output_dir",   str(ct2_dir),
        "--quantization", "int8",
        "--force",
    ]
    print(f"  Running: {' '.join(str(c) for c in cmd)}\n")
    ret = subprocess.run(cmd)

    if ret.returncode != 0:
        print(f"\n  [ERR] Conversion failed (exit {ret.returncode})")
        if ct2_dir.exists():
            shutil.rmtree(ct2_dir)
        return False

    print(f"\n  [OK]  CT2 model saved to {ct2_dir}")

    # ── Step 3: Remove raw HF download ────────────────────────────────────────
    if cleanup and hf_dir.exists():
        shutil.rmtree(hf_dir)
        print(f"  [OK]  Removed raw HF download (--no-cleanup to keep)")

    return True


def main():
    raw_args = sys.argv[1:]
    flags    = [a for a in raw_args if a.startswith("--")]
    targets  = [a for a in raw_args if not a.startswith("--")] or list(LANG_MODELS)
    cleanup  = "--no-cleanup" not in flags

    bad = [t for t in targets if t not in LANG_MODELS]
    if bad:
        print(f"Unknown language code(s): {bad}")
        print(f"Available: {list(LANG_MODELS)}")
        sys.exit(1)

    print("\nVANI Language-Specific ASR Model Downloader")
    print(f"Models dir : {MODELS_DIR}")
    print(f"Cleanup    : {'yes' if cleanup else 'no (--no-cleanup)'}")

    results = {lang: download_and_convert(lang, LANG_MODELS[lang], cleanup) for lang in targets}

    # Tidy up empty cache dir
    if HF_CACHE.exists() and not any(HF_CACHE.iterdir()):
        HF_CACHE.rmdir()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for lang, ok in results.items():
        tag = "[OK]    " if ok else "[FAILED]"
        print(f"  {tag} {lang.upper()} -> {LANG_MODELS[lang]['ct2_name']}")

    ok_langs = [lang for lang, ok in results.items() if ok]
    if ok_langs:
        print("\nAdd to config.yaml under paths:")
        for lang in ok_langs:
            info = LANG_MODELS[lang]
            print(f"  {info['config_key']}: models/{info['ct2_name']}")
        print("\nVANI auto-selects these models when MMS-LID detects the")
        print("language with >=65% confidence before ASR starts.")


if __name__ == "__main__":
    main()

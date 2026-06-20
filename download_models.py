"""
download_models.py – Download additional VANI models
=====================================================
Run ONCE with internet access before going fully offline.
Models are saved under models/ and referenced in config.yaml.

Usage:
    python download_models.py                  # all models
    python download_models.py mms              # MMS-LID only
    python download_models.py qwen             # Qwen2.5 only
    python download_models.py whisper-turbo    # Whisper Large-v3-turbo only
"""

import sys
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CATALOG = {
    "mms": {
        "repo_id":    "facebook/mms-lid-256",
        "local_name": "mms-lid-256",
        "size":       "~150 MB",
        "purpose":    "Audio-based language identification (3rd vote, 256 languages)",
    },
    "qwen": {
        "repo_id":    "Qwen/Qwen2.5-1.5B-Instruct",
        "local_name": "qwen2.5-1.5b-instruct",
        "size":       "~3.1 GB",
        "purpose":    "LLM-based ISUM generation — 1.5B gives significantly better quality than 0.5B",
    },
    "whisper-turbo": {
        "repo_id":    "openai/whisper-large-v3-turbo",
        "local_name": "whisper-large-v3-turbo",
        "size":       "~1.6 GB",
        "purpose":    "Better ASR than Whisper Medium, 3x faster than Large-v3",
    },
}


def download(key: str, info: dict):
    dest = MODELS_DIR / info["local_name"]
    if dest.exists() and any(dest.iterdir()):
        print(f"  [SKIP] {key} already exists at {dest}")
        return
    print(f"\n  Downloading {key}  ({info['size']})  ->  {dest}")
    print(f"  Purpose: {info['purpose']}")
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  [OK] {key} saved to {dest}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(CATALOG.keys())
    invalid = [t for t in targets if t not in CATALOG]
    if invalid:
        print(f"Unknown model(s): {invalid}")
        print(f"Available: {list(CATALOG.keys())}")
        sys.exit(1)

    print("VANI Model Downloader")
    print("=" * 50)
    for key in targets:
        download(key, CATALOG[key])

    print("\n" + "=" * 50)
    print("Done. Update config.yaml if using whisper-turbo:")
    print("  paths:")
    print("    whisper_model: models/whisper-large-v3-turbo")
    print("\nMMS and Qwen will be auto-detected from models/ directory.")


if __name__ == "__main__":
    main()

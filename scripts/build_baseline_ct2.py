"""
build_baseline_ct2.py – convert the TRUE openai/whisper-large-v3 to CT2 int8.

Why this exists
---------------
scripts/eval/compare_all_models.py claims to evaluate a "Whisper large-v3 baseline"
but actually loads models/whisper-large-v3-turbo-ct2. Turbo is a different model, and
for Mandarin it defaults to the *translate* task -- which is the real reason the
published zh baseline is 100.03% WER. The comparison "baseline 100% -> fine-tuned 16%"
therefore measures a transcriber against a translator.

This builds a real large-v3 baseline from the copy already in the HF cache, so no
internet is needed.

The two CT2 landmines, both handled here
----------------------------------------
1. ct2-transformers-converter does NOT copy tokenizer.json. faster-whisper then falls
   back to whisper-tiny's tokenizer, where <|transcribe|> = 50359. large-v3 expanded
   its vocab, so in large-v3 50359 is <|translate|>. The one-token shift makes the
   model silently translate to English (~100% WER against source-language refs).
2. It also omits preprocessor_config.json, without which faster-whisper crashes on a
   mel-bin shape mismatch (large-v3 needs feature_size=128).

Both files are copied from the source snapshot, and the transcribe/translate token IDs
are asserted afterwards rather than assumed.

Usage:
    python scripts/build_baseline_ct2.py
"""

import io, sys, json, shutil, subprocess
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
HUB  = Path(r"E:\VANI\datasets\hf_cache\hub\models--openai--whisper-large-v3\snapshots")
OUT  = ROOT / "models" / "whisper-large-v3-ct2"
CONVERTER = ROOT / "venv" / "Scripts" / "ct2-transformers-converter.exe"


def find_snapshot() -> Path:
    snaps = [p for p in HUB.iterdir() if p.is_dir()] if HUB.exists() else []
    if not snaps:
        sys.exit(f"No large-v3 snapshot under {HUB}")
    snap = max(snaps, key=lambda p: p.stat().st_mtime)
    for required in ("model.safetensors", "tokenizer.json", "preprocessor_config.json"):
        if not (snap / required).exists():
            sys.exit(f"snapshot {snap} is missing {required}")
    return snap


def main():
    if OUT.exists() and (OUT / "model.bin").exists():
        print(f"{OUT} already built. Delete it to rebuild.")
    else:
        snap = find_snapshot()
        print(f"source snapshot : {snap}")
        print(f"target          : {OUT}")
        if not CONVERTER.exists():
            sys.exit(f"missing converter: {CONVERTER}")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(CONVERTER), "--model", str(snap), "--output_dir", str(OUT),
               "--copy_files", "tokenizer.json", "preprocessor_config.json",
               "--quantization", "int8"]
        print("\n$ " + " ".join(cmd) + "\n")
        subprocess.run(cmd, check=True)

    # --copy_files can silently no-op depending on version; do it ourselves and verify.
    snap = find_snapshot()
    for f in ("tokenizer.json", "preprocessor_config.json"):
        dst = OUT / f
        if not dst.exists():
            shutil.copy2(snap / f, dst)
            print(f"[fix] copied {f} (converter omitted it)")
        else:
            print(f"[ok]  {f} present")

    # Landmine 2: mel bins.
    pp = json.loads((OUT / "preprocessor_config.json").read_text(encoding="utf-8"))
    fs = pp.get("feature_size")
    print(f"[check] preprocessor feature_size = {fs}  (large-v3 requires 128)")
    assert fs == 128, f"feature_size={fs}, expected 128 -- wrong preprocessor_config"

    # Landmine 1: the task tokens must be large-v3's, not whisper-tiny's.
    tok = json.loads((OUT / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    tr, tl = vocab.get("<|transcribe|>"), vocab.get("<|translate|>")
    print(f"[check] <|transcribe|> = {tr}   <|translate|> = {tl}")
    assert tr is not None and tl is not None, "task tokens missing from tokenizer.json"
    assert tr != tl, "transcribe and translate share an id -- tokenizer is wrong"
    if tr == 50359:
        sys.exit("FATAL: <|transcribe|>=50359 is the whisper-tiny layout. This tokenizer "
                 "is not large-v3's -- the model would silently translate.")

    print(f"\nBuilt {OUT}")
    print("Now point compare_all_models.py's baseline at whisper-large-v3-ct2.")


if __name__ == "__main__":
    main()

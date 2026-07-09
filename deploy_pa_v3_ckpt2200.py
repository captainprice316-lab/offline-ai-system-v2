"""
One-off deploy: merge PA v3 checkpoint-2200 (LoRA r=16, 50.62% WER) into
whisper-large-v3 and convert to CTranslate2 int8.

Safe deploy strategy:
  1. Merge adapter -> merged HF model (CPU, fp16; does not touch GPU/training).
  2. Convert to a NEW temp CT2 dir (whisper-large-v3-pa-ct2-v3new).
  3. Verify model.bin + tokenizer.json.
  4. Back up the live model, then atomically swap the new one into place.

PA uses no custom tokens, so the processor/tokenizer come straight from the
cached base model (openai/whisper-large-v3) — same tokenizer.json the working
v2 model uses (the anti-translate fix).
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

ROOT      = Path(r"C:\Users\vis15\offline_ai_system_v2")
BASE      = "openai/whisper-large-v3"
ADAPTER   = Path(r"D:\finetune_runs\pa\adapter\checkpoint-2200")
MERGED    = ROOT / "finetune_runs" / "pa" / "merged_ckpt2200"
CONVERTER = ROOT / "venv" / "Scripts" / "ct2-transformers-converter.exe"

LIVE      = ROOT / "models" / "whisper-large-v3-pa-ct2"
NEW_CT2   = ROOT / "models" / "whisper-large-v3-pa-ct2-v3new"
BACKUP    = ROOT / "models" / "whisper-large-v3-pa-ct2-v2backup"


def log(msg): print(f"[deploy] {msg}", flush=True)


def main():
    if not ADAPTER.exists():
        log(f"ERROR: adapter not found: {ADAPTER}"); sys.exit(1)

    # ── 1. Merge (CPU) ──────────────────────────────────────────────
    log("[1/4] Loading base whisper-large-v3 (fp16, CPU) ...")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16)
    log("      Applying LoRA adapter from checkpoint-2200 ...")
    merged = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED))
    WhisperProcessor.from_pretrained(BASE).save_pretrained(str(MERGED))
    log(f"      Merged model saved -> {MERGED}")

    # ── 2. Convert to NEW temp CT2 dir ──────────────────────────────
    log(f"[2/4] Converting to CT2 int8 -> {NEW_CT2}")
    if NEW_CT2.exists():
        shutil.rmtree(NEW_CT2)
    cmd = [str(CONVERTER), "--model", str(MERGED),
           "--output_dir", str(NEW_CT2), "--quantization", "int8", "--force"]
    log("      " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        log("ERROR: CT2 conversion failed. Live model untouched."); sys.exit(1)

    # copy tokenizer.json (anti-translate fix) + preprocessor_config.json
    for fname in ("tokenizer.json", "preprocessor_config.json"):
        src = MERGED / fname
        if src.exists():
            shutil.copy2(str(src), str(NEW_CT2 / fname))
            log(f"      copied {fname}")
    # vocabulary.json is emitted by the converter; if missing, derive from tokenizer
    if not (NEW_CT2 / "vocabulary.json").exists() and (LIVE / "vocabulary.json").exists():
        shutil.copy2(str(LIVE / "vocabulary.json"), str(NEW_CT2 / "vocabulary.json"))
        log("      copied vocabulary.json from live model")

    # ── 3. Verify ───────────────────────────────────────────────────
    log("[3/4] Verifying new model ...")
    mb = NEW_CT2 / "model.bin"
    tok = NEW_CT2 / "tokenizer.json"
    if not mb.exists() or mb.stat().st_size < 500 * 1048576:
        log("ERROR: model.bin missing/too small. Live model untouched."); sys.exit(1)
    if not tok.exists():
        log("ERROR: tokenizer.json missing (would translate not transcribe). Aborting."); sys.exit(1)
    log(f"      model.bin: {mb.stat().st_size/1048576:.0f} MB   tokenizer.json: OK")

    # ── 4. Back up live model, then swap ────────────────────────────
    log("[4/4] Backing up live model and swapping ...")
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    if LIVE.exists():
        shutil.move(str(LIVE), str(BACKUP))
        log(f"      live v2 model backed up -> {BACKUP}")
    shutil.move(str(NEW_CT2), str(LIVE))
    log(f"      new v3 model deployed -> {LIVE}")

    log("DONE. checkpoint-2200 (WER 50.62%) is now the live PA model.")
    log(f"Rollback if needed: delete {LIVE}, rename {BACKUP} -> {LIVE.name}")


if __name__ == "__main__":
    main()

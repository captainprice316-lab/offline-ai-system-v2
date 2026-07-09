"""Smoke-test of the KS training pipeline.
Tests: prefix setup, 1 sample preparation, and a tiny 50-sample data load.
Does NOT do a full model forward pass or full training data load."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
sys.path.insert(0, r"C:\Users\vis15\offline_ai_system_v2")

from finetune_whisper import (
    _resolve_model, _setup_ks_prefix, _decode_audio, LANG_CONFIG, RUNS_DIR
)
from transformers import WhisperProcessor
import pyarrow.parquet as pq
from pathlib import Path
import soundfile as sf
import numpy as np
import io as io_

cfg = LANG_CONFIG["ks"]
hf_model_path = _resolve_model(cfg)
print(f"Model path : {hf_model_path}")

# ── 1. Processor + prefix setup ───────────────────────────────────────────────
print("\n[1] Loading processor and setting up KS prefix ...")
processor = WhisperProcessor.from_pretrained(hf_model_path)
forced_decoder_ids = _setup_ks_prefix(processor)
print(f"  forced_decoder_ids: {forced_decoder_ids}")

bos_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
ks_id  = processor.tokenizer.convert_tokens_to_ids("<|ks|>")
tr_id  = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
nts_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
expected_prefix = [bos_id, ks_id, tr_id, nts_id]

test_ids = processor.tokenizer("سلام", return_tensors="pt").input_ids[0][:4].tolist()
print(f"  Expected prefix: {expected_prefix}")
print(f"  Got prefix     : {test_ids}")
assert test_ids == expected_prefix, f"Prefix mismatch: {test_ids} != {expected_prefix}"
print("  [OK] Prefix correct\n")

# ── 2. Load one sample from parquet and prepare ───────────────────────────────
print("[2] Loading one parquet sample and running prepare() ...")
parquet_dir = Path(cfg["indicvoices_parquet_dir"])
test_file = next(parquet_dir.glob("test-*.parquet"))
t = pq.read_table(test_file, columns=["audio", "normalized", "duration"])
raw = t.to_pydict()

# Find first valid sample
for i, (audio, text, dur) in enumerate(zip(raw["audio"], raw["normalized"], raw["duration"])):
    if dur and 2.0 <= dur <= 20.0 and text:
        break

print(f"  Sample {i}: dur={dur:.1f}s  text='{text[:60]}'")
print(f"  Audio type: {type(audio).__name__}, keys: {list(audio.keys())}")

audio_data = _decode_audio(audio, target_sr=16000)
print(f"  Decoded: shape={audio_data.shape}, dtype={audio_data.dtype}, max={audio_data.max():.3f}")

feats = processor.feature_extractor(audio_data, sampling_rate=16000)
print(f"  input_features shape: {feats.input_features[0].shape}")

label_ids = processor.tokenizer(text, max_length=448, truncation=True).input_ids
print(f"  label_ids[:6]: {label_ids[:6]}")
prefix_ok = label_ids[:4] == expected_prefix
print(f"  KS prefix in labels: {prefix_ok}")
assert prefix_ok, f"KS prefix missing from labels! Got: {label_ids[:4]}"
print("  [OK] Prepare sample works\n")

# ── 3. Quick 50-sample generator test ────────────────────────────────────────
print("[3] Generator test (50 train samples) ...")
from datasets import Dataset, Features, Value
audio_features = Features({
    "audio": {"bytes": Value("binary"), "path": Value("string")},
    "text":  Value("string"),
})

def _mini_gen(pq_files, max_samples=50):
    count = 0
    for f in pq_files:
        if count >= max_samples:
            break
        t_ = pq.read_table(f, columns=["audio", "normalized", "duration"])
        d_ = t_.to_pydict()
        del t_
        for a, tx, du in zip(d_["audio"], d_["normalized"], d_["duration"]):
            if count >= max_samples:
                break
            if du and 2.0 <= du <= 20.0 and tx:
                yield {"audio": a, "text": tx}
                count += 1
        del d_

train_files = sorted(parquet_dir.glob("train-*.parquet"))
ds = Dataset.from_generator(
    _mini_gen,
    gen_kwargs={"pq_files": train_files},
    features=audio_features,
)
print(f"  Dataset size: {len(ds)}")
print(f"  First text: '{ds[0]['text'][:60]}'")
print(f"  First audio keys: {list(ds[0]['audio'].keys())}")
print("  [OK] Generator dataset works\n")

print("[OK] All smoke tests passed.")
print("     Ready to run: python finetune_whisper.py ks --steps 3000")

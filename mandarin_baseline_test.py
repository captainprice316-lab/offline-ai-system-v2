"""
Empirically test the Mandarin baseline behaviour: does whisper-large-v3 translate
Mandarin to English (→ ~100% WER vs Chinese refs) or transcribe it?

Runs 3 Mandarin clips from FLEURS cmn test set through openai/whisper-large-v3 in:
  - DEFAULT settings (no task forced)
  - task="transcribe"  (what the fine-tuned model does)
  - task="translate"   (the behaviour that produces the ~100% baseline)
and prints each output next to the Chinese reference, with a rough char-error check.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

PARQUET = r"E:\hf_cache\hub\datasets--google--fleurs\snapshots\70bb2e84b976b7e960aa89f1c648e09c59f894dd\parquet-data\cmn_hans_cn\test-00000-of-00001.parquet"
MODEL   = "openai/whisper-large-v3"
OUT     = Path("logs/mandarin_baseline_test.txt")
N_CLIPS = 3

def han_ratio(s):
    if not s: return 0.0
    return sum('一' <= c <= '鿿' for c in s) / max(1, len(s))

# Pick device safely: only use CUDA if plenty of VRAM is free (training may still hold it)
DEVICE, DTYPE = "cpu", torch.float32
if torch.cuda.is_available():
    free_b, _ = torch.cuda.mem_get_info()
    if free_b > 4 * 1024**3:          # need >4 GB free to load large-v3 safely
        DEVICE, DTYPE = "cuda", torch.float16
print(f"Device: {DEVICE} ({DTYPE})", flush=True)

print("Loading base model openai/whisper-large-v3 (offline) ...", flush=True)
proc  = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE)
model.eval()
print("Loaded.\n", flush=True)

# read a few rows (audio bytes + reference transcription) from the parquet
t = pq.read_table(PARQUET)
cols = t.column_names
# FLEURS columns: 'audio' (dict w/ bytes), 'transcription' or 'raw_transcription'
audio_col = "audio"
text_col  = "transcription" if "transcription" in cols else "raw_transcription"

rows = t.to_pylist()[:N_CLIPS]
lines = [f"Model: {MODEL}\nColumns: {cols}\n"]

for i, row in enumerate(rows):
    ref = row.get(text_col) or row.get("raw_transcription") or ""
    audio_bytes = row[audio_col]["bytes"]
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        # simple resample via numpy interpolation
        n = int(len(wav) * 16000 / sr)
        wav = np.interp(np.linspace(0, len(wav), n, endpoint=False), np.arange(len(wav)), wav)
    feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, DTYPE)

    outs = {}
    for label, kwargs in [
        ("DEFAULT   ", {}),
        ("transcribe", {"language": "zh", "task": "transcribe"}),
        ("translate ", {"language": "zh", "task": "translate"}),
    ]:
        with torch.no_grad():
            ids = model.generate(feats, num_beams=1, max_new_tokens=200, **kwargs)
        outs[label] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    lines.append(f"\n===== Clip {i} =====")
    lines.append(f"REFERENCE  (Han {han_ratio(ref)*100:.0f}%): {ref}")
    for label, txt in outs.items():
        lines.append(f"{label} (Han {han_ratio(txt)*100:3.0f}%): {txt}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}", flush=True)

"""Smoke test: does the newly deployed PA model transcribe Gurmukhi (not translate)?"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from faster_whisper import WhisperModel

MODEL = r"C:\Users\vis15\offline_ai_system_v2\models\whisper-large-v3-pa-ct2"
AUDIO_DIR = Path(r"C:\Users\vis15\offline_ai_system_v2\eval_audio")

print(f"Loading {MODEL} ...", flush=True)
m = WhisperModel(MODEL, device="cuda", compute_type="int8")
print("Model loaded OK.\n", flush=True)

def is_gurmukhi(s):
    return any('਀' <= c <= '੿' for c in s)

out = []
for i in range(3):
    wav = AUDIO_DIR / f"Punjabi_{i:04d}.wav"
    if not wav.exists():
        continue
    segs, info = m.transcribe(str(wav), language="pa", task="transcribe", beam_size=5)
    text = " ".join(s.text for s in segs).strip()
    tag = "GURMUKHI OK" if is_gurmukhi(text) else "NOT GURMUKHI (translated?)"
    out.append(f"[{wav.name}]  ({tag})\n   {text}\n")

Path("logs/smoke_test_pa_result.txt").write_text("\n".join(out), encoding="utf-8")
print("Wrote logs/smoke_test_pa_result.txt", flush=True)

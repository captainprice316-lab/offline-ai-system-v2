"""
extract_ks_audio.py – Populate robustness_cache/ks/ from IndicVoices-R test arrows.

Reads the two IndicVoices-R Kashmiri test Arrow IPC files, extracts audio,
resamples to 16 kHz mono, and saves the first N valid samples (2–20 s) as WAV.

Usage:
    python scripts/eval/extract_ks_audio.py          # extract 30 samples
    python scripts/eval/extract_ks_audio.py --n 50   # extract 50 samples
"""

import argparse, sys, io
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyarrow as pa
import pyarrow.ipc as ipc

SCRIPT_DIR = Path(__file__).resolve().parent
VANI_ROOT  = SCRIPT_DIR.parent.parent
CACHE_DIR  = VANI_ROOT / "robustness_cache" / "ks"

ARROW_DIR = Path(
    r"E:\VANI\datasets\hf_ks_temp\datasets\ai4bharat___indicvoices_r\Kashmiri"
    r"\0.0.0\5f4495c91d500742a58d1be2ab07d77f73c0acf8"
)
ARROW_FILES = sorted(ARROW_DIR.glob("indicvoices_r-test-*.arrow"))

TARGET_SR = 16000
MIN_DUR   = 2.0
MAX_DUR   = 20.0


def iter_samples():
    """Yield (audio_bytes_or_array, sampling_rate) from all test arrow files."""
    for af in ARROW_FILES:
        with open(af, "rb") as f:
            try:
                reader = ipc.open_stream(f)
            except pa.lib.ArrowInvalid:
                f.seek(0)
                reader = ipc.open_file(f)

            batches = list(reader)
            tbl = pa.Table.from_batches(batches)
            print(f"  {af.name}: {len(tbl)} rows, columns: {tbl.schema.names}", flush=True)

            # Audio column may be a struct with bytes/array/sampling_rate
            audio_col_name = None
            for name in tbl.schema.names:
                if "audio" in name.lower():
                    audio_col_name = name
                    break

            if audio_col_name is None:
                print(f"  No audio column found, skipping {af.name}", flush=True)
                continue

            col = tbl.column(audio_col_name)
            for row in col:
                d = row.as_py()
                if isinstance(d, dict):
                    raw   = d.get("bytes") or d.get("array")
                    sr    = d.get("sampling_rate") or 48000
                    yield raw, sr
                elif isinstance(d, bytes):
                    yield d, 48000
                elif isinstance(d, list):
                    yield np.array(d, dtype=np.float32), 48000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of samples to extract")
    args = parser.parse_args()

    if not ARROW_FILES:
        sys.exit(f"No arrow files found in {ARROW_DIR}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CACHE_DIR.glob("*.wav"))
    if len(existing) >= args.n:
        print(f"Already have {len(existing)} KS WAVs in {CACHE_DIR} — nothing to do.")
        return

    saved, skipped = 0, 0
    for audio_bytes, src_sr in iter_samples():
        if saved >= args.n:
            break
        if audio_bytes is None:
            skipped += 1
            continue
        try:
            if isinstance(audio_bytes, (bytes, bytearray)):
                arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            elif isinstance(audio_bytes, (list, np.ndarray)):
                arr = np.array(audio_bytes, dtype=np.float32)
                sr  = src_sr
            else:
                skipped += 1
                continue

            if arr.ndim > 1:
                arr = arr.mean(axis=1)

            if sr != TARGET_SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
                sr  = TARGET_SR

            dur = len(arr) / sr
            if dur < MIN_DUR or dur > MAX_DUR:
                skipped += 1
                continue

            out_path = CACHE_DIR / f"{saved:04d}.wav"
            sf.write(str(out_path), arr, sr)
            saved += 1
            print(f"  saved {saved}/{args.n} ({dur:.1f}s)", end="\r", flush=True)

        except Exception as e:
            print(f"  [skip] {e}", flush=True)
            skipped += 1

    print(f"\nDone: {saved} saved, {skipped} skipped → {CACHE_DIR}")


if __name__ == "__main__":
    main()

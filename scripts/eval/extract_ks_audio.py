"""
extract_ks_audio.py – Populate robustness_cache/ks/ from IndicVoices-R test arrows.

Reads the two IndicVoices-R Kashmiri test Arrow IPC files, extracts audio,
resamples to 16 kHz mono, and saves the first N valid samples (2–20 s) as WAV
*and* their reference transcripts to refs.jsonl.

The transcript comes from the `normalized` column — the same column
finetune_whisper.py trains on (line ~306) and eval_indic_conformer_ks.py scores
against. `verbatim` is the literal utterance including mispronunciations and is
NOT the right ASR reference; it is carried through for inspection only.

Originally this script wrote audio but no transcripts, which is why Kashmiri could
not be scored for WER at all. refs.jsonl is emitted from the same loop as the WAVs,
so the two cannot drift out of alignment.

Usage:
    python scripts/eval/extract_ks_audio.py             # extract 30 samples + refs
    python scripts/eval/extract_ks_audio.py --n 50      # extract 50 samples
    python scripts/eval/extract_ks_audio.py --refs-only # WAVs exist: just write refs,
                                                        # verifying durations match
"""

import argparse, sys, io, json
from pathlib import Path

# The Windows console is cp1252 and cannot encode Perso-Arabic (or even "->" as U+2192).
# Without this the script crashes on print() *after* doing all its work.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
    """Yield (audio_bytes_or_array, sampling_rate, text) from all test arrow files."""
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

            names = tbl.schema.names
            text_col = "normalized" if "normalized" in names else "text"
            texts = tbl.column(text_col).to_pylist()
            col   = tbl.column(audio_col_name)

            for i, row in enumerate(col):
                d    = row.as_py()
                text = (texts[i] or "").strip()
                if isinstance(d, dict):
                    raw = d.get("bytes") or d.get("array")
                    sr  = d.get("sampling_rate") or 48000
                    yield raw, sr, text
                elif isinstance(d, bytes):
                    yield d, 48000, text
                elif isinstance(d, list):
                    yield np.array(d, dtype=np.float32), 48000, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of samples to extract")
    parser.add_argument("--refs-only", action="store_true",
                        help="WAVs already cached: write refs.jsonl only, verifying "
                             "each duration against the WAV on disk")
    args = parser.parse_args()

    if not ARROW_FILES:
        sys.exit(f"No arrow files found in {ARROW_DIR}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CACHE_DIR.glob("*.wav"))
    refs_path = CACHE_DIR / "refs.jsonl"

    if args.refs_only and not existing:
        sys.exit(f"--refs-only but no WAVs in {CACHE_DIR}")
    if not args.refs_only and len(existing) >= args.n and refs_path.exists():
        print(f"Already have {len(existing)} KS WAVs and refs.jsonl — nothing to do.")
        return

    saved, skipped, refs = 0, 0, []
    for audio_bytes, src_sr, text in iter_samples():
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
            if args.refs_only:
                # Prove index alignment instead of trusting it: the WAV already on disk
                # at this index must be the audio we just decoded.
                if saved >= len(existing):
                    break
                wav_dur = sf.info(str(existing[saved])).duration
                if abs(wav_dur - dur) > 0.05:
                    sys.exit(
                        f"MISALIGNED at index {saved}: {existing[saved].name} is "
                        f"{wav_dur:.2f}s but arrow row is {dur:.2f}s.\n"
                        f"Delete {CACHE_DIR} and re-run without --refs-only so audio "
                        f"and refs come from a single pass."
                    )
            else:
                sf.write(str(out_path), arr, sr)

            if not text:
                print(f"  [warn] empty transcript at index {saved} — kept, will not be scored")
            refs.append({"idx": saved, "ref": text, "dur": round(dur, 3)})
            saved += 1
            print(f"  {'checked' if args.refs_only else 'saved'} {saved}/{args.n} ({dur:.1f}s)",
                  end="\r", flush=True)

        except Exception as e:
            print(f"  [skip] {e}", flush=True)
            skipped += 1

    with refs_path.open("w", encoding="utf-8") as fh:
        for r in refs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_empty = sum(1 for r in refs if not r["ref"])
    print(f"\nDone: {saved} samples, {skipped} skipped → {CACHE_DIR}")
    print(f"      refs.jsonl: {len(refs)} rows ({n_empty} empty) — column 'normalized'")
    if args.refs_only:
        print(f"      duration alignment verified against {len(existing)} cached WAVs")


if __name__ == "__main__":
    main()

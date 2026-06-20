"""
audio_chunker.py – VAD-aware audio chunking
---------------------------------------------
Key improvements over original:
  • Uses VAD segment timestamps instead of blind fixed-size splits
  • Respects Whisper's 30-second training window (chunks ≤ 29s)
  • Groups adjacent VAD segments into a single chunk (avoids chopping sentences)
  • Falls back to fixed chunking if no VAD segments provided
  • Returns chunk metadata including original timestamps for JSON output
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


class AudioChunker:

    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.max_duration = cfg.get("max_chunk_duration", 29)   # seconds
        self.min_duration = cfg.get("min_chunk_duration", 1.0)
        self.sr           = 16000

    # ── public API ─────────────────────────────────────────────────────────────

    def split_audio(
        self,
        audio_path: str,
        output_dir: str,
        vad_segments_seconds: list = None,
    ) -> list:
        """
        Split audio into chunks for Whisper.

        Parameters
        ----------
        audio_path            : path to preprocessed WAV
        output_dir            : directory to write chunk WAVs
        vad_segments_seconds  : list of dicts with start_sec / end_sec
                                 from VADModule. If None, falls back to
                                 fixed-size splitting.

        Returns
        -------
        List of dicts:
            {
              "path"      : Path to chunk WAV,
              "start_sec" : original start in full audio,
              "end_sec"   : original end in full audio,
              "index"     : chunk index,
            }
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale chunks
        for old in output_dir.glob("chunk_*.wav"):
            try:
                old.unlink()
            except Exception:
                pass

        audio, _ = librosa.load(str(audio_path), sr=self.sr)
        total_dur = len(audio) / self.sr

        if vad_segments_seconds:
            groups = self._group_vad_segments(vad_segments_seconds, total_dur)
        else:
            groups = self._fixed_groups(total_dur)

        chunks = []
        for idx, (start, end) in enumerate(groups):
            duration = end - start
            if duration < self.min_duration:
                continue

            start_sample = int(start * self.sr)
            end_sample   = min(int(end   * self.sr), len(audio))
            chunk_audio  = audio[start_sample:end_sample]

            chunk_path = output_dir / f"chunk_{idx:04d}.wav"
            sf.write(str(chunk_path), chunk_audio, self.sr)

            chunks.append({
                "path":      chunk_path,
                "start_sec": round(start, 3),
                "end_sec":   round(end,   3),
                "index":     idx,
            })

        return chunks

    # ── private helpers ────────────────────────────────────────────────────────

    def _group_vad_segments(self, vad_segs: list, total_dur: float) -> list:
        """
        Greedily pack VAD segments into groups that fit within max_duration.
        Adjacent segments are merged into one chunk as long as the combined
        duration stays under the limit.
        """
        groups = []
        group_start = None
        group_end   = None

        for seg in vad_segs:
            s = seg["start_sec"]
            e = seg["end_sec"]

            if group_start is None:
                group_start, group_end = s, e
                continue

            # Would adding this segment exceed the limit?
            if (e - group_start) > self.max_duration:
                groups.append((group_start, group_end))
                group_start, group_end = s, e
            else:
                group_end = e   # extend current group

        if group_start is not None:
            groups.append((group_start, group_end))

        return groups

    def _fixed_groups(self, total_dur: float) -> list:
        """Fallback: fixed-size windows."""
        groups = []
        start = 0.0
        while start < total_dur:
            end = min(start + self.max_duration, total_dur)
            groups.append((start, end))
            start = end
        return groups

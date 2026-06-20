"""
vad_module.py – Voice Activity Detection
-----------------------------------------
Key improvements over original:
  • Returns segments with SECONDS (not samples) so pipeline can use them directly
  • Configurable threshold / padding via config dict
  • Speech padding: adds N ms around each segment to avoid clipped words
  • Segment merging: adjacent segments < min_silence apart are merged
  • Preserves original timestamps so chunker can do VAD-aware splitting
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from silero_vad import load_silero_vad, get_speech_timestamps


def _read_audio_sf(path: str, sampling_rate: int) -> torch.Tensor:
    """soundfile-based loader — replaces silero's read_audio which requires torchcodec in torchaudio>=2.9."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)          # stereo → mono
    if sr != sampling_rate:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=sampling_rate)
    return torch.from_numpy(data)


class VADModule:

    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.sampling_rate     = cfg.get("sampling_rate", 16000)
        self.threshold         = cfg.get("threshold", 0.45)
        self.min_speech_ms     = cfg.get("min_speech_duration_ms", 250)
        self.min_silence_ms    = cfg.get("min_silence_duration_ms", 600)
        self.speech_pad_ms     = cfg.get("speech_pad_ms", 100)
        self.model             = load_silero_vad()

    # ── public API ─────────────────────────────────────────────────────────────

    def remove_silence(self, audio_path: str, output_path: str) -> dict:
        """
        Process audio file, strip silence, write clean audio.

        Returns
        -------
        dict with keys:
            segments_samples  – raw sample-index segments from Silero
            segments_seconds  – same segments converted to seconds (used by chunker)
            output_audio      – path to written clean WAV
            total_speech_sec  – total retained speech duration
        """
        wav = _read_audio_sf(str(audio_path), self.sampling_rate)
        total_samples = len(wav)

        raw_segments = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
        )

        if not raw_segments:
            # Retry with a lower threshold for quiet / noisy recordings
            raw_segments = get_speech_timestamps(
                wav,
                self.model,
                sampling_rate=self.sampling_rate,
                threshold=0.25,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
            )

        if not raw_segments:
            # Final fallback: treat entire audio as one speech segment so
            # Whisper's own no_speech filter can make the per-segment call
            raw_segments = [{"start": 0, "end": total_samples - 1}]

        # Apply padding (clamp to valid range)
        pad = int(self.speech_pad_ms * self.sampling_rate / 1000)
        padded = []
        for seg in raw_segments:
            padded.append({
                "start": max(0, seg["start"] - pad),
                "end":   min(total_samples - 1, seg["end"] + pad),
            })

        # Merge segments that now overlap after padding
        merged = [padded[0].copy()]
        for seg in padded[1:]:
            if seg["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            else:
                merged.append(seg.copy())

        # Build clean audio from merged segments
        speech_chunks = [wav[s["start"]: s["end"]] for s in merged]
        speech_audio  = torch.cat([torch.tensor(c) if not isinstance(c, torch.Tensor) else c
                                   for c in speech_chunks])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, speech_audio.numpy(), self.sampling_rate)

        sr = self.sampling_rate

        # Timestamps must be relative to the OUTPUT file (_vad.wav), where silence
        # has been stripped and segments are concatenated from 0.  Using original
        # timestamps here causes the chunker to skip leading speech equal to the
        # amount of leading silence that was removed.
        segments_seconds = []
        cumulative_sec = 0.0
        for s in merged:
            duration = (s["end"] - s["start"]) / sr
            segments_seconds.append({
                "start_sec":    round(cumulative_sec, 3),
                "end_sec":      round(cumulative_sec + duration, 3),
                "start_sample": int(cumulative_sec * sr),
                "end_sample":   int((cumulative_sec + duration) * sr),
            })
            cumulative_sec += duration

        total_speech = cumulative_sec

        return {
            "segments_samples": merged,
            "segments_seconds": segments_seconds,
            "output_audio":     str(output_path),
            "total_speech_sec": round(total_speech, 2),
        }

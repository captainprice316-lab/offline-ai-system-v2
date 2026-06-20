"""
audio_preprocess.py – Radio-aware audio preprocessing
-------------------------------------------------------
Key improvements over original:
  • stationary=True noise reduction: far better for constant radio hiss/carrier
  • Pre-emphasis filter: compensates for HF radio high-frequency rolloff
  • Safer normalization: RMS-based instead of peak (avoids over-amplifying noise)
  • Bandpass filter: 300–3400 Hz radio telephony window removes sub-bass and HF noise
  • Configurable noise reduction strength (prop_decrease)
  • SNR before/after measurement returned in result dict
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr


class AudioPreprocessor:

    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.target_sr       = cfg.get("target_sr", 16000)
        self.normalize       = cfg.get("normalize", True)
        self.noise_reduce    = cfg.get("noise_reduce", True)
        self.stationary      = cfg.get("noise_reduce_stationary", True)
        self.prop_decrease   = cfg.get("prop_decrease", 0.75)
        self.trim_db         = cfg.get("trim_silence_db", 20)
        self.bandpass        = cfg.get("bandpass_filter", True)
        self.bandpass_low    = cfg.get("bandpass_low_hz",  300)
        self.bandpass_high   = cfg.get("bandpass_high_hz", 3400)

    # ── public API ─────────────────────────────────────────────────────────────

    def preprocess(self, input_path: str, output_path: str) -> dict:
        input_path  = Path(input_path)
        output_path = Path(output_path)

        # 1. Load + resample to 16 kHz mono
        audio, _ = librosa.load(str(input_path), sr=self.target_sr, mono=True)

        if len(audio) == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, self.target_sr)
            return {"input_path": str(input_path), "output_path": str(output_path),
                    "sample_rate": self.target_sr, "duration_sec": 0.0,
                    "snr_before_db": None, "snr_after_db": None}

        snr_before = _estimate_snr(audio)

        # 2. Pre-emphasis – compensates HF radio rolloff (boosts highs slightly)
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # 3. Bandpass filter – radio telephony bandwidth 300–3400 Hz
        #    Removes sub-bass rumble and high-frequency hiss outside voice range.
        if self.bandpass:
            audio = _bandpass_filter(
                audio, self.target_sr,
                self.bandpass_low, self.bandpass_high,
            )

        # 4. Noise reduction
        #    stationary=True: estimates noise from the whole clip (good for
        #    constant carrier hiss). stationary=False adapts per-frame (good for
        #    wind/crowd but can eat speech on quiet radio).
        if self.noise_reduce:
            audio = nr.reduce_noise(
                y=audio,
                sr=self.target_sr,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
            )

        snr_after = _estimate_snr(audio)

        # 5. RMS normalization (safer than peak for radio audio)
        if self.normalize:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 1e-6:
                target_rms = 0.1          # -20 dBFS target
                audio = audio * (target_rms / rms)
            audio = np.clip(audio, -1.0, 1.0)

        # Silence trimming intentionally omitted here — VAD (Stage 1) already
        # strips silence before this stage.  A second trim shifts audio offsets
        # and causes the chunker to skip leading speech.

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.target_sr)

        return {
            "input_path":   str(input_path),
            "output_path":  str(output_path),
            "sample_rate":  self.target_sr,
            "duration_sec": round(len(audio) / self.target_sr, 2),
            "snr_before_db": round(snr_before, 1) if snr_before is not None else None,
            "snr_after_db":  round(snr_after,  1) if snr_after  is not None else None,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bandpass_filter(
    audio: np.ndarray, sr: int,
    low_hz: float = 300, high_hz: float = 3400,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to restrict audio to voice range."""
    try:
        from scipy.signal import butter, sosfilt
        nyq  = sr / 2.0
        low  = max(low_hz  / nyq, 1e-4)
        high = min(high_hz / nyq, 1 - 1e-4)
        sos  = butter(order, [low, high], btype="band", output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        return audio   # graceful fallback if scipy unavailable


def _estimate_snr(audio: np.ndarray, frame_len: int = 512) -> float:
    """
    Estimate signal-to-noise ratio using a simple percentile energy method.
    Loud frames (top 20%) represent signal; quiet frames (bottom 20%) represent noise.
    Returns SNR in dB, or None if audio is too short.
    """
    if len(audio) < frame_len * 4:
        return None
    frames = np.array([
        np.mean(audio[i:i + frame_len] ** 2)
        for i in range(0, len(audio) - frame_len, frame_len)
    ])
    frames = frames[frames > 0]
    if len(frames) < 4:
        return None
    signal_power = np.percentile(frames, 80)
    noise_power  = np.percentile(frames, 20)
    if noise_power < 1e-12:
        return 60.0
    return float(10 * np.log10(signal_power / noise_power))

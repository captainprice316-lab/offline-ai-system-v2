"""
asr_module.py – Whisper ASR with radio/military tuning
--------------------------------------------------------
Key improvements over original:
  • beam_size=2, best_of=1, temperature=0.0 → 3-4x faster on CPU, more stable
  • condition_on_previous_text=False → prevents hallucination loops on radio noise
  • word_timestamps=True → per-word timing for precision search
  • initial_prompt → primes Whisper vocabulary for military/radio domain
  • language hint passed in (detected once, reused for all chunks)
  • Returns per-segment confidence (avg_logprob → 0-1 score) and no_speech_prob
  • Filters out hallucinated "thank you / music" segments via no_speech_prob
"""

import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# faster-whisper has a hardcoded language allowlist that excludes custom tokens
# like <|ks|> (Kashmiri). Patch it at import time so language="ks" is accepted.
# The tokenizer already has <|ks|> at ID 51866 in the vocabulary.
import faster_whisper.tokenizer as _fwt
if "ks" not in _fwt._LANGUAGE_CODES:
    _fwt._LANGUAGE_CODES = frozenset(list(_fwt._LANGUAGE_CODES) + ["ks"])

# Whisper's decoder skips speech that starts at sample 0 (cold start).
# Prepending silence gives it a "runway" so the first timestamp token lands at
# ~1.0 s; we subtract that offset from every returned timestamp.
WHISPER_PREPEND_SILENCE_S = 1.0


# Segments whose no_speech_prob exceeds this are likely noise/static – skip them
# Raised from 0.60 → 0.70 to reduce false-filtering of Punjabi/Indic segments
NO_SPEECH_THRESHOLD = 0.70

# Known Whisper hallucination phrases (common on silence/radio noise)
_HALLUCINATION_PHRASES = frozenset([
    "thank you for watching", "thanks for watching",
    "please subscribe", "like and subscribe", "don't forget to subscribe",
    "subtitles by", "transcribed by", "caption by",
    "www.", ".com", ".org",
    "[music]", "[ music ]", "[applause]", "[laughter]", "[silence]",
    "♪", "♫",
    "visit our website", "for more information",
    "this video", "this episode",
])


def _is_hallucination(text: str) -> bool:
    """Return True if text matches known Whisper hallucination patterns."""
    t = text.lower().strip()
    if not t:
        return True
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in t:
            return True
    # Detect extreme repetition: same word appearing ≥ 5 times in a row
    words = t.split()
    if len(words) >= 5:
        for i in range(len(words) - 4):
            if len(set(words[i:i + 5])) == 1:
                return True
    return False


class ASRModule:

    def __init__(self, model_path: str = "models/whisper_medium",
                 device: str = "cpu", cfg: dict = None):
        cfg = cfg or {}
        self.model_path = str(model_path)   # recorded so the pipeline can detect
                                            # a cached model that doesn't match the
                                            # language-specific selection
        # CTranslate2 (faster-whisper) only supports "cpu" and "cuda" — clamp MPS
        ct2_device  = "cpu" if device not in ("cpu", "cuda") else device
        compute     = "int8" if ct2_device == "cpu" else "float16"
        cpu_threads = cfg.get("cpu_threads", min(os.cpu_count() or 4, 8))

        self.model = WhisperModel(
            model_path, device=ct2_device,
            compute_type=compute, cpu_threads=cpu_threads,
        )

        # Decoding parameters – all overridable via config
        self.beam_size                 = cfg.get("beam_size", 2)
        self.best_of                   = cfg.get("best_of", 1)
        self.temperature               = cfg.get("temperature", 0.0)
        self.condition_on_prev         = cfg.get("condition_on_previous_text", False)
        self.word_timestamps           = cfg.get("word_timestamps", True)
        self.vad_filter                = cfg.get("vad_filter", True)
        self.vad_parameters            = cfg.get("vad_parameters", {})
        self.initial_prompt            = cfg.get("initial_prompt", None)
        self._lang_prompts             = {k[len("initial_prompt_"):]: v
                                          for k, v in cfg.items()
                                          if k.startswith("initial_prompt_")}
        self._detected_language        = None   # cached after first chunk

    # ── public API ─────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language_hint: str = None) -> dict:
        """
        Transcribe one audio file (chunk).

        Parameters
        ----------
        audio_path    : path to WAV chunk
        language_hint : ISO-639-1 code from first-chunk detection.
                        Passing this skips per-chunk language detection
                        and speeds up processing significantly.

        Returns
        -------
        dict with transcript, segments (with confidence), language info
        """
        lang = language_hint or self._detected_language

        # Use language-specific initial prompt if available (e.g. initial_prompt_pa for Punjabi)
        prompt = self._lang_prompts.get(lang, self.initial_prompt) if lang else self.initial_prompt

        # Load audio and prepend silence so Whisper doesn't skip the cold start
        audio, _ = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        pad_samples = int(WHISPER_PREPEND_SILENCE_S * 16000)
        audio_input = np.concatenate([np.zeros(pad_samples, dtype=np.float32), audio])

        segments_iter, info = self.model.transcribe(
            audio_input,            # numpy array; faster-whisper assumes 16 kHz
            language                = lang,
            beam_size               = self.beam_size,
            best_of                 = self.best_of,
            temperature             = self.temperature,
            condition_on_previous_text = self.condition_on_prev,
            word_timestamps         = self.word_timestamps,
            vad_filter              = self.vad_filter,
            vad_parameters          = self.vad_parameters or None,
            initial_prompt          = prompt,
        )

        # Cache language from first detection
        if self._detected_language is None:
            self._detected_language = info.language

        transcript_parts = []
        segment_list     = []

        for seg in segments_iter:

            # Skip likely-silence / music hallucinations
            if seg.no_speech_prob > NO_SPEECH_THRESHOLD:
                continue

            text = seg.text.strip()
            if not text:
                continue

            # Skip known Whisper hallucination phrases
            if _is_hallucination(text):
                continue

            transcript_parts.append(text)

            # avg_logprob is negative; convert to 0-1 confidence
            confidence = round(min(1.0, max(0.0, 1.0 + seg.avg_logprob / 4.0)), 3)

            # Subtract the prepended silence offset from all timestamps
            adj_start = round(max(0.0, seg.start - WHISPER_PREPEND_SILENCE_S), 3)
            adj_end   = round(max(0.0, seg.end   - WHISPER_PREPEND_SILENCE_S), 3)

            seg_dict = {
                "start":          adj_start,
                "end":            adj_end,
                "text":           text,
                "confidence":     confidence,
                "no_speech_prob": round(seg.no_speech_prob, 3),
            }

            # Attach per-word timestamps if available
            if self.word_timestamps and seg.words:
                seg_dict["words"] = [
                    {
                        "word":        w.word,
                        "start":       round(max(0.0, w.start - WHISPER_PREPEND_SILENCE_S), 3),
                        "end":         round(max(0.0, w.end   - WHISPER_PREPEND_SILENCE_S), 3),
                        "probability": round(w.probability, 3),
                    }
                    for w in seg.words
                ]

            segment_list.append(seg_dict)

        return {
            "language":             info.language,
            "language_probability": round(getattr(info, "language_probability", 0.0), 3),
            "transcript":           " ".join(transcript_parts).strip(),
            "segments":             segment_list,
        }

    def reset_language_cache(self):
        """Call between unrelated audio files."""
        self._detected_language = None

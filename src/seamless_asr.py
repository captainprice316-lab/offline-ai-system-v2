"""
seamless_asr.py — SeamlessM4T v2 ASR backend.

Drop-in alternative to ASRModule for languages where zero-shot SeamlessM4T
beats the fine-tuned Whisper (measured: Punjabi 55.7%->19.8%, Nepali
49.2%->28.5% WER). Exposes the same transcribe() / reset_language_cache()
interface so the pipeline can swap it in transparently.

Only SeamlessM4T's ASR (speech -> source-language text) is used; downstream
translation stays on NLLB (SeamlessM4T's S2TT is not used — its ASR-only
quality is the win, and FT-SM4T S2TT was found broken).
"""
import numpy as np
import soundfile as sf

# ISO-639-1 -> SeamlessM4T v2 source-language code
SEAMLESS_LANG = {
    "pa": "pan", "ne": "npi", "ur": "urd",
    "hi": "hin", "ps": "pbt", "zh": "cmn",
}


class SeamlessASR:
    """SeamlessM4T v2 speech-to-text, ASRModule-compatible."""

    def __init__(self, model_path: str, device: str = "cpu",
                 cfg: dict = None, default_lang: str = None):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.model_path = str(model_path)
        self.device     = "cpu" if device not in ("cpu", "cuda") else device
        dtype           = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model     = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            self.model_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        self._torch             = torch
        self.default_lang       = default_lang
        self._detected_language = None

    def reset_language_cache(self):
        """Call between unrelated audio files (interface parity with ASRModule)."""
        self._detected_language = None

    def to_device(self, device: str):
        """Move the model between CPU RAM and GPU (dtype preserved).

        Lets a cached instance park on CPU between files and promote to GPU
        only for the ASR stage — PCIe transfer (~1 s) vs disk reload (~8 s).
        """
        device = "cpu" if device not in ("cpu", "cuda") else device
        if next(self.model.parameters()).device.type != device:
            self.model.to(device)
        self.device = device

    # Utterances shorter than this are skipped — SeamlessM4T hallucinates on
    # sub-word fragments, and VAD's own min_speech_duration is already ~250 ms.
    _MIN_SUBSEG_S = 0.20

    def _generate(self, audio_16k, sm_lang: str) -> str:
        """Run one SeamlessM4T ASR pass over a 16 kHz mono array → text."""
        inputs = self.processor(
            audio=audio_16k, return_tensors="pt",
            sampling_rate=16000, src_lang=sm_lang,
        ).to(self.device)
        with self._torch.no_grad():
            toks = self.model.generate(**inputs, tgt_lang=sm_lang)
        return self.processor.decode(toks[0], skip_special_tokens=True).strip()

    def transcribe(self, audio_path: str, language_hint: str = None,
                   subsegments: list = None) -> dict:
        """Transcribe a chunk.

        subsegments : optional list of (start_s, end_s) CHUNK-RELATIVE windows (the
                      chunk's constituent VAD utterances). When given, SeamlessM4T runs
                      once per utterance and returns one timed segment each, so downstream
                      speaker labelling and keyword→time mapping stay per-utterance instead
                      of collapsing a 29 s chunk into a single untimed blob. When omitted,
                      the whole chunk is transcribed as one segment (prior behaviour).
        """
        lang    = language_hint or self.default_lang or self._detected_language or "pa"
        sm_lang = SEAMLESS_LANG.get(lang, "pan")

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        dur   = round(len(audio) / sr, 3)
        segs  = []
        parts = []

        if subsegments:
            for (rs, re) in subsegments:
                rs = max(0.0, rs)
                re = min(dur, re)
                if re - rs < self._MIN_SUBSEG_S:
                    continue
                clip = audio[int(rs * sr):int(re * sr)]
                if len(clip) < int(self._MIN_SUBSEG_S * sr):
                    continue
                t = self._generate(clip, sm_lang)
                if not t:
                    continue
                segs.append({"start": round(rs, 3), "end": round(re, 3), "text": t,
                             "confidence": 0.9, "no_speech_prob": 0.0})
                parts.append(t)

        # No subsegments given, or every subseg came back empty → whole-chunk fallback.
        if not segs:
            t = self._generate(audio, sm_lang)
            if t:
                segs  = [{"start": 0.0, "end": dur, "text": t,
                          "confidence": 0.9, "no_speech_prob": 0.0}]
                parts = [t]

        text = " ".join(parts).strip()

        if self._detected_language is None:
            self._detected_language = lang
        return {
            "language":             lang,
            "language_probability": 1.0,
            "transcript":           text,
            "segments":             segs,
        }

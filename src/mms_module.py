"""
src/mms_module.py – Meta MMS Language Identification
-----------------------------------------------------
Uses facebook/mms-lid-256 to identify spoken language directly from
the audio waveform. Provides a third, independent vote for language
detection alongside Whisper (ASR-based) and FastText (text-based).

MMS-LID is particularly strong for South/Central Asian languages and
performs well on noisy radio audio.
"""

import gc
import os
import torch
import numpy as np


# ── Language code mapping: MMS ISO-639-3 → internal ISO-639-1 ─────────────────
MMS_TO_INTERNAL = {
    # Target languages
    "hin": "hi",   "pan": "pa",   "urd": "ur",   "npi": "ne",
    "pbu": "ps",   "pst": "ps",   "pus": "ps",
    "cmn": "zh",   "yue": "zh",   "mya": "my",   "bod": "bo",
    "kas": "ks",   "dgo": "doi",  "mai": "mai",  "ben": "bn",
    "snd": "sd",   "sin": "si",   "tgk": "tg",   "uzb": "uz",
    "kaz": "kk",   "fas": "fa",   "ara": "ar",   "pes": "fa",
    "eng": "en",   "fra": "fr",   "deu": "de",   "spa": "es",
    "rus": "ru",   "por": "pt",   "tur": "tr",   "jpn": "ja",
    "kor": "ko",
}


class MMSLangDetector:

    def __init__(self, model_path: str, device: str = "cpu"):
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

        # Wav2Vec2 on MPS can be unstable — keep on CPU for reliability
        self.device = "cpu" if device not in ("cpu", "cuda") else device

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path, local_files_only=True,
        )
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_path, local_files_only=True,
        ).to(self.device)
        self._model.eval()

    def detect(self, audio_path: str, top_k: int = 5) -> dict:
        """
        Identify spoken language from an audio file.

        Returns
        -------
        dict with:
            language   : best-match internal language code (ISO-639-1)
            confidence : probability 0–1
            top5       : list of {mms_code, language, confidence}
            mms_code   : raw MMS 3-letter code of top result
        """
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        # Truncate to 30s max for speed (MMS-LID is frame-level)
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = self._feature_extractor(
            audio, sampling_rate=16000,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs   = torch.softmax(logits, dim=-1)[0]
        top_idx = torch.topk(probs, min(top_k, len(probs))).indices

        results = []
        for idx in top_idx:
            mms_code = self._model.config.id2label[idx.item()]
            internal = MMS_TO_INTERNAL.get(mms_code, mms_code[:2].lower())
            results.append({
                "mms_code":   mms_code,
                "language":   internal,
                "confidence": round(probs[idx].item(), 4),
            })

        return {
            "language":   results[0]["language"],
            "mms_code":   results[0]["mms_code"],
            "confidence": results[0]["confidence"],
            "top5":       results,
        }

    def unload(self):
        del self._model, self._feature_extractor
        self._model = self._feature_extractor = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

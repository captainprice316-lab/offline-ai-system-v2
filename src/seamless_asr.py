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
# ks/"kas" is NOT in the base model: it exists only via the ks_max adapter's
# custom __kas__ token (added at fine-tune time, embedding row restored by the
# adapter's trainable-token delta). It only works when that adapter is loaded.
SEAMLESS_LANG = {
    "pa": "pan", "ne": "npi", "ur": "urd",
    "hi": "hin", "ps": "pbt", "zh": "cmn",
    "ks": "kas",
}

# Per-language generate() kwargs, matching the configuration each deployed
# adapter was EVALUATED with. Kashmiri needs the decode fixes from the
# 2026-07-17 probe (early-EOS under-generation: 128->94 WER from these alone).
def _gen_kwargs(sm_lang: str, dur_s: float) -> dict:
    if sm_lang == "kas":
        return {"min_new_tokens": min(180, max(5, int(dur_s * 2.5))),
                "no_repeat_ngram_size": 3}
    return {}


class SeamlessASR:
    """SeamlessM4T v2 speech-to-text, ASRModule-compatible."""

    def __init__(self, model_path: str, device: str = "cpu",
                 cfg: dict = None, default_lang: str = None):
        import torch
        from pathlib import Path
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.model_path = str(model_path)
        self.device     = "cpu" if device not in ("cpu", "cuda") else device
        dtype           = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model     = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            self.model_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        # Optional per-language LoRA adapters (cfg asr.seamless_adapters:
        # {vani_lang: path}). The adapter is enabled ONLY for its language's
        # generate() calls; every other language runs with adapters disabled,
        # which is numerically identical to the plain base model. Deployed for
        # hi 2026-07-13: 13.94 vs zero-shot 15.44 WER (n=100 clean), and wins
        # 4/5 radio-degradation conditions incl. bandpass (16.28 vs 18.96).
        self._adapters = {}   # sm_lang code -> peft adapter name, on self.model
        self._ks_model = None       # Kashmiri gets its OWN model instance (see below)
        self._ks_processor = None
        adapters_cfg = (cfg or {}).get("seamless_adapters") or {}
        if adapters_cfg:
            from peft import PeftModel
            repo_root = Path(__file__).resolve().parents[1]

            # Kashmiri (deployed 2026-07-20) is deliberately kept OFF the
            # shared multi-adapter model. Its adapter carries a trainable
            # __kas__ embedding delta (PEFT trainable_token_indices) whose
            # wrapper module is only built when that adapter is the sole/
            # first one on a PeftModel; stacking any other adapter (hi/ne/ps)
            # alongside it — in either load order — corrupts the shared
            # state dict (KeyError on trainable_tokens_delta, confirmed by
            # direct testing 2026-07-20). A second, fully independent
            # SeamlessM4Tv2ForSpeechToText + PeftModel instance sidesteps the
            # whole class of bug, matching how every ks eval script already
            # validated this adapter (eval_ks_seamless.py, ks_ruler_study.py).
            ks_rel = adapters_cfg.get("ks")
            if ks_rel:
                ks_p = Path(ks_rel)
                if not ks_p.is_absolute():
                    ks_p = repo_root / ks_rel
                if ks_p.exists():
                    self._ks_processor = AutoProcessor.from_pretrained(str(ks_p))
                    ks_base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                        self.model_path, torch_dtype=dtype)
                    tok = self._ks_processor.tokenizer
                    kas_id = tok.convert_tokens_to_ids("__kas__")
                    if ks_base.get_input_embeddings().weight.shape[0] < len(tok):
                        ks_base.resize_token_embeddings(len(tok))
                    self._ks_model = PeftModel.from_pretrained(
                        ks_base, str(ks_p)).to(self.device)
                    self._ks_model.eval()
                    lang_map = getattr(self._ks_model.generation_config,
                                       "text_decoder_lang_to_code_id", None)
                    if lang_map is not None:
                        try:
                            lang_map["kas"] = kas_id
                        except TypeError:
                            pass
                    print(f"[SeamlessASR] ks: own model instance loaded "
                          f"(__kas__ id {kas_id}, vocab {len(tok)})")
                else:
                    print(f"[SeamlessASR] WARN: adapter for 'ks' not loaded ({ks_p})")

            for vani_lang, rel in adapters_cfg.items():
                if vani_lang == "ks":
                    continue   # handled above, on its own model
                sm = SEAMLESS_LANG.get(vani_lang)
                p  = Path(rel)
                if not p.is_absolute():
                    p = repo_root / rel
                if sm is None or not p.exists():
                    print(f"[SeamlessASR] WARN: adapter for '{vani_lang}' not loaded "
                          f"({'unknown language' if sm is None else p})")
                    continue
                name = f"lora_{vani_lang}"
                if not self._adapters:
                    self.model = PeftModel.from_pretrained(self.model, str(p),
                                                           adapter_name=name)
                else:
                    self.model.load_adapter(str(p), adapter_name=name)
                self._adapters[sm] = name
                print(f"[SeamlessASR] LoRA adapter loaded for {vani_lang} ({name})")
            if self._adapters:
                self.model.to(self.device)
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
        if self._ks_model is not None and next(self._ks_model.parameters()).device.type != device:
            self._ks_model.to(device)
        self.device = device

    # Utterances shorter than this are skipped — SeamlessM4T hallucinates on
    # sub-word fragments, and VAD's own min_speech_duration is already ~250 ms.
    _MIN_SUBSEG_S = 0.20

    def _generate(self, audio_16k, sm_lang: str) -> str:
        """Run one SeamlessM4T ASR pass over a 16 kHz mono array → text.

        If a LoRA adapter is registered for this language it is activated for
        the call; otherwise all adapters are disabled so non-adapter languages
        see the unmodified base model."""
        gen_kw = _gen_kwargs(sm_lang, len(audio_16k) / 16000.0)

        # Kashmiri: separate model/processor entirely (see __init__ note).
        if sm_lang == "kas" and self._ks_model is not None:
            inputs = self._ks_processor(
                audio=audio_16k, return_tensors="pt", sampling_rate=16000,
            ).to(self.device)
            with self._torch.no_grad():
                toks = self._ks_model.generate(**inputs, tgt_lang=sm_lang, **gen_kw)
            return self._ks_processor.decode(toks[0], skip_special_tokens=True).strip()

        # feature extraction only — "kas" is not a valid src_lang for the base
        # processor's text side, and ASR needs no source-text conditioning
        inputs = self.processor(
            audio=audio_16k, return_tensors="pt", sampling_rate=16000,
        ).to(self.device)
        adapter = self._adapters.get(sm_lang)
        with self._torch.no_grad():
            if adapter is not None:
                self.model.set_adapter(adapter)
                toks = self.model.generate(**inputs, tgt_lang=sm_lang, **gen_kw)
            elif self._adapters:
                with self.model.disable_adapter():
                    toks = self.model.generate(**inputs, tgt_lang=sm_lang, **gen_kw)
            else:
                toks = self.model.generate(**inputs, tgt_lang=sm_lang, **gen_kw)
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

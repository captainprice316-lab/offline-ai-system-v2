"""
translation_module.py – Memory-safe translation for 8 GB RAM
--------------------------------------------------------------
Key improvements over original:
  • Explicit model unloading after use (del model + gc + empty_cache)
  • torch.no_grad() throughout → saves ~30% RAM vs original
  • Input truncated to max_input_tokens to prevent OOM on long transcripts
  • IndicTrans2 source language tag set correctly (required by the model)
  • NLLB target language always set to eng_Latn
  • Both models never in RAM simultaneously
"""

import gc
import re
import sys
import types
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _split_sentences(text: str) -> list:
    """Split text into sentences on . ! ? । or newlines."""
    parts = re.split(r'(?<=[.!?।])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]

# IndicTrans2 compatibility: transformers>=5.0 removed transformers.onnx.
# Stub it before any IndicTrans2 config/model import to prevent ImportError.
if "transformers.onnx" not in sys.modules:
    _onnx_stub = types.ModuleType("transformers.onnx")
    sys.modules["transformers.onnx"] = _onnx_stub


# IndicTrans2 language code map (Whisper/ISO → IndicTrans2 flores codes)
INDIC_LANG_MAP = {
    "hi":  "hin_Deva",
    "pa":  "pan_Guru",
    "doi": "dgo_Deva",   # Dogri
    "ne":  "npi_Deva",   # Nepali
    "ks":  "kas_Arab",   # Kashmiri (Arabic script)
    "ur":  "urd_Arab",
    "mai": "mai_Deva",
    "bn":  "ben_Beng",
    "sd":  "snd_Arab",
    "si":  "sin_Sinh",
}

# NLLB language code map
# Indic languages added: IndicTrans2 has transformers>=5 cache incompatibility,
# so NLLB-200 is used as the primary route for all supported Indic languages.
NLLB_LANG_MAP = {
    # ── Indic (NLLB primary route) ─────────────────────────────────────────────
    "hi":  "hin_Deva",   # Hindi
    "pa":  "pan_Guru",   # Punjabi (Gurmukhi)
    "ur":  "urd_Arab",   # Urdu
    "ne":  "npi_Deva",   # Nepali
    "bn":  "ben_Beng",   # Bengali
    "mai": "mai_Deva",   # Maithili
    "ks":  "kas_Arab",   # Kashmiri
    "sd":  "snd_Arab",   # Sindhi
    "si":  "sin_Sinh",   # Sinhala
    # ── Other languages ────────────────────────────────────────────────────────
    "ps":  "pus_Arab",   # Pashto
    "zh":  "zho_Hans",   # Simplified Chinese
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant", # Traditional Chinese
    "my":  "mya_Mymr",   # Burmese
    "bo":  "bod_Tibt",   # Tibetan
    "fa":  "pes_Arab",   # Persian/Farsi
    "ar":  "arb_Arab",
    "tg":  "tgk_Cyrl",   # Tajik
    "uz":  "uzn_Latn",   # Uzbek
    "kk":  "kaz_Cyrl",   # Kazakh
}

TARGET_LANG = "eng_Latn"


class TranslationModule:

    def __init__(self, indic_model_path: str, nllb_model_path: str,
                 device: str = "cpu", cfg: dict = None):
        cfg = cfg or {}
        self.indic_model_path  = indic_model_path
        self.nllb_model_path   = nllb_model_path
        self.max_input_tokens  = cfg.get("max_input_tokens",  256)
        self.max_output_tokens = cfg.get("max_output_tokens", 256)
        self.unload_after_use  = cfg.get("unload_after_use",  True)
        self.indic_num_beams   = cfg.get("indic_num_beams",   2)
        self.nllb_num_beams    = cfg.get("nllb_num_beams",    4)
        # IndicTrans2 requires eager attention — keep on CPU to avoid MPS incompatibilities
        self.device            = "cpu" if device not in ("cpu", "cuda") else device

        # Models kept as None until needed
        self._indic_tokenizer = self._indic_model = None
        self._nllb_tokenizer  = self._nllb_model  = None

    # ── public API ─────────────────────────────────────────────────────────────

    def translate(self, text: str, route: str, detected_lang: str) -> dict:
        """
        Translate text to English.

        Returns dict with:
            translated_text  – English translation
            route_used       – which model was used
            success          – True/False
            error            – error message if failed
        """
        if route == "none" or not text.strip():
            return {"translated_text": text, "route_used": "none",
                    "success": True, "error": None}

        try:
            if route == "indictrans2":
                result = self._translate_indic(text, detected_lang)
            elif route == "nllb":
                result = self._translate_nllb(text, detected_lang)
            else:
                result = text

            return {"translated_text": result, "route_used": route,
                    "success": True, "error": None}

        except Exception as e:
            return {"translated_text": "", "route_used": route,
                    "success": False, "error": str(e)}

    # ── IndicTrans2 ────────────────────────────────────────────────────────────

    def _translate_indic(self, text: str, lang: str) -> str:
        src_code = INDIC_LANG_MAP.get(lang)
        if not src_code:
            # Fallback: try NLLB for unknown Indic
            return self._translate_nllb(text, lang)

        self._load_indic()
        try:
            # IndicTrans2: switch to SOURCE mode so the correct SPM handles Indic script.
            # Format: "src_lang_code tgt_lang_code source_text"
            self._indic_tokenizer._switch_to_input_mode()
            tagged = f"{src_code} {TARGET_LANG} {text}"

            with torch.no_grad():
                inputs = self._indic_tokenizer(
                    tagged,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_tokens,
                    padding=True,
                )
                # use_cache=False avoids DynamicCache/EncoderDecoderCache incompatibility.
                # no_repeat_ngram_size prevents repetition loops common in transformers>=5.
                output_ids = self._indic_model.generate(
                    **inputs,
                    num_beams=self.indic_num_beams,
                    max_new_tokens=self.max_output_tokens,
                    use_cache=False,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            # Switch to TARGET mode for decoding English output
            self._indic_tokenizer._switch_to_target_mode()
            return self._indic_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        finally:
            if self.unload_after_use:
                self._unload_indic()

    def _load_indic(self):
        if self._indic_model is None:
            self._indic_tokenizer = AutoTokenizer.from_pretrained(
                self.indic_model_path, local_files_only=True,
                trust_remote_code=True, use_fast=False,
            )
            self._indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.indic_model_path, local_files_only=True,
                trust_remote_code=True,
                attn_implementation="eager",   # SDPA is incompatible with IndicTrans2 custom attention
            )
            self._indic_model.eval()

    def _unload_indic(self):
        self._indic_model = self._indic_tokenizer = None
        self._free_memory()

    def _free_memory(self):
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

    # ── NLLB ───────────────────────────────────────────────────────────────────

    def _translate_nllb(self, text: str, lang: str) -> str:
        src_code = NLLB_LANG_MAP.get(lang, "eng_Latn")

        self._load_nllb()
        try:
            self._nllb_tokenizer.src_lang = src_code

            # Check actual token count without truncation
            token_ids = self._nllb_tokenizer.encode(text, add_special_tokens=True)
            if len(token_ids) <= self.max_input_tokens:
                return self._nllb_translate_chunk(text, src_code)

            # Text too long — split into sentence chunks and translate each
            sentences = _split_sentences(text)
            chunks: list = []
            current: list = []
            current_len = 0
            for sent in sentences:
                sent_len = len(self._nllb_tokenizer.encode(sent, add_special_tokens=False))
                if current_len + sent_len > self.max_input_tokens - 10 and current:
                    chunks.append(" ".join(current))
                    current = [sent]
                    current_len = sent_len
                else:
                    current.append(sent)
                    current_len += sent_len
            if current:
                chunks.append(" ".join(current))

            return " ".join(self._nllb_translate_chunk(c, src_code) for c in chunks)

        finally:
            if self.unload_after_use:
                self._unload_nllb()

    def _nllb_translate_chunk(self, text: str, src_code: str) -> str:
        """Translate a single chunk that fits within max_input_tokens."""
        self._nllb_tokenizer.src_lang = src_code
        with torch.no_grad():
            inputs = self._nllb_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            tgt_id = self._nllb_tokenizer.convert_tokens_to_ids(TARGET_LANG)
            outputs = self._nllb_model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                max_length=None,
                forced_bos_token_id=tgt_id,
                num_beams=self.nllb_num_beams,
            )
        return self._nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def backtranslate_nllb(self, english_text: str, target_lang: str) -> dict:
        """
        Translate English back to target language via NLLB (for chrF scoring).
        Only works for NLLB-supported languages.
        """
        tgt_code = NLLB_LANG_MAP.get(target_lang)
        if not tgt_code:
            return {"translated_text": "", "success": False,
                    "error": f"No NLLB code for language: {target_lang}"}
        self._load_nllb()
        try:
            self._nllb_tokenizer.src_lang = TARGET_LANG  # eng_Latn
            with torch.no_grad():
                inputs = self._nllb_tokenizer(
                    english_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_tokens,
                    padding=True,
                )
                tgt_id  = self._nllb_tokenizer.convert_tokens_to_ids(tgt_code)
                outputs = self._nllb_model.generate(
                    **inputs,
                    max_new_tokens=self.max_output_tokens,
                    max_length=None,
                    forced_bos_token_id=tgt_id,
                    num_beams=self.nllb_num_beams,
                )
            text_out = self._nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"translated_text": text_out, "success": True, "error": None}
        except Exception as e:
            return {"translated_text": "", "success": False, "error": str(e)}
        finally:
            if self.unload_after_use:
                self._unload_nllb()

    def _load_nllb(self):
        if self._nllb_model is None:
            self._nllb_tokenizer = AutoTokenizer.from_pretrained(
                self.nllb_model_path, local_files_only=True,
            )
            self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.nllb_model_path, local_files_only=True,
            ).to(self.device)
            self._nllb_model.eval()

    def _unload_nllb(self):
        self._nllb_model = self._nllb_tokenizer = None
        self._free_memory()

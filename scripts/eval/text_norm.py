"""
text_norm.py – the single text normaliser + WER/CER for every VANI evaluator.

Why this module exists
----------------------
Three evaluators each carried their own copy of `normalise()`:
compare_all_models.py, eval_seamless_ft.py and score_wer_robustness.py. None of them
segmented CJK. Because jiwer tokenises on whitespace and FLEURS Mandarin references are
character-spaced, a *perfect* SeamlessM4T transcript scored 100.0% WER, and the Whisper
large-v3 baseline scored 100.03%. Both went into docs/model_comparison_results.json, the
finetune report, the slide decks and the paper. Verified 2026-07-10: char-segmented,
those same clips score 0.0% and 8.48%.

Duplication was the root cause. One copy fixed is one copy of three. So: one definition,
imported everywhere. Do not re-implement `normalise` in an evaluator.

Rules
-----
* NFC first. The Kashmiri baseline was once reported at 96.87% WER purely because
  references and hypotheses used different Unicode compositions of the same
  Perso-Arabic graphemes.
* Strip punctuation, including Arabic/Urdu variants, and zero-width characters.
* Lowercase only Latin-script output (English translations).
* Character-segment scripts with no orthographic word boundaries, so that
  whitespace tokenisation == character tokenisation.

Always pass the DATASET language code (zh, ks, pa ...), never a model-specific code
like SeamlessM4T's "cmn" or "pan". compare_all_models.run_seamless_asr used to pass
"cmn" while its references were normalised as "zh" -- the two sides never went through
the same normaliser.

Known gap
---------
Number formatting is NOT normalised. SeamlessM4T writes 二零一一年八月 where FLEURS
writes 2011年8月; that is a spelling difference, not a recognition error, and it
inflates Seamless's zh error. Fixing it (e.g. cn2an) would only strengthen Seamless.
"""

import re
import unicodedata

# Scripts written without spaces between words.
NO_WORD_BOUNDARY = {"zh", "yue", "ja", "th"}

_PUNCT     = re.compile(r"[،,؟?!\.؛;:\-–—۔]")
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_WS        = re.compile(r"\s+")


def charseg(text: str) -> str:
    """Space-separate characters, dropping existing spaces."""
    return " ".join(c for c in text.replace(" ", "") if c.strip())


def normalise(text: str, lang: str) -> str:
    """Normalise `text` for scoring. `lang` is the dataset code, not a model code."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text.strip())
    text = _PUNCT.sub(" ", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _WS.sub(" ", text).strip()
    if lang in ("en", "eng"):
        text = text.lower()
    if lang in NO_WORD_BOUNDARY:
        text = charseg(text)
    return text


def _pairs(preds, refs, lang):
    valid = [(p, r) for p, r in zip(preds, refs) if p is not None and r]
    if not valid:
        return None, None
    return ([normalise(p, lang) for p, _ in valid],
            [normalise(r, lang) for _, r in valid])


def compute_wer(preds, refs, lang):
    """WER %, both sides through the same normaliser. None if nothing scorable."""
    import jiwer
    p_list, r_list = _pairs(preds, refs, lang)
    if p_list is None:
        return None
    try:
        return round(jiwer.wer(r_list, p_list) * 100, 2)
    except ValueError:
        return 100.0        # an all-empty hypothesis side is a real 100%, not a crash


def compute_cer(preds, refs, lang):
    """CER %. The honest headline for zh (whitespace WER is fragile) and for ks."""
    import jiwer
    p_list, r_list = _pairs(preds, refs, lang)
    if p_list is None:
        return None
    p_list = [p.replace(" ", "") for p in p_list]
    r_list = [r.replace(" ", "") for r in r_list]
    try:
        return round(jiwer.cer(r_list, p_list) * 100, 2)
    except ValueError:
        return 100.0

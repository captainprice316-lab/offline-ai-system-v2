"""
language_module.py – Language identification and routing
---------------------------------------------------------
Key improvements over original:
  • VOTING SYSTEM: combines Whisper's language prob + FastText score
    rather than trusting either alone (fixes the circular dependency flaw)
  • Correct ISO codes: doi (Dogri), ne (Nepali), my (Burmese), ur (Urdu)
  • Nepali and Burmese now properly routed
  • DialectDetector now explicitly handles Romanised Indic (Hinglish, Panjabi in Latin)
  • LanguageRouter uses confidence-weighted voting, not brittle if/else chains
  • Low-confidence fallback: flags "uncertain" for human review in ISUM
"""

import fasttext
from collections import Counter


# ── Chinese variant detection (no external library required) ───────────────────
# Characters that appear exclusively (or near-exclusively) in Traditional Chinese
_TRADITIONAL_CHARS = set(
    "學語國時來說個們見車東開關問電話發長從體為這還對與頭愛點書歲"
    "機樣歡義興實際農會邊號現後務動廣歷圖劃經紀幾義臺灣傳歸華鄉"
    "親愛謝謙嚴讓歲節�憶應當層總種雖雲連運過這還邊號際農務動廣"
)
# Characters that appear exclusively (or near-exclusively) in Simplified Chinese
_SIMPLIFIED_CHARS = set(
    "学语国时来说个们见车东开关问电话发长从体为这还对与头爱点书岁"
    "机样欢义兴实际农会边号现后务动广历图划经纪几义台湾传归华乡"
    "亲爱谢谦严让岁节忆应当层总种虽云连运过这还边号际农务动广"
)

def detect_chinese_variant(text: str) -> str:
    """
    Returns 'zh-tw' if Traditional Chinese characters dominate, else 'zh'.
    Uses character-set counting — no external library needed.
    Falls back to 'zh' (Simplified) on ambiguous or short text.
    """
    trad = sum(1 for ch in text if ch in _TRADITIONAL_CHARS)
    simp = sum(1 for ch in text if ch in _SIMPLIFIED_CHARS)
    if trad == 0 and simp == 0:
        return "zh"
    # Require at least 60% Traditional to call it Traditional
    if trad / (trad + simp) >= 0.60:
        return "zh-tw"
    return "zh"


# ── ISO-639 language families ──────────────────────────────────────────────────
# IndicTrans2 has transformers>=5.x cache incompatibility.
# Route all supported languages through NLLB-200 which works correctly.
INDIC_LANGS  = {"doi"}   # Dogri only — not in NLLB-200 distilled, keep IndicTrans2
NLLB_LANGS   = {"hi", "ur", "ne", "bn", "mai", "ks", "sd", "si",         # Indic via NLLB
                "ps", "zh", "my", "bo", "fa", "ar", "tg", "uz", "kk"}    # Others
# pa excluded: fine-tuned whisper-large-v3-pa-ct2 outputs English directly — no translation needed
ENGLISH_LIKE = {"en"}

# Languages that use Arabic/Nastaliq script — used by Script-Cascade Algorithm
ARABIC_SCRIPT_LANGS = {"ur", "ks", "sd", "ps", "fa", "ar"}

# Whisper uses slightly different codes for some languages
WHISPER_TO_ISO = {
    "chinese":  "zh",
    "pashto":   "ps",
    "burmese":  "my",
    "hindi":    "hi",
    "punjabi":  "pa",
    "urdu":     "ur",
    "nepali":   "ne",
    "dogri":    "doi",
    "mandarin": "zh",
}


class FastTextLangDetector:

    def __init__(self, model_path: str = "models/langid/lid.176.bin"):
        self.model = fasttext.load_model(model_path)

    def detect(self, text: str) -> dict:
        text = (text or "").replace("\n", " ").strip()
        if not text or len(text) < 5:
            return {"language": "unknown", "confidence": 0.0}

        labels, scores = self.model.predict(text, k=3)
        top_lang  = labels[0].replace("__label__", "")
        top_score = float(scores[0])

        # Normalise Whisper-style names that FastText may return
        top_lang = WHISPER_TO_ISO.get(top_lang, top_lang)

        return {
            "language":   top_lang,
            "confidence": round(top_score, 3),
            "top3": [
                {
                    "language":   l.replace("__label__", ""),
                    "confidence": round(float(s), 3),
                }
                for l, s in zip(labels, scores)
            ],
        }


class DialectDetector:
    """
    Script-based heuristic.
    Operates on Whisper output text which may be Romanised even for Indic
    languages – so we look at Unicode script ratios as soft signals only.
    """

    def detect_code_mix(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return self._empty()

        latin = devanagari = arabic = chinese = gurmukhi = 0

        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                devanagari += 1
            elif 0x0A00 <= cp <= 0x0A7F:
                gurmukhi += 1          # Punjabi Gurmukhi script
            elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F):
                arabic += 1
            elif 0x4E00 <= cp <= 0x9FFF:
                chinese += 1
            elif ch.isascii() and ch.isalpha():
                latin += 1

        total = max(len([c for c in text if c.strip()]), 1)

        lr  = latin       / total
        dr  = devanagari  / total
        ar  = arabic      / total
        cr  = chinese     / total
        gr  = gurmukhi    / total

        if cr > 0.20:
            return self._result("chinese_script",    0.95, lr, dr, ar, cr)
        if ar > 0.20:
            return self._result("arabic_script",     0.85, lr, dr, ar, cr)
        if gr > 0.20:
            return self._result("gurmukhi_indic",    0.85, lr, dr, ar, cr)
        if dr > 0.20:
            return self._result("devanagari_indic",  0.85, lr, dr, ar, cr)
        # Romanised Indic: mostly Latin but FastText/Whisper says Indic language
        if lr > 0.50:
            return self._result("romanised_latin",   0.70, lr, dr, ar, cr)

        return self._result("unknown", 0.40, lr, dr, ar, cr)

    @staticmethod
    def _result(dialect, conf, lr, dr, ar, cr):
        return {
            "dialect":          dialect,
            "dialect_confidence": conf,
            "latin_ratio":      round(lr, 3),
            "devanagari_ratio": round(dr, 3),
            "arabic_ratio":     round(ar, 3),
            "chinese_ratio":    round(cr, 3),
        }

    @staticmethod
    def _empty():
        return {
            "dialect": "unknown", "dialect_confidence": 0.0,
            "latin_ratio": 0.0, "devanagari_ratio": 0.0,
            "arabic_ratio": 0.0, "chinese_ratio": 0.0,
        }


class LanguageRouter:
    """
    3-way confidence-weighted voting: Whisper (ASR) + FastText (text) + MMS-LID (audio).

    Strategy
    --------
    1. Normalise all codes to ISO-639-1.
    2. Hard overrides for unambiguous script signals.
    3. Majority vote across all three available sources.
    4. Unanimous agreement → very high confidence.
    5. 2-of-3 agreement → majority confidence.
    6. All disagree → pick highest individual confidence source.
    7. Flag low-confidence results for human review.
    """

    CONFIDENCE_THRESHOLD = 0.60   # below this → flag as uncertain

    def __init__(self, confidence_threshold: float = None):
        if confidence_threshold is not None:
            self.CONFIDENCE_THRESHOLD = confidence_threshold

    def detect_family(
        self,
        whisper_lang:      str,
        transcript:        str,
        fasttext_lang:     str   = None,
        fasttext_conf:     float = 0.0,
        whisper_lang_prob: float = 0.0,
        dialect:           str   = None,
        mms_lang:          str   = None,
        mms_conf:          float = 0.0,
    ) -> dict:

        wl = WHISPER_TO_ISO.get((whisper_lang or "").lower(), (whisper_lang or "").lower())
        fl = WHISPER_TO_ISO.get((fasttext_lang or "").lower(), (fasttext_lang or "").lower())
        ml = (mms_lang or "").lower()
        d  = (dialect or "").lower()

        # ── Hard overrides from script ─────────────────────────────────────────
        if d == "chinese_script" or fl.startswith("zh") or wl.startswith("zh") or ml.startswith("zh"):
            variant = detect_chinese_variant(transcript)
            return self._make(variant, "nllb", 0.95, "chinese",
                              vote_note=f"chinese-script ({variant})")

        if d == "arabic_script" and (fl == "ps" or wl == "ps" or ml == "ps"):
            return self._make("ps", "nllb", 0.90, "arabic")

        # ── Build candidate list ───────────────────────────────────────────────
        candidates = []
        if wl and wl not in ("", "unknown"):
            candidates.append((wl, whisper_lang_prob, "whisper"))
        if fl and fl not in ("", "unknown"):
            candidates.append((fl, fasttext_conf, "fasttext"))
        if ml and ml not in ("", "unknown"):
            candidates.append((ml, mms_conf, "mms"))

        # ── Punjabi correction: Whisper often returns 'hi' for Punjabi audio ──
        # Gurmukhi script is unambiguous; FastText/MMS are more reliable here.
        if d == "gurmukhi_indic":
            pa_conf = max(fasttext_conf if fl == "pa" else 0.0,
                          mms_conf      if ml == "pa" else 0.0)
            return self._make("pa", "nllb",
                              max(pa_conf, 0.75), "indic", False,
                              "gurmukhi-script override")
        if (fl == "pa" or ml == "pa") and wl == "hi":
            pa_conf = max(fasttext_conf if fl == "pa" else 0.0,
                          mms_conf      if ml == "pa" else 0.0)
            if pa_conf >= 0.55:
                return self._make("pa", "nllb", pa_conf, "indic", False,
                                  "pa-override (Whisper hi/pa confusion)")

        # ── Script-Cascade: Arabic script ─────────────────────────────────────
        # Whisper frequently misidentifies Nastaliq text; FastText lid.176 has
        # limited Arabic-script coverage and returns 'unknown' for Urdu ~90% of
        # the time.  When the transcript is Arabic-script, restrict the vote to
        # Arabic-script-compatible languages {ur, ks, sd, ps, fa, ar} before
        # applying confidence-weighted majority voting.
        if d == "arabic_script":
            ar_cands = [(lang, conf, src) for lang, conf, src in candidates
                        if lang in ARABIC_SCRIPT_LANGS]
            if ar_cands:
                ac = Counter(lang for lang, _, _ in ar_cands)
                best_al, best_av = ac.most_common(1)[0]
                conf_al = (sum(c for lang, c, _ in ar_cands if lang == best_al)
                           / best_av)
                return self._make(best_al, "nllb", max(conf_al, 0.70), "arabic",
                                  False, "arabic-script filtered vote")
            # No Arabic-script source voted → default to ur (South Asian SIGINT)
            return self._make("ur", "nllb", 0.65, "arabic", True,
                              "arabic-script default→ur")

        if not candidates:
            return self._make("unknown", "none", 0.3, "unknown", True)

        # ── Vote counting ──────────────────────────────────────────────────────
        vote_counts = Counter(lang for lang, _, _ in candidates)
        majority_lang, majority_votes = vote_counts.most_common(1)[0]

        if majority_votes == len(candidates):
            # Unanimous agreement — boost confidence
            avg_conf = sum(c for _, c, _ in candidates) / len(candidates)
            conf = min(0.99, avg_conf * 1.10)
            return self._route_lang(majority_lang, conf, d,
                                    note=f"unanimous ({len(candidates)}/{len(candidates)})")

        if majority_votes >= 2:
            # Majority agreement — use average confidence of agreeing sources
            agreeing = [(lang, c) for lang, c, _ in candidates if lang == majority_lang]
            conf = sum(c for _, c in agreeing) / len(agreeing)
            return self._route_lang(majority_lang, conf, d,
                                    note=f"majority ({majority_votes}/{len(candidates)})")

        # ── All disagree: pick highest-confidence source ───────────────────────
        best_lang, best_conf, best_src = max(candidates, key=lambda x: x[1])
        return self._route_lang(best_lang, best_conf * 0.85, d,
                                note=f"no-consensus (best={best_src})")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _route_lang(self, lang: str, conf: float, dialect: str,
                    note: str = "") -> dict:
        uncertain = conf < self.CONFIDENCE_THRESHOLD

        if lang == "en":
            return self._make("en", "none", conf, "latin", uncertain, note)

        if lang in INDIC_LANGS:
            return self._make(lang, "indictrans2", conf, "indic", uncertain, note)

        if lang in NLLB_LANGS:
            return self._make(lang, "nllb", conf, _script_for(lang), uncertain, note)

        # Unknown but script gives a clue
        if dialect == "devanagari_indic":
            return self._make(lang or "hi", "indictrans2", conf * 0.8, "indic", True, note)
        if dialect == "arabic_script":
            return self._make(lang or "ar", "nllb", conf * 0.8, "arabic", True, note)

        return self._make(lang or "unknown", "none", conf, "unknown", True, note)

    @staticmethod
    def _make(lang, route, conf, script, uncertain=False, vote_note=""):
        return {
            "final_language": lang,
            "route":          route,
            "confidence":     round(conf, 3),
            "script_hint":    script,
            "uncertain":      uncertain,
            "vote_note":      vote_note,
        }


def _script_for(lang: str) -> str:
    scripts = {
        "ps": "arabic", "ar": "arabic", "fa": "arabic",
        "zh": "chinese", "my": "burmese",
    }
    return scripts.get(lang, "unknown")

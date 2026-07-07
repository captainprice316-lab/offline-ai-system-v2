"""
keyword_module.py – Unified keyword detection (replaces both keyword_module.py
                    and keyboard_module.py)
---------------------------------------------------------------------------
Key improvements over original:
  • Single unified file (original had two conflicting implementations)
  • Regex word-boundary matching on BOTH transcript and translation
  • Indic language keywords added (Hindi, Punjabi, Pashto transliterations)
  • Segment-level timing attached to every alert
  • Threat level scoring: each category has a severity weight
  • Returns structured alerts usable by ISUM generator
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class KeywordAlert:
    category:    str
    severity:    str          # critical / high / medium / low
    matched_word: str
    matched_in:  str          # "transcript" | "translation"
    start_sec:   float
    end_sec:     float
    segment_text: str
    confidence:  float = 0.0
    effective_severity: str = ""  # severity downgraded one level if ASR confidence < 0.40
    coded:          bool = False  # matched a coded-terminology term (possible tradecraft)
    decoded_meaning: str = ""     # hidden meaning, e.g. "potato" -> "grenades / explosives"


# ── Keyword dictionary ─────────────────────────────────────────────────────────
# Each entry: category → (severity, [keywords in English + Indic transliterations])
KEYWORD_MAP = {
    # ── Threat / hostile activity ──────────────────────────────────────────────
    "enemy_activity": ("critical", [
        # English
        "enemy", "hostile", "intruder", "infiltrator", "insurgent",
        "militant", "terrorist",
        # Hindi/Urdu transliteration
        "dushman", "dushman ki tukdi", "atankwadi",
        # Punjabi
        "dushman",
        # Pashto transliteration
        "dushman", "zalim",
    ]),

    # ── Attack / fire ──────────────────────────────────────────────────────────
    "attack": ("critical", [
        "attack", "attacking", "fire", "firing", "ambush", "engage",
        "assault", "strike", "rocket", "mortar", "grenade", "IED",
        # Hindi
        "hamla", "goli", "goli chalao", "aag",
        # Pashto transliteration
        "wor", "topak",
    ]),

    # ── Weapons / equipment ───────────────────────────────────────────────────
    "weapons": ("high", [
        "gun", "rifle", "AK", "RPG", "explosive", "ammunition", "ammo",
        "bomb", "mine", "weapon",
        # Hindi
        "bandook", "hathiyar", "bomb",
    ]),

    # ── Movement ──────────────────────────────────────────────────────────────
    "movement": ("high", [
        "advance", "advancing", "retreat", "retreating", "fallback",
        "withdraw", "move", "moving", "crossing", "flanking", "encircling",
        # Hindi
        "aage badho", "peeche hato", "nikal",
    ]),

    # ── Location / coordinates ────────────────────────────────────────────────
    "location": ("high", [
        "north", "south", "east", "west", "sector", "checkpoint",
        "grid", "coordinates", "border", "LOC", "LAC", "ridge", "peak",
        "nala", "village", "bridge", "road",
        # Hindi
        "uttar", "dakshin", "poorab", "paschim", "seema",
    ]),

    # ── Support / reinforcement ───────────────────────────────────────────────
    "support_request": ("high", [
        "support", "backup", "reinforcement", "casualty", "wounded",
        "medevac", "extract", "extraction", "helicopter",
        # Hindi
        "madad", "karo madad", "zakhmi",
    ]),

    # ── Command / control ─────────────────────────────────────────────────────
    "command": ("medium", [
        "orders", "command", "over", "roger", "copy", "wilco", "abort",
        "hold", "hold position", "stand by", "affirmative", "negative",
        # Hindi
        "ruko", "theek hai", "samajh",
    ]),

    # ── Communication / callsign markers ──────────────────────────────────────
    "comms": ("low", [
        "callsign", "frequency", "channel", "radio", "comms", "out",
        "send", "say again", "repeat",
    ]),
}

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


class KeywordDetector:

    def __init__(self, custom_keywords: dict = None, dictionary_path: str = None):
        self.keyword_map = dict(KEYWORD_MAP)
        self._coded_cats: set = set()          # categories flagged as coded terminology
        self._glossaries: Dict[str, dict] = {}  # category -> {word: hidden meaning}

        # Load from JSON dictionary file if provided
        if dictionary_path:
            import json
            from pathlib import Path
            try:
                with open(dictionary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for category, entry in data.get("categories", {}).items():
                    severity = entry.get("severity", "medium")
                    all_kws  = []
                    for lang_kws in entry.get("keywords", {}).values():
                        all_kws.extend(lang_kws)
                    self.keyword_map[category] = (severity, all_kws)
                    # capture coded-terminology glossary (word -> hidden meaning)
                    if entry.get("coded") and entry.get("glossary"):
                        self._coded_cats.add(category)
                        self._glossaries[category] = {
                            k.lower(): v for k, v in entry["glossary"].items()
                        }
            except Exception as e:
                import logging
                logging.getLogger("vani.keywords").warning(
                    f"Failed to load keyword dictionary '{dictionary_path}': {e} "
                    "— falling back to built-in keywords"
                )

        if custom_keywords:
            self.keyword_map.update(custom_keywords)

        # Pre-compile all patterns
        self._patterns: Dict[str, tuple] = {}
        for category, (severity, words) in self.keyword_map.items():
            # Use (?<!\w)/(?!\w) instead of \b: Indic combining vowel signs
            # (ा ी ु े …) are non-word chars, so a trailing \b cannot anchor
            # after them — \bहमला\b silently fails to match "हमला". The
            # lookaround form matches whenever the keyword is not flanked by a
            # word char, which works for both Latin and Indic/Nastaliq scripts.
            compiled = [
                (w, re.compile(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)"))
                for w in words
            ]
            self._patterns[category] = (severity, compiled)

    # ── public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        transcript:  str,
        translation: str,
        segments:    list,
    ) -> dict:
        """
        Scan both transcript (original language) and translation (English).

        Returns
        -------
        {
            alerts:          list of KeywordAlert dicts,
            summary_counts:  {category: count},
            threat_level:    "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "CLEAR",
            top_categories:  sorted list of triggered categories,
        }
        """
        alerts: List[KeywordAlert] = []

        # Build a lookup: text_fragment → segment timing
        seg_index = _build_segment_index(segments)

        for source_label, text in [("transcript", transcript), ("translation", translation)]:
            if not text:
                continue
            text_lower = text.lower()

            for category, (severity, compiled) in self._patterns.items():
                for word, pattern in compiled:
                    for match in pattern.finditer(text_lower):
                        # Find which segment this match falls in
                        seg      = _find_segment(match.start(), text_lower, seg_index, segments)
                        seg_conf = seg.get("confidence", 1.0) if seg else 1.0
                        alerts.append(KeywordAlert(
                            category=           category,
                            severity=           severity,
                            matched_word=       word,
                            matched_in=         source_label,
                            start_sec=          seg.get("start", 0.0) if seg else 0.0,
                            end_sec=            seg.get("end",   0.0) if seg else 0.0,
                            segment_text=       seg.get("text",  "")  if seg else "",
                            confidence=         seg_conf,
                            effective_severity= _downgrade_severity(severity, seg_conf),
                            coded=              category in self._coded_cats,
                            decoded_meaning=    self._glossaries.get(category, {}).get(word.lower(), ""),
                        ))

        # Deduplicate (same category + segment + word)
        unique = _deduplicate(alerts)

        summary_counts = {}
        for a in unique:
            summary_counts[a.category] = summary_counts.get(a.category, 0) + 1

        top_cats = sorted(
            summary_counts.keys(),
            key=lambda c: SEVERITY_ORDER.get(self._patterns[c][0], 0),
            reverse=True,
        )

        threat_level = _compute_threat_level(unique)

        return {
            "alerts":         [asdict(a) for a in unique],
            "summary_counts": summary_counts,
            "threat_level":   threat_level,
            "top_categories": top_cats,
        }


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_segment_index(segments: list) -> list:
    """Build list of (cumulative_char_offset, segment) for approximate mapping."""
    index = []
    offset = 0
    for seg in segments:
        t = (seg.get("text") or "").lower()
        index.append((offset, offset + len(t), seg))
        offset += len(t) + 1   # +1 for the space join
    return index


def _find_segment(char_pos: int, full_text: str, seg_index: list, segments: list):
    """Return the segment that contains the character at char_pos."""
    for start, end, seg in seg_index:
        if start <= char_pos < end:
            return seg
    return segments[0] if segments else None


def _deduplicate(alerts: List[KeywordAlert]) -> List[KeywordAlert]:
    seen = set()
    out  = []
    for a in alerts:
        key = (a.category, a.matched_word, a.start_sec, a.matched_in)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _downgrade_severity(severity: str, confidence: float) -> str:
    """Drop severity one level when ASR segment confidence is very low (<0.40)."""
    if confidence >= 0.40:
        return severity
    _order = ["low", "medium", "high", "critical"]
    idx = _order.index(severity) if severity in _order else -1
    return _order[idx - 1] if idx > 0 else severity


def _compute_threat_level(alerts: List[KeywordAlert]) -> str:
    if not alerts:
        return "CLEAR"
    # Use effective_severity (confidence-adjusted) for threat level
    max_sev = max(
        SEVERITY_ORDER.get(a.effective_severity or a.severity, 0) for a in alerts
    )
    return {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(max_sev, "CLEAR")

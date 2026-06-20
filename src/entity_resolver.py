"""
entity_resolver.py — Callsign / entity name normalisation and deduplication.

Problem: the same entity appears in who_field as many surface forms:
  "Alpha 3", "Alpha-3", "ALPHA 3", "Alpha Three", "Alpha3"  → should be one actor.

Strategy:
  1. Normalise each raw name to a canonical key:
       - lowercase
       - written numbers → digits  ("one" → "1", "three" → "3")
       - collapse all separators (-, _, /) and extra spaces to single space
       - strip residual punctuation
  2. Group all raw names that share the same canonical key.
  3. Pick the most frequent raw form as the display name.
  4. Expose an `aliases` list so the UI can show all variants.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ── Number-word → digit map ────────────────────────────────────────────────────
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_NUM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _NUM_WORDS) + r")\b",
    re.IGNORECASE,
)

# ── Common abbreviation expansions (for canonical display only) ────────────────
_ABBREV_EXPAND = {
    r"\btf\b":  "Task Force",
    r"\bsf\b":  "Special Forces",
    r"\bqrf\b": "Quick Reaction Force",
    r"\bop\b":  "OP",
}


def normalize_callsign(name: str) -> str:
    """
    Return a canonical key for deduplication — NOT for display.
    Two names with the same key are treated as the same entity.
    """
    s = name.strip().lower()

    # Written numbers → digits
    s = _NUM_PATTERN.sub(lambda m: _NUM_WORDS[m.group(1).lower()], s)

    # Collapse separators to space
    s = re.sub(r"[-_/\\.,]+", " ", s)

    # Strip non-alphanumeric (except spaces)
    s = re.sub(r"[^\w\s]", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _best_display_name(raw_names: List[Tuple[str, int]]) -> str:
    """
    Pick the display name from a list of (raw_name, frequency) tuples.
    Prefers the most frequent; breaks ties by choosing the longer/more complete form.
    """
    if not raw_names:
        return ""
    raw_names.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return raw_names[0][0]


def resolve_entities(
    actors: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Takes the raw `actors` dict from get_actor_profiles() keyed by name.lower()
    and returns a new dict keyed by canonical key, with:
      - "name"       : best display form
      - "aliases"    : sorted list of all observed surface forms
      - "appearances": merged list from all variants
      - all other fields preserved from the most-represented variant

    Input dict values must have at least: "appearances", "name", "callsign_type".
    """
    # Group raw keys by their canonical key
    canonical_groups: Dict[str, List[str]] = defaultdict(list)
    for raw_key in actors:
        canon = normalize_callsign(raw_key)
        canonical_groups[canon].append(raw_key)

    resolved: Dict[str, dict] = {}

    for canon, raw_keys in canonical_groups.items():
        # Merge appearances from all variants
        all_appearances = []
        name_freq: List[Tuple[str, int]] = []

        for rk in raw_keys:
            data = actors[rk]
            all_appearances.extend(data["appearances"])
            name_freq.append((data.get("name", rk), len(data["appearances"])))

        # Sort appearances by timestamp
        all_appearances.sort(key=lambda a: a.get("timestamp", ""))

        # Pick base data from the most-frequent raw variant
        base = max(
            (actors[rk] for rk in raw_keys),
            key=lambda d: len(d["appearances"]),
        ).copy()

        base["name"]        = _best_display_name(name_freq)
        base["appearances"] = all_appearances
        base["aliases"]     = sorted(
            {actors[rk].get("name", rk) for rk in raw_keys} - {base["name"]}
        )
        base["canonical_key"] = canon

        resolved[canon] = base

    return resolved

"""
search.py – Transcript search with fuzzy matching and ISUM-aware output
------------------------------------------------------------------------
Key improvements over original:
  • Searches both transcript AND translation (already done) AND ISUM fields
  • Fuzzy matching: handles transliteration variants (e.g. "hamla" ~ "hamlaa")
  • Returns results ranked by threat level + keyword match count
  • Time-range filter: search within a specific time window
  • Word-level timestamp search when word_timestamps are available
"""

import json
import sys
import re
from pathlib import Path
from difflib import SequenceMatcher


# ── Fuzzy similarity ───────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _fuzzy_match(keyword: str, text: str, threshold: float = 0.82) -> bool:
    """
    Returns True if keyword appears in text either exactly (word boundary)
    or with fuzzy similarity ≥ threshold for each word in text.
    """
    kw = keyword.lower().strip()
    if not kw:
        return False

    # Exact word boundary match first
    if re.search(r"\b" + re.escape(kw) + r"\b", text.lower()):
        return True

    # Fuzzy match against individual words in text
    for word in text.lower().split():
        if len(word) >= 4 and _similarity(kw, word) >= threshold:
            return True

    return False


# ── Search function ────────────────────────────────────────────────────────────

THREAT_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "CLEAR": 0}


def search_transcripts(
    keyword:     str,
    output_dir:  Path,
    fuzzy:       bool  = True,
    time_from:   float = None,   # seconds
    time_to:     float = None,
) -> list:
    """
    Search all result JSON files for keyword.

    Returns list of match dicts, ranked by threat level.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    results = []

    for json_file in sorted(output_dir.glob("*_result.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            results.append({"file": json_file.name, "error": str(e)})
            continue

        transcript  = data.get("transcript", "")
        translation = data.get("translation", {})
        trans_text  = translation.get("translated_text", "") if isinstance(translation, dict) else str(translation)
        segments    = data.get("segments", [])
        isum        = data.get("isum", {})

        match_fn = _fuzzy_match if fuzzy else lambda kw, txt: kw.lower() in txt.lower()

        in_transcript  = match_fn(keyword, transcript)
        in_translation = match_fn(keyword, trans_text)
        in_isum        = match_fn(keyword, isum.get("assessment", "") + " " +
                                           isum.get("what", "") + " " +
                                           isum.get("who", ""))

        if not (in_transcript or in_translation or in_isum):
            continue

        # Find matching segments (with optional time range filter)
        matched_segs = []
        for seg in segments:
            start = seg.get("start", 0.0)
            end   = seg.get("end",   0.0)

            if time_from is not None and end < time_from:
                continue
            if time_to is not None and start > time_to:
                continue

            seg_text = seg.get("text", "")
            if match_fn(keyword, seg_text):
                entry = {
                    "start":      start,
                    "end":        end,
                    "text":       seg_text,
                    "confidence": seg.get("confidence", 0.0),
                }
                # Word-level match if available
                if "words" in seg:
                    matched_words = [
                        w for w in seg["words"]
                        if match_fn(keyword, w.get("word", ""))
                    ]
                    if matched_words:
                        entry["matched_words"] = matched_words
                matched_segs.append(entry)

        results.append({
            "file":             json_file.name,
            "audio_file":       data.get("audio_file", ""),
            "timestamp_utc":    data.get("timestamp_utc", ""),
            "final_language":   data.get("final_language", ""),
            "threat_level":     data.get("threat_level", "CLEAR"),
            "matched_in":       (["transcript"]  if in_transcript  else []) +
                                (["translation"] if in_translation else []) +
                                (["isum"]        if in_isum        else []),
            "matched_segments": matched_segs,
            "transcript_snippet": transcript[:150],
            "translation_snippet": trans_text[:150],
            "isum_assessment":  isum.get("assessment", ""),
            "top_categories":   data.get("top_categories", []),
        })

    # Rank by threat level (highest first), then by number of segment matches
    results.sort(
        key=lambda x: (
            THREAT_ORDER.get(x.get("threat_level", "CLEAR"), 0),
            len(x.get("matched_segments", [])),
        ),
        reverse=True,
    )

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    root       = Path(__file__).resolve().parent.parent
    output_dir = root / "output"

    if len(sys.argv) < 2:
        print("Usage: python src/search.py <keyword> [--exact] [--from <sec>] [--to <sec>]")
        sys.exit(1)

    keyword = sys.argv[1]
    fuzzy   = "--exact" not in sys.argv

    time_from = time_to = None
    if "--from" in sys.argv:
        idx = sys.argv.index("--from")
        try:
            time_from = float(sys.argv[idx + 1])
        except Exception:
            pass
    if "--to" in sys.argv:
        idx = sys.argv.index("--to")
        try:
            time_to = float(sys.argv[idx + 1])
        except Exception:
            pass

    results = search_transcripts(keyword, output_dir, fuzzy=fuzzy,
                                  time_from=time_from, time_to=time_to)

    if not results:
        print(f"No matches found for: '{keyword}'")
        return

    match_type = "fuzzy" if fuzzy else "exact"
    print(f"\nSearch results for '{keyword}' [{match_type}]: {len(results)} file(s)\n")

    for item in results:
        if "error" in item:
            print(f"[ERROR] {item['file']}: {item['error']}")
            continue

        print(f"{'-'*60}")
        print(f"File          : {item['file']}")
        print(f"Audio         : {item['audio_file']}")
        print(f"Language      : {item['final_language']}")
        print(f"Threat Level  : {item['threat_level']}")
        print(f"Matched in    : {', '.join(item['matched_in'])}")
        print(f"Categories    : {', '.join(item['top_categories'])}")

        if item["isum_assessment"]:
            print(f"Assessment    : {item['isum_assessment']}")

        if item["matched_segments"]:
            print(f"Matched Segments ({len(item['matched_segments'])}):")
            for seg in item["matched_segments"][:5]:
                conf = f" [conf={seg['confidence']:.2f}]" if seg.get("confidence") else ""
                print(f"  [{seg['start']:.2f}s – {seg['end']:.2f}s]{conf} {seg['text']}")
                if "matched_words" in seg:
                    for w in seg["matched_words"]:
                        print(f"    -> word '{w['word']}' at {w['start']:.2f}s")
        else:
            print("  Matched in full transcript/translation but no specific segment.")

    print(f"{'-'*60}\n")


if __name__ == "__main__":
    main()

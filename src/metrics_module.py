"""
src/metrics_module.py – VANI Performance Metrics
--------------------------------------------------
Two tiers:
  1. Auto metrics  – computed from the pipeline result alone (no reference needed)
  2. Reference metrics – require analyst-provided ground truth
"""

import math
import statistics
from typing import Optional


# ── Tier 1: Auto metrics ───────────────────────────────────────────────────────

def compute_auto_metrics(result: dict) -> dict:
    """
    Compute all metrics that require no reference data.
    Returns a flat dict suitable for display.
    """
    return {
        "rtf":                _rtf(result),
        "segment_confidence": _segment_confidence(result),
        "model_agreement":    _model_agreement(result),
        "isum_completeness":  _isum_completeness(result),
        "stage_timings":      _stage_timings(result),
        "memory":             _memory(result),
        "vocab_richness":     _vocab_richness(result),
        "backtrans_chrf":     result.get("backtrans_chrf"),
    }


def _rtf(result: dict) -> dict:
    proc   = result.get("processing_time_s") or 0.0
    speech = result.get("total_speech_sec")  or 0.0
    if speech <= 0:
        return {"value": None, "label": "N/A", "note": "No speech detected"}
    rtf   = proc / speech
    grade = "FAST" if rtf < 0.5 else "OK" if rtf < 1.0 else "SLOW"
    return {
        "value":      round(rtf, 3),
        "proc_s":     round(proc, 2),
        "speech_s":   round(speech, 2),
        "grade":      grade,
        "note":       f"{proc:.1f}s processing / {speech:.1f}s speech",
    }


def _segment_confidence(result: dict) -> dict:
    segs  = result.get("segments", [])
    confs = [s["confidence"] for s in segs
             if isinstance(s.get("confidence"), (int, float))]
    nsp   = [s["no_speech_prob"] for s in segs
             if isinstance(s.get("no_speech_prob"), (int, float))]
    if not confs:
        return {"count": 0, "mean": None, "std": None,
                "pct_high": None, "pct_low": None,
                "mean_no_speech": None, "distribution": []}

    mean_c  = statistics.mean(confs)
    std_c   = statistics.stdev(confs) if len(confs) > 1 else 0.0
    pct_hi  = sum(1 for c in confs if c >= 0.80) / len(confs) * 100
    pct_lo  = sum(1 for c in confs if c <  0.50) / len(confs) * 100
    mean_ns = round(statistics.mean(nsp), 3) if nsp else None

    grade = ("HIGH"   if mean_c >= 0.80 else
             "MEDIUM" if mean_c >= 0.55 else "LOW")

    # Bucket into 10 bins for histogram
    buckets = [0] * 10
    for c in confs:
        idx = min(int(c * 10), 9)
        buckets[idx] += 1

    return {
        "count":          len(confs),
        "mean":           round(mean_c, 3),
        "std":            round(std_c,  3),
        "pct_high":       round(pct_hi, 1),
        "pct_low":        round(pct_lo, 1),
        "mean_no_speech": mean_ns,
        "grade":          grade,
        "distribution":   confs,        # raw list for detailed histogram
        "buckets":        buckets,       # pre-bucketed [0.0-0.1, ..., 0.9-1.0]
    }


def _model_agreement(result: dict) -> dict:
    wl   = result.get("whisper_language")   or ""
    fl   = result.get("fasttext_language")  or ""
    ml   = result.get("mms_language")       or ""
    wp   = result.get("whisper_language_probability") or 0.0
    fp   = result.get("fasttext_confidence")          or 0.0
    mp   = result.get("mms_confidence")               or 0.0
    route = result.get("translation_route", "-")

    if not wl or not fl:
        return {"agree": None, "note": "Insufficient language data"}

    # 2-way or 3-way depending on MMS availability
    langs = [l for l in [wl, fl, ml] if l]
    from collections import Counter
    majority_lang = Counter(langs).most_common(1)[0][0] if langs else wl
    n_agree = sum(1 for l in langs if l == majority_lang)
    agree   = n_agree == len(langs)

    confs   = [c for c in [wp, fp, mp] if c > 0]
    avg_c   = round(sum(confs) / len(confs), 3) if confs else 0.0
    delta   = round(max(confs) - min(confs), 3) if len(confs) > 1 else 0.0

    ensemble = round(avg_c * (1.0 if agree else 0.75 if n_agree >= 2 else 0.6), 3)
    grade    = ("STRONG"   if ensemble >= 0.80 else
                "MODERATE" if ensemble >= 0.55 else "WEAK")

    vote_note = result.get("vote_note", "")
    sources   = f"Whisper={wl} p={wp:.2f}"
    if fl:
        sources += f"  FastText={fl} p={fp:.2f}"
    if ml:
        sources += f"  MMS={ml} p={mp:.2f}"

    return {
        "agree":        agree,
        "n_sources":    len(langs),
        "n_agree":      n_agree,
        "whisper_lang": wl,
        "whisper_prob": round(wp, 3),
        "fasttext_lang": fl,
        "fasttext_conf": round(fp, 3),
        "mms_lang":     ml or None,
        "mms_conf":     round(mp, 3) if mp else None,
        "confidence_delta": delta,
        "ensemble_score":   ensemble,
        "grade":            grade,
        "route":            route,
        "vote_note":        vote_note,
        "final_language":   result.get("final_language", "-"),
        "sources":          sources,
        "note": (f"All {len(langs)} models agree" if agree else
                 f"{n_agree}/{len(langs)} agree on {majority_lang.upper()}"),
    }


def _isum_completeness(result: dict) -> dict:
    isum = result.get("isum", {})
    DEFAULT = {
        "who":   "Not identified from intercept.",
        "what":  "No significant activity detected.",
        "where": "No specific location identified.",
        "when":  "No specific time referenced in intercept.",
    }
    field_status = {}
    for key, default in DEFAULT.items():
        val = (isum.get(key) or "").strip()
        field_status[key] = val and val != default

    score   = sum(field_status.values())
    pct     = round(score / len(DEFAULT) * 100)
    grade   = ("COMPLETE"  if pct == 100 else
               "PARTIAL"   if pct >= 50  else "SPARSE")

    # Keyword density
    transcript  = result.get("transcript", "") or ""
    word_count  = len(transcript.split()) if transcript else 0
    kw          = result.get("keyword_alerts", {})
    alert_count = len(kw.get("alerts", [])) if isinstance(kw, dict) else 0
    kw_density  = round(alert_count / word_count * 100, 2) if word_count > 0 else 0.0

    return {
        "score":        score,
        "max":          len(DEFAULT),
        "pct":          pct,
        "grade":        grade,
        "fields":       field_status,   # {who: True/False, ...}
        "kw_density":   kw_density,
        "kw_count":     alert_count,
        "word_count":   word_count,
        "threat_level": result.get("threat_level", "CLEAR"),
    }


def _stage_timings(result: dict) -> dict:
    timings = result.get("stage_timings") or {}
    if not timings:
        return {"available": False, "timings": {}, "total": None, "slowest": None}
    total   = round(sum(timings.values()), 2)
    slowest = max(timings, key=timings.get) if timings else None
    pcts    = {k: round(v / total * 100, 1) if total > 0 else 0
               for k, v in timings.items()}
    return {
        "available": True,
        "timings":   timings,   # {stage: seconds}
        "pcts":      pcts,      # {stage: % of total}
        "total":     total,
        "slowest":   slowest,
    }


def _memory(result: dict) -> dict:
    start = result.get("mem_start_mb")
    peak  = result.get("mem_peak_mb")
    if peak is None:
        return {"available": False, "start_mb": None, "peak_mb": None, "delta_mb": None}
    delta = round(peak - start, 1) if start is not None else None
    grade = ("OK"   if peak < 4096 else
             "HIGH" if peak < 6144 else "CRITICAL")
    return {
        "available": True,
        "start_mb":  start,
        "peak_mb":   peak,
        "delta_mb":  delta,
        "grade":     grade,
    }


def _vocab_richness(result: dict) -> dict:
    ttr = result.get("vocab_richness_ttr")
    transcript = result.get("transcript", "") or ""
    words  = transcript.split()
    unique = len(set(w.lower() for w in words))
    grade  = ("RICH"   if (ttr or 0) >= 0.7 else
              "NORMAL" if (ttr or 0) >= 0.4 else "REPETITIVE")
    return {
        "ttr":         ttr,
        "word_count":  len(words),
        "unique_words": unique,
        "grade":       grade if ttr is not None else "N/A",
    }


# ── Tier 2: Reference-based metrics ───────────────────────────────────────────

def compute_wer_cer(hypothesis: str, reference: str) -> dict:
    """
    Word Error Rate and Character Error Rate.
    Requires jiwer. hypothesis = system output, reference = analyst ground truth.
    """
    import jiwer

    if not hypothesis or not reference:
        return {"wer": None, "cer": None, "error": "Empty input"}

    try:
        # Normalise: lowercase, strip extra whitespace
        hyp = " ".join(hypothesis.lower().split())
        ref = " ".join(reference.lower().split())

        # jiwer 4.x: use process_words for full breakdown
        word_out = jiwer.process_words(ref, hyp)
        char_out = jiwer.process_characters(ref, hyp)

        return {
            "wer":           round(word_out.wer * 100, 2),
            "cer":           round(char_out.cer * 100, 2),
            "substitutions": word_out.substitutions,
            "deletions":     word_out.deletions,
            "insertions":    word_out.insertions,
            "ref_words":     len(ref.split()),
            "hyp_words":     len(hyp.split()),
            "grade":         _asr_grade(word_out.wer),
            "error":         None,
        }
    except Exception as e:
        return {"wer": None, "cer": None, "error": str(e)}


def compute_bleu_chrf(hypothesis: str, reference: str) -> dict:
    """
    BLEU and chrF scores.
    Requires sacrebleu. hypothesis = system translation, reference = analyst ground truth.
    """
    import sacrebleu

    if not hypothesis or not reference:
        return {"bleu": None, "chrf": None, "error": "Empty input"}

    try:
        bleu  = sacrebleu.corpus_bleu([hypothesis], [[reference]])
        chrf  = sacrebleu.corpus_chrf([hypothesis], [[reference]])
        ter   = sacrebleu.corpus_ter([hypothesis],  [[reference]])

        return {
            "bleu":       round(bleu.score,  2),   # 0–100
            "chrf":       round(chrf.score,  2),
            "ter":        round(ter.score,   2),   # Translation Edit Rate
            "bleu_bp":    round(bleu.bp,     3),   # brevity penalty
            "bleu_prec":  [round(p, 2) for p in bleu.precisions],  # n-gram precisions
            "grade":      _bleu_grade(bleu.score),
            "error":      None,
        }
    except Exception as e:
        return {"bleu": None, "chrf": None, "ter": None, "error": str(e)}


# ── Grade helpers ──────────────────────────────────────────────────────────────

def _asr_grade(wer: float) -> str:
    if wer <= 0.05:  return "EXCELLENT  (WER <=5%)"
    if wer <= 0.15:  return "GOOD       (WER <=15%)"
    if wer <= 0.30:  return "FAIR       (WER <=30%)"
    if wer <= 0.50:  return "POOR       (WER <=50%)"
    return             "UNUSABLE   (WER >50%)"


def _bleu_grade(bleu: float) -> str:
    if bleu >= 50:   return "EXCELLENT  (BLEU >=50)"
    if bleu >= 30:   return "GOOD       (BLEU >=30)"
    if bleu >= 15:   return "FAIR       (BLEU >=15)"
    if bleu >= 5:    return "POOR       (BLEU >=5)"
    return             "VERY LOW   (BLEU <5)"

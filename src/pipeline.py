"""
src/pipeline.py – VANI main processing pipeline
-------------------------------------------------
Improvements over original:
  • progress_cb parameter – callable for Streamlit live progress updates
  • Sequential model loading with explicit memory release
  • VAD segments fed to chunker (VAD-aware splitting)
  • Language hint cached after first chunk (skips per-chunk detection)
  • Confidence-weighted LangID voting
  • ISUM generation integrated
  • Structured logging via utils.get_logger
  • All paths resolved via utils.ROOT
"""

import os
import json
import time
from pathlib import Path
from typing import Callable, Optional

import torch

# Force offline before any HF import
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

from utils import (
    ROOT, load_config, get_logger,
    free_memory, utc_now_iso, generate_report_id,
    elapsed, ensure_dir, clear_dir_wavs,
)
from vad_module         import VADModule
from preprocessing      import AudioPreprocessor
from chunker            import AudioChunker
from asr_module         import ASRModule
from language_module    import FastTextLangDetector, DialectDetector, LanguageRouter, INDIC_LANGS, NLLB_LANGS
from translation_module import TranslationModule
from keyword_module     import KeywordDetector
from isum_module        import ISUMGenerator

def _mms_lid_available(paths: dict) -> bool:
    mms_path = ROOT / paths.get("mms_lid_model", "models/mms-lid-256")
    return mms_path.exists() and any(mms_path.iterdir())

def _qwen_available(paths: dict) -> bool:
    qwen_path = ROOT / paths.get("qwen_model", "models/qwen2.5-0.5b-instruct")
    return qwen_path.exists() and any(qwen_path.iterdir())


def _probe_and_select_asr_model(
    paths: dict, audio_path: Path, config: dict, device: str, logger
) -> tuple:
    """
    Run a quick MMS-LID probe before ASR to pick a language-specific Whisper model.
    Returns (selected_model_path, mms_result_dict).

    mms_result is passed through to Stage 5 so MMS-LID is not loaded twice.
    Falls back to the default whisper_model path if no specialized model matches.
    """
    default_path = ROOT / paths["whisper_model"]

    # Build map: lang_code → CT2 path, for each whisper_model_<lang> in config
    lang_model_map: dict = {}
    for key, val in paths.items():
        if key.startswith("whisper_model_") and key != "whisper_model":
            lang_code = key[len("whisper_model_"):]
            candidate = ROOT / val
            if candidate.exists() and any(candidate.iterdir()):
                lang_model_map[lang_code] = candidate

    if not lang_model_map or not _mms_lid_available(paths):
        return default_path, None

    try:
        from mms_module import MMSLangDetector
        mms_lid_path = ROOT / paths.get("mms_lid_model", "models/mms-lid-256")
        mms_det      = MMSLangDetector(model_path=str(mms_lid_path), device=device)
        mms_result   = mms_det.detect(str(audio_path))
        mms_det.unload()
        del mms_det

        lang = mms_result.get("language", "")
        conf = mms_result.get("confidence", 0.0)

        if lang in lang_model_map and conf >= 0.65:
            selected = lang_model_map[lang]
            logger.info(f"  Specialized ASR model selected for {lang.upper()}: {selected.name}")
            return selected, mms_result

        return default_path, mms_result

    except Exception as e:
        logger.warning(f"  Pre-ASR language probe failed: {e}")
        return default_path, None


def run_pipeline(
    audio_file:  Path,
    config:      dict       = None,
    logger                  = None,
    progress_cb: Callable   = None,
    models:      dict       = None,
) -> dict:
    """
    Run full VANI pipeline on an audio file.

    Parameters
    ----------
    audio_file  : Path to input audio
    config      : loaded config dict (defaults to config.yaml)
    logger      : logger instance (created if None)
    progress_cb : optional callable(stage_name: str) for UI progress updates

    Returns
    -------
    Full result dict (saved to JSON and SQLite by caller)
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = get_logger("vani.pipeline")
    models = models or {}

    # ── Stage timing + memory tracking ────────────────────────────────────────
    _stage_times: dict  = {}
    _stage_order: list  = []
    _stage_t0:    float = 0.0

    try:
        import psutil as _psutil
        _proc = _psutil.Process()
        _mem_start_mb = _proc.memory_info().rss / 1_048_576
    except ImportError:
        _proc = None
        _mem_start_mb = None

    def progress(stage: str):
        nonlocal _stage_t0
        now = time.time()
        if _stage_order:
            _stage_times[_stage_order[-1]] = round(now - _stage_t0, 2)
        _stage_order.append(stage)
        _stage_t0 = now
        logger.info(f"STAGE: {stage}")
        if progress_cb:
            try:
                progress_cb(stage)
            except Exception:
                pass

    t_start   = time.time()
    device    = config.get("device", "cpu")
    paths     = config["paths"]

    whisper_path  = ROOT / paths["whisper_model"]
    indic_path    = ROOT / paths["indictrans_model"]
    nllb_path     = ROOT / paths["nllb_model"]
    fasttext_path = ROOT / paths["fasttext_model"]
    kw_dict_path  = ROOT / paths.get("keyword_dictionary", "alerts/keyword_dictionary.json")
    output_dir    = ensure_dir(ROOT / paths["output_dir"])
    chunk_dir     = ensure_dir(output_dir / f"{audio_file.stem}_chunks")

    clear_dir_wavs(chunk_dir)

    # ── STAGE 1: VAD ──────────────────────────────────────────────────────────
    progress("VAD")
    vad        = VADModule(cfg=config.get("vad", {}))
    vad_out    = output_dir / f"{audio_file.stem}_vad.wav"
    vad_result = vad.remove_silence(str(audio_file), str(vad_out))
    logger.info(f"  Speech: {vad_result['total_speech_sec']}s in "
                f"{len(vad_result['segments_seconds'])} segments")
    del vad;  free_memory(logger)

    # ── Clean-audio path decision ─────────────────────────────────────────────
    # audio_mode: "auto" (decide by SNR) | "clean" (force skip) | "noisy" (full).
    # When clean, denoise/bandpass (built for degraded radio) and diarization are
    # skipped — they add latency and can degrade clean speech.
    _audio_mode = (config.get("audio_mode") or "auto").lower()
    _snr_db = None
    try:
        import soundfile as _sf_snr
        from preprocessing import _estimate_snr
        _snr_arr, _ = _sf_snr.read(str(vad_out))
        _snr_val = _estimate_snr(_snr_arr)
        _snr_db = round(float(_snr_val), 1) if _snr_val is not None else None
    except Exception:
        pass
    _clean_thresh = config.get("preprocessing", {}).get("clean_snr_threshold", 18.0)
    if _audio_mode == "clean":
        _clean_mode = True
    elif _audio_mode == "noisy":
        _clean_mode = False
    else:  # auto
        _clean_mode = (_snr_db is not None and _snr_db >= _clean_thresh)
    _skipped_stages: list = []
    logger.info(f"  Audio mode: {_audio_mode} | SNR={_snr_db} dB | clean_path={_clean_mode}")

    # ── STAGE 2: Preprocessing ────────────────────────────────────────────────
    progress("Preprocessing")
    _pre_cfg = dict(config.get("preprocessing", {}))
    if _clean_mode:
        _pre_cfg["noise_reduce"]    = False
        _pre_cfg["bandpass_filter"] = False
        _skipped_stages.append("denoise/bandpass")
        logger.info("  Clean audio — skipping denoise + bandpass")
    pre      = AudioPreprocessor(cfg=_pre_cfg)
    pre_out  = output_dir / f"{audio_file.stem}_preprocessed.wav"
    pre_info = pre.preprocess(str(vad_out), str(pre_out))
    logger.info(f"  Duration after preprocessing: {pre_info['duration_sec']}s")
    del pre;  free_memory(logger)

    # ── STAGE 3: Chunking ─────────────────────────────────────────────────────
    progress("Chunking")
    chunker = AudioChunker(cfg=config.get("chunking", {}))
    chunks  = chunker.split_audio(
        str(pre_out), str(chunk_dir),
        vad_segments_seconds=vad_result["segments_seconds"],
    )
    logger.info(f"  {len(chunks)} chunks created")
    del chunker;  free_memory(logger)

    if not chunks:
        # Fallback: treat the entire preprocessed file as one chunk so Whisper
        # can make the final call rather than aborting with no output.
        logger.warning("No speech chunks from chunker — using full audio as single chunk.")
        import soundfile as _sf2
        _fb_audio, _fb_sr = _sf2.read(str(pre_out))
        _fb_dur = len(_fb_audio) / max(_fb_sr, 1)
        if _fb_dur < 0.5:
            logger.warning("Audio too short (<0.5 s) — aborting.")
            return {}
        _fb_chunk_path = chunk_dir / "chunk_0000.wav"
        _sf2.write(str(_fb_chunk_path), _fb_audio, _fb_sr)
        chunks = [{
            "path":      _fb_chunk_path,
            "start_sec": 0.0,
            "end_sec":   round(_fb_dur, 3),
            "index":     0,
        }]
        logger.info(f"  Fallback chunk: {_fb_dur:.1f}s")

    # ── STAGE 3.5: Pre-ASR Language Probe ────────────────────────────────────
    # Runs MMS-LID on the preprocessed audio to select a language-specific
    # Whisper model when one is configured (e.g. whisper_model_zh, whisper_model_ps).
    # The mms_result is cached and reused in Stage 5 to avoid loading MMS-LID twice.
    _pre_asr_mms_result = None
    if not config.get("language_override"):
        whisper_path, _pre_asr_mms_result = _probe_and_select_asr_model(
            paths, pre_out, config, device, logger
        )
        if _pre_asr_mms_result:
            logger.info(f"  Pre-ASR probe: {_pre_asr_mms_result['language']} "
                        f"(p={_pre_asr_mms_result['confidence']:.2f})")

    # ── STAGE 4: ASR ──────────────────────────────────────────────────────────
    progress("ASR")
    # For languages where zero-shot SeamlessM4T beats fine-tuned Whisper
    # (pa/ne per benchmark), route ASR to the SeamlessM4T backend. Controlled by
    # asr.seamless_langs in config; translation still goes through NLLB downstream.
    _probe_lang     = _pre_asr_mms_result.get("language") if _pre_asr_mms_result else None
    _seamless_langs = set(config.get("asr", {}).get("seamless_langs", []) or [])
    _use_seamless   = _probe_lang is not None and _probe_lang in _seamless_langs

    if _use_seamless:
        from seamless_asr import SeamlessASR
        _seamless_path = ROOT / paths.get("seamless_model", "models/seamless-m4t-v2-large")
        logger.info(f"  ASR backend: SeamlessM4T (zero-shot) for {_probe_lang}")
        asr = SeamlessASR(model_path=str(_seamless_path), device=device,
                          default_lang=_probe_lang)
        _asr_cached = False
    else:
        # Use the cached ASR only if it is the SAME model that Stage 3.5 selected.
        # The app pre-caches the default model; blindly reusing it here silently
        # discarded the language-specific fine-tuned model selection.
        _cached_asr = models.get("asr")
        if _cached_asr is not None and \
                str(getattr(_cached_asr, "model_path", "")) != str(whisper_path):
            logger.info(f"  Cached ASR is {Path(getattr(_cached_asr, 'model_path', '?')).name} "
                        f"— rebuilding for selected model {whisper_path.name}")
            _cached_asr = None
        _asr_cached = _cached_asr is not None
        asr = _cached_asr or ASRModule(
            model_path=str(whisper_path),
            device=device,
            cfg=config.get("asr", {}),
        )
    asr.reset_language_cache()   # always reset between files

    full_transcript   = []
    all_segments      = []
    whisper_lang      = None
    whisper_lang_prob = 0.0

    # If the pre-ASR probe confidently selected a language-specific model,
    # force that language for decoding instead of re-running Whisper's own LID.
    # Whisper LID can misfire on short radio audio (e.g. pa misread as gu at
    # p=0.38 → Gujarati-script transcript), which then poisons the FastText
    # vote in Stage 5 and kills the translation route downstream.
    if _pre_asr_mms_result and whisper_path != ROOT / paths["whisper_model"]:
        whisper_lang      = _pre_asr_mms_result["language"]
        whisper_lang_prob = _pre_asr_mms_result["confidence"]
        logger.info(f"  ASR language forced from pre-ASR probe: {whisper_lang} "
                    f"(p={whisper_lang_prob:.2f})")

    for chunk in chunks:
        chunk_label = f"ASR {chunk['index']+1}/{len(chunks)}"
        logger.info(f"  Chunk {chunk['index']+1}/{len(chunks)} "
                    f"[{chunk['start_sec']:.1f}s–{chunk['end_sec']:.1f}s]")
        if progress_cb:
            try:
                progress_cb(chunk_label)
            except Exception:
                pass
        result = asr.transcribe(str(chunk["path"]), language_hint=whisper_lang)

        if whisper_lang is None and result["language"]:
            whisper_lang      = result["language"]
            whisper_lang_prob = result["language_probability"]
            logger.info(f"  Language: {whisper_lang} (prob={whisper_lang_prob:.2f})")

        if result["transcript"]:
            full_transcript.append(result["transcript"])

        for seg in result["segments"]:
            s = dict(seg)
            s["start"]     = round(seg["start"] + chunk["start_sec"], 3)
            s["end"]       = round(seg["end"]   + chunk["start_sec"], 3)
            s["chunk_idx"] = chunk["index"]
            all_segments.append(s)

    transcript_text = " ".join(full_transcript).strip()
    logger.info(f"  Transcript: {len(transcript_text)} chars")
    if not _asr_cached:
        del asr;  free_memory(logger)

    # ── STAGE 4.5: Speaker Diarization ───────────────────────────────────────
    _diar_cfg = config.get("diarization", {})
    if _diar_cfg.get("enabled", True) and all_segments and not _clean_mode:
        progress("Diarization")
        try:
            from diarize_module import diarize as _diarize
            _diarize(str(pre_out), all_segments,
                     max_speakers=_diar_cfg.get("max_speakers", 4))
            _n_spk = len({s.get("speaker") for s in all_segments if s.get("speaker")})
            logger.info(f"  Diarization: {_n_spk} speaker(s) detected")
        except Exception as _de:
            logger.warning(f"  Diarization skipped: {_de}")
    elif _clean_mode and _diar_cfg.get("enabled", True) and all_segments:
        _skipped_stages.append("diarization")
        logger.info("  Clean audio — skipping diarization")


    # ── STAGE 5: Language ID ──────────────────────────────────────────────────
    progress("Language ID")
    _lang_override    = config.get("language_override")
    _conf_threshold   = config.get("language", {}).get("confidence_threshold", 0.60)

    if _lang_override:
        # Analyst-specified language — skip voting entirely
        if _lang_override in INDIC_LANGS:
            _ov_route = "indictrans2"
        elif _lang_override in NLLB_LANGS:
            _ov_route = "nllb"
        elif _lang_override == "en":
            _ov_route = "none"
        else:
            _ov_route = "none"
        routing      = {"final_language": _lang_override, "route": _ov_route,
                        "confidence": 1.0, "script_hint": "manual_override",
                        "uncertain": False, "vote_note": "analyst override"}
        ft_result    = {"language": _lang_override, "confidence": 1.0}
        dialect_info = {"dialect": "unknown", "dialect_confidence": 0.0}
        mms_result   = None
        logger.info(f"  Language override: {_lang_override} -> {_ov_route}")
    else:
        _ft_cached  = "fasttext" in models
        _mms_cached = "mms" in models
        lang_det  = models.get("fasttext") or FastTextLangDetector(model_path=str(fasttext_path))
        dialect   = DialectDetector()
        router    = LanguageRouter(confidence_threshold=_conf_threshold)

        ft_result    = lang_det.detect(transcript_text)
        dialect_info = dialect.detect_code_mix(transcript_text)

        # Optional MMS-LID (audio-based, 3rd vote)
        # Disabled by default on low-memory systems via memory.use_mms_lid: false
        _use_mms  = config.get("memory", {}).get("use_mms_lid", True)
        mms_result = None
        if _use_mms and models.get("mms") is not None:
            try:
                mms_result = models["mms"].detect(str(pre_out))
                logger.info(f"  MMS-LID: {mms_result['language']} "
                            f"(p={mms_result['confidence']:.2f})")
            except Exception as e:
                logger.warning(f"  MMS-LID skipped: {e}")
        elif _use_mms and _pre_asr_mms_result is not None:
            # Reuse result from pre-ASR language probe — avoids loading MMS-LID twice
            mms_result = _pre_asr_mms_result
            logger.info(f"  MMS-LID (cached from pre-ASR probe): "
                        f"{mms_result['language']} (p={mms_result['confidence']:.2f})")
        elif _use_mms and _mms_lid_available(paths):
            try:
                from mms_module import MMSLangDetector
                mms_lid_path = ROOT / paths.get("mms_lid_model", "models/mms-lid-256")
                mms_det  = MMSLangDetector(model_path=str(mms_lid_path), device=device)
                mms_result = mms_det.detect(str(pre_out))
                mms_det.unload()
                logger.info(f"  MMS-LID: {mms_result['language']} "
                            f"(p={mms_result['confidence']:.2f})")
                del mms_det
            except Exception as e:
                logger.warning(f"  MMS-LID skipped: {e}")
        else:
            logger.info("  MMS-LID skipped (use_mms_lid=false)")

        free_memory(logger)

        routing = router.detect_family(
            whisper_lang=      whisper_lang,
            transcript=        transcript_text,
            fasttext_lang=     ft_result["language"],
            fasttext_conf=     ft_result["confidence"],
            whisper_lang_prob= whisper_lang_prob,
            dialect=           dialect_info["dialect"],
            mms_lang=          mms_result["language"]   if mms_result else None,
            mms_conf=          mms_result["confidence"] if mms_result else 0.0,
        )
        logger.info(f"  Final: {routing['final_language']} via {routing['route']} "
                    f"(conf={routing['confidence']:.2f}, vote={routing.get('vote_note','')})")
        del lang_det, dialect, router;  free_memory(logger)

    # ── STAGE 6: Translation ──────────────────────────────────────────────────
    progress("Translation")
    translator  = TranslationModule(
        str(indic_path), str(nllb_path),
        device=device,
        cfg=config.get("translation", {}),
    )
    translation = translator.translate(
        text=          transcript_text,
        route=         routing["route"],
        detected_lang= routing["final_language"],
    )
    logger.info(f"  Translation success: {translation['success']}")

    # ── Back-translation via NLLB (only if enabled and NLLB-routed)
    # Disabled by default via translation.back_translation: false (saves ~1.2 GB RAM)
    _do_backtrans = config.get("translation", {}).get("back_translation", False)
    back_translation = None
    if (_do_backtrans and
            routing["route"] == "nllb" and
            translation.get("success") and
            translation.get("translated_text", "").strip()):
        try:
            back_translation = translator.backtranslate_nllb(
                translation["translated_text"],
                routing["final_language"],
            )
            logger.info(f"  Back-translation: {back_translation.get('success')}")
        except Exception as e:
            logger.warning(f"  Back-translation skipped: {e}")
    else:
        logger.info("  Back-translation skipped (back_translation=false)")

    del translator;  free_memory(logger)

    # ── STAGE 7: Keyword Detection ────────────────────────────────────────────
    progress("Keywords")
    kw_detector = KeywordDetector(dictionary_path=str(kw_dict_path))
    trans_text  = translation.get("translated_text", "")
    kw_info     = kw_detector.detect(
        transcript=  transcript_text,
        translation= trans_text,
        segments=    all_segments,
    )
    logger.info(f"  Threat: {kw_info['threat_level']} | {kw_info['top_categories']}")

    # ── STAGE 8: ISUM ─────────────────────────────────────────────────────────
    progress("ISUM")
    t_elapsed = elapsed(t_start)

    # Finalise last stage time
    if _stage_order:
        _stage_times[_stage_order[-1]] = round(time.time() - _stage_t0, 2)

    # Peak memory
    _mem_peak_mb = None
    if _proc is not None:
        try:
            _mem_peak_mb = round(_proc.memory_info().rss / 1_048_576, 1)
        except Exception:
            pass

    # Vocabulary richness (Type-Token Ratio)
    _words = transcript_text.lower().split() if transcript_text else []
    _ttr   = round(len(set(_words)) / len(_words), 3) if _words else None

    # Back-translation chrF
    _backtrans_chrf = None
    if back_translation and back_translation.get("success"):
        try:
            import sacrebleu as _sb
            _backtrans_chrf = round(
                _sb.corpus_chrf(
                    [back_translation["translated_text"]],
                    [[transcript_text]]
                ).score, 2
            )
            logger.info(f"  Back-translation chrF: {_backtrans_chrf}")
        except Exception as e:
            logger.warning(f"  chrF computation skipped: {e}")

    intermediate = {
        "audio_file":                  audio_file.name,
        "whisper_language":            whisper_lang,
        "whisper_language_probability":whisper_lang_prob,
        "final_language":              routing["final_language"],
        "route_confidence":            routing["confidence"],
        "translation_route":           routing.get("route", "-"),
        "uncertain":                   routing["uncertain"],
        "transcript":                  transcript_text,
        "translation":                 translation,
        "segments":                    all_segments,
        "keyword_alerts":              kw_info,
    }

    isum_gen   = ISUMGenerator(cfg=config.get("isum", {}), device=device)
    isum_mode  = config.get("isum", {}).get("model", "rule_based")
    qwen_path  = (ROOT / paths.get("qwen_model", "models/qwen2.5-0.5b-instruct")
                  if isum_mode != "rule_based" and _qwen_available(paths) else None)
    isum       = isum_gen.generate(intermediate, processing_time_s=t_elapsed,
                                   qwen_model_path=str(qwen_path) if qwen_path else None)

    # ── Cross-file speaker re-identification (needs report_id from ISUM) ──────
    # Default OFF: MFCC-stat cosine cannot separate speakers on bandpassed radio
    # audio (measured 2026-07-06: same-speaker mean cos 0.993 vs diff-speaker
    # 0.984 — 77% false-match rate at any usable threshold; every intercept
    # since 13 Jun collapsed into VOICE_001). Re-enable only after replacing
    # centroids with real speaker embeddings (e.g. ECAPA-TDNN).
    if config.get("diarization", {}).get("cross_file_reid", False) and all_segments:
        try:
            from diarize_module import compute_speaker_centroids
            from speaker_store import SpeakerStore

            _db_path   = ROOT / config.get("paths", {}).get("database", "database/transcripts.db")
            _store     = SpeakerStore(str(_db_path))
            _ts        = utc_now_iso()
            _reid      = isum["report_id"]
            _centroids = compute_speaker_centroids(str(pre_out), all_segments)
            _local_to_voice: dict = {}

            for _local_lbl, _centroid in _centroids.items():
                _voice_id, _is_new = _store.match_or_register(_centroid, _reid, _ts)
                _local_to_voice[_local_lbl] = _voice_id
                logger.info(
                    f"  {_local_lbl} -> {_voice_id} "
                    f"({'new voice' if _is_new else 'matched existing'})"
                )

            for seg in all_segments:
                _lbl = seg.get("speaker")
                if _lbl and _lbl in _local_to_voice:
                    seg["speaker"] = _local_to_voice[_lbl]

            logger.info(f"  Voice mapping: {_local_to_voice}")
        except Exception as _re:
            logger.warning(f"  Speaker re-ID skipped: {_re}")

    # ── Assemble final result ─────────────────────────────────────────────────
    result = {
        "report_id":               isum["report_id"],
        "audio_file":              audio_file.name,
        "timestamp_utc":           isum["timestamp_utc"],
        "processing_time_s":       round(t_elapsed, 2),
        "vad_segments_seconds":    vad_result["segments_seconds"],
        "total_speech_sec":        vad_result["total_speech_sec"],
        "preprocessing":           pre_info,
        "chunks_created":          len(chunks),
        "whisper_language":        whisper_lang,
        "whisper_language_probability": whisper_lang_prob,
        "fasttext_language":       ft_result["language"],
        "fasttext_confidence":     ft_result["confidence"],
        "dialect":                 dialect_info["dialect"],
        "dialect_confidence":      dialect_info["dialect_confidence"],
        "final_language":          routing["final_language"],
        "translation_route":       routing["route"],
        "route_confidence":        routing["confidence"],
        "script_hint":             routing["script_hint"],
        "language_uncertain":      routing["uncertain"],
        "transcript":              transcript_text,
        "translation":             translation,
        "segments":                all_segments,
        "keyword_summary":         kw_info["summary_counts"],
        "threat_level":            kw_info["threat_level"],
        "top_categories":          kw_info["top_categories"],
        "keyword_alerts":          kw_info,
        "confidence_flags":        isum["confidence_flags"],
        "isum":                    isum,
        # ── Language detection detail ──────────────────────────────────────
        "mms_language":            mms_result["language"]   if mms_result else None,
        "mms_confidence":          mms_result["confidence"] if mms_result else None,
        "vote_note":               routing.get("vote_note", ""),
        # ── Performance telemetry ──────────────────────────────────────────
        "stage_timings":           _stage_times,
        "mem_start_mb":            round(_mem_start_mb, 1) if _mem_start_mb else None,
        "mem_peak_mb":             _mem_peak_mb,
        "vocab_richness_ttr":      _ttr,
        "back_translation":        back_translation,
        "backtrans_chrf":          _backtrans_chrf,
        # ── Clean-audio path ───────────────────────────────────────────────
        "audio_mode":              _audio_mode,
        "snr_db":                  _snr_db,
        "clean_path":              _clean_mode,
        "skipped_stages":          _skipped_stages,
    }

    # Save JSON
    out_file = output_dir / f"{audio_file.stem}_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Pipeline complete in {t_elapsed}s -> {out_file}")
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cfg = load_config()
    log = get_logger("vani")

    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py input_audio/sample.wav")
        sys.exit(1)

    run_pipeline(ROOT / sys.argv[1], cfg, log)

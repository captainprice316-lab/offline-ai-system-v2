"""
wer_robustness_eval.py – VANI WER-under-degradation evaluation
==============================================================
Companion to robustness_eval.py, which scores *LangID accuracy* across radio-channel
degradations but never scores WER. This script fills that gap: it measures ASR
accuracy for each candidate ASR backend under the same degradations.

Why it exists
-------------
Every ASR routing decision in config.yaml (`asr.seamless_langs`) rests on
docs/model_comparison_results.json + docs/seamless_ft_results.json, which are scored
on *clean* FLEURS read speech. VANI's operational input is 300-3400 Hz bandpassed
radio. A clean-speech ranking can invert under noise, so the routing table has never
actually been validated on the distribution it serves.

Design notes
------------
1. **Raw hypotheses are the artefact, not the scores.** We write one JSONL row per
   (system, lang, condition, utterance) holding the reference and the hypothesis
   verbatim. Scoring lives in score_wer_robustness.py and can be re-run for free.
   This is deliberate: the Kashmiri baseline was once reported at 96.87% WER purely
   because of a Unicode normalisation mismatch. Never make a normalisation bug cost
   a GPU run.

2. **Language is forced, not detected.** We pass `language_hint=<lang>` so this
   measures ASR only. LangID robustness is robustness_eval.py's job; conflating the
   two is what makes a 0 dB result uninterpretable.

3. **VAD is disabled** so Whisper and SeamlessM4T see byte-identical audio.
   faster-whisper has an internal VAD; Seamless has none. Leaving it on would mean
   the two systems are transcribing different signals at low SNR.

4. **Resumable.** Rows already present in the JSONL are skipped. An 8 GB card running
   a multi-hour sweep will get interrupted; that must not cost the whole run.

Usage
-----
    python scripts/eval/wer_robustness_eval.py --cache-refs      # phase 0: fetch reference text
    python scripts/eval/wer_robustness_eval.py                   # full sweep
    python scripts/eval/wer_robustness_eval.py --langs hi ur --conditions clean bandpass
    python scripts/eval/wer_robustness_eval.py --systems seamless_zs --n 10
"""

import os, sys, gc, json, argparse, warnings, tempfile, io
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# Windows console (cp1252) cannot print Gurmukhi / Devanagari / Han.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Reference text lives on the Hub; models are local. Must not force offline here.
os.environ["HF_HUB_OFFLINE"]       = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # robustness_eval (degrade fns)
sys.path.insert(0, str(HERE.parent.parent / "src")) # asr_module, seamless_asr, utils

from utils import load_config, ROOT as VANI_ROOT
from robustness_eval import DATASETS, CACHE_DIR, MIN_DUR, MAX_DUR, degrade, _patch_audio_decode

OUT_JSONL = VANI_ROOT / "eval_data" / "wer_robustness_hyps.jsonl"

# The five conditions already reported in robustness_results.csv, so the WER table
# lines up row-for-row with the published LangID table.
DEFAULT_CONDITIONS = ["clean", "bandpass", "awgn_10", "awgn_0", "codec_mp3"]
DEFAULT_LANGS      = ["hi", "ur", "pa", "ne", "ps", "zh", "ks"]
DEFAULT_SYSTEMS    = ["whisper_ft", "seamless_zs"]

# SeamlessM4T v2 has no Kashmiri (kas). compare_all_models.py:109 says the same.
SEAMLESS_UNSUPPORTED = {"ks"}


# ── Phase 0: cache reference transcripts ──────────────────────────────────────

def cache_refs(langs, n_samples):
    """Stream FLEURS again and write refs.jsonl alongside the cached WAVs.

    robustness_eval.py cached audio only. We re-stream with the *identical* filter
    (MIN_DUR/MAX_DUR, same order) so sample i of the stream is 000i.wav. That
    assumption is then verified against each WAV's real duration; a mismatch aborts
    rather than silently misaligning every reference in the language.
    """
    from datasets import load_dataset
    _patch_audio_decode()

    for lang in langs:
        info     = DATASETS[lang]
        lang_dir = CACHE_DIR / lang
        wavs     = sorted(lang_dir.glob("*.wav"))[:n_samples]
        refs_p   = lang_dir / "refs.jsonl"

        if not wavs:
            print(f"  {info['name']:9} no cached WAVs — run robustness_eval.py --phase1-only first")
            continue
        if refs_p.exists():
            have = sum(1 for _ in refs_p.open(encoding="utf-8"))
            if have >= len(wavs):
                print(f"  {info['name']:9} refs.jsonl already has {have} rows — skipping")
                continue
        if info.get("local_only"):
            # ks: audio came from IndicVoices-R via extract_ks_audio.py, which did not
            # persist transcripts. Say so loudly instead of quietly dropping the language.
            print(f"  {info['name']:9} LOCAL-ONLY: no reference text available. "
                  f"Kashmiri will be SKIPPED in the sweep until refs.jsonl is supplied "
                  f"(one JSON object per line: {{\"idx\": 0, \"ref\": \"...\"}}).")
            continue

        print(f"  {info['name']:9} streaming {info['config']} for reference text ...")
        ds = load_dataset(info["hf_id"], name=info["config"], split=info["split"],
                          streaming=True, trust_remote_code=False)

        rows, saved = [], 0
        for sample in ds:
            if saved >= len(wavs):
                break
            audio = sample.get("audio")
            if not isinstance(audio, dict) or audio.get("array") is None:
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr  = int(audio.get("sampling_rate", 16000))
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            dur = len(arr) / sr
            if dur < MIN_DUR or dur > MAX_DUR:      # same filter as download_and_cache
                continue

            wav_dur = sf.info(str(wavs[saved])).duration
            if abs(wav_dur - dur) > 0.05:
                raise RuntimeError(
                    f"{lang}: stream/cache misalignment at index {saved} — "
                    f"cached WAV is {wav_dur:.2f}s but stream sample is {dur:.2f}s. "
                    f"Delete robustness_cache/{lang}/ and re-run robustness_eval.py "
                    f"--phase1-only so audio and refs come from one pass."
                )

            rows.append({"idx": saved, "ref": sample["transcription"], "dur": round(dur, 3)})
            saved += 1

        with refs_p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {info['name']:9} wrote {len(rows)} refs -> {refs_p}")


def load_refs(lang, n_samples):
    p = CACHE_DIR / lang / "refs.jsonl"
    if not p.exists():
        return None
    refs = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            refs[int(r["idx"])] = r["ref"]
    return {i: refs[i] for i in sorted(refs) if i < n_samples}


# ── ASR backends ──────────────────────────────────────────────────────────────

def _whisper_model_path(cfg, lang):
    """Per-language CT2 model, exactly as pipeline._build_lang_model_map resolves it."""
    paths = cfg["paths"]
    key   = f"whisper_model_{lang}"
    rel   = paths.get(key) or paths["whisper_model"]     # turbo fallback
    p     = VANI_ROOT / rel
    return (p, key in paths and p.exists())


def make_whisper(cfg, lang, device):
    from asr_module import ASRModule
    path, is_ft = _whisper_model_path(cfg, lang)
    if not path.exists():
        raise FileNotFoundError(f"missing ASR model for {lang}: {path}")
    asr_cfg = dict(cfg.get("asr", {}))
    asr_cfg["vad_filter"]      = False   # identical signal for both systems (see docstring)
    asr_cfg["word_timestamps"] = False   # not needed; saves time
    return ASRModule(str(path), device=device, cfg=asr_cfg), path.name, is_ft


def make_whisper_base(cfg, device):
    """The TRUE large-v3 baseline (whisper-large-v3-ct2), one model for every language.

    Built by scripts/build_baseline_ct2.py. The point of including it here: the routing
    sweep compared fine-tuned Whisper vs Seamless, but for Mandarin the un-fine-tuned
    baseline beats BOTH on clean speech (10.99% vs ft 14.22 vs seamless 11.69). This
    tests whether that clean-speech win survives radio degradation.
    """
    from asr_module import ASRModule
    path = VANI_ROOT / "models" / "whisper-large-v3-ct2"
    if not path.exists():
        raise FileNotFoundError(f"missing baseline model: {path} "
                                f"(build with scripts/build_baseline_ct2.py)")
    asr_cfg = dict(cfg.get("asr", {}))
    asr_cfg["vad_filter"]      = False
    asr_cfg["word_timestamps"] = False
    return ASRModule(str(path), device=device, cfg=asr_cfg), path.name


def make_seamless(cfg, device):
    from seamless_asr import SeamlessASR
    path = VANI_ROOT / cfg["paths"]["seamless_model"]
    if not path.exists():
        raise FileNotFoundError(f"missing SeamlessM4T at {path}")
    return SeamlessASR(str(path), device=device, cfg=cfg.get("asr", {})), path.name


def _free(model):
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ── Sweep ─────────────────────────────────────────────────────────────────────

def done_keys(path):
    """(system, lang, condition, idx) tuples already recorded, for resume."""
    seen = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                seen.add((r["system"], r["lang"], r["condition"], r["idx"]))
            except Exception:
                continue          # a torn final line from a killed run
    return seen


def transcribe_one(asr, arr, sr, lang):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, arr, sr)
        tmp_path = tmp.name
    try:
        asr.reset_language_cache()
        return asr.transcribe(tmp_path, language_hint=lang).get("transcript", "") or ""
    finally:
        os.unlink(tmp_path)


def sweep(systems, langs, conditions, n_samples, device):
    cfg  = load_config()
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(OUT_JSONL)
    if seen:
        print(f"[resume] {len(seen)} utterances already recorded — skipping those\n")

    skipped_langs = []
    fh = OUT_JSONL.open("a", encoding="utf-8")

    def emit(**row):
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()

    for system in systems:
        print(f"\n{'='*74}\nSYSTEM: {system}\n{'='*74}", flush=True)

        # seamless_zs and whisper_base are ONE model for every language; whisper_ft is
        # one model per language.
        shared = None
        if system == "seamless_zs":
            shared, model_name = make_seamless(cfg, device)
            print(f"  loaded {model_name}", flush=True)
        elif system == "whisper_base":
            shared, model_name = make_whisper_base(cfg, device)
            print(f"  loaded {model_name}", flush=True)

        for lang in langs:
            if system == "seamless_zs" and lang in SEAMLESS_UNSUPPORTED:
                msg = f"{lang}: SeamlessM4T v2 has no {lang} support — no result, not a zero"
                print(f"  [SKIP] {msg}", flush=True)
                skipped_langs.append((system, msg))
                continue

            refs = load_refs(lang, n_samples)
            if not refs:
                msg = f"{lang}: no refs.jsonl — cannot score WER"
                print(f"  [SKIP] {msg}", flush=True)
                skipped_langs.append((system, msg))
                continue

            wavs = sorted((CACHE_DIR / lang).glob("*.wav"))[:n_samples]

            if system == "whisper_ft":
                asr, model_name, is_ft = make_whisper(cfg, lang, device)
                tag = "fine-tuned" if is_ft else "TURBO FALLBACK (no fine-tuned model)"
                print(f"\n  [{lang}] {model_name}  ({tag})", flush=True)
            else:
                asr = shared
                print(f"\n  [{lang}] {model_name}", flush=True)

            for condition in conditions:
                todo = [i for i in range(len(wavs))
                        if i in refs and (system, lang, condition, i) not in seen]
                if not todo:
                    print(f"    {condition:10} done", flush=True)
                    continue

                for n, i in enumerate(todo, 1):
                    arr, sr = sf.read(str(wavs[i]), dtype="float32")
                    deg = degrade(arr, sr, condition)
                    try:
                        hyp = transcribe_one(asr, deg, sr, lang)
                    except Exception as e:
                        print(f"    [ERR] {lang}/{condition}/{i}: {e}", flush=True)
                        continue
                    emit(system=system, lang=lang, condition=condition, idx=i,
                         model=model_name, ref=refs[i], hyp=hyp,
                         dur=round(len(arr) / sr, 3))
                    print(f"    {condition:10} {n}/{len(todo)}", end="\r", flush=True)

                print(f"    {condition:10} {len(todo)} transcribed", flush=True)

            if system == "whisper_ft":
                _free(asr)

        if shared is not None:
            _free(shared)

    fh.close()

    print(f"\n{'='*74}")
    print(f"hypotheses -> {OUT_JSONL}")
    if skipped_langs:
        print("\nSKIPPED (reported, not silently dropped):")
        for sysname, msg in skipped_langs:
            print(f"  {sysname}: {msg}")
    print("\nNow score with:  python scripts/eval/score_wer_robustness.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-refs", action="store_true",
                    help="phase 0: fetch reference transcripts into robustness_cache/<lang>/refs.jsonl")
    ap.add_argument("--langs",      nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--systems",    nargs="+", default=DEFAULT_SYSTEMS,
                    choices=["whisper_ft", "seamless_zs", "whisper_base"])
    ap.add_argument("--n",          type=int, default=30, help="samples per language")
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    if args.cache_refs:
        print("Phase 0: caching reference transcripts\n")
        cache_refs(args.langs, args.n)
        return

    print(f"systems={args.systems}  langs={args.langs}")
    print(f"conditions={args.conditions}  n={args.n}  device={args.device}")
    total = len(args.systems) * len(args.langs) * len(args.conditions) * args.n
    print(f"upper bound: {total} transcriptions (minus skips/resume)\n")
    sweep(args.systems, args.langs, args.conditions, args.n, args.device)


if __name__ == "__main__":
    main()

"""
explore_seamless.py  --  SeamlessM4T v2 exploration for VANI
=============================================================
Tests Meta's SeamlessM4T-v2-large on:
  1. ASR (speech -> text in source language)
  2. S2TT (speech -> English translation directly)
  3. Kashmiri support check
  4. Speed comparison vs faster-whisper

SeamlessM4T v2 supports 100+ languages in a single model including
Kashmiri (kas), replacing Whisper+NLLB with one inference pass.

Usage:
    python scripts/utils/explore_seamless.py              # download + capability check
    python scripts/utils/explore_seamless.py --audio <file.wav>   # test on audio file
    python scripts/utils/explore_seamless.py --compare   # compare vs VANI pipeline
"""

import argparse
import time
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_ID   = "facebook/seamless-m4t-v2-large"
MODEL_DIR  = ROOT / "models" / "seamless-m4t-v2-large"
CACHE_DIR  = ROOT / "models" / "seamless-m4t-v2-large" / "cache"

# SeamlessM4T language codes for VANI's target languages
# (different from ISO 639-1 used by Whisper)
LANG_MAP = {
    "ks":  ("kas", "Kashmiri"),
    "ur":  ("urd", "Urdu"),
    "hi":  ("hin", "Hindi"),
    "pa":  ("pan", "Punjabi"),
    "ps":  ("pbt", "Pashto (Southern)"),
    "ne":  ("npi", "Nepali"),
    "zh":  ("cmn", "Mandarin Chinese"),
    "en":  ("eng", "English"),
}

# Languages SeamlessM4T supports for speech input
S2T_SUPPORTED = {
    "kas", "urd", "hin", "pan", "npi", "cmn", "pbt", "eng",
    "fra", "deu", "spa", "arb", "ben", "tam", "tel", "mar",
}


def download_model():
    print(f"\n{'='*60}")
    print(f"  Downloading {MODEL_ID}")
    print(f"  -> {MODEL_DIR}")
    print(f"  Size: ~4.4 GB (large-v2)")
    print(f"{'='*60}\n")

    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    t0 = time.time()
    print("[1/2] Downloading processor ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=str(CACHE_DIR)
    )
    processor.save_pretrained(str(MODEL_DIR))
    print(f"  -> processor saved ({time.time()-t0:.0f}s)")

    t1 = time.time()
    print("[2/2] Downloading model weights ...")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        MODEL_ID, cache_dir=str(CACHE_DIR)
    )
    model.save_pretrained(str(MODEL_DIR))

    size_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*.safetensors")) / 1e9
    print(f"  -> model saved  ({time.time()-t1:.0f}s)  weights: {size_gb:.1f} GB")
    print(f"\n[OK] SeamlessM4T v2 ready at: {MODEL_DIR}\n")
    return processor, model


def load_model(device="cpu"):
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    print(f"Loading SeamlessM4T v2 from {MODEL_DIR} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(MODEL_DIR), torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s  (device={device})")
    return processor, model


def load_audio(path: str, target_sr: int = 16000):
    import torch, torchaudio
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, target_sr


def transcribe(processor, model, wav, src_lang_code: str, device="cpu"):
    """ASR — speech -> text in source language."""
    import torch
    inputs = processor(
        audios=wav.squeeze().numpy(),
        return_tensors="pt",
        sampling_rate=16000,
        src_lang=src_lang_code,
    ).to(device)
    t0 = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            tgt_lang=src_lang_code,
            generate_speech=False,
        )
    text = processor.decode(tokens[0].tolist()[0], skip_special_tokens=True)
    return text, time.time() - t0


def translate(processor, model, wav, src_lang_code: str, tgt_lang: str = "eng", device="cpu"):
    """S2TT — speech -> English translation (one pass, no separate MT model)."""
    import torch
    inputs = processor(
        audios=wav.squeeze().numpy(),
        return_tensors="pt",
        sampling_rate=16000,
        src_lang=src_lang_code,
    ).to(device)
    t0 = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            tgt_lang=tgt_lang,
            generate_speech=False,
        )
    text = processor.decode(tokens[0].tolist()[0], skip_special_tokens=True)
    return text, time.time() - t0


def capability_check():
    """Print language support summary without loading audio."""
    print("\n" + "="*60)
    print("  SeamlessM4T v2 — VANI Language Support Check")
    print("="*60)
    print(f"\n{'Lang':<6} {'Seamless Code':<16} {'S2T Input':<12} {'S2TT->Eng':<12} {'Name'}")
    print("-"*65)
    for vani_code, (sm_code, name) in LANG_MAP.items():
        s2t  = "YES" if sm_code in S2T_SUPPORTED else "no"
        s2tt = "YES" if sm_code in S2T_SUPPORTED else "no"
        print(f"  {vani_code:<4} {sm_code:<16} {s2t:<12} {s2tt:<12} {name}")

    print("""
Key advantages over Whisper+NLLB:
  - Single model for ASR + translation (no separate NLLB pass)
  - Native Kashmiri (kas) support — no proxy token needed
  - ~4.4 GB vs ~1.5 GB (Whisper) + ~1.2 GB (NLLB) = ~2.7 GB (VANI current)
  - Unified model simplifies pipeline Stages 4+6

Potential drawbacks:
  - Slower than faster-whisper (no CT2/int8 optimization)
  - Float32 on CPU — high RAM usage (~9 GB loaded)
  - No LoRA fine-tuning ecosystem (yet)
  - Less tested on noisy radio audio vs domain-adapted Whisper
""")


def run_audio_test(audio_path: str, src_lang: str = "ur"):
    import torch
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    sm_code = LANG_MAP.get(src_lang, (src_lang, src_lang))[0]

    if not MODEL_DIR.exists():
        print("Model not downloaded yet. Run without --audio first.")
        sys.exit(1)

    processor, model = load_model(device)
    wav, sr = load_audio(audio_path)
    duration = wav.shape[-1] / sr

    print(f"\nAudio: {audio_path}  ({duration:.1f}s)")
    print(f"Source language: {src_lang} -> SeamlessM4T code: {sm_code}\n")

    # ASR
    asr_text, asr_t = transcribe(processor, model, wav, sm_code, device)
    print(f"[ASR  -> {src_lang}] ({asr_t:.1f}s):\n  {asr_text}")

    # S2TT -> English
    en_text, en_t = translate(processor, model, wav, sm_code, "eng", device)
    print(f"\n[S2TT -> en]  ({en_t:.1f}s):\n  {en_text}")

    rtf = (asr_t + en_t) / duration
    print(f"\nRTF: {rtf:.2f}x  (Real-Time Factor — lower is faster)")


def compare_pipeline(audio_path: str, src_lang: str = "ur"):
    """Compare SeamlessM4T vs current VANI Whisper+NLLB pipeline."""
    import torch, yaml
    from faster_whisper import WhisperModel
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm_code = LANG_MAP.get(src_lang, (src_lang, src_lang))[0]
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    wav, sr = load_audio(audio_path)
    duration = wav.shape[-1] / sr
    print(f"\nAudio: {audio_path}  ({duration:.1f}s, lang={src_lang})\n")

    # ── Path A: VANI current (Whisper CT2 + NLLB) ─────────────────────────────
    print("="*50)
    print("PATH A: Whisper (CT2 int8) + NLLB-200")
    print("="*50)
    whisper_key = f"whisper_model_{src_lang}"
    whisper_path = ROOT / cfg["paths"].get(whisper_key, cfg["paths"]["whisper_model"])
    t0 = time.time()
    wm = WhisperModel(str(whisper_path), device=device, compute_type="int8")
    segs, _ = wm.transcribe(audio_path, language=src_lang, task="transcribe")
    whisper_text = " ".join(s.text for s in segs)
    whisper_t = time.time() - t0
    print(f"Whisper transcript ({whisper_t:.1f}s):\n  {whisper_text}")

    nllb_path = ROOT / cfg["paths"]["nllb_model"]
    nllb_src = {"ur": "urd_Arab", "hi": "hin_Deva", "ne": "npi_Deva",
                "pa": "pan_Guru", "ps": "pbt_Arab", "zh": "zho_Hans",
                "ks": "kas_Arab"}.get(src_lang, "urd_Arab")
    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(str(nllb_path), src_lang=nllb_src)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(str(nllb_path))
    forced = tok.lang_code_to_id["eng_Latn"]
    enc = tok(whisper_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    out = nllb.generate(**enc, forced_bos_token_id=forced, max_new_tokens=256)
    nllb_text = tok.decode(out[0], skip_special_tokens=True)
    nllb_t = time.time() - t1
    print(f"NLLB translation ({nllb_t:.1f}s):\n  {nllb_text}")
    path_a_total = whisper_t + nllb_t
    print(f"Total Path A: {path_a_total:.1f}s  (RTF {path_a_total/duration:.2f}x)")

    # ── Path B: SeamlessM4T (single model) ────────────────────────────────────
    print("\n" + "="*50)
    print("PATH B: SeamlessM4T v2 (ASR + translation, single model)")
    print("="*50)
    processor, sm_model = load_model(device)
    asr_text, asr_t = transcribe(processor, sm_model, wav, sm_code, device)
    en_text,  en_t  = translate(processor, sm_model, wav, sm_code, "eng", device)
    path_b_total = asr_t + en_t
    print(f"Seamless ASR  ({asr_t:.1f}s):\n  {asr_text}")
    print(f"Seamless S2TT ({en_t:.1f}s):\n  {en_text}")
    print(f"Total Path B: {path_b_total:.1f}s  (RTF {path_b_total/duration:.2f}x)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"  Whisper+NLLB total : {path_a_total:.1f}s  (RTF {path_a_total/duration:.2f}x)")
    print(f"  SeamlessM4T total  : {path_b_total:.1f}s  (RTF {path_b_total/duration:.2f}x)")
    print(f"\n  Whisper transcript : {whisper_text[:120]}")
    print(f"  NLLB translation   : {nllb_text[:120]}")
    print(f"\n  Seamless ASR       : {asr_text[:120]}")
    print(f"  Seamless S2TT->EN  : {en_text[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeamlessM4T v2 exploration for VANI")
    parser.add_argument("--audio",   type=str, help="Path to audio file for testing")
    parser.add_argument("--lang",    type=str, default="ur", help="Source language code (ur/hi/ks/pa/...)")
    parser.add_argument("--compare", action="store_true", help="Compare vs VANI Whisper+NLLB pipeline")
    parser.add_argument("--download-only", action="store_true", help="Just download the model")
    args = parser.parse_args()

    # Always show capability check
    capability_check()

    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        print("\nModel not found locally. Downloading now (~4.4 GB)...")
        download_model()
    else:
        size_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*.safetensors")) / 1e9
        print(f"\nModel already downloaded at {MODEL_DIR}  ({size_gb:.1f} GB)")

    if args.download_only:
        sys.exit(0)

    if args.compare and args.audio:
        compare_pipeline(args.audio, args.lang)
    elif args.audio:
        run_audio_test(args.audio, args.lang)
    else:
        print("\nRun with --audio <file.wav> to test transcription.")
        print("Run with --audio <file.wav> --compare to compare vs Whisper+NLLB.")
        print("Run with --download-only to just fetch the model weights.")

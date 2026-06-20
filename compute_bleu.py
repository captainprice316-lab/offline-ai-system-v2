"""
compute_bleu.py – Cascade ASR+MT BLEU for Hindi evaluation
===========================================================
For each Hindi sample in the evaluation set:
  1. Reference text (ground-truth Hindi) → NLLB-200 → reference_en  (MT ceiling)
  2. Audio → Whisper ASR → transcript   → NLLB-200 → system_en      (cascade output)
  3. Corpus BLEU(system_en, [[reference_en]]) and chrF2

This measures translation quality degradation introduced by ASR errors.
Reference translations are machine-generated (NLLB ceiling), not human,
so the metric is reported as "cascade BLEU vs MT ceiling".
"""

import os, sys, gc, warnings, json
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, ROOT as VANI_ROOT
import numpy as np
import tempfile, soundfile as sf
import sacrebleu

N_SAMPLES = 30

def load_nllb(paths, device):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    model_path = str(VANI_ROOT / paths["nllb_model"])
    print(f"  Loading NLLB from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

def translate_nllb(text, tokenizer, model, src_lang="hin_Deva", tgt_lang="eng_Latn"):
    import torch
    if not text or not text.strip():
        return ""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=256,
            num_beams=2,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    config = load_config()
    paths  = config["paths"]
    device = config.get("device", "cpu")

    print("\nLoading ASR model...")
    from asr_module import ASRModule
    asr = ASRModule(
        model_path=str(VANI_ROOT / paths["whisper_model"]),
        device=device,
        cfg=config.get("asr", {}),
    )

    print("Loading NLLB translation model...")
    tokenizer, nllb = load_nllb(paths, device)

    print("Loading Hindi dataset (streaming)...")
    from datasets import load_dataset
    ds = load_dataset(
        "MatrixSpeechAI/All_Hindi_ASR_v1.2",
        split="train",
        streaming=True,
    )

    system_translations  = []   # cascade: ASR → NLLB
    reference_translations = []  # ceiling: ref text → NLLB
    wer_scores = []

    collected = 0
    for sample in ds:
        if collected >= N_SAMPLES:
            break

        audio_data = sample.get("audio")
        if audio_data is None:
            continue
        try:
            if hasattr(audio_data, "get_all_samples"):
                s   = audio_data.get_all_samples()
                arr = np.array(s.data[0], dtype=np.float32)
                sr  = int(s.sample_rate)
            elif isinstance(audio_data, dict):
                arr = np.array(audio_data["array"], dtype=np.float32)
                sr  = int(audio_data["sampling_rate"])
            else:
                continue
        except Exception:
            continue

        dur = len(arr) / sr
        if dur < 2.0 or dur > 20.0:
            continue

        ref_text = sample.get("transcription", "")
        if not ref_text:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, arr, sr)
            audio_path = tmp.name

        try:
            # 1. ASR
            asr.reset_language_cache()
            asr_result  = asr.transcribe(audio_path, language_hint="hi")
            asr_text    = asr_result.get("transcript", "")

            # 2. Translate reference Hindi → English
            ref_en  = translate_nllb(ref_text,  tokenizer, nllb, src_lang="hin_Deva")
            # 3. Translate ASR output → English
            sys_en  = translate_nllb(asr_text,  tokenizer, nllb, src_lang="hin_Deva")

            reference_translations.append(ref_en)
            system_translations.append(sys_en)

            collected += 1
            print(f"  [{collected:2d}/{N_SAMPLES}]  ref_en: {ref_en[:60]}...")

        except Exception as e:
            print(f"  [ERR] {e}")
        finally:
            os.unlink(audio_path)

    del asr, nllb, tokenizer; gc.collect()

    if len(system_translations) < 5:
        print("Not enough samples for BLEU computation.")
        sys.exit(1)

    # Corpus BLEU and chrF2
    bleu  = sacrebleu.corpus_bleu(system_translations,
                                   [reference_translations])
    chrf  = sacrebleu.corpus_chrf(system_translations,
                                   [reference_translations])

    print(f"\n{'='*55}")
    print(f"  Hindi Cascade ASR+MT Translation Quality")
    print(f"  Samples : {len(system_translations)}")
    print(f"  BLEU    : {bleu.score:.1f}")
    print(f"  chrF2   : {chrf.score:.1f}")
    print(f"  (Reference = NLLB MT ceiling on ground-truth Hindi)")
    print(f"{'='*55}")

    result = {
        "n_samples": len(system_translations),
        "bleu":  round(bleu.score,  1),
        "chrf2": round(chrf.score,  1),
        "system":    system_translations[:5],
        "reference": reference_translations[:5],
    }
    Path("bleu_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nResults saved to bleu_results.json")

if __name__ == "__main__":
    main()

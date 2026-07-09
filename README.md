# VANI – Voice Analysis & Neural Intelligence

**Offline AI-based Radio Intercept Analysis System**  
M.Tech Research Project · IIT Indore

---

## Overview

VANI processes radio intercepts in multiple Indic and foreign languages, producing:
- Real-time transcription via language-specific fine-tuned Whisper ASR models
- Automatic language identification (MMS-LID, 256 languages)
- English translation via NLLB-200
- Keyword-based threat detection
- Structured Intelligence Summary (ISUM) reports via Gemma 3:12B
- Searchable transcript database

All processing is **fully offline** — no internet connection required after setup.

---

## Supported Languages

| Language | ISO | ASR Model | Eval WER (FT) | Eval WER (Baseline) | Translation |
|----------|-----|-----------|---------------|---------------------|-------------|
| Punjabi | pa | whisper-large-v3-pa-ct2 ★ | 55.67% | 105.79% | NLLB-200 |
| Pashto | ps | whisper-medium-pashto-ct2 ★ | 38.55% | 94.23% | NLLB-200 |
| Urdu | ur | whisper-large-v3-ur-ct2 ★ | 19.82% | 24.44% | NLLB-200 |
| Nepali | ne | whisper-large-v3-ne-ct2 ★ | 49.24% | 94.55% | NLLB-200 |
| Mandarin | zh | whisper-large-v3-zh-ct2 ★ | 16.03% | 100.03%† | NLLB-200 |
| Hindi | hi | whisper-large-v3-hi-ct2 ★ | 19.78% | 30.29% | NLLB-200 |
| Kashmiri | ks | whisper-large-v3-ks-ct2 ★ | —‡ | —‡ | NLLB-200 |
| Dogri | doi | whisper-large-v3-turbo-ct2 | — | — | IndicTrans2 |
| Burmese | my | whisper-large-v3-turbo-ct2 | — | — | NLLB-200 |

★ Fine-tuned with LoRA (r=8, α=16) on FLEURS / IndicVoices. See `docs/FINETUNE_REPORT.md`.  
† Baseline turbo model translates Mandarin to English by default; fine-tuned large-v3 transcribes correctly.  
‡ Whisper has no `ks` vocabulary token — trained with `whisper_lang="ur"` Nastaliq proxy on IndicVoices (20k samples). WER vs Kashmiri refs is not meaningful; eval loss 0.936 at checkpoint-1500.

### Cross-model comparison (100-sample test, 26 June 2026)

| Language | FT Whisper WER | ZS SM4T WER | FT SM4T WER | Best ASR |
|----------|--------------|------------|------------|---------|
| Punjabi | 55.67% | **19.77%** | 19.77% | ZS/FT SM4T |
| Pashto | **38.55%** | 44.4% | 41.22% | **FT Whisper** |
| Urdu | 19.82% | **16.9%** | 17.26% | ZS SM4T |
| Nepali | 49.24% | **28.46%** | 28.92% | ZS SM4T |
| Mandarin | **16.03%** | 100.0%§ | 60.53% | **FT Whisper** |
| Hindi | 19.78% | 15.44% | **13.43%** | FT SM4T |

§ ZS SM4T Mandarin WER = 100.0% is a script-normalisation mismatch, not a model failure.  
⚠ FT SM4T translation (S2TT chrF) is near zero — ASR-only LoRA fine-tuning breaks multi-task language conditioning.  
Full results: `docs/model_comparison_results.json` · `docs/seamless_ft_results.json` · `docs/model_comparison_report.md`

---

## Project Structure

```
offline_ai_system_v2/
├── app.py                           # Streamlit UI entry point
├── finetune_whisper.py              # LoRA fine-tuning script (all languages)
├── run_full_pipeline_batch.py       # Batch pipeline runner
├── config.yaml                      # All configuration
│
├── docs/
│   ├── FINETUNE_REPORT.md           # Full fine-tuning + eval report
│   ├── model_comparison_results.json# Raw 7-language cross-model eval data
│   └── model_comparison_report.md   # Cross-model comparison report
│
├── models/                          # Deployed CT2 models (int8 quantized)
│   ├── whisper-large-v3-pa-ct2/     # Punjabi   — eval WER 55.67%
│   ├── whisper-medium-pashto-ct2/   # Pashto    — eval WER 38.55%
│   ├── whisper-large-v3-ur-ct2/     # Urdu      — eval WER 19.82%
│   ├── whisper-large-v3-ne-ct2/     # Nepali    — eval WER 49.24%
│   ├── whisper-large-v3-zh-ct2/     # Mandarin  — eval WER 16.03%
│   ├── whisper-large-v3-hi-ct2/     # Hindi     — eval WER 19.78%
│   ├── whisper-large-v3-ks-ct2/     # Kashmiri  — eval_loss 0.936
│   ├── whisper-large-v3-turbo-ct2/  # Dogri/Burmese fallback
│   ├── nllb-200-distilled-600M/     # Translation model
│   ├── mms-lid-256/                 # MMS language ID (256 languages)
│   └── langid/lid.176.bin           # FastText language ID fallback
│
├── finetune_runs/                   # LoRA checkpoints
│   ├── pa/adapter/                  # Punjabi   (ckpt-1000)
│   ├── ps/adapter/                  # Pashto    (ckpt-1000)
│   ├── ur/adapter/                  # Urdu      (ckpt-1000)
│   ├── ne/adapter/                  # Nepali    (ckpt-1000)
│   ├── zh/adapter/checkpoint-400/   # Mandarin  (best ckpt, step 400)
│   ├── hi/adapter/                  # Hindi     (ckpt-600)
│   └── ks/adapter/                  # Kashmiri  (ckpt-1500, ur proxy)
│
├── scripts/
│   ├── eval/
│   │   ├── compare_all_models.py    # Cross-model evaluation (Whisper vs SeamlessM4T)
│   │   ├── eval_fleurs.py           # WER evaluation on FLEURS
│   │   ├── ablation_eval.py         # Ablation study
│   │   └── robustness_eval.py       # Noise/distortion robustness eval
│   ├── paper/                       # Paper and presentation generators
│   └── utils/
│       ├── download_models.py       # Download base models from HuggingFace
│       ├── download_lang_models.py  # Download language-specific models
│       └── download_fleurs.py       # Pre-download FLEURS datasets
│
├── src/                             # Core pipeline modules
│   ├── pipeline.py                  # 10-stage orchestration pipeline
│   ├── asr_module.py                # Whisper ASR (faster-whisper)
│   ├── language_module.py           # MMS-LID + script-cascade routing
│   ├── translation_module.py        # NLLB-200 translation
│   ├── vad_module.py                # Silero VAD
│   ├── preprocessing.py             # Bandpass + noise reduction
│   ├── chunker.py                   # VAD-aware chunking
│   ├── keyword_module.py            # Keyword/entity detection
│   ├── isum_module.py               # ISUM via Gemma 3:12B (Ollama)
│   ├── database.py                  # SQLite storage
│   └── search.py                    # Transcript search
│
├── alerts/keyword_dictionary.json   # Multilingual keyword dictionary
├── input_audio/                     # Drop input WAV files here
├── output/                          # Pipeline JSON outputs
├── database/transcripts.db          # SQLite transcript database
└── logs/                            # System logs
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# PyTorch with CUDA (RTX 5060 / CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### 2. Download models (one time, requires internet)

```bash
# Base Whisper model (CTranslate2 format)
python scripts/utils/download_models.py

# Language-specific fine-tuned models
python scripts/utils/download_lang_models.py

# Pre-download FLEURS eval datasets (optional)
python scripts/utils/download_fleurs.py
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Fine-Tuning

All 7 language models were fine-tuned using LoRA (r=8, α=16, target=q_proj+v_proj, 0.25% trainable params) on an RTX 5060 8 GB:

```bash
# Fine-tune a language (e.g. Urdu)
python finetune_whisper.py --lang ur

# Fine-tune with custom settings
python finetune_whisper.py --lang hi --steps 600 --max-grad-norm 0.5
```

The script automatically merges LoRA weights, converts to CT2 int8, and copies `tokenizer.json` into the CT2 directory (required for correct task-token lookup on large-v3 models).

See `docs/FINETUNE_REPORT.md` for full training details, eval results, and the CT2 tokenizer bug fix.

---

## Pipeline Architecture

```
Audio Input
  → Stage 1:  VAD (Silero)
  → Stage 2:  Preprocessing (bandpass 300–3400 Hz, noise reduction)
  → Stage 3:  MMS-LID language detection (256-language model)
  → Stage 3.5: Language-specific Whisper model selection
       pa → whisper-large-v3-pa-ct2   → NLLB-200
       ps → whisper-medium-pashto-ct2 → NLLB-200
       ur → whisper-large-v3-ur-ct2   → NLLB-200
       ne → whisper-large-v3-ne-ct2   → NLLB-200
       zh → whisper-large-v3-zh-ct2   → NLLB-200
       hi → whisper-large-v3-hi-ct2   → NLLB-200
       ks → whisper-large-v3-ks-ct2   → NLLB-200
       doi → whisper-large-v3-turbo   → IndicTrans2
  → Stage 4:  ASR transcription
  → Stage 5:  Script-cascade override (Arabic-script detection for Urdu/Kashmiri)
  → Stage 6:  Translation → English
  → Stage 7:  Speaker diarization
  → Stage 8:  Keyword/entity detection
  → Stage 9:  ISUM summary (Gemma 3:12B via Ollama)
  → Stage 10: SQLite DB + JSON report export
```

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5060 8 GB VRAM |
| OS | Windows 11 |
| Runtime | CUDA · faster-whisper · CTranslate2 int8 |
| Fine-tuning | ~5.5 GB VRAM per model · 1–3h per language |

---

*Developed as M.Tech Research Project, IIT Indore*

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

| Language | ISO | Deployed ASR Backend | Deployed WER | Whisper FT WER | Whisper Baseline WER† | Translation |
|----------|-----|----------------------|--------------|----------------|------------------------|-------------|
| Punjabi | pa | SeamlessM4T v2 (zero-shot) | 19.77% | 57.39% | 77.60% | NLLB-200 |
| Pashto | ps | SeamlessM4T v2 + LoRA ★ | 36.91% | 38.55% | 89.76% | NLLB-200 |
| Urdu | ur | SeamlessM4T v2 (zero-shot) | 16.90% | 19.82% | 21.23% | NLLB-200 |
| Nepali | ne | SeamlessM4T v2 + LoRA ★ | 24.34% | 50.92% | 88.85% | NLLB-200 |
| Mandarin | zh | SeamlessM4T v2 (zero-shot) | 11.69% | 14.22%‡ | 10.99% | NLLB-200 |
| Hindi | hi | SeamlessM4T v2 + LoRA ★ | 12.91% | 19.78% | 26.34% | NLLB-200 |
| Kashmiri | ks | whisper-large-v3-ks-ct2 ★ | 74.02%¶ | 74.02% | 96.87% | NLLB-200 |
| Dogri | doi | whisper-large-v3-turbo-ct2 | — | — | — | IndicTrans2 |
| Burmese | my | whisper-large-v3-turbo-ct2 | — | — | — | NLLB-200 |

★ Fine-tuned with LoRA. Whisper models: r=8, α=16 on FLEURS / IndicVoices. Hindi and Nepali are SeamlessM4T LoRA adapters trained on FLEURS + IndicVoices-R (deployed 2026-07-18); Pashto is a noise-augmented SeamlessM4T LoRA adapter (r=32 incl. MLP, FLEURS + Common Voice, deployed 2026-07-19) — each deployed only after winning the radio-degradation sweep. See `docs/FINETUNE_REPORT.md`.  
† True `openai/whisper-large-v3` baseline, CJK-aware scoring (corrected eval, 2026-07-10; the previously published baselines used a mislabelled turbo model and whitespace WER on character-spaced Han).  
‡ Mandarin fine-tuning *regressed* vs the 10.99% baseline (+3.2 pp); the fine-tuned zh model is retained but not deployed.  
¶ Kashmiri: custom `<|ks|>` vocab token (ID 51866, embedding initialised from `<|ur|>`), IndicVoices-R (20k samples), best checkpoint step 2400. SeamlessM4T has no Kashmiri, so ks stays on fine-tuned Whisper.

**Backend routing (since 2026-07-11, adapters updated 2026-07-19):** `asr.seamless_langs: [pa, ne, hi, ur, zh, ps]` routes six languages to SeamlessM4T — Hindi, Nepali, and Pashto with deployed LoRA adapters, the rest zero-shot. Pashto took five attempts: the domain-pretrained Whisper-medium base survived data scaling, capacity scaling, and decode tuning, and finally fell to **noise-augmented training** (degrading training audio with the evaluation's own bandpass/noise/codec pipeline), which fixed the 0 dB SNR collapse (87.2 → 56.0 vs Whisper's 64.8) while *improving* clean accuracy (36.91 vs 38.55). Whisper keeps only Kashmiri (SM4T has no `kas`; custom-token attempts lost by 14+ points).

### Cross-model comparison (100-sample FLEURS test, corrected re-run 2026-07-10)

| Language | Whisper Baseline | FT Whisper | ZS SM4T | FT SM4T | Deployed |
|----------|------------------|------------|---------|---------|----------|
| Punjabi | 77.60% | 57.39% | **19.77%** | 19.77% | ZS SM4T |
| Pashto | 89.76% | 38.55% | 44.40% | **36.91%** | **FT SM4T (LoRA)** |
| Urdu | 21.23% | 19.82% | **16.90%** | 17.26% | ZS SM4T |
| Nepali | 88.85% | 50.92% | 28.46% | **24.34%** | **FT SM4T (LoRA)** |
| Mandarin | **10.99%** | 14.22% | 11.69% | 18.68% | ZS SM4T |
| Hindi | 26.34% | 19.78% | 15.44% | **12.91%** | **FT SM4T (LoRA)** |

Note: the FT SM4T translation collapse reported earlier (chrF ≈ 0) was a **label-encoding bug** in the training script (targets tokenised in source mode with an `__eng__` prefix), not a property of LoRA fine-tuning — corrected-label retrains recover chrF (ps 2.12→37.60, hi 0.06→43.24) with unchanged ASR. Moot in production: VANI translates with NLLB-200 regardless.  
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
├── models/                          # Deployed models
│   ├── seamless-m4t-v2-large/       # SeamlessM4T v2 — DEPLOYED for pa/ne/hi/ur/zh/ps (hi/ne/ps with LoRA)
│   ├── whisper-medium-pashto-ct2/   # Pashto    — retained for rollback; SM4T+LoRA serves ps
│   ├── whisper-large-v3-ks-ct2/     # Kashmiri  — DEPLOYED, eval WER 74.02% (ckpt-2400)
│   ├── whisper-large-v3-pa-ct2/     # Punjabi   — retained; SM4T serves pa
│   ├── whisper-large-v3-ur-ct2/     # Urdu      — retained; SM4T serves ur
│   ├── whisper-large-v3-ne-ct2/     # Nepali    — retained; SM4T serves ne
│   ├── whisper-large-v3-zh-ct2/     # Mandarin  — retained; SM4T serves zh (FT regressed)
│   ├── whisper-large-v3-hi-ct2/     # Hindi     — retained; SM4T+LoRA serves hi
│   ├── whisper-large-v3-turbo-ct2/  # Dogri/Burmese fallback
│   ├── whisper-large-v3-ct2/        # True large-v3 baseline (evaluation only)
│   ├── nllb-200-distilled-600M/     # Translation model
│   ├── mms-lid-256/                 # MMS language ID (256 languages)
│   └── langid/lid.176.bin           # FastText language ID fallback
│
├── finetune_runs/                   # Whisper LoRA checkpoints
│   ├── pa/adapter/                  # Punjabi   (v3 best = ckpt-4000)
│   ├── ps/adapter/                  # Pashto    (ckpt-1000)
│   ├── ur/adapter/                  # Urdu      (ckpt-800)
│   ├── ne/adapter/                  # Nepali    (ckpt-3000)
│   ├── zh/adapter/checkpoint-400/   # Mandarin  (best ckpt, step 400)
│   ├── hi/adapter/                  # Hindi     (ckpt-600)
│   └── ks/adapter/                  # Kashmiri  (best = ckpt-2400, custom <|ks|> token)
│
├── finetune_runs_seamless/          # SeamlessM4T LoRA adapters (hi_iv + ne_iv + ps_aug DEPLOYED)
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

All 7 language-specific Whisper models were fine-tuned using LoRA (r=8, α=16, target=q_proj+v_proj, 0.25% trainable params) on an RTX 5060 8 GB. Note that after the corrected evaluation (2026-07-10), only the Pashto and Kashmiri Whisper models serve ASR in the pipeline; Hindi is served by a SeamlessM4T LoRA adapter, and the rest run zero-shot SeamlessM4T:

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
  → Stage 3.5: Per-language ASR backend selection
       pa → SeamlessM4T v2 (zero-shot)      → NLLB-200
       ne → SeamlessM4T v2 + ne LoRA        → NLLB-200
       hi → SeamlessM4T v2 + hi LoRA        → NLLB-200
       ur → SeamlessM4T v2 (zero-shot)      → NLLB-200
       zh → SeamlessM4T v2 (zero-shot)      → NLLB-200
       ps → SeamlessM4T v2 + ps LoRA        → NLLB-200
       ks → whisper-large-v3-ks-ct2 (FT)    → NLLB-200
       doi → whisper-large-v3-turbo         → IndicTrans2
  → Stage 4:  ASR transcription (SeamlessM4T runs per VAD utterance → timed segments)
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

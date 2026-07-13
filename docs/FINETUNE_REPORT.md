# VANI — ASR Backend Selection & Whisper Fine-Tuning Report
**Voice Analysis & Neural Intelligence System**
**Date:** 22 June 2026 · **Corrected:** 11–12 July 2026 (see Corrections)
**Hardware:** Windows 11 · NVIDIA RTX 5060 8 GB VRAM · CUDA

---

## Overview

Seven language-specific Whisper ASR models were fine-tuned using LoRA (Low-Rank Adaptation) for border-region radio intercept languages, then evaluated head-to-head against zero-shot SeamlessM4T v2 on held-out test sets, both clean and under radio-channel degradation. **The corrected evaluation shows that per-language fine-tuning is justified only for Pashto and Kashmiri; for the other five languages, zero-shot SeamlessM4T wins outright and is now the deployed backend** (`asr.seamless_langs: [pa, ne, hi, ur, zh]`). All Whisper models are deployed in int8 quantized CTranslate2 (CT2) format via faster-whisper.

---

## Corrections (2026-07-11)

The originally published numbers contained two independent scoring defects; both are fixed and all figures below are from the corrected re-run (n=100 per language, `docs/model_comparison_results.json` + `docs/seamless_ft_results.json`; pre-fix values archived at `docs/*_PRE_FIX_2026-07-10.json`).

1. **The "Whisper large-v3 baseline" was actually large-v3-turbo.** The comparison script loaded `whisper-large-v3-turbo-ct2` while labelling it large-v3. Every baseline was inflated, so the report systematically overstated fine-tuning gains (e.g. Punjabi's gain shrinks from −50 pp to −20 pp; Urdu's nearly vanishes at −1.4 pp). The corrected baseline is the true `openai/whisper-large-v3`.
2. **Mandarin WER was a whitespace-tokenisation artefact.** FLEURS Mandarin references are character-spaced; WER tokenises on whitespace; un-fine-tuned models emit unspaced Han — so a near-perfect transcript scored ~100%. The previously reported "baseline 100.03% → fine-tuned 16.03% (−84 pp)" was entirely this artefact. Re-scored with CJK character segmentation: **baseline 10.99%, fine-tuned 14.22% — fine-tuning REGRESSED Mandarin by +3.2 pp.** The earlier claim that the baseline "translates Chinese to English by default" was wrong (the task token was forced to transcribe throughout) and is withdrawn. The same artefact affected zero-shot SeamlessM4T Mandarin (published 100.0%, true 11.69%) and fine-tuned SeamlessM4T (published 60.53%, true 18.68%).

---

## Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Method | LoRA (PEFT) |
| LoRA rank (r) | 8 (Punjabi v3: r=16, α=32) |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Trainable params | ~3.9M / 1.55B total (0.25%) |
| Batch size | 2 (effective) |
| Learning rate | 5×10⁻⁵ with linear warmup |
| Quantization | int8 (CT2) |
| Eval metric | Word Error Rate (WER) |
| Best model selection | `load_best_model_at_end=True` |

---

## Results Summary

| # | Language | Script | Base Model | Dataset | Train Samples | Best Train-Val WER | Held-Out Eval WER† | Deployed ASR Backend |
|---|----------|--------|------------|---------|--------------|--------------------|--------------------|----------------------|
| 1 | Punjabi (pa) | Gurmukhi | whisper-large-v3 | FLEURS pa_in + IndicVoices-R | 21,923 (v3) | 49.31% @ step 4000 | 57.39% | **SeamlessM4T (19.77%)** |
| 2 | Pashto (ps) | Nastaliq | pashto-ghag-whisper-medium | FLEURS ps_af | 2,082 | 38.55% @ step 2000 | 38.55% | **FT Whisper** |
| 3 | Urdu (ur) | Nastaliq | whisper-large-v3 | FLEURS ur_pk | 2,109 | 22.27% @ step 800 | 19.82% | **SeamlessM4T (16.90%)** |
| 4 | Nepali (ne) | Devanagari | whisper-large-v3 | FLEURS ne_np + IndicVoices-R | 13,332 | 50.82% @ step 3000 | 50.92% | **SeamlessM4T (28.46%)** |
| 5 | Mandarin (zh) | Simplified Han | whisper-large-v3 | FLEURS cmn_hans_cn | 3,246 | 8.97% @ step 400‡ | 14.22%§ | **SeamlessM4T (11.69%)** |
| 6 | Hindi (hi) | Devanagari | whisper-large-v3 | FLEURS hi_in | 2,120 | 23.13% @ step 600 | 19.78% | **SeamlessM4T + LoRA (13.94%)** |
| 7 | Kashmiri (ks) | Nastaliq | whisper-large-v3 + `<\|ks\|>` token | IndicVoices-R Kashmiri | 20,000 | 74.02% @ step 2400 | 74.02%¶ | **FT Whisper** |

† Held-out eval: 100-sample cross-model evaluation on the FLEURS test split (pa/ps/ur/ne/zh/hi), one CJK-aware normaliser, true large-v3 baseline. See Cross-Model Evaluation.
‡ Training diverged at step ~820 (grad_norm spike in fp16); best checkpoint at step 400 used. Subsequent runs use `max_grad_norm=0.5`.
§ Mandarin's strong training-val WER (8.97%) did not generalise: held-out test WER (14.22%) is *worse* than the un-fine-tuned large-v3 baseline (10.99%) — a +3.2 pp regression, likely from the small single-domain training set narrowing the model. The fine-tuned Mandarin model is not deployed.
¶ Kashmiri: the n=100 cross-model re-run was not repeated for ks (its loader pulls the full 18 GB IndicVoices train split); 74.02% is the training-eval on the IndicVoices-R test split. Baseline 96.87% (−22.85 pp gain).

---

## Cross-Model Evaluation

100 samples per language · FLEURS test set (pa/ps/ur/ne/zh/hi) · corrected re-run 2026-07-10 · true `openai/whisper-large-v3` baseline · CJK-aware normaliser

### ASR Word Error Rate (source-language transcription, lower is better)

Cells give **WER% (CER%)**. CER is reported alongside WER because two of the seven scripts (Han, Perso-Arabic Kashmiri) have orthographies where whitespace tokenisation misleads — the defect class behind the original Mandarin numbers. Mandarin is scored with character segmentation, so its WER and CER coincide by construction. **Never compare a character-level number against a word-level one** — CER runs ~2–3× lower than WER on the space-delimited languages.

| Language | Whisper Baseline (large-v3) | Whisper Fine-Tuned | SeamlessM4T v2 (zero-shot) | FT Gain vs Baseline (WER) | Deployed Backend |
|----------|------------------------------|--------------------|----------------------------|---------------------|------------------|
| Punjabi (pa) | 77.60 (39.73) | 57.39 (32.52) | **19.77 (9.97)** | −20.2 pp | SeamlessM4T |
| Pashto (ps) | 89.76 (37.60) | **38.55 (17.65)** | 44.40 (22.92) | −51.2 pp | FT Whisper |
| Urdu (ur) | 21.23 (8.12) | 19.82 (7.29) | **16.90 (7.00)** | −1.4 pp | SeamlessM4T |
| Nepali (ne) | 88.85 (29.26) | 50.92 (18.83) | **28.46 (11.22)** | −37.9 pp | SeamlessM4T |
| Mandarin (zh) | **10.99 (10.99)** | 14.22 (14.22) | 11.69 (11.69) | **+3.2 pp (regression)** | SeamlessM4T |
| Hindi (hi) | 26.34 (10.55) | 19.78 (7.46) | **15.44 (9.12)** | −6.6 pp | SeamlessM4T + LoRA |
| Kashmiri (ks) | 96.87 (—) | **74.02 (—)**† | —‡ | −22.85 pp | FT Whisper |

† Kashmiri values are the training-time eval on the IndicVoices-R test split (custom `<|ks|>` token model, best checkpoint step 2400); the cross-model re-run was skipped for ks (18 GB loader issue), and that eval did not record CER (hence the dash). On the 30-clip robustness set (clean condition, `eval_data/wer_robustness_results.csv`) the deployed ks model scores **81.46% WER / 47.95% CER** — the wide gap reflects Perso-Arabic orthographic variation that WER over-penalises. That is a different sample set than the 74.02% column, so the two must not be read as one WER (CER) pair.
‡ SeamlessM4T v2 has no Kashmiri (`kas` absent from the model vocabulary). A Urdu-token proxy was tested and failed (109% WER — fluent but unrelated Urdu), so Kashmiri necessarily stays on fine-tuned Whisper.

### Translation Quality — chrF → English (higher is better)

| Language | Whisper+NLLB-200 | SeamlessM4T S2TT | Winner |
|----------|-----------------|-----------------|--------|
| Punjabi (pa) | 40.15 | **54.53** | SeamlessM4T |
| Pashto (ps) | **44.48** | 40.15 | Whisper+NLLB |
| Urdu (ur) | **51.34** | 50.73 | Whisper+NLLB |
| Nepali (ne) | 45.55 | **51.67** | SeamlessM4T |
| Mandarin (zh) | 42.00 | **49.15** | SeamlessM4T |
| Hindi (hi) | **53.71** | 51.54 | Whisper+NLLB |
| Kashmiri (ks) | — | — | — (no English refs; SM4T lacks `kas`) |

**Key observations:**
- Zero-shot SeamlessM4T beats fine-tuned Whisper on clean-speech ASR for **five of six** comparable languages (pa, ne, hi, ur, zh); fine-tuned Whisper wins only **Pashto**.
- For **Mandarin**, even the *un-fine-tuned* large-v3 baseline (10.99%) beats the fine-tuned model (14.22%); SeamlessM4T (11.69%) is deployed because it also wins every degradation condition (see next section).
- chrF winners split 3–3: SeamlessM4T S2TT for pa/ne/zh, Whisper+NLLB for ps/ur/hi. VANI keeps NLLB downstream regardless, using only SeamlessM4T's ASR output.
- SeamlessM4T adds ~10 GB to the deployment footprint but was already resident for pa/ne, so routing hi/ur/zh to it costs no additional VRAM.

### ASR WER under Radio-Channel Degradation

Because VANI processes degraded radio rather than clean speech, the backend choice must survive channel effects. The table gives SeamlessM4T's WER advantage over fine-tuned Whisper (FT WER − SeamlessM4T WER, in points; positive = SeamlessM4T better) on the same 30 FLEURS clips per language under five conditions (source: `eval_data/wer_robustness_results.csv`):

| Language | Clean | Bandpass 300–3400 Hz | Noise 10 dB | Noise 0 dB | MP3 codec | Winner |
|----------|-------|----------------------|-------------|------------|-----------|--------|
| Punjabi (pa) | +39.0 | +37.7 | +41.7 | +36.8 | +41.8 | **SeamlessM4T** |
| Nepali (ne) | +31.6 | +30.5 | +34.1 | +36.4 | +32.0 | **SeamlessM4T** |
| Hindi (hi) | +5.0 | +3.9 | +10.8 | +19.5 | +5.6 | **SeamlessM4T** |
| Urdu (ur) | +2.3 | +2.4 | +4.8 | +4.2 | +3.1 | **SeamlessM4T** |
| Mandarin (zh) | +3.6 | +5.0 | +2.9 | +17.2 | +3.0 | **SeamlessM4T** |
| Pashto (ps) | −6.7 | −2.1 | −1.9 | +5.6 | −2.8 | **FT Whisper**\* |

SeamlessM4T's advantage is positive in every condition for all five routed languages and **widens as the channel worsens** — Hindi grows from +5.0 (clean) to +19.5 (0 dB), Mandarin from +3.6 to +17.2. The clean-speech ranking does not invert under noise, so the routing is safe for operational radio.

\* Pashto: fine-tuned Whisper wins clean/bandpass/10 dB/MP3; SeamlessM4T edges ahead only at 0 dB SNR, so Pashto stays on Whisper.

---

## Per-Language Detail

### 1. Punjabi (pa) — `whisper-large-v3-pa-ct2` *(retained, not serving ASR)*
- **Base:** `openai/whisper-large-v3`
- **Dataset:** FLEURS `pa_in` + IndicVoices-R Punjabi — v3: 21,923 train samples
- **Training:** v1 FLEURS-only (best 56.67%); v2 +IndicVoices-R (best 52.55% @ step 3000); v3 LoRA r=16 α=32, completed 4,000 steps, best train-val WER **49.31% @ step 4000**
- **Held-out eval WER:** 57.39% fine-tuned vs 77.60% true large-v3 baseline (−20.2 pp) · **SeamlessM4T: 19.77%**
- **Translation:** Whisper+NLLB chrF 40.15 · SeamlessM4T S2TT chrF 54.53
- **Pipeline role:** since 2026-07-11 routing change, `pa` → **SeamlessM4T** ASR → NLLB-200 → English. The fine-tuned model is retained but not serving.
- **Note:** CT2 tokenizer fix (2026-06-23, Finding 7) restored Gurmukhi transcription after the model initially appeared to "output English directly."

### 2. Pashto (ps) — `whisper-medium-pashto-ct2` *(DEPLOYED)*
- **Base:** `Nasimbahar/pashto-ghag-whisper-medium-asr` (domain-specific Pashto model)
- **Dataset:** FLEURS `ps_af` — 2,082 train / 251 val
- **Training:** 2000 steps, ~3.5 h
- **Held-out eval WER:** **38.55%** fine-tuned vs 89.76% baseline (−51.2 pp) · SeamlessM4T: 44.40%
- **Translation:** Whisper+NLLB chrF 44.48 · SeamlessM4T S2TT chrF 40.15 — Whisper+NLLB wins
- **Pipeline role:** MMS-LID routes `ps` → FT Whisper (Nastaliq) → NLLB-200 → English
- **Note:** The one clear per-language fine-tuning win among SeamlessM4T-supported languages — beats SeamlessM4T on both ASR (4 of 5 degradation conditions) and translation.

### 3. Urdu (ur) — `whisper-large-v3-ur-ct2` *(retained, not serving ASR)*
- **Base:** `openai/whisper-large-v3`
- **Dataset:** FLEURS `ur_pk` — 2,109 train / 267 val
- **Training:** 1000 steps, ~6 h; best train-val WER 22.27% @ step 800
- **Held-out eval WER:** 19.82% fine-tuned vs 21.23% baseline (**−1.4 pp** — the smallest real gain; the previously published −4.6 pp was inflated by the turbo mislabel) · **SeamlessM4T: 16.90%**
- **Translation:** Whisper+NLLB chrF 51.34 · SeamlessM4T S2TT chrF 50.73 — Whisper+NLLB wins narrowly
- **Pipeline role:** since 2026-07-11, `ur` → **SeamlessM4T** ASR → NLLB-200 → English
- **Script detection:** Arabic-script cascade (>20% Nastaliq chars) catches low-confidence MMS-LID detections

### 4. Nepali (ne) — `whisper-large-v3-ne-ct2` *(retained, not serving ASR)*
- **Base:** `openai/whisper-large-v3`
- **Dataset:** FLEURS `ne_np` + IndicVoices-R Nepali — 13,332 train / 572 val
- **Training:** v1 FLEURS-only (best 52.14%); v2 +IndicVoices-R, 3000 steps, best train-val WER 50.82% @ step 3000
- **Held-out eval WER:** 50.92% fine-tuned vs 88.85% baseline (−37.9 pp) · **SeamlessM4T: 28.46%**
- **Translation:** Whisper+NLLB chrF 45.55 · SeamlessM4T S2TT chrF 51.67
- **Pipeline role:** since 2026-07-11, `ne` → **SeamlessM4T** ASR → NLLB-200 → English
- **Note:** Large fine-tuning gain, but SeamlessM4T's zero-shot lead (22.5 pp) is larger still.

### 5. Mandarin Chinese (zh) — `whisper-large-v3-zh-ct2` *(retained, not serving ASR — fine-tuning regressed)*
- **Base:** `openai/whisper-large-v3`
- **Dataset:** FLEURS `cmn_hans_cn` — 3,246 train / 409 val
- **Training:** 400 steps used (best checkpoint), ~3 h 10 m; diverged at step ~820 (fp16 gradient spike, grad_norm=12.9); subsequent training uses `max_grad_norm=0.5`
- **Held-out eval WER:** 14.22% fine-tuned vs **10.99% baseline — fine-tuning REGRESSED Mandarin (+3.2 pp)** · SeamlessM4T: 11.69%
- **Translation:** Whisper+NLLB chrF 42.00 · SeamlessM4T S2TT chrF 49.15
- **Pipeline role:** since 2026-07-11, `zh` → **SeamlessM4T** ASR → NLLB-200 → English
- **Note (CORRECTED):** The previously reported "baseline 100.03% → fine-tuned 16.03% (−84 pp)" was a whitespace-tokenisation artefact on character-spaced Han references, and the "turbo translates Chinese by default" explanation was wrong (the transcribe task was always forced). With character segmentation, the un-fine-tuned baseline is 10.99% and the fine-tune is a regression — likely the small single-domain training set narrowing the model. The strong training-val WER (8.97% @ step 400) did not generalise. SeamlessM4T is deployed because it also wins all five degradation conditions.

### 6. Hindi (hi) — `whisper-large-v3-hi-ct2` *(retained, not serving ASR)*
- **Base:** `openai/whisper-large-v3`
- **Dataset:** FLEURS `hi_in` — 2,120 train / 239 val
- **Training:** 600 steps, ~6 h 45 m; `max_grad_norm=0.5` applied (lesson from Mandarin); fully stable; near-best WER by step 200
- **Held-out eval WER:** 19.78% fine-tuned vs 26.34% baseline (−6.6 pp, a genuine gain) · **SeamlessM4T: 15.44%; + LoRA adapter 13.94% (DEPLOYED 2026-07-13)**
- **Translation:** Whisper+NLLB chrF 53.71 · SeamlessM4T S2TT chrF 51.54 — Whisper+NLLB wins
- **Pipeline role:** since 2026-07-11, `hi` → **SeamlessM4T** ASR → NLLB-200 → English; since 2026-07-13 with the corrected-label **LoRA adapter** enabled for Hindi calls (13.94 vs 15.44 clean; wins 4/5 degradation conditions incl. bandpass 16.28 vs 18.96)

### 7. Kashmiri (ks) — `whisper-large-v3-ks-ct2` *(DEPLOYED)*
- **Base:** `openai/whisper-large-v3` + custom `<|ks|>` token (ID 51866)
- **Dataset:** IndicVoices-R Kashmiri — 20,000 train / 372 val
- **Training:** 3000 steps, ~18 h (incl. 2 power outages, PD-charger recovery)
- **Language token:** Whisper has no native `ks` token. A custom `<|ks|>` token was added to the vocab, with its embedding initialised from `<|ur|>` (Urdu — same Nastaliq script); forced prefix `[<|startoftranscript|>, <|ks|>, <|transcribe|>, <|notimestamps|>]` injected via TemplateProcessing. faster-whisper patched at import time to accept the `ks` language code.
- **Eval WER:** **74.02% @ step 2400** vs 96.87% baseline (−22.85 pp), on the IndicVoices-R test split
- **CT2 source:** checkpoint-2400 (trainer's best by WER)
- **Pipeline role:** MMS-LID routes `ks` → FT Whisper (Nastaliq) → NLLB-200 (`kas_Arab`) → English
- **Notes:**
  - An earlier Urdu-proxy run (`whisper_lang="ur"`, ckpt-1500, eval 103.58% — not a true accuracy figure) is superseded by the custom-token model.
  - SeamlessM4T v2 does not support Kashmiri (`kas` absent); the Urdu-proxy trick fails on SeamlessM4T too (109% WER). Whisper owns Kashmiri; the hybrid architecture is forced.
  - FLEURS has no Kashmiri config; Common Voice has no Kashmiri data → IndicVoices-R used.
  - The 96.87% baseline is itself inflated by a Unicode-normalisation mismatch in the reference text; CER should be reported alongside WER before further ks training decisions.

---

## VANI Pipeline Integration

All models integrate into the existing 10-stage VANI pipeline. **Since the 2026-07-11 routing change, Stage 3.5 selects the ASR backend per language:**

```
Audio Input
  → Stage 1: VAD (voice activity detection)
  → Stage 2: Preprocessing (bandpass 300–3400 Hz, noise reduction)
  → Stage 3: MMS-LID language detection (256-language model)
  → Stage 3.5: Per-language ASR backend selection
       MMS lang=pa  → SeamlessM4T v2 (zero-shot)      → NLLB-200
       MMS lang=ne  → SeamlessM4T v2 (zero-shot)      → NLLB-200
       MMS lang=hi  → SeamlessM4T v2 + hi LoRA        → NLLB-200
       MMS lang=ur  → SeamlessM4T v2 (zero-shot)      → NLLB-200
       MMS lang=zh  → SeamlessM4T v2 (zero-shot)      → NLLB-200
       MMS lang=ps  → whisper-medium-pashto-ct2 (FT)  → NLLB-200
       MMS lang=ks  → whisper-large-v3-ks-ct2 (FT)    → NLLB-200
  → Stage 4: ASR transcription (SeamlessM4T runs per VAD utterance → timed segments)
  → Stage 5: Script-cascade override (Arabic-script detection for Urdu/Kashmiri)
  → Stage 6: Translation (NLLB-200 → English)
  → Stage 7: Speaker diarization
  → Stage 8: Keyword/entity detection
  → Stage 9: ISUM summary (Gemma 3 via Ollama)
  → Stage 10: Database + report export
```

---

## Model File Locations

```
offline_ai_system_v2/
├── models/
│   ├── whisper-large-v3-pa-ct2/     (1479 MB) — Punjabi   (retained; SM4T serves pa)
│   ├── whisper-medium-pashto-ct2/   ( 734 MB) — Pashto    (DEPLOYED)
│   ├── whisper-large-v3-ur-ct2/     (1479 MB) — Urdu      (retained; SM4T serves ur)
│   ├── whisper-large-v3-ne-ct2/     (1479 MB) — Nepali    (retained; SM4T serves ne)
│   ├── whisper-large-v3-zh-ct2/     (1479 MB) — Mandarin  (retained; SM4T serves zh)
│   ├── whisper-large-v3-hi-ct2/     (1479 MB) — Hindi     (retained; SM4T serves hi)
│   ├── whisper-large-v3-ks-ct2/     (1479 MB) — Kashmiri  (DEPLOYED, ckpt-2400)
│   ├── whisper-large-v3-ct2/                  — true large-v3 baseline (evaluation)
│   └── seamless-m4t-v2-large/       (~10 GB)  — SeamlessM4T v2 (DEPLOYED: pa/ne/hi/ur/zh)
├── finetune_runs/
│   ├── pa/adapter/   — LoRA checkpoints (Punjabi, v3 best=ckpt-4000)
│   ├── ps/adapter/   — LoRA checkpoints (Pashto)
│   ├── ur/adapter/   — LoRA checkpoints (Urdu)
│   ├── ne/adapter/   — LoRA checkpoints (Nepali)
│   ├── zh/adapter/   — LoRA checkpoints (Mandarin, best=ckpt-400)
│   ├── hi/adapter/   — LoRA checkpoints (Hindi)
│   └── ks/adapter/   — LoRA checkpoints (Kashmiri, best=ckpt-2400)
├── finetune_runs_seamless/{hi,ne,pa,ps,ur,zh,ks}/adapter/ — SM4T LoRA (hi DEPLOYED 2026-07-13; rest not)
└── finetune_whisper.py              — Training script
```

---

## SeamlessM4T v2 Fine-Tuning

SeamlessM4T v2 large was fine-tuned with LoRA (r=8, α=16, target=q_proj+v_proj, 0.10% trainable params) on the same FLEURS data as Whisper, **for ASR only** (tgt_lang = src_lang). 1000 steps per language on RTX 5060.

### ASR WER — Fine-tuned SM4T vs Zero-shot SM4T vs Fine-tuned Whisper (corrected, n=100 held-out)

| Language | ZS SM4T WER | FT SM4T WER | Δ vs ZS | FT Whisper WER |
|----------|------------|------------|---------|---------------|
| Punjabi (pa) | 19.77% | 19.77% | 0.0 pp | 57.39% |
| Pashto (ps) | 44.40% | **41.30%**‡ | −3.1 pp | **38.55%** |
| Urdu (ur) | **16.90%** | 17.26% | +0.4 pp | 19.82% |
| Nepali (ne) | **28.46%** | 28.92% | +0.5 pp | 50.92% |
| Mandarin (zh) | **11.69%** | 18.68%† | +7.0 pp | 14.22% |
| Hindi (hi) | 15.44% | **13.94%**‡ | −1.5 pp | 19.78% |

† The previously published FT SM4T Mandarin figure (60.53%) carried the same whitespace artefact as the other Mandarin numbers; the corrected value is 18.68% — still a large regression vs zero-shot, so the zh adapter is not deployed.
‡ Pashto and Hindi are 2026-07-13 **corrected-label retrains** (see the label-bug note below); their wrong-label predecessors scored 41.22% and 13.43% — statistically identical ASR in both cases. The pa/ur/ne/zh rows are the original wrong-label adapters (not retrained).

**Fine-tuning SeamlessM4T is a wash for ASR except Hindi** (−1.5 pp vs zero-shot): Pashto improves but still loses to fine-tuned Whisper (with or without the label fix), and Mandarin regresses badly. **The hi adapter is DEPLOYED (2026-07-13)** — it also wins 4/5 radio-degradation conditions incl. bandpass (16.28 vs 18.96, 30-clip sweep; only 10 dB noise is a marginal +0.6 loss). It is enabled only for Hindi `generate()` calls via `asr.seamless_adapters`; every other language runs with adapters disabled, verified byte-identical to the plain base model. The wrong-label runs are archived as `*_PRE_LABELFIX_*` alongside the retrains in `finetune_runs_seamless/`.

### Translation (S2TT chrF) — Fine-tuned SM4T (broken)

| Language | ZS SM4T chrF | FT SM4T chrF | FT Whisper+NLLB chrF |
|----------|-------------|-------------|----------------------|
| Punjabi (pa) | 54.53 | 0.31 | 40.15 |
| Pashto (ps) | 40.15 | **37.60**‡ | 44.48 |
| Urdu (ur) | 50.73 | 0.55 | 51.34 |
| Nepali (ne) | 51.67 | 0.10 | 45.55 |
| Mandarin (zh) | 49.15 | 0.62 | 42.00 |
| Hindi (hi) | 51.54 | **43.24**‡ | 53.71 |

‡ Corrected-label retrains (2026-07-13); the wrong-label predecessors scored 2.12 (ps) and 0.06 (hi).

**Finding (CORRECTED 2026-07-13):** the near-zero chrF rows were **not** an inherent property of ASR-only LoRA fine-tuning — they were a label-encoding bug. The original training script tokenised targets as plain `text`, which SeamlessM4T's tokenizer encodes in *source* mode (prefix `__eng__`) regardless of the `tgt_lang` argument; the model therefore learned to emit source-language text after *any* prefix, destroying translation at inference. With labels correctly encoded in target mode (`text_target=`, prefix `[eos, __lang__]`), the retrained adapters' chrF recovers — Pashto 2.12 → **37.60**, Hindi 0.06 → **43.24** — while ASR WER is unchanged in both (41.30 vs 41.22; 13.94 vs 13.43). Two independent confirmations. The earlier explanation ("LoRA adaptation overrides multi-task language conditioning") is withdrawn. ASR-focused fine-tuning costs a few chrF points (vs zero-shot: ps −2.6, hi −8.3), not all of them. The pa/ur/ne/zh adapters still carry wrong-label training and their chrF rows remain artefactual. (Moot in production: VANI uses NLLB-200 for translation regardless.)

---

## Key Findings

1. **Backend selection beats blanket fine-tuning.** Against the corrected true large-v3 baseline, zero-shot SeamlessM4T wins clean-speech ASR for 5 of 6 supported languages and every degradation condition for those five. Per-language Whisper fine-tuning is justified only for **Pashto** (beats SM4T in 4/5 conditions) and **Kashmiri** (SM4T has no `kas`).
2. **Fine-tuning helps in proportion to how poorly the base model covers the language.** Real gains vs the true baseline: ps −51.2 pp, ne −37.9, ks −22.85, pa −20.2, hi −6.6, ur −1.4, zh **+3.2 (regression)**. The turbo-mislabelled baseline had inflated every one of these.
3. **A strong training-val WER can fail to generalise.** Mandarin's 8.97% train-val became 14.22% held-out — worse than the 10.99% un-fine-tuned baseline. Small single-domain training sets can narrow a strong base model.
4. **Scoring methodology is a result in itself.** Two silent defects (wrong baseline model; whitespace WER on character-spaced Han) produced an −84 pp headline gain that was entirely artefactual. All WER now goes through one CJK-aware normaliser, raw hypotheses are persisted to JSONL so re-scoring never needs a GPU re-run, and pre-fix results are archived for the erratum trail.
5. **The SeamlessM4T lead survives — and widens under — radio degradation.** Its advantage grows at 0 dB SNR (hi +19.5, zh +17.2), killing the "Whisper is more robust to noise" hypothesis.
6. **fp16 instability at low learning rates** — Mandarin training showed a gradient spike at step ~820 when LR decayed to ~1.5×10⁻⁵. `max_grad_norm=0.5` resolved this in subsequent training.
7. **CT2 models need `tokenizer.json` from the source model** — `ct2-transformers-converter` does not copy it; faster-whisper falls back to `whisper-tiny`'s tokenizer where `<|transcribe|>=50359`, but large-v3 uses `<|transcribe|>=50360` / `<|translate|>=50359`. The off-by-one made fine-tuned large-v3 CT2 models *translate* instead of transcribe. **Fix:** `finetune_whisper.py merge_and_convert()` now copies `tokenizer.json` into each CT2 directory.
8. **Kashmiri required a custom vocab token** — `<|ks|>` (ID 51866) added to large-v3 with embedding initialised from `<|ur|>`; 74.02% WER at step 2400 (−22.85 pp). The earlier Urdu-proxy approach (eval 103.58%, not a true accuracy figure) is superseded. SeamlessM4T cannot cover Kashmiri, so the hybrid Whisper+SeamlessM4T architecture is forced.
9. **Fine-tuning SeamlessM4T buys almost nothing on ASR except Hindi** (−1.5 pp vs zero-shot, and the win holds in 4/5 degradation conditions — so the hi LoRA adapter is deployed as of 2026-07-13; all other languages run zero-shot). The dramatic "fine-tuning breaks translation" effect reported earlier was a label-encoding bug (targets tokenised in source mode with an `__eng__` prefix), not a property of LoRA: corrected-label retrains recover chrF from 2.12 to 37.60 (ps) and 0.06 to 43.24 (hi) with statistically identical ASR in both. Scoring methodology strikes again — see Finding 4.
10. **Script-based cascade prevents misidentification** — the Arabic-script detection fallback correctly catches Urdu even when MMS-LID confidence is below threshold.

---

*Generated: 23 June 2026 · Cross-model eval: 23 June 2026 · Extended training + re-eval: 25 June 2026 · SeamlessM4T FT: 26 June 2026 · **Corrected eval (true large-v3 baseline, CJK-aware scoring) + backend routing change: 10–11 July 2026** · VANI v2 · RTX 5060 8 GB*

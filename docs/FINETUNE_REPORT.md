# VANI — ASR Backend Selection & Whisper Fine-Tuning Report
**Voice Analysis & Neural Intelligence System**
**Date:** 22 June 2026 · **Corrected:** 11–12 July 2026 (see Corrections)
**Hardware:** Windows 11 · NVIDIA RTX 5060 8 GB VRAM · CUDA

---

## Overview

Seven language-specific Whisper ASR models were fine-tuned using LoRA (Low-Rank Adaptation) for border-region radio intercept languages, then evaluated head-to-head against SeamlessM4T v2 on held-out test sets, both clean and under radio-channel degradation. **After the corrected evaluation and the 2026-07 extra-data campaign, SeamlessM4T serves all seven languages** (`asr.seamless_langs: [pa, ne, hi, ur, zh, ps, ks]`) — zero-shot for pa/ur/zh, and with per-language LoRA adapters for Hindi, Nepali, Pashto (via noise-augmented training) and, finally, Kashmiri (via a custom `__kas__` token made trainable, plus a scoring-ruler correction). **Fine-tuned Whisper is no longer the deployed backend for any language**; all seven Whisper models are retained on disk for rollback and are in int8 quantized CTranslate2 (CT2) format via faster-whisper.

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
| 2 | Pashto (ps) | Nastaliq | pashto-ghag-whisper-medium | FLEURS ps_af | 2,082 | 38.55% @ step 2000 | 38.55% | **SeamlessM4T + LoRA (36.91%)** |
| 3 | Urdu (ur) | Nastaliq | whisper-large-v3 | FLEURS ur_pk | 2,109 | 22.27% @ step 800 | 19.82% | **SeamlessM4T (16.90%)** |
| 4 | Nepali (ne) | Devanagari | whisper-large-v3 | FLEURS ne_np + IndicVoices-R | 13,332 | 50.82% @ step 3000 | 50.92% | **SeamlessM4T + LoRA (24.34%)** |
| 5 | Mandarin (zh) | Simplified Han | whisper-large-v3 | FLEURS cmn_hans_cn | 3,246 | 8.97% @ step 400‡ | 14.22%§ | **SeamlessM4T (11.69%)** |
| 6 | Hindi (hi) | Devanagari | whisper-large-v3 | FLEURS hi_in | 2,120 | 23.13% @ step 600 | 19.78% | **SeamlessM4T + LoRA (12.91%)** |
| 7 | Kashmiri (ks) | Nastaliq | whisper-large-v3 + `<\|ks\|>` token | IndicVoices-R Kashmiri | 20,000 | 74.02% @ step 2400 | 74.02%¶ | **SeamlessM4T + LoRA (64.31%)**◊ |

† Held-out eval: 100-sample cross-model evaluation on the FLEURS test split (pa/ps/ur/ne/zh/hi), one CJK-aware normaliser, true large-v3 baseline. See Cross-Model Evaluation.
‡ Training diverged at step ~820 (grad_norm spike in fp16); best checkpoint at step 400 used. Subsequent runs use `max_grad_norm=0.5`.
§ Mandarin's strong training-val WER (8.97%) did not generalise: held-out test WER (14.22%) is *worse* than the un-fine-tuned large-v3 baseline (10.99%) — a +3.2 pp regression, likely from the small single-domain training set narrowing the model. The fine-tuned Mandarin model is not deployed.
¶ Kashmiri: the n=100 cross-model re-run was not repeated for ks (its loader pulls the full 18 GB IndicVoices train split); 74.02% is the training-eval on the IndicVoices-R test split. Baseline 96.87% (−22.85 pp gain).  
◊ Deployed backend figure is the SeamlessM4T ks_max adapter's WER on diacritic-normalised Perso-Arabic text (372-clip IndicVoices test split, identical clips and scorer against the *actually deployed* Whisper CT2 artefact, which itself scores 79.29% raw on this ruler — not the training-eval's 74.02%). Raw, undiacritised WER is 80.91% for ks_max — see the Extra-Data Campaign section and `docs/ks_ruler_study.json` for the full normalisation-ladder result.

---

## Cross-Model Evaluation

100 samples per language · FLEURS test set (pa/ps/ur/ne/zh/hi) · corrected re-run 2026-07-10 · true `openai/whisper-large-v3` baseline · CJK-aware normaliser

### ASR Word Error Rate (source-language transcription, lower is better)

Cells give **WER% (CER%)**. CER is reported alongside WER because two of the seven scripts (Han, Perso-Arabic Kashmiri) have orthographies where whitespace tokenisation misleads — the defect class behind the original Mandarin numbers. Mandarin is scored with character segmentation, so its WER and CER coincide by construction. **Never compare a character-level number against a word-level one** — CER runs ~2–3× lower than WER on the space-delimited languages.

| Language | Whisper Baseline (large-v3) | Whisper Fine-Tuned | SeamlessM4T v2 (zero-shot) | FT Gain vs Baseline (WER) | Deployed Backend |
|----------|------------------------------|--------------------|----------------------------|---------------------|------------------|
| Punjabi (pa) | 77.60 (39.73) | 57.39 (32.52) | **19.77 (9.97)** | −20.2 pp | SeamlessM4T |
| Pashto (ps) | 89.76 (37.60) | **38.55 (17.65)** | 44.40 (22.92) | −51.2 pp | SeamlessM4T + LoRA |
| Urdu (ur) | 21.23 (8.12) | 19.82 (7.29) | **16.90 (7.00)** | −1.4 pp | SeamlessM4T |
| Nepali (ne) | 88.85 (29.26) | 50.92 (18.83) | **28.46 (11.22)** | −37.9 pp | SeamlessM4T + LoRA |
| Mandarin (zh) | **10.99 (10.99)** | 14.22 (14.22) | 11.69 (11.69) | **+3.2 pp (regression)** | SeamlessM4T |
| Hindi (hi) | 26.34 (10.55) | 19.78 (7.46) | **15.44 (9.12)** | −6.6 pp | SeamlessM4T + LoRA |
| Kashmiri (ks) | 96.87 (—) | 74.02 (—)† | —‡ | −22.85 pp | **SeamlessM4T + LoRA◊** |

† Kashmiri values are the training-time eval on the IndicVoices-R test split (custom `<|ks|>` token model, best checkpoint step 2400); the cross-model re-run was skipped for ks (18 GB loader issue), and that eval did not record CER (hence the dash). Note this "Whisper Fine-Tuned" column is now the *rollback* model, not the deployed one — see ◊.
‡ SeamlessM4T v2 has no *zero-shot* Kashmiri (`kas` absent from the model's pretrained vocabulary) — this column is genuinely empty. Its LoRA adapter adds a custom `__kas__` token instead (◊); a zero-shot Urdu-token proxy was tested separately and failed (109% WER — fluent but unrelated Urdu).
◊ **Deployed 2026-07-20.** The ks_max SeamlessM4T LoRA adapter (custom `__kas__` token, r=32 incl. MLP, plus — a first for this project — a *trainable* `__kas__` embedding row) scores 80.91% raw WER on the same 372-clip test split, nominally behind. Two corrections changed the picture: (1) the *actually deployed* Whisper CT2 artefact scores 79.29% raw on this same ruler, not the training-eval's 74.02% quoted in this table; (2) stripping the Perso-Arabic diacritics that saturate the references (which both models drop, so raw WER penalises both per-word) puts ks_max ahead on WER (64.31% vs 65.19%) and CER at every normalisation level. It also won the 5-condition radio-degradation sweep 4/5, with Whisper's CER exceeding 100% at 0 dB SNR. See the Extra-Data Campaign section and `docs/ks_ruler_study.json`.

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

\* Pashto (this table is the *zero-shot* comparison): fine-tuned Whisper wins clean/bandpass/10 dB/MP3 vs zero-shot SM4T and held the routing until 2026-07-19, when the noise-augmented **ps_aug adapter** beat Whisper in 4/5 conditions (bandpass 40.8 vs 41.2, 10 dB 41.5 vs 45.5, 0 dB 56.0 vs 64.8, MP3 40.0 vs 40.3) — see the Extra-Data Campaign section.

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

### 2. Pashto (ps) — `whisper-medium-pashto-ct2` *(retained for rollback; SM4T + LoRA serves ps since 2026-07-19)*
- **Base:** `Nasimbahar/pashto-ghag-whisper-medium-asr` (domain-specific Pashto model)
- **Dataset:** FLEURS `ps_af` — 2,082 train / 251 val
- **Training:** 2000 steps, ~3.5 h
- **Held-out eval WER:** **38.55%** fine-tuned vs 89.76% baseline (−51.2 pp) · zero-shot SeamlessM4T: 44.40% · **SeamlessM4T + ps_aug LoRA: 36.91% (DEPLOYED 2026-07-19)**
- **Translation:** Whisper+NLLB chrF 44.48 · SeamlessM4T S2TT chrF 40.15
- **Pipeline role:** since 2026-07-19, `ps` → **SeamlessM4T + ps_aug adapter** → NLLB-200 → English
- **Note:** For a year of this project, Pashto was the one language where per-language Whisper fine-tuning beat SeamlessM4T — its domain-pretrained base survived data scaling (42.47), capacity scaling (37.29 clean but 87.2 @ 0 dB), and decode tuning (38.88). It finally lost to a **noise-augmented** SeamlessM4T adapter (r=32 incl. MLP; training audio degraded with the evaluation's own bandpass/noise/codec pipeline): 36.91% clean and 4/5 degradation conditions incl. 0 dB SNR 56.0 vs 64.8. See the Extra-Data Campaign section.

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
- **Held-out eval WER:** 50.92% fine-tuned vs 88.85% baseline (−37.9 pp) · **SeamlessM4T: 28.46%; + LoRA adapter 24.34% (DEPLOYED 2026-07-18)**
- **Translation:** Whisper+NLLB chrF 45.55 · SeamlessM4T S2TT chrF 51.67
- **Pipeline role:** since 2026-07-11, `ne` → **SeamlessM4T** ASR → NLLB-200 → English; since 2026-07-18 with the **ne_iv LoRA adapter** (FLEURS + IndicVoices-R, cap 20k) — Nepali's first working adapter (the original ne adapter carried the label bug and was never retrained). 24.34 vs 28.46 clean; wins **all 5** degradation conditions (e.g. bandpass 30.6 vs 34.3).
- **Note:** Large Whisper fine-tuning gain, but SeamlessM4T's zero-shot lead (22.5 pp) is larger still — and adding IndicVoices-R data to a SeamlessM4T adapter beat even zero-shot.

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
- **Held-out eval WER:** 19.78% fine-tuned vs 26.34% baseline (−6.6 pp, a genuine gain) · **SeamlessM4T: 15.44%; + LoRA adapter 13.94% (2026-07-13); + IndicVoices data 12.91% (DEPLOYED 2026-07-18)**
- **Translation:** Whisper+NLLB chrF 53.71 · SeamlessM4T S2TT chrF 51.54 — Whisper+NLLB wins
- **Pipeline role:** since 2026-07-11, `hi` → **SeamlessM4T** ASR → NLLB-200 → English; since 2026-07-13 with a corrected-label **LoRA adapter** (13.94); since 2026-07-18 the adapter was replaced by **hi_iv** (FLEURS + IndicVoices-R, cap 20k): 12.91 vs 13.94 clean, beats the previous adapter in 4/5 degradation conditions (e.g. bandpass 15.0 vs 16.3). The 2026-07-13 adapter is retained for rollback.

### 7. Kashmiri (ks) — `whisper-large-v3-ks-ct2` *(retained for rollback; SM4T + ks_max LoRA serves ks since 2026-07-20)*
- **Base:** `openai/whisper-large-v3` + custom `<|ks|>` token (ID 51866)
- **Dataset:** IndicVoices-R Kashmiri — 20,000 train / 372 val
- **Training:** 3000 steps, ~18 h (incl. 2 power outages, PD-charger recovery)
- **Language token:** Whisper has no native `ks` token. A custom `<|ks|>` token was added to the vocab, with its embedding initialised from `<|ur|>` (Urdu — same Nastaliq script); forced prefix `[<|startoftranscript|>, <|ks|>, <|transcribe|>, <|notimestamps|>]` injected via TemplateProcessing. faster-whisper patched at import time to accept the `ks` language code.
- **Held-out eval WER:** **74.02% @ step 2400** vs 96.87% baseline (−22.85 pp), on the IndicVoices-R test split — but this was the *training-time* eval of the merged fp16 model; the *deployed* CT2 artefact scores 79.29% raw WER on the same 372 test clips when re-measured for the 2026-07-20 ruler study, and CT2 int8 is what the pipeline actually runs.
- **CT2 source:** checkpoint-2400 (trainer's best by WER)
- **Pipeline role:** since 2026-07-20, `ks` → **SeamlessM4T + ks_max LoRA adapter** → NLLB-200 (`kas_Arab`) → English. This model is retained on disk for rollback.
- **Robustness (30-clip set, clean condition, `eval_data/wer_robustness_results.csv`):** this rollback model scores **81.46% WER / 47.95% CER** — the wide gap between them, on the same clip set, is itself an early sign of the Perso-Arabic diacritic sensitivity that the 2026-07-20 ruler study later confirmed on the SeamlessM4T side too.
- **Notes:**
  - An earlier Urdu-proxy run (`whisper_lang="ur"`, ckpt-1500, eval 103.58% — not a true accuracy figure) is superseded by the custom-token model.
  - FLEURS has no Kashmiri config; Common Voice has no Kashmiri data → IndicVoices-R used (also the training corpus for the SeamlessM4T adapter that superseded this model).
  - The 96.87% baseline is itself inflated by a Unicode-normalisation mismatch in the reference text; the same class of scoring fragility (this time Perso-Arabic diacritics, not Unicode normalisation) is what the 2026-07-20 ruler study uncovered on the SeamlessM4T side of the comparison — see the Extra-Data Campaign section.

### 7b. Kashmiri (ks) — `ks_max` SeamlessM4T LoRA adapter *(DEPLOYED 2026-07-20)*
- **Base:** SeamlessM4T v2 large + custom `__kas__` token (added the same way as Whisper's, embedding initialised from `__urd__`)
- **What's new vs three earlier Seamless attempts:** every prior attempt (1 epoch → 129.29% WER; +decode fixes → 92.09%; r=16 LoRA, 3 epochs → 88.42%) trained with that `__kas__` embedding row **frozen** as a copy of Urdu's, because standard LoRA cannot touch embedding weights. `ks_max` is the first run to make that single 1024-dim row **trainable**, via PEFT's `trainable_token_indices` — alongside a capacity increase to r=32 incl. MLP layers (fc1/fc2), the same rung that won Pashto. Full IndicVoices-R data (24,000 samples, all available), 7,500 steps (~2.5 epochs), early-stopped.
- **Training:** eval_loss fell from 2.421 to 1.079 across the run — 31% lower than the r=16 attempt's floor and by far the steepest Kashmiri training curve of the campaign.
- **Held-out eval (raw, 372-clip IndicVoices test split):** 80.91% WER / 39.33% CER — nominally behind the training-eval figure of 74.02%, but ahead of the deployed CT2 artefact's own re-measured 79.29% by only 1.6 pp.
- **The scoring correction that changed the routing decision:** Kashmiri references are densely diacritised (Perso-Arabic combining marks), and *both* systems drop diacritics in their output — so raw WER penalises both per-word, symmetrically. A normalisation-ladder study (`scripts/eval/ks_ruler_study.py`, `docs/ks_ruler_study.json`) stripping diacritics (and, further, folding orthographic variants like yeh/kaf/alef-maqsura) found ks_max **already wins WER** at that level (64.31% vs Whisper's 65.19%) and **wins CER at every normalisation level on both test sets** (e.g. 32.24% vs 39.36%; boundary-free CER 34.85% vs 42.91%).
- **Degradation sweep (`docs/ks_max_degradation.json`, 5 conditions × 30 clips, same decode fixes):** wins 4 of 5 conditions — loses only clean speech (raw, +1.3 pp; normalised it's closer still), and wins bandpass/10 dB/0 dB/codec by 1.5–8.4 raw points (up to 12.6 pp normalised). Whisper's CER *exceeds 100%* at 0 dB SNR (108.41), while ks_max stays at 60.32.
- **Pipeline role:** `ks` → SeamlessM4T v2 + ks_max LoRA adapter (its own model instance in `SeamlessASR`, since it needs a resized vocabulary the other adapters don't share) → NLLB-200 → English.
- **Note:** this is the fifth independent scoring-methodology finding of this project (after the turbo-baseline mislabel, the CJK whitespace artefact, the label-encoding bug, and the sweep-harness adapter-override bug) — each one initially looked like a real model result.

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
       MMS lang=ne  → SeamlessM4T v2 + ne LoRA        → NLLB-200
       MMS lang=hi  → SeamlessM4T v2 + hi LoRA        → NLLB-200
       MMS lang=ur  → SeamlessM4T v2 (zero-shot)      → NLLB-200
       MMS lang=zh  → SeamlessM4T v2 (zero-shot)      → NLLB-200
       MMS lang=ps  → SeamlessM4T v2 + ps LoRA        → NLLB-200
       MMS lang=ks  → SeamlessM4T v2 + ks LoRA        → NLLB-200
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
│   ├── whisper-medium-pashto-ct2/   ( 734 MB) — Pashto    (retained for rollback; SM4T+LoRA serves ps)
│   ├── whisper-large-v3-ur-ct2/     (1479 MB) — Urdu      (retained; SM4T serves ur)
│   ├── whisper-large-v3-ne-ct2/     (1479 MB) — Nepali    (retained; SM4T serves ne)
│   ├── whisper-large-v3-zh-ct2/     (1479 MB) — Mandarin  (retained; SM4T serves zh)
│   ├── whisper-large-v3-hi-ct2/     (1479 MB) — Hindi     (retained; SM4T serves hi)
│   ├── whisper-large-v3-ks-ct2/     (1479 MB) — Kashmiri  (retained for rollback; SM4T+LoRA serves ks)
│   ├── whisper-large-v3-ct2/                  — true large-v3 baseline (evaluation)
│   └── seamless-m4t-v2-large/       (~10 GB)  — SeamlessM4T v2 (DEPLOYED: ALL 7 langs)
├── finetune_runs/
│   ├── pa/adapter/   — LoRA checkpoints (Punjabi, v3 best=ckpt-4000)
│   ├── ps/adapter/   — LoRA checkpoints (Pashto)
│   ├── ur/adapter/   — LoRA checkpoints (Urdu)
│   ├── ne/adapter/   — LoRA checkpoints (Nepali)
│   ├── zh/adapter/   — LoRA checkpoints (Mandarin, best=ckpt-400)
│   ├── hi/adapter/   — LoRA checkpoints (Hindi)
│   └── ks/adapter/   — LoRA checkpoints (Kashmiri, best=ckpt-2400)
├── finetune_runs_seamless/{hi,ne,pa,ps,ur,zh,ks}/adapter/     — SM4T LoRA (FLEURS-only generation)
├── finetune_runs_seamless/{hi_iv,ne_iv,ps_aug}/adapter/  — extra-data adapters (DEPLOYED 2026-07-18/19)
├── finetune_runs_seamless/ks_max/adapter/                — DEPLOYED 2026-07-20 (r=32+MLP, trainable __kas__)
├── finetune_runs_seamless/{ps_cv,ps_bal,ps_bal2,ks_r16}/adapter/  — extra-data experiments (not deployed)
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

## Extra-Data Campaign — Can SeamlessM4T Replace Whisper Entirely? (2026-07-17/20)

A systematic attempt to move the last two Whisper languages (Pashto, Kashmiri) onto
SeamlessM4T, and to push the SeamlessM4T languages further with more training data.
**Result: it succeeded for both — Whisper is retained on disk for rollback but is no
longer the deployed backend for any VANI language.**
All adapters LoRA on SeamlessM4T v2; n=100 held-out FLEURS eval unless noted.

| Experiment | Recipe | Result | Verdict |
|---|---|---|---|
| ps_cv | FLEURS + CV-20 Pashto (~50k, 95% CV), r=8 | 42.47 | **Lost** — worse than FLEURS-only (41.30); CV domain drift |
| ps_bal | FLEURS ×8 + CV cap 10k, r=16 α=32 q/k/v/out | 39.72; best decode (beam 5, lp 0.8) **38.88** | **Lost by 0.33** to Whisper-medium 38.55 |
| ps_bal2 | ps_bal data, r=32 α=64 + MLP (fc1/fc2) | **37.29** clean (first Seamless win) — but **87.2 at 0 dB** vs Whisper 64.8; loses 3/5 sweep conditions | **Failed the robustness gate** |
| ps_aug | ps_bal2 recipe + **noise-augmented training** (the sweep's own degradation family: bandpass / AWGN 0–15 dB / MP3) | **36.91** clean (36.50 w/ beams) AND **56.0 at 0 dB** vs Whisper 64.8 — wins 4/5 sweep conditions | **WON — deployed 2026-07-19; Pashto flips to SeamlessM4T** |
| ks continuation | r=8 resumed to 7,500 steps + decode fixes | 92.09 (test) | **Lost** to Whisper-ks 74.02 |
| ks_r16 | r=16 α=32, all 24k IndicVoices samples | 88.42 (test) | **Lost** — a stronger loss curve, but the gap barely closed |
| ks_max | r=32+MLP **AND** a trainable `__kas__` embedding row (PEFT `trainable_token_indices` — every prior attempt froze it) | 80.91 raw (test) — nominally behind, but the *deployed* Whisper CT2 artefact re-measures at 79.29 raw (not the published 74.02); **diacritic-normalised, ks_max already wins WER (64.31 vs 65.19) and CER at every level on both test sets** | **WON — deployed 2026-07-20; Kashmiri flips to SeamlessM4T** (see `docs/ks_ruler_study.json`, `docs/ks_max_degradation.json`) |
| hi_iv | FLEURS + IndicVoices-R Hindi (cap 20k), r=8 | **12.91** vs prior adapter 13.94 | **WON** — deployed 2026-07-18 (4/5 sweep conditions) |
| ne_iv | FLEURS + IndicVoices-R Nepali (cap 20k), r=8 | **24.34** vs zero-shot 28.46 | **WON** — deployed 2026-07-18 (5/5 sweep conditions) |

**What the campaign established:**

1. **Each language needed a different lever, and the sweep decided every deployment.**
   Hindi and Nepali (well-covered in SM4T pretraining) needed only more data (IndicVoices-R).
   Pashto (thin coverage) needed three levers stacked: balanced domain mixing, adapter
   capacity (r=32 incl. MLP), and finally **noise-augmented training** — degrading training
   audio with the evaluation's own bandpass/noise/codec pipeline. That last step fixed the
   0 dB collapse (87.2 → 56.0, beating Whisper's 64.8) while *improving* clean accuracy
   (37.29 → 36.91), turning attempt #4's robustness-gate failure into attempt #5's 4/5 win.
   Kashmiri (no coverage at all — custom `__kas__` token) needed the most: three attempts of
   epochs, decode fixes, and doubled rank closed most of the gap but not all of it — the
   final win needed a **trainable embedding row** (the conditioning vector itself, not just
   the adapter, learning) plus a **correction to the scoring ruler** (below).
2. **Validation loss is not a model-selection metric for ASR — confirmed a fourth time.**
   Four times in this campaign a clearly better eval_loss produced an equal-or-worse *raw*
   WER (ps_cv: −6% loss, +1.2 WER; ks continuation: −17% loss, −2 WER; ks_r16: −23% loss,
   −3.7 WER; ks_max: −31% loss vs ks_r16, but still behind on the raw ruler). Free-running
   generation quality decouples from teacher-forced cross-entropy. Held-out WER decides —
   but see finding 4 for what happens when even the WER ruler is wrong.
3. **The hybrid Whisper+SeamlessM4T architecture is not needed after all — for ASR
   *deployment*.** A custom-token LoRA adapter *can* match and exceed a native-vocab Whisper
   fine-tune, even for a language the base model never saw, given enough of the right
   levers: adapter capacity, a trainable conditioning token, and (see below) a scoring ruler
   that accounts for the script's own quirks. Every VANI language now runs on SeamlessM4T;
   the fine-tuned Whisper models remain on disk purely as a rollback path, not a production
   dependency.
4. **The Kashmiri scoring correction is the campaign's most important finding, and the
   fifth independent scoring-methodology bug this project has caught** (after the
   turbo-baseline mislabel, the CJK whitespace artefact, the S2TT label-encoding bug, and
   the sweep-harness adapter-override bug in finding 5 below, which happened first
   chronologically). ks_max's apparent
   WER loss (80.91 vs the published 74.02) was compounded from two separate errors: (a) the
   published 74.02 was a training-time eval of the merged fp16 model, never re-measured on
   the actually-deployed int8 CT2 artefact, which scores 79.29 on the identical clips; (b)
   raw WER on Perso-Arabic Kashmiri is dominated by diacritic mismatches that *both* systems
   make symmetrically (both drop the dense combining marks in the references), inflating
   WER for both models roughly equally and masking the real ordering underneath. A
   normalisation-ladder study (`scripts/eval/ks_ruler_study.py`) stripping diacritics first,
   then folding orthographic variants, reversed the WER verdict and confirmed a CER
   advantage that had been visible from the very first ks_max eval. Neither correction was
   a training change — both were corrections to how the existing hypotheses were being read.
5. **The fourth silent methodology bug (chronologically first of the two above) was caught
   by its own gate.** The first hi_iv/ne_iv
   degradation sweeps scored identical-to-the-decimal with the deployed adapter / zero-shot:
   the production per-language adapter switch inside `SeamlessASR.generate()` was silently
   overriding the experimental adapter under test. The sweep harness now builds a clean
   model. (Lesson repeated across this whole campaign: any number that looks like an exact
   copy of another number, or a result that looks too clean to be true, is a bug until
   proven otherwise — the Kashmiri routing decision itself would have been wrong under the
   first ruler.)

---

## Key Findings

1. **Backend selection beats blanket fine-tuning.** Against the corrected true large-v3 baseline, zero-shot SeamlessM4T wins clean-speech ASR for 5 of 6 supported languages and every degradation condition for those five. Per-language Whisper fine-tuning is **no longer the deployed backend for any VANI language**: Pashto fell on 2026-07-19 to a noise-augmented SeamlessM4T adapter, and Kashmiri — despite having no native SeamlessM4T vocabulary at all — fell the following day to a SeamlessM4T adapter with a trainable custom token, once the WER scoring ruler itself was corrected (see the Extra-Data Campaign section).
2. **Fine-tuning helps in proportion to how poorly the base model covers the language.** Real gains vs the true baseline: ps −51.2 pp, ne −37.9, ks −22.85, pa −20.2, hi −6.6, ur −1.4, zh **+3.2 (regression)**. The turbo-mislabelled baseline had inflated every one of these.
3. **A strong training-val WER can fail to generalise.** Mandarin's 8.97% train-val became 14.22% held-out — worse than the 10.99% un-fine-tuned baseline. Small single-domain training sets can narrow a strong base model.
4. **Scoring methodology is a result in itself.** Two silent defects (wrong baseline model; whitespace WER on character-spaced Han) produced an −84 pp headline gain that was entirely artefactual. All WER now goes through one CJK-aware normaliser, raw hypotheses are persisted to JSONL so re-scoring never needs a GPU re-run, and pre-fix results are archived for the erratum trail.
5. **The SeamlessM4T lead survives — and widens under — radio degradation.** Its advantage grows at 0 dB SNR (hi +19.5, zh +17.2), killing the "Whisper is more robust to noise" hypothesis.
6. **fp16 instability at low learning rates** — Mandarin training showed a gradient spike at step ~820 when LR decayed to ~1.5×10⁻⁵. `max_grad_norm=0.5` resolved this in subsequent training.
7. **CT2 models need `tokenizer.json` from the source model** — `ct2-transformers-converter` does not copy it; faster-whisper falls back to `whisper-tiny`'s tokenizer where `<|transcribe|>=50359`, but large-v3 uses `<|transcribe|>=50360` / `<|translate|>=50359`. The off-by-one made fine-tuned large-v3 CT2 models *translate* instead of transcribe. **Fix:** `finetune_whisper.py merge_and_convert()` now copies `tokenizer.json` into each CT2 directory.
8. **Kashmiri required a custom vocab token** — `<|ks|>` (ID 51866) added to large-v3 with embedding initialised from `<|ur|>`; 74.02% WER at step 2400 (−22.85 pp). The earlier Urdu-proxy approach (eval 103.58%, not a true accuracy figure) is superseded. SeamlessM4T cannot cover Kashmiri, so the hybrid Whisper+SeamlessM4T architecture is forced.
9. **Fine-tuning SeamlessM4T on FLEURS alone buys little — the wins came from data, capacity, and noise augmentation** (hi 15.44 → 12.91 with IndicVoices data; ne 28.46 → 24.34 likewise; ps 44.40 zero-shot → 36.91 with balanced data + r=32 MLP LoRA + noise-augmented training; all three deployed after passing the degradation sweep — see the Extra-Data Campaign section; pa/ur/zh still run zero-shot, and every ks attempt lost to Whisper). The dramatic "fine-tuning breaks translation" effect reported earlier was a label-encoding bug (targets tokenised in source mode with an `__eng__` prefix), not a property of LoRA: corrected-label retrains recover chrF from 2.12 to 37.60 (ps) and 0.06 to 43.24 (hi) with statistically identical ASR in both. Scoring methodology strikes again — see Finding 4.
10. **Script-based cascade prevents misidentification** — the Arabic-script detection fallback correctly catches Urdu even when MMS-LID confidence is below threshold.

---

*Generated: 23 June 2026 · Cross-model eval: 23 June 2026 · Extended training + re-eval: 25 June 2026 · SeamlessM4T FT: 26 June 2026 · **Corrected eval (true large-v3 baseline, CJK-aware scoring) + backend routing change: 10–11 July 2026** · **Extra-data campaign + hi_iv/ne_iv deployment: 17–18 July 2026** · **Noise-augmented ps_aug deployment (Pashto → SeamlessM4T): 19 July 2026** · **ks_max deployment + ruler correction (Kashmiri → SeamlessM4T, all 7 languages now on SM4T): 20 July 2026** · VANI v2 · RTX 5060 8 GB*

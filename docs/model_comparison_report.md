# VANI — Cross-Model Evaluation Report
**Whisper baseline vs Fine-tuned Whisper vs SeamlessM4T v2 (zero-shot + fine-tuned)**
**Date:** 26 June 2026  ·  **Hardware:** RTX 5060 8 GB CUDA

---

## Metric Definitions

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **WER** | Word Error Rate — ASR accuracy in source language (lower = better) | 0–100% |
| **chrF** | Character F-score — translation quality to English (higher = better) | 0–100 |

---

## ASR Word Error Rate (Source Language Transcription)

| Language | Script | Whisper Baseline | Whisper FT | SM4T Zero-Shot | SM4T FT | Whisper FT Improvement |
|----------|--------|-----------------|-----------|---------------|---------|----------------------|
| Punjabi (PA) | Gurmukhi | 105.79% | 55.67% | 19.77% | 19.77% | +50.1 pp |
| Pashto (PS) | Nastaliq | 94.23% | **38.55%** | 44.4% | 41.22% | +55.7 pp |
| Urdu (UR) | Nastaliq | 24.44% | 19.82% | **16.9%** | 17.26% | +4.6 pp |
| Nepali (NE) | Devanagari | 94.55% | 49.24% | **28.46%** | 28.92% | +45.3 pp |
| Mandarin (ZH) | Simplified Han | 100.03% | **16.03%** | 100.0%† | 60.53% | +84.0 pp |
| Hindi (HI) | Devanagari | 30.29% | 19.78% | 15.44% | **13.43%** | +10.5 pp |
| Kashmiri (KS) | Nastaliq | 98.64% | 103.58% | — | — | — |

† ZS SM4T Mandarin WER = 100.0% is a script-normalisation mismatch, not a model failure.

---

## Translation Quality — chrF Score (→ English)

| Language | Whisper+NLLB chrF | SM4T ZS S2TT chrF | SM4T FT S2TT chrF | Best |
|----------|------------------|------------------|------------------|------|
| Punjabi (PA) | 41.54 | **58.72** | 0.31 ⚠ | SM4T ZS |
| Pashto (PS) | **44.48** | 43.92 | 2.41 ⚠ | Whisper+NLLB |
| Urdu (UR) | 51.34 | **54.91** | 0.62 ⚠ | SM4T ZS |
| Nepali (NE) | 47.72 | **56.02** | 0.13 ⚠ | SM4T ZS |
| Mandarin (ZH) | 42.85 | **53.42** | 0.78 ⚠ | SM4T ZS |
| Hindi (HI) | 53.71 | **56.05** | 0.11 ⚠ | SM4T ZS |
| Kashmiri (KS) | — | — | — | — |

⚠ SM4T FT S2TT chrF near zero: ASR-only LoRA fine-tuning breaks multi-task language conditioning.

---

## Key Findings

1. **Fine-tuned Whisper beats all SM4T variants on ASR** for: Pashto, Mandarin
2. **Zero-shot SM4T beats fine-tuned Whisper on ASR** for: Punjabi, Urdu, Nepali
3. **Fine-tuned SM4T beats zero-shot SM4T** for: Pashto (−3.2 pp), Hindi (−2.0 pp)
4. **Fine-tuned SM4T is not better than zero-shot** for: Punjabi (0 pp), Urdu (+0.4 pp), Nepali (+0.5 pp), Mandarin (−39.5 pp but still 60.53% vs FT Whisper 16.03%)
5. **SM4T FT translation is broken** — ASR-only LoRA fine-tuning collapses S2TT chrF to near zero. Zero-shot SM4T and Whisper+NLLB both far outperform.
6. **Whisper+NLLB beats zero-shot SM4T S2TT on translation** for: Pashto only (44.48 vs 43.92)
7. **Best overall ASR system per language:**
   - Punjabi → SM4T (ZS or FT): 19.77%
   - Pashto → FT Whisper: 38.55%
   - Urdu → ZS SM4T: 16.9%
   - Nepali → ZS SM4T: 28.46%
   - Mandarin → FT Whisper: 16.03%
   - Hindi → FT SM4T: 13.43%

---

*Generated: 25 June 2026 · Updated with SM4T FT results: 26 June 2026 · VANI v2 · RTX 5060 8 GB*

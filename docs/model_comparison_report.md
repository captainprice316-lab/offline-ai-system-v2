# VANI — Cross-Model Evaluation Report
**Whisper baseline vs Fine-tuned Whisper vs SeamlessM4T v2**
**Date:** 23 June 2026  ·  **Hardware:** RTX 5060 8 GB CUDA

---

## Metric Definitions

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **WER** | Word Error Rate — ASR accuracy in source language (lower = better) | 0–100% |
| **chrF** | Character F-score — translation quality to English (higher = better) | 0–100 |

---

## ASR Word Error Rate (Source Language Transcription)

| Language | Script | Whisper Baseline | Whisper Fine-Tuned | SeamlessM4T v2 | Improvement |
|----------|--------|-----------------|-------------------|----------------|-------------|
| Punjabi (PA) | Gurmukhi | 105.83% | 59.94% | 19.77% | +45.9pp |
| Pashto (PS) | Nastaliq | 95.07% | 39.72% | 44.4% | +55.3pp |
| Urdu (UR) | Nastaliq | 24.44% | 19.82% | 16.9% | +4.6pp |
| Nepali (NE) | Devanagari | 94.55% | 53.92% | 28.46% | +40.6pp |
| Mandarin (ZH) | Simplified Han | 100.03% | 16.03% | 100.0% | +84.0pp |
| Hindi (HI) | Devanagari | 30.29% | 19.78% | 15.44% | +10.5pp |
| Kashmiri (KS) | Nastaliq | 98.64% | 103.58% | — | +-4.9pp |

---

## Translation Quality — chrF Score (→ English)

| Language | Whisper+NLLB chrF | SeamlessM4T S2TT chrF | Winner |
|----------|------------------|----------------------|--------|
| Punjabi (PA) | 39.09 | 58.72 | SeamlessM4T |
| Pashto (PS) | 44.4 | 43.92 | Whisper+NLLB |
| Urdu (UR) | 51.34 | 54.91 | SeamlessM4T |
| Nepali (NE) | 47.67 | 56.02 | SeamlessM4T |
| Mandarin (ZH) | 42.85 | 53.42 | SeamlessM4T |
| Hindi (HI) | 53.71 | 56.05 | SeamlessM4T |
| Kashmiri (KS) | — | — | — |

---

## Key Findings

1. **Fine-tuned Whisper beats SeamlessM4T on ASR WER** for: Pashto, Mandarin
2. **SeamlessM4T beats fine-tuned Whisper on ASR WER** for: Punjabi, Urdu, Nepali, Hindi
3. **SeamlessM4T S2TT beats Whisper+NLLB on translation** for: Punjabi, Urdu, Nepali, Mandarin, Hindi
4. **Whisper+NLLB beats SeamlessM4T on translation** for: Pashto

---

*Generated: 23 June 2026 22:20  ·  VANI v2  ·  RTX 5060 8 GB*
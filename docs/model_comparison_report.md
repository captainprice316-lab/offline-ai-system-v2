# VANI — Cross-Model Evaluation Report
**Whisper baseline vs Fine-tuned Whisper vs SeamlessM4T v2**
**Date:** 25 June 2026  ·  **Hardware:** RTX 5060 8 GB CUDA

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
| Urdu (UR) | Nastaliq | 24.44% | 19.82% | 16.9% | +4.6pp |
| Mandarin (ZH) | Simplified Han | 100.03% | 16.03% | 100.0% | +84.0pp |
| Hindi (HI) | Devanagari | 30.29% | 19.78% | 15.44% | +10.5pp |
| Kashmiri (KS) | Nastaliq | 98.64% | 103.58% | — | +-4.9pp |
| Punjabi (PA) | Gurmukhi | 105.79% | 55.67% | 19.77% | +50.1pp |
| Nepali (NE) | Devanagari | 94.55% | 49.24% | 28.46% | +45.3pp |
| Pashto (PS) | Nastaliq | 94.23% | 38.55% | 44.4% | +55.7pp |

---

## Translation Quality — chrF Score (→ English)

| Language | Whisper+NLLB chrF | SeamlessM4T S2TT chrF | Winner |
|----------|------------------|----------------------|--------|
| Urdu (UR) | 51.34 | 54.91 | SeamlessM4T |
| Mandarin (ZH) | 42.85 | 53.42 | SeamlessM4T |
| Hindi (HI) | 53.71 | 56.05 | SeamlessM4T |
| Kashmiri (KS) | — | — | — |
| Punjabi (PA) | 41.54 | 58.72 | SeamlessM4T |
| Nepali (NE) | 47.72 | 56.02 | SeamlessM4T |
| Pashto (PS) | 44.48 | 43.92 | Whisper+NLLB |

---

## Key Findings

1. **Fine-tuned Whisper beats SeamlessM4T on ASR WER** for: Mandarin, Pashto
2. **SeamlessM4T beats fine-tuned Whisper on ASR WER** for: Urdu, Hindi, Punjabi, Nepali
3. **SeamlessM4T S2TT beats Whisper+NLLB on translation** for: Urdu, Mandarin, Hindi, Punjabi, Nepali
4. **Whisper+NLLB beats SeamlessM4T on translation** for: Pashto

---

*Generated: 25 June 2026 22:56  ·  VANI v2  ·  RTX 5060 8 GB*
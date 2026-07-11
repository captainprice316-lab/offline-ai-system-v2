# VANI — Cross-Model Evaluation Report
**Whisper baseline vs Fine-tuned Whisper vs SeamlessM4T v2**
**Date:** 10 July 2026  ·  **Hardware:** RTX 5060 8 GB CUDA

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
| Punjabi (PA) | Gurmukhi | 77.6% | 57.39% | 19.77% | +20.2pp |
| Pashto (PS) | Nastaliq | 89.76% | 38.55% | 44.4% | +51.2pp |
| Urdu (UR) | Nastaliq | 21.23% | 19.82% | 16.9% | +1.4pp |
| Nepali (NE) | Devanagari | 88.85% | 50.92% | 28.46% | +37.9pp |
| Mandarin (ZH) | Simplified Han | 10.99% | 14.22% | 11.69% | +-3.2pp |
| Hindi (HI) | Devanagari | 26.34% | 19.78% | 15.44% | +6.6pp |

---

## Translation Quality — chrF Score (→ English)

| Language | Whisper+NLLB chrF | SeamlessM4T S2TT chrF | Winner |
|----------|------------------|----------------------|--------|
| Punjabi (PA) | 40.15 | 54.53 | SeamlessM4T |
| Pashto (PS) | 44.48 | 40.15 | Whisper+NLLB |
| Urdu (UR) | 51.34 | 50.73 | Whisper+NLLB |
| Nepali (NE) | 45.55 | 51.67 | SeamlessM4T |
| Mandarin (ZH) | 42.0 | 49.15 | SeamlessM4T |
| Hindi (HI) | 53.71 | 51.54 | Whisper+NLLB |

---

## Key Findings

1. **Fine-tuned Whisper beats SeamlessM4T on ASR WER** for: Pashto
2. **SeamlessM4T beats fine-tuned Whisper on ASR WER** for: Punjabi, Urdu, Nepali, Mandarin, Hindi
3. **SeamlessM4T S2TT beats Whisper+NLLB on translation** for: Punjabi, Nepali, Mandarin
4. **Whisper+NLLB beats SeamlessM4T on translation** for: Pashto, Urdu, Hindi

---

*Generated: 10 July 2026 21:24  ·  VANI v2  ·  RTX 5060 8 GB*
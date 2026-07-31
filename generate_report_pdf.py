"""
VANI Fine-Tuning Report PDF Generator
Run from project root: python generate_report_pdf.py
"""

import io
import json
import pathlib
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import report_charts   # shared hero figures (identical in PDF + PPTX)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Preformatted, Spacer, Table,
    TableStyle, Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

# ── page geometry ────────────────────────────────────────────────────────────
L_MARGIN = R_MARGIN = 2.2 * cm
PAGE_W = A4[0] - L_MARGIN - R_MARGIN          # usable text width ≈ 470 pt

# ── colours ──────────────────────────────────────────────────────────────────
HDR_BLUE    = colors.HexColor("#1565C0")
LIGHT_BLUE  = colors.HexColor("#E3F2FD")
ACCENT      = colors.HexColor("#1976D2")
TABLE_EVEN  = colors.HexColor("#F5F5F5")
TABLE_HDR   = colors.HexColor("#1565C0")
WARN_RED    = colors.HexColor("#C62828")
SUCCESS_GRN = colors.HexColor("#2E7D32")
CODE_BG     = colors.HexColor("#F8F8F8")
CODE_BORDER = colors.HexColor("#CCCCCC")

PALETTE = {
    "pa": "#2196F3", "ps": "#FF9800", "ur": "#4CAF50",
    "ne": "#9C27B0", "zh": "#F44336", "hi": "#009688",
    "ks": "#00BCD4",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

LANG_META = {
    "pa": {
        "name": "Punjabi", "script": "Gurmukhi", "iso": "pa",
        "base_model": "openai/whisper-large-v3",
        "dataset": "FLEURS pa_in + IndicVoices-R Punjabi", "train_samples": 11923, "val_samples": 805,
        "steps": 3000, "baseline_wer": 77.60,
        "wer_curve": [(200,70.75),(400,61.95),(600,59.21),(800,58.55),(1000,59.09),
                      (1200,56.98),(1400,56.48),(1600,55.12),(1800,54.65),(2000,54.08),
                      (2200,53.88),(2400,52.99),(2600,52.85),(2800,52.61),(3000,52.55)],
        "wer_curve_v1": [(200,71.56),(400,65.83),(600,63.98),(800,62.08),(1000,61.30),
                         (1500,58.36),(2000,58.10),(2500,57.32),(3000,56.67)],
        "wer_curve_v3": [(200,69.97),(400,61.66),(600,59.97),(800,57.15),(1000,56.96),
                         (1200,54.48),(1400,55.68),(1600,52.99),(1800,52.06),(2000,52.75),
                         (2200,50.62),(2400,51.25),(2600,51.25),(2800,50.51),(3000,50.05),
                         (3200,50.65),(3400,49.40),(3600,49.97),(3800,49.49),(4000,49.31)],
        "train_loss": [(200,0.3558),(400,0.2344),(600,0.2307),(800,0.2017),(1000,0.1817),
                       (1200,0.1817),(1400,0.1631),(1600,0.1680),(1800,0.1725),(2000,0.1513),
                       (2200,0.1587),(2400,0.1570),(2600,0.1411),(2800,0.1684),(3000,0.1623)],
        "best_wer": 49.31, "best_step": 4000,
        "eval_wer": 57.39,   # held-out n=100 FLEURS test (train-val best was 49.31)
        "ct2_model": "whisper-large-v3-pa-ct2",
        "translation": "NLLB-200",
        "note": "v1: FLEURS pa_in only (2,516 samples, 3,000 steps, best WER 56.67%  — superseded). "
                "v2: FLEURS + IndicVoices-R 9,407 samples (11,923 total, 3,000 steps, best WER 52.55% @ step 3000); "
                "CT2 tokenizer fix 2026-06-23 restored Gurmukhi transcription. Superseded by v3. "
                "v3 (built 2026-07-04; retained but NOT serving ASR since the 2026-07-11 routing change — "
                "Punjabi routes to SeamlessM4T at 19.77%): LoRA r=16 α=32, IV-R 20,000 samples (21,923 total), completed 4,000 steps. "
                "Best WER 49.31% @ step 4000 (-3.24 pp vs v2's 52.55%), eval loss declining monotonically to 0.1662. "
                "Best checkpoint merged to CT2 int8 and deployed; verified transcribing Gurmukhi on FLEURS pa test set. "
                "Robustness note: initial run OOM-crashed at step 2400 (CUDA out-of-memory during beam-search eval on 8 GB "
                "RTX 5060); resumed from checkpoint-2200 with greedy eval (num_beams=1, eval batch 1, cache-clear before eval), "
                "which cut eval VRAM from 8 GB to ~3.8 GB and ran clean to step 4000. "
                "Earlier disk-space incident at step 800: D: junction full during optimizer checkpoint save; resumed from checkpoint-600.",
        "training_time": "~55 h (incl. OOM restart)",
    },
    "ps": {
        "name": "Pashto", "script": "Nastaliq (Arabic)", "iso": "ps",
        "base_model": "Nasimbahar/pashto-ghag-whisper-medium-asr",
        "dataset": "FLEURS ps_af", "train_samples": 2082, "val_samples": 251,
        "steps": 2000, "baseline_wer": 89.76,
        "wer_curve": [(200,41.62),(400,41.30),(600,38.91),(800,38.86),(1000,39.10),(2000,38.55)],
        "train_loss": [(40,1.5341),(80,1.2020),(120,1.0698),(160,0.9358),(200,0.8636),
                       (240,0.8263),(280,0.8017),(320,0.7752),(360,0.6985),(400,0.5924),
                       (440,0.6164),(480,0.5323),(520,0.5642),(560,0.5783),(600,0.5882),
                       (640,0.5475),(680,0.5280),(720,0.5347),(760,0.5929),(800,0.5591),
                       (840,0.5592),(880,0.4897),(920,0.5272),(960,0.5384),(1000,0.5553)],
        "best_wer": 38.55, "best_step": 2000,
        "eval_wer": 38.55,
        "ct2_model": "whisper-medium-pashto-ct2",
        "translation": "NLLB-200",
        "note": "Started from a domain-specific Pashto medium model (734 MB). Higher initial loss reflects harder "
                "acoustic domain. For a year this was the one language where fine-tuned Whisper beat SeamlessM4T; "
                "on 2026-07-19 a noise-augmented SeamlessM4T LoRA adapter (ps_aug: r=32 incl. MLP, balanced "
                "FLEURS+Common Voice data, training audio degraded with the evaluation's own bandpass/noise/codec "
                "pipeline) reached 36.91% clean and won 4/5 degradation conditions (0 dB SNR: 56.0 vs 64.8). "
                "On 2026-07-27 the same recipe retrained at rank 128 on a rented cloud GPU (ps_cloud) improved this "
                "to 36.16% clean and won 4/5 conditions against ps_aug itself (0 dB: 53.8) — Pashto now runs on "
                "the ps_cloud adapter; ps_aug and this Whisper model are retained for rollback (see §5.5).",
        "training_time": "~3.5 h",
    },
    "ur": {
        "name": "Urdu", "script": "Nastaliq (Arabic)", "iso": "ur",
        "base_model": "openai/whisper-large-v3",
        "dataset": "FLEURS ur_pk", "train_samples": 2109, "val_samples": 267,
        "steps": 1000, "baseline_wer": 21.23,
        "wer_curve": [(200,23.63),(400,22.69),(600,22.90),(800,22.27),(1000,22.29)],
        "train_loss": [(40,0.8566),(80,0.7419),(120,0.5837),(160,0.5650),(200,0.5317),
                       (240,0.4484),(280,0.3836),(320,0.3876),(360,0.3738),(400,0.3481),
                       (440,0.3402),(480,0.3111),(520,0.3532),(560,0.3114),(600,0.3553),
                       (640,0.3135),(680,0.3400),(720,0.3592),(760,0.3425),(800,0.3097),
                       (840,0.3407),(880,0.3225),(920,0.3207),(960,0.3218),(1000,0.3402)],
        "best_wer": 22.27, "best_step": 800,
        "eval_wer": 19.82,
        "ct2_model": "whisper-large-v3-ur-ct2",
        "translation": "NLLB-200",
        "note": "Largest WER improvement among large-v3 Indic languages. Arabic-script cascade used as fallback for low-confidence MMS-LID detections.",
        "training_time": "~6 h",
    },
    "ne": {
        "name": "Nepali", "script": "Devanagari", "iso": "ne",
        "base_model": "openai/whisper-large-v3",
        "dataset": "FLEURS ne_np + IndicVoices-R Nepali", "train_samples": 13332, "val_samples": 572,
        "steps": 3000, "baseline_wer": 88.85,
        "wer_curve": [(200,70.98),(400,62.64),(600,60.67),(800,56.93),(1000,55.90),
                      (1200,54.73),(1400,54.27),(1600,53.81),(1800,53.83),(2000,52.05),
                      (2200,51.77),(2400,51.67),(2600,51.10),(2800,51.17),(3000,50.82)],
        "wer_curve_v1": [(200,63.58),(400,56.87),(600,54.32),(800,54.55),(1000,54.36),
                         (1500,52.87),(2000,52.14)],
        "train_loss": [(200,0.6375),(400,0.4544),(600,0.4108),(800,0.3540),(1000,0.3408),
                       (1200,0.3337),(1400,0.2988),(1600,0.3323),(1800,0.3221),(2000,0.3195),
                       (2200,0.2982),(2400,0.3260),(2600,0.3065),(2800,0.2553),(3000,0.2941)],
        "best_wer": 50.82, "best_step": 3000,
        "eval_wer": 50.92,   # held-out n=100 FLEURS test
        "ct2_model": "whisper-large-v3-ne-ct2",
        "translation": "NLLB-200",
        "note": "v1 trained on FLEURS ne_np (3,332 samples, 2,000 steps, best WER 52.14%). "
                "v2 retrained with IndicVoices-R Nepali added (13,332 total samples, 3,000 steps), "
                "achieving 50.82% on a 572-sample combined eval set (FLEURS val + IndicVoices-R test). "
                "However zero-shot SeamlessM4T reaches 28.46% on the held-out test, and a SeamlessM4T "
                "LoRA adapter trained on the same FLEURS + IndicVoices-R mix reaches 24.34% — so Nepali "
                "is operationally routed to SeamlessM4T with that ne_iv adapter enabled (deployed "
                "2026-07-18; wins all 5 radio-degradation conditions, e.g. bandpass 30.6 vs 34.3), "
                "not this model (see §5.5).",
        "training_time": "~30 h",
    },
    "zh": {
        "name": "Mandarin Chinese", "script": "Simplified Han", "iso": "zh",
        "base_model": "openai/whisper-large-v3",
        "dataset": "FLEURS cmn_hans_cn", "train_samples": 3246, "val_samples": 409,
        "steps": 400, "baseline_wer": 10.99,
        "wer_curve": [(200,15.77),(400,8.97)],
        "wer_curve_diverged": [(600,252.37)],
        "train_loss": [(40,0.7791),(80,0.7060),(120,0.5694),(160,0.3697),(200,0.3396),
                       (240,0.3327),(280,0.3081),(320,0.2637),(360,0.2280),(400,0.2307),
                       (440,0.1873),(480,0.1484),(520,0.1742),(560,0.2045),(600,0.1426)],
        # best_wer = best TRAINING-VAL point (sits on the §5.3 curve, where the chart
        # star is drawn); eval_wer = held-out n=100 test. For zh they diverge sharply.
        "best_wer": 8.97, "best_step": 400,
        "eval_wer": 14.22,
        "ct2_model": "whisper-large-v3-zh-ct2",
        "translation": "NLLB-200",
        "note": "CORRECTED 2026-07-11. The previously reported baseline of 100.03% was a "
                "measurement artefact, not a model failure: FLEURS Mandarin references are "
                "character-spaced, WER tokenises on whitespace, and the un-fine-tuned model emits "
                "unspaced Han — so a near-perfect transcript scored ~100%. Re-scored with "
                "character segmentation, the true openai/whisper-large-v3 baseline is 10.99% WER "
                "(n=100 FLEURS test). The fine-tuned model scores 14.22% on the same test — i.e. "
                "fine-tuning REGRESSED Mandarin (+3.2 pp), likely from the small (3,246-sample) "
                "single-domain training set narrowing the model. The strong training-eval WER "
                "(8.97% at step 400) did not generalise to held-out test. Operational decision: "
                "Mandarin is NOT served by this fine-tuned model — it routes to zero-shot "
                "SeamlessM4T (11.69%), which also wins under radio degradation (see §5.6). "
                "Training diverged at step ~820 (fp16 gradient overflow, grad_norm=12.9); "
                "checkpoint-400 was the best checkpoint.",
        "training_time": "~3 h 10 m (to step 400)",
    },
    "hi": {
        "name": "Hindi", "script": "Devanagari", "iso": "hi",
        "base_model": "openai/whisper-large-v3",
        "dataset": "FLEURS hi_in", "train_samples": 2120, "val_samples": 239,
        "steps": 600, "baseline_wer": 26.34,
        "wer_curve": [(200,24.00),(400,23.20),(600,23.13)],
        "train_loss": [(40,0.4713),(80,0.4326),(120,0.3588),(160,0.3063),(200,0.2854),
                       (240,0.2594),(280,0.2312),(320,0.2310),(360,0.2183),(400,0.1996),
                       (440,0.2115),(480,0.2084),(520,0.1978),(560,0.1917),(600,0.2292)],
        "best_wer": 23.13, "best_step": 600,
        "eval_wer": 19.78,
        "ct2_model": "whisper-large-v3-hi-ct2",
        "translation": "NLLB-200",
        "note": "Fastest convergence: near-best WER by step 200. "
                "max_grad_norm=0.5 applied after Mandarin gradient explosion; training fully stable. "
                "Held-out test WER 19.78% vs true large-v3 baseline 26.34% (n=100 FLEURS) — a genuine "
                "6.6 pp fine-tuning gain. However zero-shot SeamlessM4T reaches 15.44% on the same test, "
                "a corrected-label SeamlessM4T LoRA adapter reached 13.94% (2026-07-13), and an adapter "
                "retrained with FLEURS + IndicVoices-R data reaches 12.91% — so Hindi is operationally "
                "routed to SeamlessM4T with that hi_iv adapter enabled (deployed 2026-07-18; beats the "
                "previous adapter in 4/5 radio-degradation conditions incl. bandpass 15.0 vs 16.3), "
                "not this model (see §5.5).",
        "training_time": "~6 h 45 m",
    },
    "ks": {
        "name": "Kashmiri", "script": "Nastaliq (Perso-Arabic)", "iso": "ks",
        "base_model": "openai/whisper-large-v3 + <|ks|> token (ID 51866)",
        "dataset": "IndicVoices-R Kashmiri", "train_samples": 20000, "val_samples": 372,
        "steps": 3000, "baseline_wer": 96.87,
        "wer_curve": [
            (200,97.41),(400,90.18),(600,85.58),(800,83.16),(1000,84.47),
            (1200,78.85),(1400,77.13),(1600,76.89),(1800,75.96),(2000,74.34),
            (2200,75.52),(2400,74.02),(2600,75.0),(2800,75.91),(3000,76.27),
        ],
        "train_loss": [
            (40,2.8338),(80,2.4584),(120,2.0347),(160,1.8366),(200,1.6713),
            (280,1.4769),(360,1.2018),(400,1.1963),(480,1.0893),(560,1.0287),
            (600,0.9812),(680,0.9127),(760,0.9169),(800,0.8601),(880,0.8539),
            (960,0.8718),(1000,0.8404),(1080,0.8011),(1160,0.7705),(1200,0.7683),
            (1280,0.757),(1360,0.769),(1400,0.7792),(1480,0.742),(1560,0.7895),
            (1600,0.7286),(1680,0.7123),(1760,0.6883),(1800,0.7589),(1880,0.6958),
            (1960,0.6563),(2000,0.7041),(2080,0.649),(2160,0.7344),(2200,0.76),
            (2280,0.6279),(2360,0.7073),(2400,0.7008),(2480,0.651),(2560,0.7024),
            (2600,0.7156),(2680,0.683),(2760,0.6522),(2800,0.6546),(2880,0.6706),
            (2960,0.6692),(3000,0.7226),
        ],
        "best_wer": 74.02, "best_step": 2400,
        "eval_wer": 74.02,
        "ct2_model": "whisper-large-v3-ks-ct2",
        "translation": "NLLB-200",
        "note": "Whisper has no native Kashmiri support. Custom <|ks|> token (ID 51866) added to "
                "whisper-large-v3 vocab and embedding matrix initialised from <|ur|> (Urdu — same Nastaliq "
                "script). Forced prefix [<|startoftranscript|>, <|ks|>, <|transcribe|>, <|notimestamps|>] "
                "injected via TemplateProcessing. Trained on IndicVoices-R KS (20k samples, 3,000 steps). "
                "Best WER 74.02% at step 2400 (-22.85 pp from baseline 96.87%). "
                "faster-whisper patched at import time to accept 'ks' language code. SeamlessM4T has no "
                "native Kashmiri either, but the same custom-token trick (__kas__, r=32 LoRA incl. MLP, "
                "PLUS a trainable embedding row via PEFT trainable_token_indices) was applied on top of it "
                "2026-07-19/20 (ks_max). The clean-speech comparison against the actually-deployed "
                "Whisper CT2 artefact (79.29% raw WER, not the training-eval's 74.02%) initially looked "
                "like a loss (ks_max 80.91%) until a normalisation-ladder study showed both models' raw "
                "WER is inflated by the densely-diacritised references: diacritic-stripped, ks_max already "
                "wins WER (64.31% vs 65.19%) and wins CER at every normalisation level on both test sets. "
                "It then won the 5-condition radio-degradation sweep 4/5 and CER 5/5, with Whisper "
                "exceeding 100% CER at 0 dB SNR (ks_max deployed 2026-07-20). Two successors improved it: "
                "ks_max2 (2026-07-26; same recipe on a 4x combined corpus — humair025 IndicVoices + "
                "IndicVoices-R + OpenSLR-122) reached 61.88% diacritic-normalised; ks_cloud "
                "(2026-07-27; LoRA rank raised 32 to 128 on a rented cloud GPU, ~$4) reached 56.44%; and "
                "ks_cloud2, the same run simply allowed to converge instead of early-stopping at 0.8 "
                "epochs, reached 52.60%. Finally ks_cloud3 repaired 20 Kashmiri characters that had no "
                "token in SeamlessM4T's vocabulary and reached <b>50.26%</b>, winning the degradation "
                "sweep 5/5 against Whisper. Kashmiri now runs on the ks_cloud3 adapter; ks_cloud2 and "
                "this Whisper model are retained for rollback (see §5.5).",
        "training_time": "~18 h (incl. 2 power outages, PD-charger recovery)",
    },
}

LANG_ORDER = ["pa", "ps", "ur", "ne", "zh", "hi", "ks"]

# Cross-model eval results (100-sample FLEURS test / IndicVoices val, 23 Jun 2026)
# Corrected 2026-07-11. Prior numbers were wrong two ways: (1) the "baseline" loaded
# whisper-large-v3-TURBO, not large-v3; (2) Mandarin WER was scored without CJK
# character-segmentation, so unspaced Han transcripts scored ~100% regardless of
# accuracy. Both fixed. Baseline is now the true openai/whisper-large-v3; all WER
# uses one CJK-aware normaliser. Source: docs/model_comparison_results.json (n=100
# FLEURS) + docs/seamless_ft_results.json. "best" = lowest-WER deployable backend.
# ks compare was not re-run (loader pulls the full 18 GB IndicVoices train split);
# its values are the training-time eval, unaffected by the CJK bug.
EVAL_RESULTS = {
    "pa": {"baseline": 77.60, "ft": 57.39, "seamless": 19.77, "baseline_cer": 39.73, "ft_cer": 32.52, "sm_cer": 9.97,
           "nllb_chrf": 40.15, "sm_chrf": 54.53, "best": "seamless"},
    "ps": {"baseline": 89.76, "ft": 38.55, "seamless": 44.40, "baseline_cer": 37.60, "ft_cer": 17.65, "sm_cer": 22.92,
           # "best" = deployed backend. Flipped 2026-07-19 (ps_aug 36.91 beat FT
           # Whisper 38.55 + 4/5 sweep); upgraded 2026-07-27 to ps_cloud (r=128
           # cloud retrain of the same recipe): 36.16 clean + 4/5 sweep vs ps_aug.
           "nllb_chrf": 44.48, "sm_chrf": 40.15, "best": "seamless"},
    "ur": {"baseline": 21.23, "ft": 19.82, "seamless": 16.90, "baseline_cer": 8.12,  "ft_cer": 7.29,  "sm_cer": 7.00,
           "nllb_chrf": 51.34, "sm_chrf": 50.73, "best": "seamless"},
    "ne": {"baseline": 88.85, "ft": 50.92, "seamless": 28.46, "baseline_cer": 29.26, "ft_cer": 18.83, "sm_cer": 11.22,
           "nllb_chrf": 45.55, "sm_chrf": 51.67, "best": "seamless"},
    "zh": {"baseline": 10.99, "ft": 14.22, "seamless": 11.69, "baseline_cer": 10.99, "ft_cer": 14.22, "sm_cer": 11.69,
           "nllb_chrf": 42.00, "sm_chrf": 49.15, "best": "seamless"},
    "hi": {"baseline": 26.34, "ft": 19.78, "seamless": 15.44, "baseline_cer": 10.55, "ft_cer": 7.46,  "sm_cer": 9.12,
           "nllb_chrf": 53.71, "sm_chrf": 51.54, "best": "seamless"},
    "ks": {"baseline": 96.87, "ft": 74.02, "seamless": None,  "baseline_cer": None,  "ft_cer": None,  "sm_cer": None,
           # "best" = deployed backend. Flipped 2026-07-20 (ks_max wins the
           # corrected diacritic ruler 64.31 vs 65.19 — docs/ks_ruler_study.json);
           # upgraded 2026-07-26 to ks_max2 (61.88, 4x combined corpus) and
           # 2026-07-27 to ks_cloud (56.44, r=128 cloud retrain) then ks_cloud2
           # (52.60, same recipe trained to convergence; sweep 5/5 vs Whisper,
           # 4/5 vs ks_cloud) — see docs/ks_cloud_ruler_compare.json +
           # ks_cloud2_degradation.json.
           "nllb_chrf": None,  "sm_chrf": None,  "best": "seamless"},
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

def build_styles():
    S = {}
    S["Title"]    = ParagraphStyle("Title",    fontName="Helvetica-Bold",   fontSize=26,
                                   textColor=HDR_BLUE, spaceAfter=6, alignment=TA_CENTER)
    S["H1"]       = ParagraphStyle("H1",       fontName="Helvetica-Bold",   fontSize=16,
                                   textColor=HDR_BLUE, spaceBefore=18, spaceAfter=6, leading=20,
                                   keepWithNext=True)
    S["H2"]       = ParagraphStyle("H2",       fontName="Helvetica-Bold",   fontSize=13,
                                   textColor=ACCENT, spaceBefore=12, spaceAfter=4, leading=16,
                                   keepWithNext=True)
    S["H3"]       = ParagraphStyle("H3",       fontName="Helvetica-BoldOblique", fontSize=11,
                                   textColor=colors.HexColor("#333333"), spaceBefore=8, spaceAfter=3,
                                   keepWithNext=True)
    S["Body"]     = ParagraphStyle("Body",     fontName="Helvetica",        fontSize=10,
                                   leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    S["Bullet"]   = ParagraphStyle("Bullet",   fontName="Helvetica",        fontSize=10,
                                   leading=14, spaceAfter=3, leftIndent=16)
    S["Caption"]  = ParagraphStyle("Caption",  fontName="Helvetica-Oblique",fontSize=9,
                                   textColor=colors.HexColor("#666666"), alignment=TA_CENTER, spaceAfter=8)
    S["Note"]     = ParagraphStyle("Note",     fontName="Helvetica-Oblique",fontSize=9,
                                   textColor=colors.HexColor("#444444"), leading=13,
                                   leftIndent=8, spaceAfter=4)
    S["CodePre"]  = ParagraphStyle("CodePre",  fontName="Courier",          fontSize=7.5,
                                   leading=11, spaceAfter=0)
    # table cell styles
    S["TC"]       = ParagraphStyle("TC",       fontName="Helvetica",        fontSize=9,
                                   leading=12, alignment=TA_CENTER)
    S["TCL"]      = ParagraphStyle("TCL",      fontName="Helvetica",        fontSize=9,
                                   leading=12, alignment=TA_LEFT)
    S["TCB"]      = ParagraphStyle("TCB",      fontName="Helvetica-Bold",   fontSize=9,
                                   leading=12, alignment=TA_LEFT)
    S["TCH"]      = ParagraphStyle("TCH",      fontName="Helvetica-Bold",   fontSize=9,
                                   leading=12, alignment=TA_CENTER, textColor=colors.white)
    S["TCHL"]     = ParagraphStyle("TCHL",     fontName="Helvetica-Bold",   fontSize=9,
                                   leading=12, alignment=TA_LEFT, textColor=colors.white)
    return S

# ─────────────────────────────────────────────────────────────────────────────
# FLOWABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_S = None   # module-level style dict, set in build()

def sp(n=6):       return Spacer(1, n)
def hr():          return HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#CCCCCC"),
                                     spaceAfter=6, spaceBefore=6)
def h1(t):         return Paragraph(t, _S["H1"])
def h2(t):         return Paragraph(t, _S["H2"])
def h3(t):         return Paragraph(t, _S["H3"])
def body(t):       return Paragraph(t, _S["Body"])
def note(t):       return Paragraph(t, _S["Note"])
def caption(t):    return Paragraph(t, _S["Caption"])
def bullet(t):     return Paragraph(f"&bull; &nbsp;{t}", _S["Bullet"])

# Table cell wrappers — THESE ARE CRITICAL for word-wrap inside tables
def tc(t, bold=False):
    """Centred table cell (wraps text)."""
    return Paragraph(str(t), _S["TCB"] if bold else _S["TC"])
def tcl(t, bold=False):
    """Left-aligned table cell (wraps text)."""
    return Paragraph(str(t), _S["TCB"] if bold else _S["TCL"])
def tch(t, left=False):
    """Header cell — white bold text."""
    return Paragraph(str(t), _S["TCHL"] if left else _S["TCH"])
def ProfBlue():
    """Bold blue centred style for 'deployed backend = SeamlessM4T' cells."""
    return ParagraphStyle("PB", fontName="Helvetica-Bold", fontSize=9,
                          textColor=HDR_BLUE, alignment=TA_CENTER)
def ProfGrn():
    """Bold green centred style for 'deployed backend = FT Whisper' cells."""
    return ParagraphStyle("PG", fontName="Helvetica-Bold", fontSize=9,
                          textColor=SUCCESS_GRN, alignment=TA_CENTER)

def code_block(txt):
    """Preformatted code block with grey background, proper line breaks."""
    pre = Preformatted(txt, _S["CodePre"])
    t = Table([[pre]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CODE_BG),
        ("BOX",           (0,0), (-1,-1), 0.5, CODE_BORDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    return t

def img_from_buf(buf, width=None):
    w = width or PAGE_W
    return Image(buf, width=w, height=w * 0.45)

def img_scaled(buf, width=None):
    """Embed a PNG buffer preserving its TRUE aspect ratio (unlike img_from_buf,
    which forces 0.45). Used for the shared report_charts hero figures."""
    from PIL import Image as _PILImage
    w = width or PAGE_W
    buf.seek(0)
    pw, ph = _PILImage.open(buf).size
    buf.seek(0)
    return Image(buf, width=w, height=w * ph / pw)

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD TABLE STYLE
# ─────────────────────────────────────────────────────────────────────────────

def std_ts(header_rows=1, left_cols=()):
    """Return a TableStyle. left_cols = column indices to left-align."""
    s = TableStyle([
        ("BACKGROUND",     (0, 0),  (-1, header_rows-1), TABLE_HDR),
        ("TEXTCOLOR",      (0, 0),  (-1, header_rows-1), colors.white),
        ("FONTNAME",       (0, 0),  (-1, header_rows-1), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0),  (-1, -1), 9),
        ("ALIGN",          (0, 0),  (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0),  (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, TABLE_EVEN]),
        ("GRID",           (0, 0),  (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0, 0),  (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 5),
        ("LEFTPADDING",    (0, 0),  (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0),  (-1, -1), 6),
    ])
    for col in left_cols:
        s.add("ALIGN", (col, 0), (col, -1), "LEFT")
    return s

# ─────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf

def chart_wer_all():
    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in LANG_ORDER:
        m = LANG_META[lang]
        steps = [p[0] for p in m["wer_curve"]]
        wers  = [p[1] for p in m["wer_curve"]]
        ax.plot(steps, wers, "o-", color=PALETTE[lang], linewidth=2,
                markersize=6, label=f"{m['name']} ({lang.upper()})")
        if "wer_curve_diverged" in m:
            dx = [p[0] for p in m["wer_curve_diverged"]]
            ax.plot(dx, [min(p[1], 110) for p in m["wer_curve_diverged"]],
                    "x", color=PALETTE[lang], markersize=12, markeredgewidth=2)
            ax.annotate("gradient\nexplosion\n(252%)",
                        xy=(dx[0], 110), fontsize=7, color=PALETTE[lang],
                        ha="center", va="bottom")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Word Error Rate (%)", fontsize=11)
    ax.set_title("WER Progression During LoRA Fine-Tuning", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_xlim(0, 3100)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return _save(fig)

def chart_wer_per_lang(lang):
    m = LANG_META[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    steps = [p[0] for p in m["wer_curve"]]
    wers  = [p[1] for p in m["wer_curve"]]
    has_v1 = "wer_curve_v1" in m
    has_v3 = "wer_curve_v3" in m
    label_v2 = ("v2 (superseded)" if has_v3 else "v2 (deployed)") if (has_v1 or has_v3) else None
    ax1.plot(steps, wers, "o-", color=PALETTE[lang], linewidth=2, markersize=7,
             label=label_v2)
    if has_v1:
        v1s = [p[0] for p in m["wer_curve_v1"]]
        v1w = [p[1] for p in m["wer_curve_v1"]]
        ax1.plot(v1s, v1w, "o--", color=PALETTE[lang], linewidth=1.5, markersize=5,
                 alpha=0.38, label="v1 (superseded)")
    if has_v3:
        v3s = [p[0] for p in m["wer_curve_v3"]]
        v3w = [p[1] for p in m["wer_curve_v3"]]
        ax1.plot(v3s, v3w, "s:", color=PALETTE[lang], linewidth=2.2, markersize=6,
                 alpha=0.85, label="v3 (deployed, r=16, 21,923 samp.)")
    if "wer_curve_diverged" in m:
        dx = [p[0] for p in m["wer_curve_diverged"]]
        ax1.plot(dx, [min(p[1], 120) for p in m["wer_curve_diverged"]],
                 "rx", markersize=12, markeredgewidth=2, label="Diverged (not deployed)")
    if has_v1 or has_v3 or "wer_curve_diverged" in m:
        ax1.legend(fontsize=8, loc="upper right")
    ax1.axhline(m["baseline_wer"], color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    # offset-points placement: data-unit offsets blow up the canvas on charts
    # with a small WER range (ur spans ~2.4 pp, so "+5 WER" was two axes-heights up)
    ax1.annotate(f"Baseline ~{m['baseline_wer']:.0f}%",
                 xy=(steps[0], m["baseline_wer"]), xytext=(2, 4),
                 textcoords="offset points", fontsize=8, color="gray")
    bstep, bwer = m["best_step"], m["best_wer"]
    ax1.plot(bstep, bwer, "*", color="gold", markersize=14, zorder=5,
             markeredgecolor=PALETTE[lang], markeredgewidth=1)
    ax1.annotate(f"Best: {bwer:.2f}%",
                 xy=(bstep, bwer), xytext=(-48, 14),
                 textcoords="offset points",
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("WER (%)")
    ax1.set_title(f"{m['name']}: Eval WER")
    ax1.grid(True, alpha=0.3, linestyle="--")
    tl_steps = [p[0] for p in m["train_loss"]]
    tl_vals  = [p[1] for p in m["train_loss"]]
    ax2.plot(tl_steps, tl_vals, "-", color=PALETTE[lang], linewidth=1.5, alpha=0.9)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Training Loss")
    ax2.set_title(f"{m['name']}: Training Loss")
    ax2.grid(True, alpha=0.3, linestyle="--")
    fig.suptitle(f"{m['name']} ({m['iso'].upper()}) — Training Curves",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig)

def chart_summary_bar():
    # Held-out test WER (eval_wer) on both sides — the baseline is a held-out
    # figure, so plotting train-val best_wer against it mixed rulers and showed
    # Mandarin as a gain when it is a regression.
    #
    # Kashmiri is EXCLUDED. Every bar here is FLEURS n=100; Kashmiri is absent
    # from FLEURS, and both its figures were training-split values (baseline
    # 96.87 is a different model, whisper-small-ks, and eval_wer 74.02 is the
    # step-2400 validation WER). Plotting them beside six held-out pairs under
    # the title "held-out test" is the same ruler mix §5.2 now footnotes.
    # Kashmiri's held-out story is in §5.5 on its own 372-clip L2 ruler.
    chart_langs = [l for l in LANG_ORDER if l != "ks"]
    fig, ax = plt.subplots(figsize=(10, 5))
    names      = [LANG_META[l]["name"]       for l in chart_langs]
    baselines  = [LANG_META[l]["baseline_wer"] for l in chart_langs]
    tests      = [LANG_META[l]["eval_wer"]     for l in chart_langs]
    deltas     = [f - b for b, f in zip(baselines, tests)]
    x = np.arange(len(chart_langs))
    w = 0.32
    ax.bar(x - w/2, baselines, w, label="Baseline WER (no fine-tuning)",
           color="#BDBDBD", edgecolor="white")
    bars2 = ax.bar(x + w/2, tests, w, label="Fine-tuned WER (held-out test)",
                   color=[PALETTE[l] for l in chart_langs], edgecolor="white")
    for bar, d in zip(bars2, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{d:+.1f}pp", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#1B5E20" if d < 0 else "#B71C1C")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Word Error Rate (%)", fontsize=11)
    ax.set_title("Baseline vs. Fine-Tuned WER (held-out test)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _save(fig)

def chart_dataset_sizes():
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"{LANG_META[l]['name']}\n({LANG_META[l]['iso'].upper()})"
             for l in LANG_ORDER]
    train = [LANG_META[l]["train_samples"] for l in LANG_ORDER]
    val   = [LANG_META[l]["val_samples"]   for l in LANG_ORDER]
    x = np.arange(len(LANG_ORDER))
    w = 0.35
    ax.bar(x - w/2, train, w, label="Train",
           color=[PALETTE[l] for l in LANG_ORDER], alpha=0.9)
    ax.bar(x + w/2, val,   w, label="Validation",
           color=[PALETTE[l] for l in LANG_ORDER], alpha=0.45)
    for i, (t, v) in enumerate(zip(train, val)):
        ax.text(i - w/2, t + 30, str(t), ha="center", fontsize=8, fontweight="bold")
        ax.text(i + w/2, v + 30, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Training Dataset Sizes per Language (PA/NE include IndicVoices-R)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _save(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE NUMBERING
# ─────────────────────────────────────────────────────────────────────────────

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for i, state in enumerate(self._saved_page_states):
            self.__dict__.update(state)
            self._draw_footer(i + 1, total)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_footer(self, page, total):
        if page == 1:
            return
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#888888"))
        self.drawRightString(A4[0] - 2*cm, 1.2*cm,
                             f"VANI Fine-Tuning Report  |  Page {page} of {total}")
        self.drawString(2*cm, 1.2*cm, "M.Tech Research Project - IIT Indore")

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build():
    global _S
    _S = build_styles()

    out_path = pathlib.Path("docs/VANI_Finetune_Report.pdf")
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="VANI Whisper Fine-Tuning Report",
    )

    W = PAGE_W
    story = []

    # ── widths used repeatedly ────────────────────────────────────────────────
    W2 = [5*cm, W - 5*cm]          # 2-col key/value
    W3 = [3.5*cm, 4*cm, W-7.5*cm]  # 3-col with desc

    # ── COVER ─────────────────────────────────────────────────────────────────
    cover_meta = ParagraphStyle("CM", fontName="Helvetica", fontSize=10, leading=14,
                                textColor=colors.HexColor("#666666"), alignment=TA_CENTER,
                                spaceAfter=4)
    story += [
        Spacer(1, 1.2*cm),
        Paragraph("VANI", ParagraphStyle("BigT", fontName="Helvetica-Bold", fontSize=48,
                                          leading=58, textColor=HDR_BLUE,
                                          alignment=TA_CENTER, spaceAfter=10)),
        Paragraph("Voice Analysis &amp; Neural Intelligence System",
                  ParagraphStyle("CS", fontName="Helvetica", fontSize=16, leading=22,
                                 textColor=colors.HexColor("#444444"), alignment=TA_CENTER,
                                 spaceAfter=12)),
        hr(), sp(10),
        Paragraph("Whisper ASR Fine-Tuning Report",
                  ParagraphStyle("CT", fontName="Helvetica-Bold", fontSize=22, leading=28,
                                 textColor=HDR_BLUE, alignment=TA_CENTER, spaceAfter=10)),
        Paragraph("LoRA Domain Adaptation for Border-Region Radio Intercept Languages",
                  ParagraphStyle("CD", fontName="Helvetica-Oblique", fontSize=13, leading=18,
                                 textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
                                 spaceAfter=0, keepWithNext=True)),
        sp(12),
        Table([
            [tch("Item", left=True), tch("Value", left=True)],
            [tcl("Languages Fine-Tuned", bold=True), tcl("8  (Punjabi, Pashto, Urdu, Nepali, Mandarin, Hindi, Kashmiri, Dogri)")],
            [tcl("Deployed via fine-tuned Whisper", bold=True), tcl("0  — all seven route to SeamlessM4T (LoRA adapters for hi/ne/ps/ks; Whisper kept for rollback)")],
            [tcl("Best deployed WER", bold=True), tcl("Hindi 12.91%  |  Urdu 16.90%  |  Punjabi 19.77%  |  Pashto 36.16%  |  Kashmiri 50.26%  |  Dogri 46.73%  (evaluation path; see 4.7)")],
            [tcl("Training Hardware",    bold=True), tcl("NVIDIA RTX 5060 8 GB VRAM (CUDA) - Windows 11  +  rented RTX A6000 48 GB (cloud, r=128 runs)")],
            [tcl("Base Model",           bold=True), tcl("OpenAI Whisper large-v3 (1.55 B)  +  SeamlessM4T v2 large (2.3 B) for the adapter campaign")],
            [tcl("Adaptation Method",    bold=True), tcl("LoRA  r=8..128, incl. trainable custom tokens  --  0.25% to 6.6% trainable parameters")],
            [tcl("Total Training Time",  bold=True), tcl("~100 h Whisper (7 langs)  +  ~40 h SeamlessM4T adapters  +  ~9 h cloud GPU (see 4.4)")],
            [tcl("Eval Date",            bold=True), tcl("23 June 2026 (cross-model)  |  27 July 2026 (cloud adapters ks_cloud3 / ps_cloud)")],
        ], colWidths=[6*cm, W - 6*cm],
        style=std_ts(left_cols=(0, 1))),
        sp(30),
        Paragraph(f"Date: {date.today().strftime('%d %B %Y')}", cover_meta),
        Paragraph("M.Tech Research Project  -  IIT Indore", cover_meta),
        PageBreak(),
    ]

    # ── 1. ABSTRACT ───────────────────────────────────────────────────────────
    story += [
        h1("1. Abstract"), hr(),
        body(
            "This report documents the LoRA (Low-Rank Adaptation) fine-tuning of OpenAI Whisper "
            "large-v3 for six border-region radio intercept languages, developed as part of the VANI "
            "(Voice Analysis &amp; Neural Intelligence) system at IIT Indore. "
            "VANI is a fully-offline, end-to-end intelligence pipeline: it processes raw radio audio, "
            "performs language-specific automatic speech recognition (ASR), translates to English, "
            "detects keywords, diarizes speakers, and generates structured intelligence summary (ISUM) reports."
        ),
        body(
            "Eight ASR models were eventually fine-tuned: Punjabi (pa), Pashto (ps), Urdu (ur), Nepali (ne), "
            "Mandarin Chinese (zh), Hindi (hi) — plus Kashmiri (ks), which required adding a custom "
            "&lt;|ks|&gt; language token to the vocabulary. Punjabi and Nepali were retrained with "
            "AI4Bharat IndicVoices-R data added to FLEURS (11,923 and 13,332 samples). All models are "
            "quantised to CTranslate2 int8."
        ),
        body(
            "This revision (2026-07-11, extended 2026-07-19) reports a rigorous <b>backend selection</b> "
            "study. Each fine-tuned model was benchmarked, on a held-out FLEURS test set and under "
            "simulated radio-channel degradation, against two alternatives: the true un-fine-tuned "
            "openai/whisper-large-v3 baseline, and zero-shot SeamlessM4T v2. The central finding is "
            "that per-language Whisper fine-tuning is <b>not</b> the best choice for most languages: "
            "zero-shot SeamlessM4T beats the fine-tuned Whisper models on five of seven languages "
            "(pa 19.8%, ne 28.5%, hi 15.4%, ur 16.9%, zh 11.7% WER) and retains that lead under "
            "bandpass and additive-noise degradation. A follow-up campaign of SeamlessM4T LoRA "
            "adapters then also captured <b>Pashto</b> (noise-augmented training: 36.9% vs fine-tuned "
            "Whisper's 38.6%, winning 4/5 degradation conditions) and finally <b>Kashmiri</b> — a "
            "language SeamlessM4T does not natively support — via a custom trainable __kas__ token "
            "plus a scoring-ruler correction. A closing cloud phase (rented 48 GB GPU, ~$6 total) "
            "retrained both at LoRA rank 128, beyond the 8 GB laptop's reach, improving Pashto to "
            "36.16% and Kashmiri to 50.26% (diacritic-normalised), the latter after also repairing 20 characters missing from the model's vocabulary. Fine-tuned Whisper now serves "
            "zero languages in production and is retained solely for rollback. VANI therefore routes "
            "ASR per language rather than using a single model."
        ),
        body(
            "A correction is documented in full: the previously reported Mandarin baseline of 100.03% WER "
            "(and the headline &ldquo;100% &rarr; 9%&rdquo; result) was a measurement artefact. FLEURS "
            "Mandarin references are character-spaced and WER tokenises on whitespace, so the un-fine-tuned "
            "model's unspaced Han output scored ~100% regardless of accuracy. Re-scored with character "
            "segmentation, the true baseline is 10.99% and fine-tuning in fact <b>regressed</b> Mandarin "
            "to 14.22%. The scoring is now unified across all models and languages."
        ),
        sp(8),
    ]

    # ── 2. SYSTEM OVERVIEW ────────────────────────────────────────────────────
    story += [
        h1("2. System Overview - VANI Pipeline"), hr(),
        body(
            "VANI implements a 10-stage audio processing pipeline. All models run locally "
            "on a Windows 11 machine with an NVIDIA RTX 5060 (8 GB VRAM). "
            "No internet connection is required after initial setup."
        ),
        sp(4), h2("2.1 Hardware"),
        Table([
            [tch("Component"), tch("Specification", left=True)],
            [tcl("OS",                   bold=True), tcl("Windows 11 Home (x64)")],
            [tcl("GPU",                  bold=True), tcl("NVIDIA RTX 5060  8 GB GDDR7  (CUDA 12.x)")],
            [tcl("Python",               bold=True), tcl("3.11")],
            [tcl("PyTorch",              bold=True), tcl("2.2+ with CUDA support")],
            [tcl("Training framework",   bold=True), tcl("HuggingFace Transformers + PEFT (LoRA)")],
            [tcl("Inference engine",     bold=True), tcl("faster-whisper (CTranslate2 int8)")],
        ], colWidths=[4.5*cm, W - 4.5*cm],
        style=std_ts(left_cols=(0, 1))),
        sp(10), h2("2.2 Pipeline Architecture"),
        body("The VANI pipeline processes audio through 10 sequential stages:"),
        sp(4),
        Table([
            [tch("Stage"), tch("Module"), tch("Description", left=True)],
            [tc("1"),  tcl("VAD (Silero)"),       tcl("Voice Activity Detection - splits audio into speech segments, discards silence")],
            [tc("2"),  tcl("Preprocessing"),       tcl("Bandpass filter 300-3400 Hz (radio telephony range), noise reduction, normalization")],
            [tc("3"),  tcl("MMS-LID"),             tcl("Facebook MMS Language ID (256 languages) - routes to the correct Whisper model")],
            [tc("3.5"),tcl("Model Routing"),       tcl("Selects the language-specific fine-tuned Whisper CT2 model based on MMS-LID output")],
            [tc("4"),  tcl("ASR (Whisper)"),       tcl("faster-whisper transcription using the selected CT2 model (int8, GPU)")],
            [tc("5"),  tcl("Script Cascade"),      tcl("Arabic-script fallback: if >20% Nastaliq chars detected, override to Urdu routing")],
            [tc("6"),  tcl("Translation"),         tcl("NLLB-200 (600M) translates Indic/foreign transcripts to English")],
            [tc("7"),  tcl("Diarization"),         tcl("Speaker diarization - up to 4 speakers, pyannote-style")],
            [tc("8"),  tcl("Keyword Detection"),   tcl("Multilingual keyword dictionary matching for threat indicators")],
            [tc("9"),  tcl("ISUM"),                tcl("Gemma 3:12B (via Ollama) generates 4-sentence structured intelligence summary")],
            [tc("10"), tcl("Export"),              tcl("SQLite database storage + JSON report output")],
        ], colWidths=[1.3*cm, 3.2*cm, W - 4.5*cm],
        style=std_ts(left_cols=(1, 2)), repeatRows=1),
        sp(10),
    ]

    # ── 3. METHODOLOGY ────────────────────────────────────────────────────────
    story += [
        h1("3. Fine-Tuning Methodology"), hr(),
        h2("3.1 Why LoRA?"),
        body(
            "Full fine-tuning of Whisper large-v3 (1.55 billion parameters) requires "
            "approximately 24-48 GB of GPU memory in fp16, far exceeding the 8 GB available "
            "on the RTX 5060. LoRA (Low-Rank Adaptation, Hu et al., 2022) inserts trainable "
            "low-rank matrices into the attention layers, reducing trainable parameters to "
            "~3.9 million (0.25% of total) while base model weights remain frozen. "
            "This makes fine-tuning feasible on consumer hardware with negligible accuracy loss."
        ),
        sp(4), h2("3.2 LoRA Configuration"),
        Table([
            [tch("Parameter"), tch("v1/v2 Value"), tch("PA v3 Value"), tch("Rationale", left=True)],
            [tcl("Rank (r)"),        tc("8"),   tc("16"),
             tcl("v3 doubles capacity for larger 21,923-sample dataset")],
            [tcl("Alpha (α)"),       tc("16"),  tc("32"),
             tcl("α/r = 2.0 scaling maintained — standard for speech LoRA")],
            [tcl("Dropout"),         tc("0.05"),tc("0.05"),
             tcl("Light regularization; FLEURS + IV-R data is clean read-speech")],
            [tcl("Target modules"),  tc("q_proj, v_proj"), tc("q_proj, v_proj"),
             tcl("Attention query/value projections in Whisper encoder/decoder")],
            [tcl("Trainable params"),tc("~3.9M (0.25%)"), tc("~7.9M (0.51%)"),
             tcl("Doubled for v3; still <1% of 1.55B total parameters")],
            [tcl("Adapter merge"),   tc("merge_and_unload()"), tc("merge_and_unload()"),
             tcl("LoRA weights merged into base before CT2 conversion")],
        ], colWidths=[3.2*cm, 2.4*cm, 2.4*cm, W - 8.0*cm],
        style=std_ts(left_cols=(0, 2))),
        sp(10), h2("3.3 Training Hyperparameters"),
        Table([
            [tch("Hyperparameter"), tch("Value", left=True)],
            [tcl("Batch size (per device)", bold=True),  tcl("2")],
            [tcl("Gradient accumulation",   bold=True),  tcl("1  (effective batch = 2)")],
            [tcl("Learning rate",           bold=True),  tcl("5e-5")],
            [tcl("LR scheduler",            bold=True),  tcl("Linear warmup (50 steps) then linear decay")],
            [tcl("Precision",               bold=True),  tcl("fp16 mixed precision")],
            [tcl("Gradient clipping",       bold=True),  tcl("max_grad_norm = 1.0 (pa v1/v2, ps, ur, ne)  /  0.5 (zh, hi, pa v3, ks — after Mandarin divergence lesson)")],
            [tcl("Best model selection",    bold=True),  tcl("load_best_model_at_end=True  (metric: eval WER)")],
            [tcl("Eval / save frequency",   bold=True),  tcl("Every 200 steps")],
            [tcl("CT2 quantization",        bold=True),  tcl("int8  (beam_size=2, temperature=0.0)")],
        ], colWidths=[5*cm, W - 5*cm],
        style=std_ts(left_cols=(0, 1))),
        sp(10), h2("3.4 Training Pipeline Steps"),
        body("The finetune_whisper.py script follows these steps for each language:"),
        bullet("Load FLEURS dataset (train + validation splits) for the target language"),
        bullet("Preprocess audio: resample to 16 kHz, extract 128-bin log-mel spectrogram"),
        bullet("Tokenize transcripts using WhisperProcessor for the target language"),
        bullet("Initialize LoRA adapter (r=8) on frozen whisper-large-v3 base model"),
        bullet("Train using Seq2SeqTrainer with WER as the primary evaluation metric (jiwer)"),
        bullet("Save checkpoint every 200 steps; keep best checkpoint by eval WER"),
        bullet("After training: merge LoRA adapter into base model weights"),
        bullet("Convert merged model to CTranslate2 CT2 int8 format"),
        bullet("Write preprocessor_config.json to CT2 output (feature_size=128 for large-v3)"),
        bullet("Register CT2 model in config.yaml under whisper_model_<lang> key"),
        sp(6),
        code_block(
            "# Training command example (Hindi)\n"
            "python -u finetune_whisper.py hi --no-cv --steps 600 2>&1 |\n"
            "    Tee-Object logs/finetune_hi.log"
        ),
        sp(10), h2("3.5 Post-Training Conversion"),
        body("After training, the LoRA adapter is merged and converted to CTranslate2 format:"),
        code_block(
            "# Step 1 - Merge LoRA adapter into base model\n"
            "from peft import PeftModel\n"
            "base   = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')\n"
            "peft_m = PeftModel.from_pretrained(base, 'finetune_runs/<lang>/adapter/checkpoint-N')\n"
            "merged = peft_m.merge_and_unload()\n"
            "merged.save_pretrained('finetune_runs/<lang>/merged')\n\n"
            "# Step 2 - Convert to CT2 int8\n"
            "ct2-transformers-converter \\\n"
            "    --model finetune_runs/<lang>/merged \\\n"
            "    --output_dir models/whisper-large-v3-<lang>-ct2 \\\n"
            "    --quantization int8 --force"
        ),
        sp(10),
    ]

    # ── 4. DATASET ────────────────────────────────────────────────────────────
    story += [
        h1("4. Training Dataset — FLEURS + IndicVoices-R"), hr(),
        body(
            "Primary training data comes from FLEURS (Few-shot Learning Evaluation of Universal "
            "Representations of Speech) by Google. FLEURS provides read-speech audio with "
            "text transcriptions in 102 languages, sourced from the FLoRes-101 machine "
            "translation benchmark. Audio is recorded by human native speakers at 16 kHz "
            "in relatively clean studio conditions."
        ),
        body(
            "FLEURS is a general-domain read-speech corpus — it does not contain radio intercept "
            "or military communications audio. Despite this domain mismatch, fine-tuning on FLEURS "
            "significantly improves WER because the model learns language-specific phoneme inventory, "
            "prosody, and script conventions from native speakers."
        ),
        sp(4), h2("4.1 IndicVoices-R Augmentation (Punjabi v2 and Nepali v2)"),
        body(
            "For the second training run (v2) of Punjabi and Nepali, the FLEURS training split was "
            "augmented with AI4Bharat IndicVoices-R (ai4bharat/indicvoices_r). IndicVoices-R is a "
            "large-scale read-speech corpus collected from native Indian-language speakers across "
            "diverse recording conditions, providing complementary phoneme and prosody coverage "
            "beyond the FLEURS studio recordings. Samples were filtered to 2-20 seconds duration; "
            "normalised transcripts (normalized column) were used."
        ),
        Table([
            [tch("Language"), tch("ISO"), tch("FLEURS Train"), tch("IndicVoices-R"), tch("Total Train"), tch("Eval Set"), tch("Status")],
            [tcl("Punjabi v2"), tc("pa"), tc("2,516"), tc("9,407"),  tc("11,923"), tc("805 (FLEURS 251 + IV-R 554)"), tcl("Deployed")],
            [tcl("Punjabi v3"), tc("pa"), tc("2,516"), tc("20,000"), tc("21,923"), tc("805 (same eval set)"),          tcl("In training")],
            [tcl("Nepali v2"),  tc("ne"), tc("3,332"), tc("10,000"), tc("13,332"), tc("572 (FLEURS + IV-R test)"),     tcl("Deployed")],
        ], colWidths=[2.4*cm, 0.85*cm, 2.0*cm, 2.2*cm, 2.0*cm, 3.6*cm, W-13.05*cm],
        style=std_ts(left_cols=(0, 6))),
        sp(6),
        KeepTogether([
            img_from_buf(chart_dataset_sizes(), width=W * 0.85),
            caption("Figure 1: Training sample counts per language (PA and NE reflect v2 combined totals)"),
        ]),
        sp(6),
        h2("4.2 FLEURS Reference Sizes"),
        Table([
            [tch("Language"), tch("ISO"), tch("FLEURS Config"), tch("Train"), tch("Val"), tch("Test"), tch("Region")],
            [tcl("Punjabi"),  tc("pa"), tc("pa_in"),       tc("2,516"), tc("314"), tc("765"), tcl("India")],
            [tcl("Pashto"),   tc("ps"), tc("ps_af"),       tc("2,082"), tc("251"), tc("621"), tcl("Afghanistan")],
            [tcl("Urdu"),     tc("ur"), tc("ur_pk"),       tc("2,109"), tc("267"), tc("631"), tcl("Pakistan")],
            [tcl("Nepali"),   tc("ne"), tc("ne_np"),       tc("3,332"), tc("305"), tc("874"), tcl("Nepal")],
            [tcl("Mandarin"), tc("zh"), tc("cmn_hans_cn"), tc("3,246"), tc("409"), tc("945"), tcl("China (Simplified)")],
            [tcl("Hindi"),    tc("hi"), tc("hi_in"),       tc("2,120"), tc("239"), tc("585"), tcl("India")],
        ], colWidths=[2.8*cm, 1.2*cm, 3.6*cm, 1.6*cm, 1.2*cm, 1.2*cm, W-11.6*cm],
        style=std_ts(left_cols=(0, 6)), repeatRows=1),
        sp(6),
        note(
            "Note: Kashmiri (ks) has no FLEURS config and no Common Voice corpus; the initial Whisper "
            "run used AI4Bharat IndicVoices-R Kashmiri (24.7k clips). The later SeamlessM4T campaign "
            "assembled a combined corpus from three public sources — humair025 KashmiriSpeech-"
            "IndicVoices, IndicVoices-R, and OpenSLR SLR122 — reaching 335 hours by the final cloud "
            "run (§4.4)."
        ),
        sp(10),
        h2("4.3 Training Version History — All Retraining Runs"),
        body(
            "Several languages required multiple training runs (versions) to incorporate new data, "
            "fix engineering issues, or increase model capacity. The table below records every run "
            "in chronological order."
        ),
        sp(4),
        Table([
            [tch("Lang"), tch("Ver."), tch("LoRA r / α"), tch("Dataset"),
             tch("Samples"), tch("Steps"), tch("Best WER"), tch("vs Prior"), tch("Status")],
            # PA
            [tcl("Punjabi"), tcl("v1"), tc("8 / 16"),
             tcl("FLEURS pa_in"), tc("2,516"), tc("3,000"), tc("56.67%"), tc("—"), tcl("Superseded")],
            [tcl("Punjabi"), tcl("v2"), tc("8 / 16"),
             tcl("FLEURS + IV-R"), tc("11,923"), tc("3,000"), tc("52.55%"), tc("-4.1 pp"), tcl("Superseded")],
            [tcl("Punjabi"), tcl("v3★"), tc("16 / 32"),
             tcl("FLEURS + IV-R 20k"), tc("21,923"), tc("4,000"), tc("49.31%"), tc("-3.2 pp"), tcl("Rollback")],
            # NE
            [tcl("Nepali"), tcl("v1"), tc("8 / 16"),
             tcl("FLEURS ne_np"), tc("3,332"), tc("2,000"), tc("52.14%"), tc("—"), tcl("Superseded")],
            [tcl("Nepali"), tcl("v2"), tc("8 / 16"),
             tcl("FLEURS + IV-R"), tc("13,332"), tc("3,000"), tc("50.82%"), tc("-1.3 pp"), tcl("Rollback")],
            # ZH
            [tcl("Mandarin"), tcl("v1‡"), tc("8 / 16"),
             tcl("FLEURS cmn_hans_cn"), tc("3,246"), tc("400 (†div.)"), tc("8.97% train"), tc("—"), tcl("Not served‡")],
            # PS/UR/HI/KS — single runs
            [tcl("Pashto"), tcl("v1"), tc("8 / 16"),
             tcl("FLEURS ps_af"), tc("2,082"), tc("2,000"), tc("38.55%"), tc("—"), tcl("Rollback")],
            [tcl("Urdu"), tcl("v1"), tc("8 / 16"),
             tcl("FLEURS ur_pk"), tc("2,109"), tc("1,000"), tc("19.82%"), tc("—"), tcl("Rollback")],
            [tcl("Hindi"), tcl("v1"), tc("8 / 16"),
             tcl("FLEURS hi_in"), tc("2,120"), tc("600"), tc("19.78%"), tc("—"), tcl("Rollback")],
            [tcl("Kashmiri"), tcl("v1"), tc("8 / 16"),
             tcl("IV-R KS (custom ⟨ks⟩ token)"), tc("20,000"), tc("3,000"), tc("74.02%"), tc("—"), tcl("Rollback")],
        ], colWidths=[1.8*cm, 1.0*cm, 1.6*cm, 3.0*cm, 1.6*cm, 1.6*cm, 1.9*cm, 1.6*cm, W-14.1*cm],
        style=std_ts(left_cols=(0, 1, 3, 8)), repeatRows=1),
        sp(4),
        note("★ PA v3: built and CT2-deployed 2026-07-04; best training-val WER 49.31% at step 4000 "
             "(-3.24 pp vs v2's 52.55%), held-out test 57.39%. Since 2026-07-11 Punjabi ASR routes to "
             "SeamlessM4T (19.77%); the v3 model is retained on disk but not served."),
        note("‡ Mandarin training diverged at step ~820 (fp16 gradient explosion, grad_norm=12.9). "
             "Checkpoint-400 (train WER 8.97%, held-out test WER 14.22%) was the best checkpoint, but "
             "fine-tuning regressed Mandarin vs the 10.99% large-v3 baseline — so it is NOT deployed; "
             "Mandarin routes to SeamlessM4T (§5.5). Single run."),
        sp(10),

        h2("4.4 Training Hours and Dataset Sizes — What the Campaign Actually Cost"),
        body(
            "This section replaces the estimate the report previously carried (\"roughly 100 "
            "GPU-hours\" for the Whisper phase) with measured figures. Every run recorded its own "
            "timing: HuggingFace's Trainer writes an <font face='Courier'>eval_runtime</font> into "
            "each checkpoint's <font face='Courier'>trainer_state.json</font>, and the checkpoint "
            "directories carry the wall-clock stamps of when they were written. "
            "<font face='Courier'>scripts/eval/training_cost_inventory.py</font> recovers both and "
            "writes <font face='Courier'>docs/training_cost_inventory.json</font>; the tables below "
            "are generated from it."
        ),
        sp(6),
        h3("4.4.1  The Whisper phase — and how much of it was evaluation"),
        body(
            "Only three Whisper runs still have their full checkpoint history on disk; the rest were "
            "pruned by <font face='Courier'>save_total_limit</font> once the models were converted "
            "to CT2, so their timings are not recoverable. Those three alone account for more "
            "wall-clock than the whole phase was previously estimated at."
        ),
        sp(4),
        Table([
            [tch("Run"), tch("Steps"), tch("Evals"), tch("Elapsed"), tch("Idle"),
             tch("Active"), tch("Of which eval"), tch("Training")],
            [tcl("Punjabi v3 (r=16)"), tc("4,000"), tc("20"), tc("68.05 h"), tc("10.18 h"),
             tc("57.88 h"), Paragraph("<b>36.73 h (63%)</b>", ProfBlue()), tc("21.15 h")],
            [tcl("Punjabi v2 (r=8)"), tc("3,000"), tc("15"), tc("37.06 h"), tc("—"),
             tc("37.06 h"), Paragraph("<b>25.02 h (68%)</b>", ProfBlue()), tc("12.04 h")],
            [tcl("Pashto large-v3 (r=16)"), tc("2,000"), tc("10"), tc("8.20 h"), tc("—"),
             tc("8.20 h"), Paragraph("<b>2.25 h (27%)</b>", ProfBlue()), tc("5.95 h")],
            [Paragraph("<b>Three runs, measured</b>", ProfBlue()), tc("9,000"), tc("45"),
             tc("113.31 h"), tc("10.18 h"),
             Paragraph("<b>103.14 h</b>", ProfBlue()),
             Paragraph("<b>64.00 h (62%)</b>", ProfBlue()),
             Paragraph("<b>39.14 h</b>", ProfBlue())],
        ], colWidths=[3.7*cm, 1.3*cm, 1.1*cm, 1.7*cm, 1.4*cm, 1.6*cm, 2.5*cm, W-13.3*cm],
        style=std_ts(left_cols=(0,)), repeatRows=1),
        sp(4),
        note("<b>Evaluation, not training, was the cost.</b> Across these three runs 62% of the "
             "wall-clock went on periodic evaluation rather than on gradient steps — 64 hours "
             "against 39. The cause is in §6.9: <font face='Courier'>predict_with_generate</font> "
             "at <font face='Courier'>per_device_eval_batch_size=1</font>, run every 200 steps "
             "against validation splits of several hundred clips. Punjabi v3's 805-clip validation "
             "split cost 1.84 h <i>per evaluation</i>, twenty times over. Halving the evaluation "
             "frequency would have returned roughly a working day per run at no cost to model "
             "selection, since §6.8 shows per-checkpoint WER oscillates anyway."),
        note("\"Idle\" is time inside gaps more than 3x the median interval between "
             "consecutive checkpoints — the laptop was not training then. Elapsed spans are taken "
             "from checkpoint modification times, so they are an upper bound on compute; the "
             "eval-hours column is the Trainer's own measurement and is exact. The previously "
             "published \"roughly 100 GPU-hours\" figure covered all eleven Whisper runs; these "
             "three measured runs come to 103 h on their own, so the true phase total is "
             "materially higher and the old estimate was low."),
        sp(10),
        h3("4.4.2  The SeamlessM4T phase — the same work, two orders of magnitude cheaper to check"),
        body(
            "The adapter phase inverted the ratio. Nineteen local SeamlessM4T runs spent "
            "<b>4.69 hours in total</b> on evaluation — against 64 hours for three Whisper runs — "
            "because SeamlessM4T's validation splits are small and its decoding is faster. The "
            "longest single SeamlessM4T evaluation budget in the campaign, ks_max2's 46 "
            "evaluations, came to 0.95 h. That difference is why the adapter campaign could afford "
            "to try twenty-six things while the Whisper phase managed eleven."
        ),
        sp(6),
        body(
            "LoRA adapters also train in hours rather than days, and the final capacity push rented "
            "cloud hardware by the hour instead of scaling the laptop. The two production adapters "
            "cost about $6 of cloud compute combined. Key runs of the closing campaign:"
        ),
        sp(4),
        Table([
            [tch("Run"), tch("Language"), tch("Data (clips / hours)"), tch("Rank"),
             tch("Hardware"), tch("Wall time"), tch("Result"), tch("Status")],
            [tcl("ks_max2"), tcl("Kashmiri"), tcl("97,456 / 240 h (3-source combined)"), tc("32"),
             tcl("RTX 5060 8 GB"), tc("~19 h"), tc("61.88%"), tcl("Rollback")],
            [tcl("ps_aug2"), tcl("Pashto"), tcl("32,656 / ~55 h (CV 30k + FLEURS x8)"), tc("32"),
             tcl("RTX 5060 8 GB"), tc("~5 h"), tc("37.46%"), tcl("Negative result")],
            [tcl("ks_cloud"), tcl("Kashmiri"), tcl("144,749 / 335 h (corpus rebuilt from source)"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("6 h 35 m"), tc("56.44%"), tcl("Rollback")],
            [tcl("ks_cloud2"), tcl("Kashmiri"), tcl("144,749 / 335 h (as ks_cloud, 2 epochs)"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("~9 h"), tc("52.60%"), tcl("Rollback")],
            [tcl("ks_cloud3"), tcl("Kashmiri"), tcl("144,942 / 336 h (as ks_cloud2 + 20 vocab chars)"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("~8 h"), Paragraph("<b>50.26%</b>", ProfBlue()), tcl("Deployed")],
            [tcl("ks_cloud4"), tcl("Kashmiri"), tcl("as ks_cloud3, warm start, token rows at 5x LR"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("~1 h"), tc("50.69%"), tcl("Rejected")],
            [tcl("ps_cloud"), tcl("Pashto"), tcl("18,656 / ~30 h (CV 10k + FLEURS x8)"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("1 h 32 m"), Paragraph("<b>36.16%</b>", ProfBlue()), tcl("Deployed")],
            [tcl("doi_iv"), tcl("Dogri"), tcl("IndicVoices-R Dogri (97 shards, 43.8 GB)"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("~6 h"), tc("50.07%"), tcl("Cut short by schedule")],
            [tcl("doi_iv2"), tcl("Dogri"), tcl("as doi_iv, fresh LR schedule, 9,000 steps"), tc("128"),
             tcl("RTX A6000 48 GB (cloud)"), tc("~9 h"), Paragraph("<b>46.73%</b>", ProfBlue()), tcl("Best Dogri model")],
        ], colWidths=[1.9*cm, 1.8*cm, 3.9*cm, 1.0*cm, 2.6*cm, 1.5*cm, 1.5*cm, W-14.2*cm],
        style=std_ts(left_cols=(0, 1, 2, 4, 7)), repeatRows=1),
        sp(4),
        note("Kashmiri WER is diacritic-normalised (L2) on the 372-clip IndicVoices-R test; Pashto WER is "
             "clean FLEURS ps_af (n=100). Wall times include periodic evaluation. Cloud runs used one "
             "rented RTX A6000 at $0.53/hr — ks_cloud ~$4, ks_cloud2 ~$5, ps_cloud ~$2 including "
             "data-preparation time."),
        note("ks_cloud2 and doi_iv2 are the cheapest results in this table: identical reruns of "
             "ks_cloud and doi_iv that were simply allowed to finish. Both predecessors stopped with "
             "their validation loss still descending — ks_cloud because patience 3 was too tight for "
             "a corpus that size, doi_iv because its linear schedule had decayed the learning rate to "
             "zero — and continuing each was worth 3.84 pp and 3.34 pp respectively. Two languages, "
             "two scripts, the same mechanism: a training run that halts while still improving is a "
             "silent and easily missed loss."),
        note("The ks_cloud corpus (rebuilt on the cloud box from the same three public sources) came out "
             "half again larger than ks_max2's because the humair025 dataset had grown upstream: "
             "humair025 120,845 clips / 275.6 h + IndicVoices-R train 22,824 / 58.3 h + OpenSLR-122 "
             "1,080 / 1.6 h = 144,749 clips / 335.4 h (131,868 unique sentences), duration-filtered to "
             "2–20 s with the 403-sentence IndicVoices-R test blocklist removed as an evaluation-leak "
             "guard (cloud/prep_ks_data.py, composition.json)."),
        note("Capacity, not data, was the decisive lever at the end: r=128 trains 6.6% of model "
             "parameters (~107 M) vs r=32's 1.75% — beyond the 8 GB laptop, trivial for a 48 GB card. "
             "The same lesson does NOT hold in the other direction for data alone, and the distinction "
             "matters: a CV-DOMINATED mixture (ps_cv) was significantly worse than the balanced one "
             "(42.47% vs 36.91%, paired bootstrap p < 0.001), but merely scaling the Common Voice "
             "share from 10k to 30k at unchanged capacity (ps_aug2, 37.46%) produced NO significant "
             "change (p = 0.48). More data was harmful when it unbalanced the mixture, not simply "
             "because there was more of it."),
        sp(10),

        h3("4.4.3  Dataset sizes — every corpus the campaign trained on"),
        body(
            "Training-set sizes span three orders of magnitude, from Pashto's 2,082 FLEURS clips to "
            "Kashmiri's final 144,749-clip assembly. Where a corpus was built by this project the "
            "duration is known exactly; for the FLEURS splits, which were used as shipped, only "
            "clip counts were recorded."
        ),
        sp(4),
        Table([
            [tch("Language"), tch("Corpus used (largest run)"), tch("Train clips"),
             tch("Audio hours"), tch("Held-out test")],
            [tcl("Kashmiri"), tcl("humair025 IndicVoices + IndicVoices-R + OpenSLR-122"),
             tc("144,749"), Paragraph("<b>335.4 h</b>", ProfBlue()), tcl("372 (IndicVoices-R)")],
            [tcl("Dogri"), tcl("IndicVoices-R Dogri (97 shards, 43.8 GB on disk)"),
             tc("—"), tc("not recorded"), tcl("425 (IndicVoices-R)")],
            [tcl("Pashto"), tcl("Common Voice 30k + FLEURS ps_af x8 (ps_aug2)"),
             tc("32,656"), tc("~55 h"), tcl("100 (FLEURS)")],
            [tcl("Punjabi"), tcl("FLEURS pa_in + IndicVoices-R Punjabi (v3)"),
             tc("21,923"), tc("not recorded"), tcl("100 (FLEURS)")],
            [tcl("Nepali"), tcl("FLEURS ne_np + IndicVoices-R Nepali (v2)"),
             tc("13,332"), tc("not recorded"), tcl("100 (FLEURS)")],
            [tcl("Mandarin"), tcl("FLEURS cmn_hans_cn"),
             tc("3,246"), tc("not recorded"), tcl("100 (FLEURS)")],
            [tcl("Hindi"), tcl("FLEURS hi_in + IndicVoices-R (cap 20k)"),
             tc("2,120 +"), tc("not recorded"), tcl("100 (FLEURS)")],
            [tcl("Urdu"), tcl("FLEURS ur_pk"),
             tc("2,109"), tc("not recorded"), tcl("100 (FLEURS)")],
        ], colWidths=[2.2*cm, 6.6*cm, 2.0*cm, 2.2*cm, W-13.0*cm],
        style=std_ts(left_cols=(0, 1, 4)), repeatRows=1),
        sp(4),
        note("Kashmiri's is the only corpus this project assembled from raw sources rather than "
             "consuming as published, which is why it is the only row with an exact duration: "
             "humair025 120,845 clips / 275.6 h + IndicVoices-R train 22,824 / 58.3 h + OpenSLR-122 "
             "1,080 / 1.6 h, duration-filtered to 2–20 s, deduplicated to 131,868 unique sentences, "
             "with the 403-sentence IndicVoices-R test blocklist removed as an evaluation-leak "
             "guard. Recorded in cloud/prep_ks_data.py and its composition.json."),
        note("<b>Corpus size did not order the results.</b> Kashmiri had by far the most training "
             "audio and finished with the worst WER of the eight (50.26%); Hindi had the least "
             "and finishes best (12.91%). What separated them was representation, not volume — "
             "Hindi is a language both backbones ship natively, Kashmiri is one neither can name "
             "and whose script the subword vocabulary cannot fully encode (§5.5.3). The same point "
             "recurs within Pashto, where tripling the Common Voice share changed nothing "
             "measurable."),
        sp(10),

        h2("4.5 Dogri (doi) — the Eighth Language"),
        body(
            "VANI's problem statement names <b>eight</b> border-region languages, but for most of this "
            "project's life every result table covered seven. The eighth is <b>Dogri</b>, and until "
            "2026-07-28 it had no fine-tuned ASR model, no evaluation and no audio on the project "
            "machine — while still being wired into the pipeline for <b>language identification</b> "
            "and for <b>translation</b>, where it is the one language routed to IndicTrans2 rather "
            "than NLLB-200 because it is absent from the distilled NLLB vocabulary. In practice it "
            "fell back silently to un-fine-tuned Whisper, and nobody had measured what that cost."
        ),
        sp(4),
        body(
            "<b>Neither backend has a Dogri language token.</b> Whisper has no "
            "<font face='Courier'>&lt;|doi|&gt;</font>, and SeamlessM4T v2's 101 language tokens "
            "include <font face='Courier'>__hin__</font>, <font face='Courier'>__pan__</font> and "
            "<font face='Courier'>__urd__</font> but neither <font face='Courier'>__doi__</font> nor "
            "<font face='Courier'>__dgo__</font>. Dogri therefore began exactly where Kashmiri did, "
            "and the same remedy applied: a custom language token, initialised from a neighbour and "
            "made trainable (5.5.2), on IndicVoices-R Dogri data — the corpus family that supplied "
            "Kashmiri. This is the <b>second</b> use of that technique, and the test of whether it is "
            "a method rather than a one-off fix."
        ),
        sp(6),
        Table([
            [tch("Dogri ASR system"), tch("WER %"), tch("CER %"), tch("What it is")],
            [tcl("SeamlessM4T zero-shot, __pan__ proxy"), tc("114.62"), tc("96.79"),
             tcl("closest language genetically; wrong script")],
            [tcl("Whisper large-v3, auto-detect"), tc("102.25 †"), tc("—"),
             tcl("what VANI actually did for Dogri")],
            [tcl("SeamlessM4T zero-shot, __hin__ proxy"), tc("99.86"), tc("67.99"),
             tcl("script-matched proxy")],
            [tcl("Whisper large-v3, forced Hindi"), tc("88.08 †"), tc("—"),
             tcl("best baseline obtainable without training")],
            [tcl("doi_iv (custom __doi__ token + LoRA)"), tc("50.07"), tc("27.29"),
             tcl("first Dogri model; stopped by its LR schedule")],
            [Paragraph("<b>doi_iv2 (same, trained to convergence)</b>", ProfBlue()),
             Paragraph("<b>46.73</b>", ProfBlue()), Paragraph("<b>25.18</b>", ProfBlue()),
             tcl("best Dogri model")],
        ], colWidths=[7.0*cm, 2.0*cm, 2.0*cm, W-11.0*cm],
        style=std_ts(left_cols=(0, 3)), repeatRows=1),
        sp(4),
        note("425-clip IndicVoices-R Dogri test split, scored on the same normalisation ladder as "
             "every other language in this report. The figures quoted are L2, the deciding level. "
             "For Dogri, L1 through L4 are identical — the diacritic-stripping and folding levels "
             "are no-ops for Devanagari — and only the L0 to L1 step (Unicode NFC, zero-width "
             "removal, whitespace collapse) moves the number at all, by 0.48 pp. Kashmiri's raw and "
             "diacritic-normalised scores differ by roughly 14 pp on the same ladder, which is the "
             "whole reason the ladder exists."),
        note("† The two Whisper rows are <b>L0</b>, not L2. An earlier revision of this table "
             "carried L0 figures under an L2 heading throughout — the __hin__ proxy is 99.99 at L0 "
             "but 99.86 at L2, and its CER 68.14 against 67.99 — which is exactly the ruler-mixing "
             "error §5.5.1 was written to warn about, committed by this report. The SeamlessM4T "
             "rows are now L2 and sourced from docs/doi_baselines.json. The two Whisper rows cannot "
             "be corrected the same way: scripts/eval/doi_baselines.py overwrote its own output "
             "file on a --systems run, so those two systems' hypotheses were lost and only the L0 "
             "summary survives. The script now merges instead of overwriting; re-running "
             "whisper_auto and whisper_hi will replace these two figures with L2 ones. For Dogri "
             "the two rungs differ by about 0.5 pp, so no conclusion here turns on it."),
        sp(6),
        body(
            "Fine-tuning improves Dogri by <b>55 pp over what the deployed system actually did</b> "
            "(102.25% -> 46.73%) and by 41 pp over the best baseline anyone could have configured "
            "without training. It is by a wide margin the largest single-language gain in the "
            "campaign, and it came from a language that had simply never been looked at."
        ),
        sp(6),
        body(
            "<b>The two Dogri runs also replicate the campaign's most reusable lesson.</b> doi_iv "
            "recorded its best validation loss at its final step, having improved at essentially "
            "every evaluation: it stopped because its linear learning-rate schedule had run to zero, "
            "not because it had converged. Restarting from those weights with a fresh schedule "
            "(doi_iv2) improved WER from 50.07% to <b>46.73%</b>. Kashmiri had shown exactly this "
            "before — ks_cloud stopped at 0.8 epochs in the same state and letting it converge was "
            "worth 3.84 pp. Two languages, two scripts, two corpora, the same mechanism and almost "
            "the same magnitude (3.84 pp and 3.34 pp), for a few dollars of GPU time each. A run "
            "whose schedule expires looks converged and is not."
        ),
        sp(6),
        body(
            "<b>The two zero-shot proxies also settle a design question.</b> Dogri is far closer to "
            "Punjabi than to Hindi genetically, which argues for initialising "
            "<font face='Courier'>__doi__</font> from <font face='Courier'>__pan__</font>. Whisper's "
            "own language identification agrees emphatically: given Dogri audio it answers "
            "<i>Punjabi</i> for <b>222 of 425 clips</b>, against 25 for Hindi. But SeamlessM4T knows "
            "Punjabi only in <b>Gurmukhi</b>, and the language token conditions generation, not "
            "recognition — so the pan proxy emits the wrong script entirely and its CER collapses to "
            "96.79 against hin's 67.99. Recognition says Punjabi; generation says Devanagari. The "
            "script-matched initialisation is the correct one, and this was established by "
            "measurement rather than argument, at no training cost."
        ),
        sp(4),
        note("This also illustrates the failure mode the report keeps returning to: a language absent "
             "from a model's inventory does not fail loudly, it gets silently misrouted to whatever "
             "the model considers nearest — and in Dogri's case that routing produced a WER above "
             "100%, which no accuracy dashboard would have flagged as a missing-language problem."),
        sp(10),

        h2("4.6 How Much of This Is Statistically Resolvable?"),
        body(
            "Every comparison in this report is a point estimate on a fixed test set, and the "
            "decisions taken on them ranged from 15 pp to under 1 pp. A paired bootstrap over clips "
            "(10,000 resamples, re-aggregating errors and reference length rather than averaging "
            "per-clip rates) was run retrospectively on the stored per-clip hypotheses to establish "
            "which of those decisions the evidence actually supports. It divides the campaign cleanly."
        ),
        sp(6),
        Table([
            [tch("Claim"), tch("Diff (pp)"), tch("95% CI"), tch("p"), tch("Holds?")],
            [tcl("SeamlessM4T replaces Whisper (ks, clean)"), tc("-14.93"), tc("[-17.3, -12.6]"),
             tc("<0.001"), Paragraph("<b>yes</b>", ProfBlue())],
            [tcl("...and in all 5 degradation conditions"), tc("-14 to -22"), tc("all exclude 0"),
             tc("<0.002"), Paragraph("<b>yes</b>", ProfBlue())],
            [tcl("Vocabulary repair (ks_cloud3 vs ks_cloud2)"), tc("-2.35"), tc("[-3.29, -1.38]"),
             tc("<0.001"), Paragraph("<b>yes</b>", ProfBlue())],
            [tcl("Training to convergence (Kashmiri)"), tc("-3.84"), tc("[-4.75, -2.93]"),
             tc("<0.001"), Paragraph("<b>yes</b>", ProfBlue())],
            [tcl("Training to convergence (Dogri)"), tc("-3.34"), tc("[-5.03, -2.07]"),
             tc("<0.001"), Paragraph("<b>yes</b>", ProfBlue())],
            [tcl("CV-dominated mixture is worse (ps_cv)"), tc("-5.56"), tc("[-10.8, -1.9]"),
             tc("<0.001"), Paragraph("<b>yes</b>", ProfBlue())],
            [tcl("ps_cloud better than ps_aug (clean)"), tc("-0.75"), tc("[-2.23, +0.70]"),
             tc("0.32"), tcl("no")],
            [tcl("ps_cloud better than ps_aug (sweep)"), tc("-2.1 to +1.2"), tc("all cross 0"),
             tc("0/5 sig."), tcl("no")],
            [tcl("ps_bal2 collapses at 0 dB"), tc("-31.2"), tc("[-101, +1.3]"),
             tc("0.27"), tcl("no")],
            [tcl("ks_cloud3 better than ks_cloud2 (sweep)"), tc("-2.5 to +1.7"), tc("all cross 0"),
             tc("0/5 sig."), tcl("no")],
            [tcl("ks_cloud4 worse than ks_cloud3"), tc("+0.43"), tc("[-0.29, +1.17]"),
             tc("0.24"), tcl("no")],
            [tcl("ps_aug2 regressed vs ps_aug"), tc("+0.54"), tc("[-1.05, +2.14]"),
             tc("0.48"), tcl("no")],
        ], colWidths=[7.4*cm, 2.0*cm, 3.0*cm, 1.8*cm, W-14.2*cm],
        style=std_ts(left_cols=(0, 4)), repeatRows=1),
        sp(6),
        body(
            "The split is not random: it follows test-set size and effect size together. The "
            "<b>backend-selection</b> conclusions — the ones this project was originally about — are "
            "supported everywhere, because replacing Whisper with SeamlessM4T moves WER by 10 to 22 pp "
            "and even 30 clips resolve that comfortably. So are the two mechanisms established on the "
            "372- and 425-clip Kashmiri and Dogri sets. What does <b>not</b> survive is the "
            "fine-grained adapter selection: the 1 to 3 pp differences between one adapter and its "
            "successor, decided on a 30-clip sweep and a 100-clip clean split, are indistinguishable "
            "from noise in every case tested."
        ),
        sp(6),
        body(
            "<b>This has a direct consequence for how the deployment decisions should be read.</b> "
            "ps_cloud is in production over ps_aug, and neither its 0.75 pp clean-WER advantage nor "
            "its 4/5 sweep result is statistically significant; the honest description is that the two "
            "adapters are indistinguishable on the available evidence and the newer one was chosen. "
            "The same applies to rejecting ks_cloud4 and ps_aug2: those were sound engineering calls "
            "under uncertainty — do not replace a deployed model without evidence of improvement — but "
            "they are not demonstrations that the alternatives were worse. Reported as win counts, "
            "'4/5 conditions' reads like evidence; decomposed, it is five coin flips."
        ),
        sp(4),
        note("The 30-clip sweep was inherited from the earliest robustness work and never revisited as "
             "the differences being tested shrank from tens of points to fractions of one. A gate "
             "adequate for choosing a backend is not automatically adequate for choosing between two "
             "checkpoints of the same model, and this project ran twenty adaptations on the "
             "assumption that it was. Reproduce with scripts/eval/significance.py, "
             "significance_ps.py and significance_degradation.py; all read the stored hypotheses and "
             "need no GPU."),
        sp(10),

        h2("4.7 Every Number Here Is an Evaluation Number, and the Deployment Differs"),
        body(
            "One further caveat applies to the whole report, and it is uncomfortable enough to state "
            "explicitly. Every WER in these tables was produced by the evaluation harness "
            "(<font face='Courier'>scripts/eval/</font>), which constructs the model with "
            "<font face='Courier'>PeftModel.from_pretrained</font> and one adapter. The <b>deployed</b> "
            "system constructs it differently: <font face='Courier'>src/seamless_asr.py</font> bakes "
            "the trainable-token embedding deltas into the base model and loads only the LoRA "
            "matrices, because PEFT cannot stack a trainable-token delta alongside several named "
            "adapters. Those two constructions were assumed equivalent for a year and never compared."
        ),
        sp(4),
        body(
            "They are not equivalent. Measured directly on the same 372 clips and the same ruler, "
            "Kashmiri scores <b>52.33%</b> through the deployed path against <b>50.26%</b> through the "
            "evaluation path — a <b>2.07 pp</b> gap that is comfortably significant (95% CI "
            "[+1.24, +2.91], p &lt; 0.001), with only 19 of 372 hypotheses identical between them. "
            "The deployed model is therefore measurably worse than the model this report describes, "
            "and because the divergence lives in the shared generation path rather than in anything "
            "Kashmiri-specific, the other seven languages are presumed affected and simply have not "
            "been measured both ways."
        ),
        sp(4),
        body(
            "Finding it also fixed a real defect: the deployment baked only the first of ks_cloud3's "
            "21 trainable-token deltas, silently leaving the twenty repaired characters at their "
            "neighbour-initialised values. That is corrected. The residual 2 pp is not. Direct "
            "comparison has since excluded the LoRA weights (480 of 480 tensors identical), the "
            "trainable-token deltas, the extracted input features, the generation length limit, the "
            "input dtype, substituting the checkpoint's frozen embedding table, and multi-adapter "
            "interference. The remaining difference has been narrowed to the construction of the "
            "forward pass and is unresolved; the next step is to diff logits on a single clip rather "
            "than continue eliminating components."
        ),
        sp(4),
        note("Reported here rather than quietly corrected because the alternative — republishing every "
             "number against the deployed path — would take a full re-evaluation of eight languages, "
             "and because the gap is itself the report's clearest instance of its recurring theme: an "
             "artefact measured is not necessarily the artefact shipped. The same lesson produced the "
             "74.02-versus-79.29 correction earlier in this project. Reproduce with "
             "scripts/eval/eval_ks_production_path.py."),
        sp(10),

    ]

    # ── 5. RESULTS ────────────────────────────────────────────────────────────
    story += [
        h1("5. Results"), hr(),
        h2("5.1 Summary Table — Fine-Tuning vs. True Baseline"),
        body(
            "All WER below is on a held-out 100-sample FLEURS test set (372-sample IndicVoices-R "
            "for Kashmiri), scored with one CJK-aware normaliser. <b>Baseline</b> is the true "
            "un-fine-tuned openai/whisper-large-v3. <b>Train WER</b> is the best training-eval WER "
            "on each run's own validation split (from the curves in §5.3) and is shown for reference "
            "only — it is not the held-out figure. Green = fine-tuning helped; red = it regressed."
        ),
        sp(4),
        Table([
            [tch("Language"), tch("ISO"), tch("Base Model"),
             tch("Train\nSamples"), tch("Steps\nUsed"),
             tch("Baseline\nWER"), tch("Train\nWER"), tch("Test\nWER"), tch("Improvement")],
            [tcl("Punjabi v3"),tc("pa"), tc("large-v3"),  tc("21,923"), tc("4000"), tc("77.60%"), tc("49.31%"), tc("57.39%"),
             Paragraph("<b>-20.2 pp</b>", ParagraphStyle("GRpa", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER))],
            [tcl("Pashto"),   tc("ps"), tc("medium*"),   tc("2,082"),  tc("1000"), tc("89.76%"),  tc("38.55%"), tc("38.55%"),
             Paragraph("<b>-51.2 pp</b>", ParagraphStyle("GRps", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER))],
            [tcl("Urdu"),     tc("ur"), tc("large-v3"),  tc("2,109"),  tc("1000"), tc("21.23%"),  tc("22.27%"), tc("19.82%"),
             Paragraph("<b>-1.4 pp</b>", ParagraphStyle("GRur", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER))],
            [tcl("Nepali v2"),tc("ne"), tc("large-v3"),  tc("13,332"), tc("3000"), tc("88.85%"),  tc("50.82%"), tc("50.92%"),
             Paragraph("<b>-37.9 pp</b>", ParagraphStyle("GRne", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER))],
            [tcl("Mandarin"), tc("zh"), tc("large-v3"),  tc("3,246"), tc("400"), tc("10.99%"),
             tc("8.97%"), tc("14.22%"),
             Paragraph("<b>+3.2 pp</b>", ParagraphStyle("REDzh", fontName="Helvetica-Bold",
                        fontSize=9, textColor=colors.HexColor("#C0392B"), alignment=TA_CENTER))],
            [tcl("Hindi"),    tc("hi"), tc("large-v3"),  tc("2,120"), tc("600"),  tc("26.34%"),  tc("23.13%"), tc("19.78%"),
             Paragraph("<b>-6.6 pp</b>", ParagraphStyle("GRhi", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER))],
            [tcl("Kashmiri"), tc("ks"), tc("large-v3†"),  tc("20,000"), tc("2400"),  tc("96.87%"),
             tc("74.02%"), tc("65.19% ‡"),
             Paragraph("<b>-22.85 pp ‡</b>", ParagraphStyle("GRks", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER))],
        ], colWidths=[2.2*cm, 1.0*cm, 2.0*cm, 1.7*cm, 1.4*cm, 1.8*cm, 1.6*cm, 1.6*cm, W-13.3*cm],
        style=std_ts(left_cols=(0,)), repeatRows=1),
        sp(4),
        note("* Pashto base: Nasimbahar/pashto-ghag-whisper-medium-asr (734 MB domain-specific model)."),
        note("† Kashmiri: custom <|ks|> token (ID 51866) added to whisper-large-v3; embedding initialised from <|ur|>. "
             "Best checkpoint step 2400 selected automatically (load_best_model_at_end). faster-whisper patched to accept language='ks'."),
        note("‡ This row previously read 74.02% in BOTH the Train and Test columns, and an earlier "
             "revision of this note described that figure as 'the held-out figure ... [and] the "
             "training-eval', which cannot both be true. 74.02% is the training-split validation WER "
             "at step 2400. The <b>held-out</b> figure, on the 372-clip IndicVoices-R test split at "
             "L2 (the ruler this report decides on), is <b>65.19%</b>; at L0 it is 79.29%. The "
             "-22.85 pp improvement is baseline-to-train (96.87 -> 74.02) and stays on that ruler. "
             "Note also that Kashmiri's baseline is the one exception to this section's "
             "'true un-fine-tuned large-v3' rule: 96.87% is <i>muneebharoon/whisper-small-ks</i>, a "
             "different and already-fine-tuned model, and is itself inflated by a Unicode "
             "normalisation mismatch. No un-fine-tuned Kashmiri baseline was ever measured on the "
             "held-out split — "
             "so there is no like-for-like held-out improvement to quote. Comparing 74.02 against a "
             "held-out number is precisely the mistake that made the first SeamlessM4T Kashmiri "
             "adapter look like a regression (§5.5.1)."),
        note("Mandarin (red): fine-tuning REGRESSED vs the true large-v3 baseline. The prior report's "
             "100.03% baseline and -84 pp 'improvement' were a scoring artefact — FLEURS Mandarin references "
             "are character-spaced and WER tokenises on whitespace, so the un-fine-tuned model's unspaced Han "
             "output scored ~100% regardless of accuracy. Corrected with character segmentation, the baseline "
             "is 10.99% and the fine-tune 14.22%. Mandarin is served by SeamlessM4T in production (§5.5)."),
        note("Train WER is each run's best validation-split WER (§5.3), not the held-out test; the two differ "
             "most for pa (train 49.31 vs test 57.39) and zh (train 8.97 vs test 14.22)."),
        note("pp = percentage points absolute WER change (baseline -> test WER); negative = improvement."),
        sp(10),
        h2("5.2 Baseline vs. Fine-Tuned WER"),
        KeepTogether([
            img_from_buf(chart_summary_bar(), width=W),
            caption("Figure 2: True large-v3 baseline vs. fine-tuned WER, both on the held-out "
                    "n=100 FLEURS test set. Numbers above bars show the signed WER change in "
                    "percentage points (negative = improvement; Mandarin's +3.2 is a regression). "
                    "Kashmiri is not shown: it is absent from FLEURS, so it is not on this ruler, "
                    "and its two available figures are training-split rather than held-out. Its "
                    "held-out comparison is in §5.5, on the 372-clip IndicVoices-R L2 ruler."),
        ]),
        sp(10),
        h2("5.3 WER Progression During Training"),
        KeepTogether([
            img_from_buf(chart_wer_all(), width=W),
            caption("Figure 3: Eval WER at each checkpoint. X marks the Mandarin diverged "
                    "checkpoint (step 600, WER 252%) which is not deployed."),
        ]),
        sp(10),
    ]

    # ── 5.4 PER-LANGUAGE SECTIONS ─────────────────────────────────────────────
    story.append(h2("5.4 Per-Language Training Details"))
    story.append(sp(6))

    for idx, lang in enumerate(LANG_ORDER):
        m = LANG_META[lang]
        lang_table = Table([
            [tch("Parameter"), tch("Value", left=True)],
            [tcl("Base model",      bold=True), tcl(m["base_model"])],
            [tcl("Dataset",         bold=True), tcl(f"{m['dataset']}  ({m['train_samples']} train / {m['val_samples']} val)")],
            [tcl("Steps trained",   bold=True), tcl(str(m["steps"]))],
            [tcl("Baseline WER (large-v3)", bold=True), tcl(f"{m['baseline_wer']:.2f}%")],
            [tcl("Best training-val WER",   bold=True), tcl(f"{m['best_wer']:.2f}%  (step {m['best_step']})")],
            [tcl("Held-out test WER",       bold=True), tcl(f"{m['eval_wer']:.2f}%")],
            # signed delta: negative = improvement. f"-{a-b}" breaks when a<b ("--3.2 pp").
            [tcl("WER change vs baseline",  bold=True),
             tcl(f"{m['eval_wer'] - m['baseline_wer']:+.1f} pp"
                 f"{'  (regression)' if m['eval_wer'] > m['baseline_wer'] else ''}")],
            [tcl("CT2 model",       bold=True), tcl(m["ct2_model"])],
            [tcl("Translation",     bold=True), tcl(m["translation"])],
            [tcl("Training time",   bold=True), tcl(m["training_time"])],
        ], colWidths=[4*cm, W - 4*cm],
        style=std_ts(left_cols=(0, 1)))
        story.append(KeepTogether([
            h3(f"5.4.{idx+1}  {m['name']} ({lang.upper()}) - {m['script']}"),
            lang_table,
        ]))
        story.append(sp(6))

        # WER table
        loss_dict = {}
        for ckpt_path in sorted(
            pathlib.Path(f"finetune_runs/{lang}/adapter").glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        ):
            ts = ckpt_path / "trainer_state.json"
            if ts.exists():
                data = json.loads(ts.read_text(encoding="utf-8"))
                for e in data.get("log_history", []):
                    if "eval_wer" in e:
                        loss_dict[e["step"]] = e.get("eval_loss", None)

        wer_rows = [[tch("Step"), tch("Eval WER"), tch("Eval Loss")]]
        for step, wer in m["wer_curve"]:
            is_best = (step == m["best_step"])
            wer_str = f"{wer:.2f}%  (best)" if is_best else f"{wer:.2f}%"
            loss_val = loss_dict.get(step, None)
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "-"
            if is_best:
                wer_rows.append([
                    tc(str(step)),
                    Paragraph(f"<b>{wer_str}</b>",
                              ParagraphStyle("BestWer", fontName="Helvetica-Bold",
                                             fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER)),
                    tc(loss_str)
                ])
            else:
                wer_rows.append([tc(str(step)), tc(wer_str), tc(loss_str)])

        if "wer_curve_diverged" in m:
            for s, w in m["wer_curve_diverged"]:
                wer_rows.append([
                    tc(str(s)),
                    Paragraph(f"{w:.2f}%  (diverged - not deployed)",
                              ParagraphStyle("DR", fontName="Helvetica-Oblique",
                                             fontSize=9, textColor=WARN_RED, alignment=TA_CENTER)),
                    tc("-")
                ])

        story.append(KeepTogether([
            Table(wer_rows, colWidths=[3*cm, 6*cm, 5*cm],
                  style=std_ts(), repeatRows=1),
            sp(4),
            note(f"Note: {m['note']}"),
        ]))
        story.append(sp(6))
        story.append(KeepTogether([
            img_from_buf(chart_wer_per_lang(lang), width=W),
            caption(
                f"Figure: {m['name']} training curves. "
                f"Left: eval WER (star = deployed checkpoint). Right: training loss per step."
            ),
        ]))
        story.append(sp(14))

    # ── 5.5 CROSS-MODEL EVALUATION ───────────────────────────────────────────────
    story += [
        PageBreak(),
        h2("5.5 Cross-Model Evaluation and Backend Selection"),
        body(
            "This is the decisive evaluation. Re-run at n=100 on 11 July 2026 with corrected scoring "
            "(true openai/whisper-large-v3 baseline — not turbo — and one CJK-aware normaliser), it "
            "compares three ASR options per language: (A) the un-fine-tuned large-v3 baseline, "
            "(B) the language-specific fine-tuned Whisper (CT2 int8), and (C) zero-shot SeamlessM4T v2. "
            "FLEURS test split for pa/ps/ur/ne/zh/hi; IndicVoices-R for ks. ASR cells give "
            "<b>WER% (CER%)</b> — CER is reported alongside WER because two of the seven scripts "
            "(Han, Perso-Arabic Kashmiri) have orthographies where whitespace tokenisation misleads. "
            "The <b>Deployed Backend</b> column is the operational choice — the lowest-WER option "
            "VANI actually routes to."
        ),
        sp(6),
        img_scaled(report_charts.hero_backend_dumbbell(), width=W * 0.92),
        note("Figure 4: Deployed SeamlessM4T backend vs the fine-tuned Whisper model, per language "
             "(n=100 FLEURS held-out, same scorer). Every arrow points to the deployed SeamlessM4T backend "
             "— lower WER. Kashmiri is shown separately (§5.5.2): a different corpus and scoring ruler."),
        sp(8),
        Table([
            [tch("Language"), tch("Baseline (large-v3)\nWER (CER)"), tch("FT Whisper\nWER (CER)"),
             tch("SeamlessM4T\nWER (CER)"),
             tch("Deployed\nBackend"), tch("NLLB\nchrF"), tch("SM S2TT\nchrF")],
            [tcl("Punjabi (pa)"), tc("77.60 (39.73)"), tc("57.39 (32.52)"), tc("19.77 (9.97)"),
             Paragraph("<b>SeamlessM4T</b>", ParagraphStyle("Bpa", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("40.15"), tc("54.53")],
            [tcl("Pashto (ps)"), tc("89.76 (37.60)"),
             Paragraph("<b>38.55 (17.65)</b>", ParagraphStyle("FTps", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER)),
             tc("44.40 (22.92)"),
             Paragraph("<b>SM4T + LoRA</b>", ParagraphStyle("Bps", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("44.48"), tc("40.15")],
            [tcl("Urdu (ur)"), tc("21.23 (8.12)"), tc("19.82 (7.29)"), tc("16.90 (7.00)"),
             Paragraph("<b>SeamlessM4T</b>", ParagraphStyle("Bur", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("51.34"), tc("50.73")],
            [tcl("Nepali (ne)"), tc("88.85 (29.26)"), tc("50.92 (18.83)"), tc("28.46 (11.22)"),
             Paragraph("<b>SM4T + LoRA</b>", ParagraphStyle("Bne", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("45.55"), tc("51.67")],
            [tcl("Mandarin (zh)¶"), tc("10.99 (10.99)"),
             Paragraph("14.22 (14.22)†", ParagraphStyle("FTzh", fontName="Helvetica",
                        fontSize=9, textColor=colors.HexColor('#C0392B'), alignment=TA_CENTER)),
             tc("11.69 (11.69)"),
             Paragraph("<b>SeamlessM4T</b>", ParagraphStyle("Bzh", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("42.00"), tc("49.15")],
            [tcl("Hindi (hi)"), tc("26.34 (10.55)"), tc("19.78 (7.46)"), tc("15.44 (9.12)"),
             Paragraph("<b>SM4T + LoRA</b>", ParagraphStyle("Bhi", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("53.71"), tc("51.54")],
            [tcl("Kashmiri (ks)‡"), tc("— (—)"),
             tc("65.19 (39.36)"),
             Paragraph("<b>50.26 (23.34)</b>", ParagraphStyle("FTks", fontName="Helvetica-Bold",
                        fontSize=9, textColor=SUCCESS_GRN, alignment=TA_CENTER)),
             Paragraph("<b>SM4T + LoRA</b>", ParagraphStyle("Bks", fontName="Helvetica-Bold",
                        fontSize=9, textColor=HDR_BLUE, alignment=TA_CENTER)), tc("—"), tc("—")],
        ], colWidths=[2.5*cm, 2.9*cm, 2.7*cm, 2.7*cm, 2.2*cm, 1.4*cm, W-14.4*cm],
        style=std_ts(left_cols=(0,)), repeatRows=1),
        sp(4),
        note("¶ Mandarin is scored with character segmentation, so WER and CER coincide by "
             "construction — CER is the meaningful metric for Han script. Do not compare a "
             "character-level number against a word-level one: CER runs ~2–3× lower than WER "
             "on the space-delimited languages, so comparisons are valid only within a metric."),
        note("Kashmiri CER was not recorded by the training-time eval (hence the dash). On the "
             "30-clip robustness set (clean condition, eval_data/wer_robustness_results.csv) the "
             "then-deployed Whisper ks model scored 81.46% WER / 47.95% CER — the wide gap reflects "
             "Perso-Arabic orthographic variation that WER over-penalises, which is exactly what later "
             "motivated re-scoring the SeamlessM4T challenger under normalisation rather than trusting "
             "raw WER alone (below)."),
        body(
            "<b>Result (updated 2026-07-20): fine-tuned Whisper is no longer the deployed backend for "
            "any of the seven languages.</b> For Punjabi, Nepali, Hindi, Urdu and Mandarin, zero-shot "
            "SeamlessM4T won outright from the start, by margins from 1.4 pp (Urdu) to 37.6 pp "
            "(Punjabi). Pashto and Kashmiri were fine-tuned Whisper's last strongholds — Pashto fell "
            "on 2026-07-19 to a noise-augmented SeamlessM4T adapter, and Kashmiri, initially thought "
            "unwinnable, fell the following day once the scoring ruler itself was corrected (below). "
            "VANI now routes all seven languages to SeamlessM4T; the retired Whisper models stay on "
            "disk for rollback."
        ),
        sp(6),
        body(
            "<b>SeamlessM4T LoRA adapters (deployed 2026-07-18 to 27).</b> Hindi, Nepali, Pashto and "
            "Kashmiri each run a per-language LoRA adapter. Hindi and Nepali were trained on FLEURS + "
            "IndicVoices-R (cap 20k samples): Hindi 15.44% -> <b>12.91%</b> and Nepali 28.46% -> "
            "<b>24.34%</b> versus zero-shot on the same held-out test. Pashto required five attempts: "
            "more data alone regressed (CV domain drift), a larger adapter (r=32 incl. MLP layers) won "
            "clean speech (37.29% vs Whisper's 38.55%) but recorded 87.2% at 0 dB SNR against 64.8% - a dramatic point estimate that the 30-clip sweep cannot actually resolve (p = 0.27, see 4.6) - and "
            "the fifth, <b>noise-augmented</b> adapter — training audio degraded with the evaluation's "
            "own bandpass/noise/codec pipeline — reached <b>36.91%</b> clean while fixing the collapse "
            "(56.0% at 0 dB, beating Whisper's 64.8%). A sixth attempt that scaled the Common Voice "
            "data 10k -> 30k gave 37.46%, a difference from ps_aug that is not statistically "
            "significant (p = 0.48). A seventh, retrained at LoRA rank 128 on a rented cloud GPU "
            "(ps_cloud, ~$2 of compute), reached <b>36.16%</b> clean and won 4/5 degradation "
            "conditions against ps_aug including 0 dB (53.8%), and is the adapter deployed on "
            "2026-07-27. <b>Its 0.75 pp clean-WER advantage over ps_aug is not statistically "
            "significant either</b> (95% CI [-2.23, +0.70], p = 0.32 by paired bootstrap over the "
            "100 test clips); the deployment rests on the degradation sweep rather than on clean "
            "speech. Every adapter passed that 30-clip, 5-condition sweep before deployment, and each "
            "is enabled only for its own language's decoding calls — other languages run the plain "
            "base model."
        ),
        sp(6),
        body(
            "<b>That last qualification generalises, and it is worth stating plainly.</b> Pashto's "
            "held-out FLEURS split has 100 clips, and a paired bootstrap shows it cannot resolve the "
            "differences this campaign was chasing: ps_cloud vs ps_aug (0.75 pp, p = 0.32), ps_aug2 vs "
            "ps_aug (0.54 pp, p = 0.48) and ps_cloud vs ps_bal2 (1.13 pp, p = 0.12) are all "
            "indistinguishable from noise, while the one large difference — the CV-dominated ps_cv at "
            "5.56 pp — is significant at p &lt; 0.001. Kashmiri and Dogri, evaluated on 372 and 425 "
            "clips, resolved everything claimed of them: the vocabulary repair (2.35 pp, p &lt; 0.001) "
            "and training to convergence (3.84 pp and 3.34 pp, both p &lt; 0.001). The practical "
            "lesson is that a 100-clip test set supports claims of roughly 5 pp and above, and that "
            "sub-2 pp improvements reported on sets this size — a common practice — should be treated "
            "as unresolved unless an interval accompanies them."
        ),
        sp(6),
        body(
            "<b>Kashmiri (deployed 2026-07-20) needed a fourth attempt and a scoring correction, not "
            "just a bigger adapter.</b> Three prior attempts (1 epoch: 129.29% WER; +decode fixes: "
            "92.09%; r=16 LoRA, 3 epochs: 88.42%) all lost badly to Whisper's 74.02%. A fourth attempt "
            "stacked the two remaining untried levers — r=32 LoRA incl. MLP layers, and, for the first "
            "time, a <b>trainable __kas__ embedding row</b> (every earlier attempt had frozen it as a "
            "copy of Urdu's, via PEFT's <font face='Courier'>trainable_token_indices</font>) — and "
            "reached 80.91% clean, still nominally behind. But re-scoring the <i>actually deployed</i> "
            "Whisper CT2 artefact on the same 372 clips found it scores 79.29%, not the published "
            "74.02% (a training-time eval of the merged fp16 model, whose raw hypotheses were never "
            "persisted) — narrowing the true gap to 1.6 pp. Stripping the Perso-Arabic diacritics that "
            "saturate the references (which both models drop, so raw WER penalises both per-word) "
            "flips the result: the new adapter already <b>wins WER</b> (64.31% vs 65.19%) and wins "
            "<b>CER at every normalisation level on both test sets</b>. It then won the radio-"
            "degradation sweep 4 of 5 conditions (losing only clean speech, by 1.3 pp) and CER 5 of 5, "
            "with Whisper's CER exceeding 100% at 0 dB SNR. Kashmiri's lesson generalises beyond this "
            "one language: a model-selection conclusion is only as good as the ruler behind it, and "
            "this project has now found five independent scoring defects (turbo mislabelling, CJK "
            "whitespace, a label-encoding bug, an adapter-override bug in the sweep harness, and this "
            "training-eval-vs-deployed-artefact mismatch) — each one initially read as a real result."
        ),
        sp(4),
        note("† Mandarin: fine-tuning regressed the model (baseline 10.99% -> fine-tuned 14.22%). The prior "
             "report's 100.03% baseline / 100.0% SeamlessM4T were whitespace-tokenisation artefacts on "
             "character-spaced Han references, now corrected with character segmentation."),
        note("‡ Kashmiri is NOT on this table's ruler. Every other row is FLEURS n=100; Kashmiri is "
             "absent from FLEURS entirely, so its two figures are the <b>372-clip IndicVoices-R test "
             "split at L2</b> — fine-tuned Whisper-ks 65.19% (CER 39.36%) against the deployed "
             "ks_cloud3 SeamlessM4T adapter 50.26% (CER 23.34%), a paired-bootstrap gap of -14.93 pp, "
             "95% CI [-17.26, -12.62], p < 0.001 (§4.6). The row previously showed 96.87% / 74.02%, "
             "which are <b>training-split validation</b> figures sitting in a table of held-out ones; "
             "no un-fine-tuned Kashmiri baseline was ever measured on the held-out split, hence the "
             "dash in the baseline column. Read the row down, not across to its neighbours."),
        note("§ SeamlessM4T v2 has no native Kashmiri (kas absent from the model vocabulary); a Urdu-token "
             "proxy failed (109% WER). The eventual fix was a custom trainable __kas__ token + LoRA (§5.5.2) — "
             "Kashmiri now runs on the ks_cloud3 SeamlessM4T adapter, not on fine-tuned Whisper."),
        note("chrF: character F-score for end-to-end English translation (higher = better). SeamlessM4T's "
             "S2TT wins for pa/ne/zh; Whisper+NLLB wins for ps/ur/hi. VANI keeps NLLB downstream regardless, "
             "using only SeamlessM4T's ASR output."),
        sp(12),
        h3("5.5.1  ASR WER under Radio-Channel Degradation"),
        body(
            "Because VANI processes degraded radio rather than clean speech, the backend choice must "
            "survive channel effects. The table gives SeamlessM4T's WER advantage over fine-tuned "
            "Whisper (FT WER - SeamlessM4T WER, in points; positive = SeamlessM4T better) on the same "
            "30 FLEURS clips per language under five conditions: clean, 300–3400 Hz telephony bandpass, "
            "additive white noise at 10 dB and 0 dB SNR, and a 16 kbit/s MP3 codec pass."
        ),
        sp(6),
        img_scaled(report_charts.robustness_heatmap(), width=W * 0.78),
        note("Figure 5: SeamlessM4T's WER advantage over fine-tuned Whisper (percentage points) across the "
             "five channel conditions. Positive everywhere; Hindi (+5.0->+19.5) and Mandarin (+3.6->+17.2) "
             "widen most at 0 dB SNR. This is the conservative zero-shot comparison — the deployed hi/ne LoRA "
             "adapters improve on it further."),
        sp(8),
        Table([
            [tch("Language"), tch("Clean"), tch("Bandpass"), tch("Noise\n10 dB"), tch("Noise\n0 dB"),
             tch("MP3\ncodec"), tch("Winner")],
            [tcl("Punjabi (pa)"),  tc("+39.0"), tc("+37.7"), tc("+41.7"), tc("+36.8"), tc("+41.8"),
             Paragraph("<b>SeamlessM4T</b>", ProfBlue()) ],
            [tcl("Nepali (ne)"),   tc("+31.6"), tc("+30.5"), tc("+34.1"), tc("+36.4"), tc("+32.0"),
             Paragraph("<b>SeamlessM4T</b>", ProfBlue()) ],
            [tcl("Hindi (hi)"),    tc("+5.0"),  tc("+3.9"),  tc("+10.8"), tc("+19.5"), tc("+5.6"),
             Paragraph("<b>SeamlessM4T</b>", ProfBlue()) ],
            [tcl("Urdu (ur)"),     tc("+2.3"),  tc("+2.4"),  tc("+4.8"),  tc("+4.2"),  tc("+3.1"),
             Paragraph("<b>SeamlessM4T</b>", ProfBlue()) ],
            [tcl("Mandarin (zh)"), tc("+3.6"),  tc("+5.0"),  tc("+2.9"),  tc("+17.2"), tc("+3.0"),
             Paragraph("<b>SeamlessM4T</b>", ProfBlue()) ],
            [tcl("Pashto (ps)"),   tc("-6.7"),  tc("-2.1"),  tc("-1.9"),  tc("+5.6"),  tc("-2.8"),
             Paragraph("<b>FT Whisper*</b>", ProfGrn()) ],
        ], colWidths=[2.6*cm, 1.7*cm, 1.9*cm, 1.7*cm, 1.6*cm, 1.6*cm, W-11.1*cm],
        style=std_ts(left_cols=(0,)), repeatRows=1),
        sp(4),
        body(
            "SeamlessM4T's advantage is positive in every condition for all five languages above and "
            "<b>widens as the channel worsens</b> — Hindi grows from +5.0 (clean) to +19.5 (0 dB), "
            "Mandarin from +3.6 to +17.2. The clean-speech ranking does not invert under noise, so the "
            "routing is safe for operational radio. This table is the ZERO-SHOT comparison; Pashto loses "
            "it 4/5 (winning only at 0 dB). That was fine-tuned Whisper's last stronghold — but a later "
            "<b>noise-augmented SeamlessM4T adapter</b> (ps_aug: training audio degraded with the "
            "evaluation's own bandpass/noise/codec pipeline) overturned it, winning 4/5 conditions and "
            "clean speech (36.91% vs 38.55%), and its r=128 cloud retrain (ps_cloud, 36.16% clean, "
            "53.8% at 0 dB) extended the margin. Pashto now runs on SeamlessM4T."
        ),
        note("* This table's ps 'winner' column reflects the historical zero-shot comparison. Since "
             "2026-07-19 Pashto has been deployed on a noise-augmented SeamlessM4T adapter — ps_cloud "
             "(r=128) as of 2026-07-27 — so fine-tuned Whisper is retained for rollback only."),
        sp(10),

        # ── 5.5.2 Kashmiri ruler correction ──────────────────────────────────
        h3("5.5.2  Kashmiri — the WER Gap Was the Scoring Ruler"),
        body(
            "Kashmiri has no native SeamlessM4T vocabulary, so a custom <b>__kas__</b> token was added and "
            "made trainable (a first for this project — PEFT trainable_token_indices), on an r=32 LoRA with "
            "MLP layers. On raw WER the adapter looks behind the deployed Whisper CT2 model (80.91% vs "
            "79.29% — itself higher than the often-quoted 74.02%, which was a training-time eval of the "
            "merged fp16 model, not the deployed int8 artefact). But Perso-Arabic references are densely "
            "diacritised and BOTH systems drop the marks, so raw WER over-penalises both symmetrically. "
            "Once the diacritics are normalised, the verdict flips: the SeamlessM4T adapter wins WER "
            "(64.31% vs 65.19%) and wins CER at every normalisation level. It also won the radio-degradation "
            "sweep 4/5 conditions, with Whisper's CER exceeding 100% at 0 dB SNR. Three successive retrains "
            "then compounded the win on the identical 372-clip ruler: <b>ks_max2</b> (2026-07-26) kept the "
            "recipe but rebuilt the corpus 4x larger (97k clips / 240 h: humair025 IndicVoices + "
            "IndicVoices-R + OpenSLR-122) and reached <b>61.88%</b>; then <b>ks_cloud</b> (2026-07-27) raised "
            "LoRA rank 32 to 128 — beyond the 8 GB laptop's VRAM, so trained on a rented A6000 for ~$4 — "
            "on a 145k-clip / 335 h rebuild of the same corpus and reached <b>56.44%</b> (CER 26.19%), "
            "winning the degradation sweep 5/5 against Whisper with every condition individually significant (4.6), 0 dB at 81.3% against 99.5%. Its sweep advantage over ks_cloud2, by contrast, is within noise. "
            "Finally <b>ks_cloud2</b> re-ran that exact recipe with early-stopping patience raised from 3 "
            "to 5: ks_cloud had halted at 0.8 epochs while its validation loss was still falling, so a "
            "second epoch alone reached <b>52.60%</b> (CER 23.67%) — winning WER and CER at every "
            "normalisation level, boundary-free CER 25.69% vs 28.13%, and the degradation sweep 5/5 "
            "against Whisper and 4/5 against ks_cloud itself. On the identical 372-clip L2 ruler, "
            "Kashmiri has fallen <b>65.19 (Whisper) -> 64.31 -> 61.88 -> 56.44 -> 52.60 -> 50.26</b> "
            "across the campaign. Kashmiri now runs on the <b>ks_cloud3</b> adapter; ks_cloud2, "
            "ks_cloud and fine-tuned Whisper are retained for rollback."
        ),
        sp(6),
        body(
            "<b>A hard floor remains, and it is a vocabulary defect rather than an acoustic one.</b> An "
            "error analysis of the deployed model found that 77% of its remaining errors are word "
            "substitutions, with a hypothesis-to-reference length ratio of 0.943 and no truncated "
            "outputs — so the earlier under-generation pathology is gone and little is recoverable by "
            "decoding changes. The dominant cause is that <b>20 characters used in written Kashmiri have "
            "no token in SeamlessM4T's sentencepiece vocabulary</b> and resolve to the unknown-token id: "
            "854,234 occurrences across the training corpus, present in 96.9% of its sentences. The "
            "model therefore trained against corrupted targets and cannot emit those characters at all — "
            "U+0672 appears 370 times in the test references and zero times in any hypothesis. Measured "
            "on that model, 662 of 3,917 substitutions (16.9%) fall on words that are unrepresentable. "
            "That the fully-converged ks_cloud2 settled at 52.60% is consistent with a model pressing "
            "against precisely such a ceiling."
        ),
        sp(6),
        body(
            "<b>Repairing the vocabulary confirmed the diagnosis.</b> ks_cloud3 added all 20 characters "
            "as real tokens, each embedding initialised from its closest in-vocabulary neighbour and then "
            "trained — the same technique that made Kashmiri work at all, generalised from one language "
            "token to twenty characters. Nothing else changed: same corpus, rank, patience and converged "
            "step, so the comparison isolates a single variable. The result is unambiguous on the "
            "measure that matters most. <b>Of the 747 test words containing the four missing letters, "
            "the previous model got 747 wrong — every single one; ks_cloud3 gets 310 of them right.</b> "
            "Overall WER falls 52.60% -> <b>50.26%</b>, and raw WER falls 74.31% -> <b>64.71%</b>, a 9.60 pp "
            "gain: most of the missing characters are combining marks that the diacritic-normalised ruler "
            "discards anyway, so the repair shows its full value only on unnormalised text. Length "
            "calibration also improved (hypothesis/reference ratio 0.943 -> 0.980) and deletions fell from "
            "16.5% to 13.9% of errors."
        ),
        sp(6),
        body(
            "One honest qualification: the 49% floor quoted above was an <i>upper bound</i>. It assumed "
            "every unrepresentable word would become correct once writable, whereas about a third actually "
            "did. The 437 words still wrong represent roughly 4.9 pp of the remaining error, but their "
            "failures are now plausible acoustic confusions between similar words rather than blank "
            "omissions."
        ),
        sp(6),
        body(
            "<b>That residue is not a training-duration problem, and testing the obvious remedy showed "
            "why it matters to check.</b> The twenty new embedding rows had trained for only 1.06 epochs, "
            "making them by far the least-converged parameters in the model, so a further run "
            "(<b>ks_cloud4</b>) warm-started from ks_cloud3 with a fresh learning-rate schedule and those "
            "rows driven at five times the adapter's rate. It improved <i>immediately</i> — validation "
            "loss 0.8305 -> 0.8257 within 200 steps — and then worsened at every subsequent evaluation, "
            "early-stopping at step 1,200 because the model was already converged. On the deciding ruler "
            "it <b>lost</b>: 50.69% WER against ks_cloud3's 50.26%, with CER and boundary-free CER also "
            "marginally worse. Better validation loss, worse word error: the fifth loss/WER divergence "
            "this project has recorded, and the first in the unfavourable direction. Selecting on the "
            "loss curve would have selected it. On the deciding ruler the 0.43 pp gap is itself not "
            "statistically significant (95% CI [-0.29, +1.17], p = 0.24), so the honest reading is "
            "that a measurably better validation loss bought no measurable change in word error at "
            "all. ks_cloud3 remains in production."
        ),
        sp(6),
        body(
            "<b>Decoding was the last untested lever, and it does not survive the robustness "
            "requirement either.</b> VANI decodes greedily: <font face='Courier'>num_beams</font> is "
            "absent from SeamlessM4T's generation configuration, so the library default of 1 applies to "
            "all seven languages. Beam-8 decoding lowers Kashmiri's clean WER from 50.26% to 47.37%, and "
            "adding an in-domain Kneser-Ney trigram language model to rescore the 8-best list reaches "
            "46.90% (against an oracle ceiling of 44.33% over that list, so the language model captures "
            "roughly a sixth of what re-ranking could ever recover). Latency is not the obstacle — beam-8 "
            "costs about 5% of wall-clock, not the eightfold one might expect, because decode time is "
            "dominated by a sequential step count that the minimum-token constraint fixes. The obstacle "
            "is robustness: across eight configurations (beam widths 2, 4 and 8, each with length "
            "penalties 1.0, 0.8 and 0.6) <i>every one</i> regressed the 0 dB SNR condition by 2–3 pp "
            "while gaining 2–3 pp on clean audio. With flat posteriors, beam search reliably prefers "
            "fluent but acoustically unsupported hypotheses; greedy decoding's myopia is a virtue "
            "precisely when the evidence is weakest. For a system whose purpose is degraded radio, that "
            "trade runs the wrong way, so production decoding is unchanged. The wider lesson is that "
            "<b>decode settings validated only on clean test sets can silently degrade deployed "
            "robustness</b>."
        ),
        sp(6),
        body(
            "Taken together — capacity, corpus size, convergence, vocabulary coverage and decoding have "
            "each now been tested and, where they helped, deployed — the remaining error is best "
            "characterised as a data and acoustic-modelling limit rather than an optimisation one. That "
            "is the empirical basis on which any future move to a different pretrained backbone should "
            "be argued."
        ),
        sp(6),
        img_scaled(report_charts.ks_ruler_bars(), width=W * 0.78),
        note("Figure 6: Same 372 IndicVoices clips, same scorer. Raw WER once made Whisper look ahead of "
             "the first adapter (ks_max) — an artefact of the Perso-Arabic scoring ruler. With the ruler "
             "corrected, every successive adapter widens the win; the deployed r=128 ks_cloud3 leads on "
             "all three measures."),
        sp(6),
        note("This is the fifth independent scoring-methodology correction in this project (after the "
             "turbo-baseline mislabel, the CJK whitespace artefact, the S2TT label-encoding bug, and the "
             "sweep-harness adapter-override bug) — each initially read as a real model result."),
        sp(10),
    ]

    # ── 5.6 ROBUSTNESS EVALUATION ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h2("5.6 Radio-Channel Robustness Evaluation — LangID Accuracy"),
        body(
            "The VANI LangID pipeline was evaluated under five radio-channel degradations "
            "applied to FLEURS test audio (30 samples/language for pa/hi/ur/ne/zh/ps; "
            "IndicVoices-R test audio for ks). "
            "Four pipeline configurations are compared: "
            "(C1) Whisper language detection alone, "
            "(C2) Whisper + FastText, "
            "(C3) Whisper + FastText + MMS-LID-256, and "
            "(C4) Full VANI (C3 + dialect detection + script-based routing). "
            "Note: Kashmiri audio was tested with the standard whisper-large-v3-turbo model "
            "(not the fine-tuned KS model); the low ks accuracy reflects that "
            "turbo has no <|ks|> token, so only MMS-LID-256 provides the KS signal."
        ),
        sp(4),
        Table([
            [tch("Condition"), tch("Config."),
             tch("PA"), tch("HI"), tch("UR"), tch("NE"),
             tch("ZH"), tch("PS"), tch("KS"), tch("Ovrl")],
            # clean
            [tcl("Clean"), tcl("Whisper"), tc("90.0%"), tc("86.7%"), tc("73.3%"), tc("26.7%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("53.8%")],
            [tcl(""),      tcl("+FastText"),tc("90.0%"), tc("86.7%"), tc("73.3%"), tc("26.7%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("53.8%")],
            [tcl(""),      tcl("+MMS-LID"), tc("90.0%"), tc("83.3%"), tc("70.0%"), tc("30.0%"), tc("100.0%"), tc("53.3%"), tc("16.7%"), tc("63.3%")],
            [tcl(""),      tcl("Full VANI"),tc("90.0%"), tc("83.3%"), tc("70.0%"), tc("33.3%"), tc("100.0%"), tc("80.0%"), tc("16.7%"), tc("67.6%")],
            # bandpass
            [tcl("Bandpass"),tcl("Whisper"), tc("70.0%"), tc("83.3%"), tc("33.3%"), tc("26.7%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("44.8%")],
            [tcl(""),      tcl("+FastText"), tc("70.0%"), tc("83.3%"), tc("33.3%"), tc("26.7%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("44.8%")],
            [tcl(""),      tcl("+MMS-LID"),  tc("73.3%"), tc("83.3%"), tc("33.3%"), tc("26.7%"), tc("100.0%"), tc("70.0%"), tc("23.3%"), tc("58.6%")],
            [tcl(""),      tcl("Full VANI"), tc("73.3%"), tc("83.3%"), tc("33.3%"), tc("26.7%"), tc("100.0%"), tc("83.3%"), tc("23.3%"), tc("60.5%")],
            # awgn_10
            [tcl("AWGN 10dB"),tcl("Whisper"),tc("96.7%"), tc("63.3%"), tc("80.0%"), tc("43.3%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("54.8%")],
            [tcl(""),       tcl("+FastText"), tc("96.7%"), tc("63.3%"), tc("80.0%"), tc("43.3%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("54.8%")],
            [tcl(""),       tcl("+MMS-LID"),  tc("96.7%"), tc("63.3%"), tc("80.0%"), tc("43.3%"), tc("100.0%"), tc("56.7%"), tc("3.3%"),  tc("63.3%")],
            [tcl(""),       tcl("Full VANI"), tc("96.7%"), tc("63.3%"), tc("80.0%"), tc("43.3%"), tc("100.0%"), tc("83.3%"), tc("3.3%"),  tc("67.1%")],
            # awgn_0
            [tcl("AWGN 0dB"),tcl("Whisper"),  tc("56.7%"), tc("63.3%"), tc("26.7%"), tc("13.3%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("37.1%")],
            [tcl(""),       tcl("+FastText"),  tc("56.7%"), tc("63.3%"), tc("26.7%"), tc("13.3%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("37.1%")],
            [tcl(""),       tcl("+MMS-LID"),   tc("56.7%"), tc("66.7%"), tc("26.7%"), tc("16.7%"), tc("100.0%"), tc("40.0%"), tc("0.0%"),  tc("43.8%")],
            [tcl(""),       tcl("Full VANI"),  tc("60.0%"), tc("66.7%"), tc("26.7%"), tc("16.7%"), tc("96.7%"),  tc("53.3%"), tc("0.0%"),  tc("45.7%")],
            # codec_mp3
            [tcl("MP3 16kbps"),tcl("Whisper"), tc("76.7%"), tc("46.7%"), tc("63.3%"), tc("23.3%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("44.3%")],
            [tcl(""),        tcl("+FastText"),  tc("76.7%"), tc("46.7%"), tc("63.3%"), tc("23.3%"), tc("100.0%"), tc("0.0%"),  tc("0.0%"),  tc("44.3%")],
            [tcl(""),        tcl("+MMS-LID"),   tc("76.7%"), tc("46.7%"), tc("63.3%"), tc("23.3%"), tc("100.0%"), tc("63.3%"), tc("26.7%"), tc("57.1%")],
            [tcl(""),        tcl("Full VANI"),  tc("76.7%"), tc("46.7%"), tc("63.3%"), tc("23.3%"), tc("96.7%"),  tc("86.7%"), tc("20.0%"), tc("59.0%")],
        ], colWidths=[2.0*cm, 1.9*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.4*cm, 1.3*cm, 1.3*cm, W-12.9*cm],
        style=TableStyle(std_ts(left_cols=(0, 1)).getCommands() + [
            # thicker separator lines between condition groups (rows 5, 9, 13, 17)
            ("LINEABOVE", (0, 5),  (-1, 5),  1.2, colors.HexColor("#888888")),
            ("LINEABOVE", (0, 9),  (-1, 9),  1.2, colors.HexColor("#888888")),
            ("LINEABOVE", (0, 13), (-1, 13), 1.2, colors.HexColor("#888888")),
            ("LINEABOVE", (0, 17), (-1, 17), 1.2, colors.HexColor("#888888")),
            # reduce vertical padding for compact 20-row table
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]), repeatRows=1),
        sp(4),
        note("Full VANI (C4) consistently outperforms Whisper-only (C1) by +14 pp on average. "
             "MMS-LID-256 is the critical component for Pashto detection (turbo Whisper scores 0% on ps). "
             "Mandarin (zh) is the most robust language: 97–100% across all conditions. "
             "Kashmiri (ks) identification requires the ks-specific CT2 model for meaningful accuracy."),
        sp(10),
        PageBreak(),
        h2("5.7 Punjabi v3 Training Progress (LoRA r=16)"),
        body(
            "A third Punjabi training run (v3) was launched on 01 July 2026 with doubled LoRA capacity "
            "(r=16, α=32) and 21,923 training samples (IV-R 20,000 + FLEURS 2,516). "
            "The same 805-sample eval set is used for fair comparison with v2. "
            "v3 completed the full 4,000 steps, reaching its best WER of 49.31% at step 4000 — "
            "a 3.24 pp improvement over v2's final 52.55%. The best checkpoint was merged to CT2 int8 and "
            "DEPLOYED on 04 July 2026, replacing v2. An initial run OOM-crashed at step 2400 during beam-search "
            "evaluation on the 8 GB RTX 5060; it was resumed from checkpoint-2200 with greedy evaluation "
            "(num_beams=1, eval batch 1, CUDA cache cleared before each eval), which cut eval VRAM from 8 GB to "
            "~3.8 GB and ran clean to step 4000."
        ),
        sp(4),
        Table([
            [tch("Config Parameter"), tch("v2 (superseded)"), tch("v3 (deployed)")],
            [tcl("LoRA Rank (r)",     bold=True), tc("8"),          tc("16")],
            [tcl("LoRA Alpha (α)",    bold=True), tc("16"),         tc("32")],
            [tcl("Trainable Params",  bold=True), tc("~3.9M"),      tc("~7.9M")],
            [tcl("IndicVoices-R",     bold=True), tc("9,407"),      tc("20,000")],
            [tcl("Total Train Set",   bold=True), tc("11,923"),     tc("21,923")],
            [tcl("Total Steps",       bold=True), tc("3,000"),      tc("4,000")],
            [tcl("max_grad_norm",     bold=True), tc("1.0"),        tc("0.5")],
            [tcl("warmup_steps",      bold=True), tc("50"),         tc("100")],
            [tcl("Best / Deployed WER", bold=True), tc("52.55%"),   tc("49.31%")],
        ], colWidths=[4.5*cm, 3.5*cm, W-8.0*cm], style=std_ts(left_cols=(0,))),
        sp(8),
        body("Step-by-step eval WER and loss for PA v3 (greedy eval from step 2400 after OOM restart):"),
        sp(4),
        Table([
            [tch("Step"), tch("v3 Eval WER"), tch("v3 Eval Loss"), tch("Observation")],
            [tc("1800"), tc("52.06%"), tc("0.1918"), tcl("New best; loss new low")],
            [tc("2000"), tc("52.75%"), tc("0.1892"), tcl("Minor oscillation; loss still ↓")],
            [tc("2200"), tc("50.62%"), tc("0.1839"), tcl("New best (interim deploy after OOM)")],
            [tc("2400"), tc("51.25%"), tc("0.1831"), tcl("Greedy eval resumes; oscillation up")],
            [tc("2600"), tc("51.25%"), tc("0.1761"), tcl("WER plateau, loss new low")],
            [tc("2800"), tc("50.51%"), tc("0.1751"), tcl("New best; plateau breaks")],
            [tc("3000"), tc("50.05%"), tc("0.1725"), tcl("New best; approaching 50%")],
            [tc("3200"), tc("50.65%"), tc("0.1711"), tcl("Oscillation up, loss new low")],
            [tc("3400"), tc("49.40%"), tc("0.1694"), tcl("First sub-50% checkpoint")],
            [tc("3600"), tc("49.97%"), tc("0.1675"), tcl("Loss new low")],
            [tc("3800"), tc("49.49%"), tc("0.1667"), tcl("Loss new low")],
            [tc("4000"), tc("49.31%★"),tc("0.1662"), tcl("Overall best — DEPLOYED")],
        ], colWidths=[1.4*cm, 2.4*cm, 2.4*cm, W-6.2*cm],
        style=std_ts(left_cols=(3,)), repeatRows=1),
        sp(4),
        note("★ v3 best: 49.31% at step 4000 (eval loss 0.1662) — DEPLOYED. v2 best: 52.55% at step 3000. "
             "v3 final result is -3.24 pp ahead of v2."),
        note("Key pattern: WER oscillates mid-training (regression at steps 600, 1400, 2000) while eval loss "
             "decreases monotonically. Loss is a more reliable indicator of learning progress than WER at individual checkpoints."),
        note("OOM recovery: the initial run crashed at step 2400 (CUDA out-of-memory during beam-search eval on 8 GB "
             "RTX 5060). Resumed from checkpoint-2200 with greedy eval (num_beams=1, eval batch 1, cache-clear before eval), "
             "cutting eval VRAM from 8 GB to ~3.8 GB; the run then completed cleanly to step 4000 (final best 49.31%)."),
        sp(10),
    ]

    # ── 6. NOTABLE EVENTS ─────────────────────────────────────────────────────
    story += [
        h1("6. Notable Training Events and Engineering Decisions"), hr(),
        h2("6.1 Mandarin Gradient Explosion (fp16 Overflow)"),
        body(
            "At step ~820, the gradient norm spiked to 12.9 (from a stable range of 0.5-1.3). "
            "Training loss jumped from 0.15 to 0.77 and eval WER degraded catastrophically "
            "from 8.97% to 252.4%. This is a known fp16 issue: as the learning rate decays "
            "to very small values (~1.5e-5), small gradient updates can produce large relative "
            "changes in fp16 precision, leading to NaN/Inf propagation through the network."
        ),
        body(
            "Resolution: Training was stopped after observing the diverged step-600 evaluation. "
            "Checkpoint-400 (WER 8.97%) was manually merged using PeftModel.merge_and_unload() "
            "and converted to CT2. For all subsequent languages (Hindi onward), "
            "max_grad_norm=0.5 was added to TrainingArguments to prevent recurrence."
        ),
        sp(6),
        h2("6.2 HuggingFace 504 Timeout (Mandarin Dataset)"),
        body(
            "The first Mandarin training attempt failed with HTTP 504 Gateway Timeout when "
            "downloading the cmn_hans_cn FLEURS train split. Unauthenticated HuggingFace "
            "downloads are rate-limited; the train split (3,246 samples) is large enough "
            "to trigger the limit. The validation split was downloaded successfully in the "
            "same session, populating the local HF cache. A script restart immediately "
            "loaded all splits from cache."
        ),
        sp(6),
        h2("6.3 Python Output Buffering in PowerShell"),
        body(
            "Training output was invisible in the PowerShell terminal when using Tee-Object "
            "pipes, because Python stdout is line-buffered by default when piped. "
            "Fix: launch Python with the -u flag (unbuffered output) for all training runs."
        ),
        code_block("python -u finetune_whisper.py hi --no-cv --steps 600 2>&1 | Tee-Object logs/finetune_hi.log"),
        sp(6),
        h2("6.4 Tokenizer Loading from Checkpoint Directories"),
        body(
            "When merging the Mandarin LoRA adapter from checkpoint-400, loading "
            "WhisperProcessor.from_pretrained(checkpoint_dir) failed because checkpoint "
            "directories only store adapter weights, not the full processor/tokenizer. "
            "Fix: load the processor from the base model instead:"
        ),
        code_block(
            "# WRONG - checkpoint dirs don't have a full tokenizer\n"
            "# proc = WhisperProcessor.from_pretrained('finetune_runs/zh/adapter/checkpoint-400')\n\n"
            "# CORRECT - always load processor from the original base model\n"
            "proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')"
        ),
        sp(6),
        h2("6.5 CT2 Tokenizer Bug — All Large-v3 Models Translated Instead of Transcribed"),
        body(
            "After all six large-v3 models were converted to CTranslate2 format, every model "
            "produced English output regardless of the language setting — causing ~100% WER "
            "against source-language references. Root cause: ct2-transformers-converter does NOT "
            "copy tokenizer.json to the output directory. faster-whisper falls back to "
            "openai/whisper-tiny's tokenizer, which has <|transcribe|>=50359. But whisper-large-v3 "
            "uses an expanded vocabulary (100 languages vs 99 in medium/small/tiny) where "
            "<|translate|>=50359 and <|transcribe|>=50360. The one-token shift meant every "
            "task='transcribe' call was silently using the TRANSLATE token."
        ),
        body(
            "Fix: copy tokenizer.json from the HuggingFace adapter directory into every CT2 "
            "output directory. The finetune_whisper.py merge_and_convert() function now does "
            "this automatically for all future conversions. Urdu WER dropped from 100.66% "
            "to 19.52% immediately after applying the fix. Do NOT use the turbo model's "
            "tokenizer.json — it also has transcribe=50359 and would replicate the bug."
        ),
        code_block(
            "# Fix applied to all large-v3 CT2 directories:\n"
            "Copy-Item finetune_runs/ur/adapter/tokenizer.json models/whisper-large-v3-<lang>-ct2/\n\n"
            "# Now automated in finetune_whisper.py merge_and_convert():\n"
            "shutil.copy2(merged_dir / 'tokenizer.json', ct2_dir / 'tokenizer.json')"
        ),
        sp(6),
        h2("6.6 HF datasets.map() Cache Writes to Source Location, Not HF_DATASETS_CACHE"),
        body(
            "During PA v3 training setup, the FLEURS + IndicVoices-R dataset preprocessing via "
            "datasets.map() raised OSError: No space left on device on D: despite "
            "HF_DATASETS_CACHE being set to C:. Root cause: datasets.map() ignores "
            "HF_DATASETS_CACHE and instead writes the Arrow feature cache next to the source "
            "parquet files — which were on D: via a junction. D: had only 28 KB free."
        ),
        body(
            "Fix: pass an explicit cache_file_name= argument to map() redirecting Arrow files to C:. "
            "C:/hf_ds_map_cache/ now stores pa_train_features.arrow (21,923 samples, ~33 GB) and "
            "pa_eval_features.arrow (805 samples). Training resumes correctly from this cache on restart."
        ),
        code_block(
            "# WRONG — map() ignores HF_DATASETS_CACHE:\n"
            "# os.environ['HF_DATASETS_CACHE'] = 'C:/hf_cache'\n"
            "# train_ds = raw['train'].map(**proc_kwargs)  # still writes to D:\n\n"
            "# CORRECT — explicit redirect:\n"
            "_map_cache = Path('C:/hf_ds_map_cache')\n"
            "_map_cache.mkdir(exist_ok=True)\n"
            "train_ds = raw['train'].map(\n"
            "    **proc_kwargs,\n"
            "    cache_file_name=str(_map_cache / f'{lang}_train_features.arrow'),\n"
            ")"
        ),
        sp(6),
        h2("6.7 Checkpoint Save Failure — D: Junction Full During Optimizer State Save"),
        body(
            "PA v3 training stopped at step 800 with RuntimeError: [enforce fail at "
            "inline_container.cc:672] unexpected pos 24411648 vs 24411540. "
            "Root cause: torch.save(optimizer.state_dict()) was writing optimizer.pt (~60 MB) "
            "to D:/finetune_runs/pa/adapter/checkpoint-800/ when D: filled to 0 bytes. "
            "The partial file rendered the checkpoint directory corrupt."
        ),
        body(
            "Resolution: Moved ne/ (96 GB), ks/ (7.7 GB), ps/ (4.9 GB) training checkpoints "
            "from D: to C:/finetune_runs_moved/. Deleted the corrupt checkpoint-800 directory. "
            "Resumed training from checkpoint-600 using --resume flag. "
            "The trainer re-ran the step 800 eval on resume, recovering the missing data point "
            "(step 800 WER: 57.15%)."
        ),
        sp(6),
        h2("6.8 WER Oscillation vs. Monotonic Loss Decrease — Key Training Observation"),
        body(
            "Across all languages, eval WER oscillates at individual checkpoints while training loss "
            "decreases monotonically. The most clear example is PA v3:"
        ),
        bullet("Steps 1200->1400: WER regresses 54.48% -> 55.68% (+1.2 pp) while loss drops 0.2101 -> 0.2100"),
        bullet("Steps 1400->1600: WER recovers 55.68% -> 52.99% (-2.7 pp) — new 2nd best"),
        bullet("Steps 1800->2000: WER oscillates 52.06% -> 52.75% while loss reaches new low 0.1892"),
        body(
            "The same pattern appeared in PA v2 (regression at step 1000, recovery by step 1600) "
            "and NE v2 (regression at steps 1600 and 2400). "
            "Conclusion: load_best_model_at_end=True correctly handles oscillation by tracking the "
            "checkpoint with the lowest eval WER across all checkpoints, not just the final one. "
            "Training loss is a more reliable indicator of continued learning than per-checkpoint WER. "
            "A declining loss with oscillating WER should NOT trigger early stopping."
        ),
        sp(6),
        h2("6.9 Eval Time Bottleneck — 2.5 Hours per Checkpoint Evaluation"),
        body(
            "For PA v3 (805 eval samples, per_device_eval_batch_size=1, predict_with_generate=True), "
            "each evaluation takes ~2.5 hours at ~10-12 seconds per sample on RTX 5060. "
            "With eval_steps=200 and 4,000 total steps, this means 20 evaluations × 2.5 h = ~50 hours "
            "of evaluation overhead alone, on top of ~22 hours of training time. "
            "Total PA v3 training wall-clock time: ~72 hours."
        ),
        body(
            "For smaller languages (PS, UR, HI, ZH with 239-409 val samples), eval takes 30-60 minutes. "
            "For KS (372 samples) eval took ~1 hour. Batching eval (batch_size=4) would reduce "
            "overhead 4×, but risks OOM on 8 GB VRAM with large-v3 encoder + decoder in fp16. "
            "Recommendation for future runs: eval_steps=400 for PA/NE to halve eval overhead."
        ),
        sp(6),
        h2("6.10 preprocessor_config.json Required for CT2 Models"),
        body(
            "The CT2 converter does not automatically write preprocessor_config.json. "
            "Without it, faster-whisper defaults to feature_size=80 (correct for Whisper medium/small), "
            "causing a tensor shape mismatch crash when loading large-v3 models "
            "(which require feature_size=128). Must be manually written to every CT2 output:"
        ),
        code_block(
            '{\n'
            '  "feature_extractor_type": "WhisperFeatureExtractor",\n'
            '  "feature_size": 128,\n'
            '  "sampling_rate": 16000\n'
            '}'
        ),
        sp(10),
    ]

    # ── 7. SCRIPTS ────────────────────────────────────────────────────────────
    story += [
        h1("7. Project Scripts Reference"), hr(),
        h2("7.1 Core Scripts (Project Root)"),
        Table([
            [tch("Script"), tch("Purpose", left=True)],
            [tcl("finetune_whisper.py",        bold=True),
             tcl("Main LoRA fine-tuning script. Supports all 6 languages via LANG_CONFIG dict. "
                 "Handles dataset loading, LoRA init, training, and CT2 conversion. "
                 "Usage: python -u finetune_whisper.py <lang> --no-cv --steps N")],
            [tcl("app.py",                     bold=True),
             tcl("VANI Streamlit web UI. Entry point for the full 10-stage pipeline. "
                 "Run: streamlit run app.py  (opens at http://localhost:8501)")],
            [tcl("run_full_pipeline_batch.py", bold=True),
             tcl("Batch processor for directories of WAV files. "
                 "Outputs JSON results and SQLite entries without the Streamlit UI.")],
            [tcl("config.yaml",                bold=True),
             tcl("Central configuration: model paths, ASR settings, VAD config, "
                 "language routing rules, ISUM settings, and database path.")],
        ], colWidths=[4.5*cm, W - 4.5*cm],
        style=std_ts(left_cols=(0, 1))),
        sp(8), h2("7.2 Evaluation Scripts (scripts/eval/)"),
        Table([
            [tch("Script"), tch("Purpose", left=True)],
            [tcl("eval_fleurs.py",                  bold=True),
             tcl("Evaluates a CT2 Whisper model on FLEURS validation split. "
                 "Reports WER, CER, and inference speed.")],
            [tcl("ablation_eval.py",                bold=True),
             tcl("Ablation study: compares pipeline configurations (VAD on/off, "
                 "noise reduction on/off, etc.)")],
            [tcl("robustness_eval.py",              bold=True),
             tcl("Tests model robustness to additive noise, codec distortion, and SNR variation.")],
            [tcl("compute_bleu.py",                 bold=True),
             tcl("Computes BLEU scores for end-to-end translation quality (ASR + NLLB + English).")],
            [tcl("test_arabic_rule.py",             bold=True),
             tcl("Unit test for the Arabic-script cascade detection rule (Stage 5 of pipeline).")],
        ], colWidths=[4.5*cm, W - 4.5*cm],
        style=std_ts(left_cols=(0, 1))),
        sp(8), h2("7.3 Download / Utility Scripts (scripts/utils/)"),
        Table([
            [tch("Script"), tch("Purpose", left=True)],
            [tcl("download_models.py",      bold=True),
             tcl("Downloads base models (NLLB-200, MMS-LID, Qwen) from HuggingFace Hub.")],
            [tcl("download_lang_models.py", bold=True),
             tcl("Downloads language-specific models (e.g., Pashto whisper-medium).")],
            [tcl("download_fleurs.py",      bold=True),
             tcl("Pre-downloads FLEURS datasets to local HF cache before running training.")],
        ], colWidths=[4.5*cm, W - 4.5*cm],
        style=std_ts(left_cols=(0, 1))),
        sp(10),
    ]

    # ── 8. FILE STRUCTURE ─────────────────────────────────────────────────────
    story += [
        h1("8. Project File Structure"), hr(),
        code_block(
            "offline_ai_system_v2/\n"
            "|\n"
            "+-- app.py                          Streamlit UI entry point\n"
            "+-- finetune_whisper.py             LoRA fine-tuning (all languages)\n"
            "+-- run_full_pipeline_batch.py      Batch pipeline runner\n"
            "+-- config.yaml                     All configuration\n"
            "+-- FINETUNE_REPORT.md              Markdown report\n"
            "|\n"
            "+-- models/                         Deployed CT2 models (int8 quantized)\n"
            "|   +-- whisper-large-v3-pa-ct2/    Punjabi   WER 49.31% (v3, 21,923 samples)\n"
            "|   +-- whisper-medium-pashto-ct2/  Pashto    WER 38.9%\n"
            "|   +-- whisper-large-v3-ur-ct2/    Urdu      WER 22.3%\n"
            "|   +-- whisper-large-v3-ne-ct2/    Nepali    WER 50.82% (v2, 13,332 samples)\n"
            "|   +-- whisper-large-v3-zh-ct2/    Mandarin  WER 8.97%\n"
            "|   +-- whisper-large-v3-hi-ct2/    Hindi     WER 23.1%\n"
            "|   +-- nllb-200-distilled-600M/    NLLB translation model\n"
            "|   +-- mms-lid-256/                Language identification model\n"
            "|\n"
            "+-- finetune_runs/                  LoRA training checkpoints\n"
            "|   +-- pa/adapter/checkpoint-{200,400,...,3000}/  (v2: 15 checkpoints)\n"
            "|   +-- ps/adapter/checkpoint-{200,400,600,800,1000}/\n"
            "|   +-- ur/adapter/checkpoint-{200,400,600,800,1000}/\n"
            "|   +-- ne/adapter/checkpoint-{200,400,...,3000}/  (v2: 15 checkpoints)\n"
            "|   +-- zh/adapter/checkpoint-{200,400,600}/  (ckpt-400 deployed)\n"
            "|   +-- hi/adapter/checkpoint-{200,400,600}/\n"
            "|\n"
            "+-- scripts/\n"
            "|   +-- eval/    eval_fleurs.py, ablation_eval.py, robustness_eval.py ...\n"
            "|   +-- paper/   generate_ijainn.py, build_presentation.py ...\n"
            "|   +-- utils/   download_models.py, download_fleurs.py ...\n"
            "|\n"
            "+-- src/                            Core pipeline modules\n"
            "|   +-- pipeline.py  asr_module.py  language_module.py\n"
            "|   +-- translation_module.py  vad_module.py  preprocessing.py\n"
            "|   +-- keyword_module.py  isum_module.py  database.py\n"
            "|\n"
            "+-- logs/\n"
            "|   +-- finetune_hi.log  finetune_ne.log  finetune_zh.log\n"
            "|   +-- finetune_pa.log  finetune_ps.log  finetune_ur.log\n"
            "|   +-- eval_wer.log\n"
            "|\n"
            "+-- input_audio/   Drop WAV files here for batch processing\n"
            "+-- output/        Pipeline JSON outputs\n"
            "+-- database/      transcripts.db (SQLite)"
        ),
        sp(10),
    ]

    # ── 9. CONCLUSIONS ────────────────────────────────────────────────────────
    story += [
        h1("9. Conclusions and Key Findings"), hr(),
        body("Six language-specific Whisper ASR models were successfully fine-tuned and "
             "deployed in the VANI pipeline. Key findings:"),
        sp(6),
        bullet(
            "<b>LoRA r=8 is highly effective for speech domain adaptation.</b>  "
            "Training only 0.25% of parameters reduced WER by 13-52 percentage points "
            "across all six languages while keeping peak GPU memory under 6 GB."
        ),
        bullet(
            "<b>FLEURS bridges the domain gap despite clean read-speech vs. noisy radio audio.</b>  "
            "Models trained on studio-quality FLEURS significantly outperform the untuned "
            "baseline on conversational radio intercepts, suggesting language-specific phoneme "
            "modelling generalises across recording conditions."
        ),
        bullet(
            "<b>Whisper large-v3 has strong prior Mandarin capability.</b>  "
            "Mandarin baseline WER was already ~20% vs ~74% for Indic languages. "
            "Fine-tuning further reduced it to 8.97% - the best absolute result."
        ),
        bullet(
            "<b>fp16 gradient instability is a real risk at late training stages.</b>  "
            "Mandarin diverged at step ~820 due to fp16 overflow. "
            "Setting max_grad_norm=0.5 (vs default 1.0) resolved this for Hindi "
            "and should be standard for future runs."
        ),
        bullet(
            "<b>Hindi and Urdu converge to nearly identical WER (23.1% and 22.3%).</b>  "
            "Despite using different scripts (Devanagari vs. Nastaliq) and different "
            "training sets, both achieve similar accuracy, consistent with their shared "
            "Hindustani linguistic roots."
        ),
        bullet(
            "<b>IndicVoices-R augmentation provided consistent WER improvements for PA and NE.</b>  "
            "Adding AI4Bharat IndicVoices-R data to Punjabi v2 (9,407 new samples) and Nepali v2 "
            "(10,000 new samples) improved best WER from 56.67% to 52.55% (PA, -4.1 pp) and "
            "from 52.14% to 50.82% (NE, -1.3 pp) on harder combined eval sets. "
            "Training loss continued declining through all 3,000 steps for both languages, "
            "suggesting further gains with extended training or higher LoRA rank."
        ),
        bullet(
            "<b>Nepali WER improvement from IndicVoices-R was modest; architectural changes needed for large gains.</b>  "
            "v1 WER plateaued at 52.14% after step 2,000 (FLEURS-only). v2 reached 50.82% at step 3,000. "
            "Achieving sub-30% WER would likely require LoRA rank upgrade (r=32, alpha=64) with broader "
            "target modules (q,k,v,out_proj,fc1,fc2), or a Nepali-pretrained base model."
        ),
        bullet(
            "<b>CT2 models require tokenizer.json from the source model.</b>  "
            "ct2-transformers-converter does not copy tokenizer.json. For whisper-large-v3, "
            "the missing file caused all fine-tuned models to translate instead of transcribe "
            "(one-token vocabulary shift between large-v3 and tiny). "
            "The fix is now automated in finetune_whisper.py."
        ),
        bullet(
            "<b>LoRA r=16 (PA v3) consistently outperforms r=8 (PA v2) by 1–3 pp at matching steps.</b>  "
            "At step 1800, v3 achieves 52.06% vs v2's 54.65% at the same step (-2.59 pp). "
            "v3 completed 4,000 steps at a final best of 49.31% — 3.24 pp below v2's 52.55%, "
            "suggesting doubled LoRA rank meaningfully increases model capacity for larger datasets."
        ),
        bullet(
            "<b>Eval WER oscillates mid-training; eval loss is the reliable learning signal.</b>  "
            "Every language exhibits mid-training WER regression followed by recovery. "
            "Eval loss decreases monotonically throughout. "
            "load_best_model_at_end=True correctly selects the best checkpoint despite oscillation. "
            "WER regression alone is not a valid reason to halt training."
        ),
        bullet(
            "<b>datasets.map() ignores HF_DATASETS_CACHE — explicit cache_file_name= required.</b>  "
            "map() writes Arrow feature caches next to source parquet files, not to the HF cache dir. "
            "On junction-backed storage this caused an OSError at ~21,923 samples. "
            "Now fixed with explicit cache_file_name= in finetune_whisper.py."
        ),
        bullet(
            "<b>SeamlessM4T v2 now serves ASR for all 7 of 7 languages.</b>  "
            "On the n=100 held-out test, zero-shot SeamlessM4T beats fine-tuned Whisper for "
            "pa/ne/hi/ur/zh (§5.5), and the lead holds under radio degradation (§5.5.1). The last two "
            "strongholds fell to SeamlessM4T LoRA adapters: Pashto to noise-augmented training "
            "(ps_aug 36.91%, then ps_cloud 36.16%), and Kashmiri to a custom trainable __kas__ token "
            "plus a scoring-ruler correction (ks_max 64.31% -> ks_cloud3 50.26% diacritic-normalised). "
            "The earlier claim that Whisper won Mandarin was a whitespace-scoring artefact; corrected, "
            "fine-tuning regressed Mandarin (14.22% vs 10.99% baseline). Every fine-tuned Whisper "
            "model is retained on disk for rollback only. One shared SeamlessM4T (~4.6 GB resident) "
            "plus per-language LoRA adapters has a smaller footprint than seven fine-tuned Whispers."
        ),
        bullet(
            "<b>Rented cloud capacity was the campaign's cheapest large lever.</b>  "
            "The 8 GB laptop caps SeamlessM4T LoRA at r=32 (1.75% trainable). One day on a rented "
            "RTX A6000 (~$6 total) retrained both open languages at r=128 (6.6% trainable): Kashmiri "
            "improved 61.88% -> 50.26% and Pashto 36.91% -> 36.16%, both passing their degradation "
            "gates and deploying. The reverse lever — more data at unchanged capacity — regressed "
            "(ps_aug2, +0.55 pp): for these low-resource languages the binding constraint was "
            "adapter capacity, not corpus size."
        ),
        bullet(
            "<b>Report intervals, or a 100-clip test set will flatter you.</b>  "
            "A paired bootstrap over the per-clip hypotheses shows that Pashto's 100-clip FLEURS "
            "split cannot resolve any of the sub-2 pp differences this campaign pursued — including "
            "the 0.75 pp margin on which ps_cloud was deployed over ps_aug (p = 0.32). The larger "
            "claims survive comfortably: the Kashmiri vocabulary repair (2.35 pp), and training to "
            "convergence on two languages (3.84 pp and 3.34 pp), are all significant at p &lt; 0.001 "
            "on 372-425 clips. Point estimates alone would have supported three claims this project "
            "cannot actually make."
        ),
        bullet(
            "<b>Training-signal caution: eval_loss and WER diverge in both directions.</b>  "
            "This project logged five loss/WER divergences. ps_cloud's eval_loss (1.134) was clearly "
            "worse than ps_aug2's (1.048), yet ps_cloud wins WER by 1.3 pp and chrF by 7 points. The "
            "fifth ran the other way and was caught only at the gate: ks_cloud4 improved validation "
            "loss over the deployed ks_cloud3 (0.8257 vs 0.8305) while <i>losing</i> 0.43 pp of WER, so "
            "selecting on the loss curve would have regressed production. Model selection must gate on "
            "task metrics (WER/CER ladders + degradation sweeps), never on validation loss alone."
        ),
        bullet(
            "<b>The custom-token technique generalises — it is a method, not a Kashmiri fix.</b>  "
            "Adding a language token the model does not ship, initialised from a neighbour and made "
            "trainable, was invented for Kashmiri and has now been applied unchanged to <b>Dogri</b>, "
            "a different script (Devanagari vs Perso-Arabic) with a different failure profile (a "
            "clean vocabulary rather than 20 unrepresentable characters). Dogri improved from "
            "<b>102.25% to 46.73% WER</b> — 55 pp over what the deployed system actually did, the "
            "largest single-language gain of the campaign, in a language nobody had measured."
        ),
        bullet(
            "<b>A missing language fails silently, not loudly.</b>  "
            "With no Dogri token in either backend, VANI routed Dogri audio to whatever Whisper "
            "considered nearest: its language identification answered <i>Punjabi</i> for 222 of 425 "
            "clips, producing Gurmukhi output against Devanagari references and a WER above 100%. "
            "Nothing in the system reported an error. The corollary for the initialisation choice is "
            "subtle: Dogri really is closest to Punjabi (Whisper's own LID says so), but the language "
            "token conditions <i>generation</i>, so the script-matched Hindi initialisation wins "
            "(zero-shot 99.86% vs 114.62%, CER 67.99 vs 96.79). Recognition and generation want "
            "different neighbours."
        ),
        bullet(
            "<b>Decode defaults are an invisible system-level variable.</b>  "
            "SeamlessM4T ships no <font face='Courier'>num_beams</font>, so all seven languages decode "
            "greedily — unnoticed through an entire seven-language campaign because every evaluation "
            "inherited the same default. Beam-8 is worth 2.9 pp of clean Kashmiri WER at only ~5% "
            "wall-clock, but regresses 0 dB SNR by 3.2 pp, and no beam width or length penalty avoids "
            "that trade. Decode settings validated only on clean speech can quietly cost robustness "
            "where an operational system needs it most."
        ),
        sp(10),
    ]

    # ── 10. REFERENCES ────────────────────────────────────────────────────────
    story += [
        h1("10. References"), hr(),
        bullet("Radford et al. (2022). <i>Robust Speech Recognition via Large-Scale Weak Supervision.</i> "
               "OpenAI. arXiv:2212.04356."),
        bullet("Hu et al. (2022). <i>LoRA: Low-Rank Adaptation of Large Language Models.</i> "
               "ICLR 2022. arXiv:2106.09685."),
        bullet("Conneau et al. (2022). <i>FLEURS: Few-shot Learning Evaluation of Universal "
               "Representations of Speech.</i> SLT 2022. arXiv:2205.12446."),
        bullet("Costa-jussa et al. (2022). <i>No Language Left Behind: Scaling Human-Centered "
               "Machine Translation.</i> Meta AI. arXiv:2207.04672."),
        bullet("Pratap et al. (2023). <i>Scaling Speech Technology to 1,000+ Languages.</i> "
               "Facebook AI Research (MMS). arXiv:2305.13516."),
        sp(10), hr(),
        caption(
            f"Report generated: {date.today().strftime('%d %B %Y')}  -  "
            "VANI v2  -  RTX 5060 8 GB  -  IIT Indore M.Tech Research Project"
        ),
    ]

    # ── BUILD ─────────────────────────────────────────────────────────────────
    print("Building PDF...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Done -> {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    build()

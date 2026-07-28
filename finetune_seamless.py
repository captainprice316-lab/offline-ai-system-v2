"""
finetune_seamless.py — LoRA fine-tuning for SeamlessM4T v2 large on FLEURS
===========================================================================
Fine-tunes SeamlessM4T v2 for ASR in one target language using LoRA (PEFT).
Eval metric is eval_loss (cross-entropy); WER is measured post-training via
the existing compare_all_models.py --lang <lang> --skip-baseline run.

Usage:
    python finetune_seamless.py pa              # Punjabi, 1000 steps
    python finetune_seamless.py ur --steps 500
    python finetune_seamless.py ne --resume
    python finetune_seamless.py all             # all 6 languages sequentially
"""

import argparse
import io
import os
import pathlib
import sys

import numpy as np
import torch

ROOT         = pathlib.Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
SEAMLESS_DIR = pathlib.Path(os.environ.get(
    "VANI_SEAMLESS_DIR", str(MODELS_DIR / "seamless-m4t-v2-large")))
# ^ cloud runs (ks_cloud) set VANI_SEAMLESS_DIR=facebook/seamless-m4t-v2-large to
#   pull the base from HF; local default is unchanged.
RUNS_DIR     = ROOT / "finetune_runs_seamless"

LANG_CFG = {
    "pa": {"fleurs": "pa_in",       "sm_lang": "pan", "name": "Punjabi"},
    "ps": {"fleurs": "ps_af",       "sm_lang": "pbt", "name": "Pashto"},
    "ur": {"fleurs": "ur_pk",       "sm_lang": "urd", "name": "Urdu"},
    "ne": {"fleurs": "ne_np",       "sm_lang": "npi", "name": "Nepali"},
    "zh": {"fleurs": "cmn_hans_cn", "sm_lang": "cmn", "name": "Mandarin"},
    "hi": {"fleurs": "hi_in",       "sm_lang": "hin", "name": "Hindi"},
    "ks": {
        # SeamlessM4T has no Kashmiri: __kas__ is ADDED to the tokenizer at train
        # time, embedding initialised from __urd__ (same Nastaliq script) — the
        # same surgery that took Whisper ks from ~97% to 74.02% WER. Zero-shot
        # urd-proxy was tested and failed (109% WER), but that says nothing about
        # a fine-tune: Seamless's encoder is very strong on Indic audio (zero-shot
        # pa 19.77 vs Whisper-FT 57.39) and ks is the one language with headroom.
        "sm_lang": "kas", "name": "Kashmiri",
        "indicvoices_parquet_dir": r"E:\VANI\datasets\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri",
        "train_cap": 20000,
    },
    "ps_cv": {
        # Experimental: Pashto with Common Voice 20 added (~50k clips, upvote-
        # filtered) on top of FLEURS ps_af (2,082). The fixed-label FLEURS-only
        # adapter scored 41.30 vs deployed Whisper-medium 38.55 — this probes
        # whether 24x more data closes the 2.75 pp gap. Own run dir; the ps/
        # adapter and its results are untouched. Val stays FLEURS-only so model
        # selection aims at the held-out FLEURS ruler.
        "fleurs": "ps_af", "sm_lang": "pbt", "name": "Pashto (FLEURS + CV-20)",
        "cv_dataset": "SherwinDesouza/pashto-common-voice-20",
    },
    "ks_r16": {
        # Kashmiri attempt #3 (after 1-epoch r=8 129.29 greedy and 3-epoch r=8
        # 92.09 with decode fixes, both losing to Whisper-ks 74.02). First
        # capacity increase for ks: r=16 a=32 on q/k/v/out_proj — SM4T never saw
        # Kashmiri, so encoder adaptation should matter more here than anywhere.
        # Cap raised to all available IndicVoices (24.7k total). Fresh run, not
        # a resume (different adapter shape). If this moves WER but falls short,
        # the next surgical lever is a trainable __kas__ embedding row (frozen
        # urd-init under LoRA — the suspected conditioning bottleneck).
        "sm_lang": "kas", "name": "Kashmiri (r16, full data)",
        "indicvoices_parquet_dir": r"E:\VANI\datasets\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri",
        "train_cap": 24000,
        "lora_r": 16, "lora_alpha": 32,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
    },
    "ps_bal": {
        # Pashto attempt #3 (after FLEURS-only 41.30 and CV-dominated 42.47 both
        # lost to Whisper-medium 38.55). Fixes ps_cv's failure mode: FLEURS is
        # oversampled 8x (~20k effective) against a 10k CV cap, keeping ~2/3 of
        # batches on the eval domain. Also the first run with more adapter
        # capacity: r=16 α=32 on q/k/v/out_proj — Pashto is thin in SM4T's
        # pretraining, so encoder adaptation is the suspected binding constraint.
        "fleurs": "ps_af", "sm_lang": "pbt", "name": "Pashto (bal. FLEURSx8 + CV10k, r16)",
        "cv_dataset": "SherwinDesouza/pashto-common-voice-20",
        "cv_cap": 10000, "fleurs_repeat": 8,
        "lora_r": 16, "lora_alpha": 32,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
    },
    "ps_bal2": {
        # Pashto attempt #4 (after ps_bal: 39.72 greedy / 38.88 best-decode vs
        # Whisper-medium 38.55 — lost by 0.33). Same balanced data recipe as
        # ps_bal (proven; keeps attribution clean); the only change is the next
        # rung of the capacity ladder, which is the one lever that has reliably
        # paid (r8→r16 bought 1.6 pp): r=32 α=64 and LoRA extended to the MLP
        # (fc1/fc2) alongside q/k/v/out_proj.
        "fleurs": "ps_af", "sm_lang": "pbt", "name": "Pashto (bal, r32+MLP)",
        "cv_dataset": "SherwinDesouza/pashto-common-voice-20",
        "cv_cap": 10000, "fleurs_repeat": 8,
        "lora_r": 32, "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    },
    "ks_max": {
        # Kashmiri attempt #4 (after r=8 129.29 → +decode 92.09 → r=16 88.42,
        # all losing to Whisper-ks 74.02). Stacks the two untried levers:
        # (1) the r=32+MLP capacity rung that won Pashto, and (2) a TRAINABLE
        # __kas__ embedding row via PEFT trainable_token_indices — in every
        # prior attempt the Kashmiri conditioning vector was a FROZEN copy of
        # Urdu's (LoRA cannot touch embeddings), the suspected bottleneck
        # behind three loss/WER divergences. Full data, ~2.5 epochs.
        "sm_lang": "kas", "name": "Kashmiri (r32+MLP, trainable __kas__)",
        "indicvoices_parquet_dir": r"E:\VANI\datasets\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri",
        "train_cap": 24000,
        "lora_r": 32, "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "trainable_kas_token": True,
    },
    "ks_max2": {
        # Kashmiri attempt #5 (2026-07-25) — the WINNING ks_max recipe (r=32+MLP,
        # trainable __kas__ token) retrained on the COMBINED corpus built by
        # scratchpad/build_ks_combined.py: 97,456 clips / 239.9 h (humair025
        # IndicVoices 72,810 + IndicVoices-R 23,364 + OpenSLR-122 1,282), ~4x the
        # 24k ks_max saw. dur>=2 s; every train row whose text is in the IVR-R
        # TEST set was dropped (eval-leak guard). Val = the SAME IVR-R test split
        # as ks_max, so eval_loss/WER are directly comparable (ks_max: 80.91 raw /
        # 64.31 diacritic-normalised). Gate the result on the ruler study + the
        # 5-condition degradation sweep, not clean WER alone.
        "sm_lang": "kas", "name": "Kashmiri (r32+MLP, trainable __kas__, combined 240 h)",
        "combined_manifest_dir": r"E:\VANI\datasets\ks_combined",
        "indicvoices_parquet_dir": r"E:\VANI\datasets\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri",
        "train_cap": None,     # use all ~97k (ks_max was capped at 24000)
        "lora_r": 32, "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "trainable_kas_token": True,
    },
    "ks_cloud": {
        # CLOUD high-capacity Kashmiri run (rent a 24-48 GB GPU) — the lever the
        # 8 GB laptop can't reach. Same ks_max2 recipe + combined 97k/240h corpus,
        # but HIGH-RANK: r=128 (alpha 256) on all attn+MLP + trainable __kas__,
        # and a bigger batch via env (VANI_TRAIN_BS/VANI_GRAD_ACCUM). Data is
        # rebuilt cloud-side by cloud/prep_ks_data.py (pulls humair025 +
        # IndicVoices-R + OpenSLR-122 from source); paths come from env so the
        # same code runs on any box. Val = IVR-R test (== ks_max2 → eval_loss
        # directly comparable to 1.040). To sweep rank, edit lora_r/lora_alpha.
        # See cloud/README.md for the full workflow.
        "sm_lang": "kas", "name": "Kashmiri (CLOUD r128+MLP, trainable __kas__)",
        "combined_manifest_dir": os.environ.get("KS_COMBINED_DIR", "ks_data"),
        "indicvoices_parquet_dir": os.environ.get(
            "KS_IVR_DIR", "ks_data/indicvoices_r/Kashmiri"),
        "train_cap": None,
        "lora_r": 128, "lora_alpha": 256,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "trainable_kas_token": True,
    },
    "ks_cloud2": {
        # ks_cloud rerun #2 (2026-07-26 night): identical recipe, TWO fixes to the
        # early exit — ks_cloud early-stopped at step 7200 = epoch 0.8, i.e. it
        # never saw 20% of the corpus and its eval curve was still descending
        # (best 0.9268, monotonic). This run: 18000 steps (~2 epochs) and
        # patience 5 via VANI_ES_PATIENCE. Own run dir; ks_cloud adapter kept.
        "sm_lang": "kas", "name": "Kashmiri (CLOUD r128, 2-epoch, patience 5)",
        "combined_manifest_dir": os.environ.get("KS_COMBINED_DIR", "ks_data"),
        "indicvoices_parquet_dir": os.environ.get(
            "KS_IVR_DIR", "ks_data/indicvoices_r/Kashmiri"),
        "train_cap": None,
        "lora_r": 128, "lora_alpha": 256,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "trainable_kas_token": True,
    },
    "ks_cloud3": {
        # THE TOKENIZER FIX (2026-07-27). Diagnostic on deployed ks_cloud found
        # 20 Kashmiri characters absent from SeamlessM4T's SPM vocab -> <unk>:
        # 854,234 occurrences, 96.9% of training sentences carried at least one,
        # so nearly every target was corrupted AND the model could never emit
        # them (U+0672: 370x in test refs, 0x in hypotheses). Measured cost:
        # ~7.4 pp of the 56.44% WER, i.e. a hard floor near 49% for ANY model on
        # the old vocab — including ks_cloud2. KS_EXTRA_CHARS adds all 20 with
        # neighbour-initialised, TRAINABLE embedding rows (the __kas__ trick,
        # generalised). Verified locally: 39,768 <unk> -> 0 on 5k sentences,
        # round-trip exact modulo NFC mark ordering. Otherwise identical to
        # ks_cloud2 so the comparison isolates the vocabulary repair.
        "sm_lang": "kas", "name": "Kashmiri (CLOUD r128, 2-epoch, +20 vocab chars)",
        "combined_manifest_dir": os.environ.get("KS_COMBINED_DIR", "ks_data"),
        "indicvoices_parquet_dir": os.environ.get(
            "KS_IVR_DIR", "ks_data/indicvoices_r/Kashmiri"),
        "train_cap": None,
        "lora_r": 128, "lora_alpha": 256,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "trainable_kas_token": True,
        "ks_extra_chars": True,
    },
    "ks_cloud4": {
        # WARM START from ks_cloud3 (deployed, 50.26 L2 WER). Diagnosis: the 21
        # trainable token rows (__kas__ + the 20 repaired characters) saw only
        # 1.06 epochs before early-stopping, and they are measurably the weak
        # part — of the 747 test words containing the 4 previously-missing
        # LETTERS, ks_cloud3 still gets 437 wrong (~4.9 pp of WER), and those
        # failures are now ordinary acoustic confusions rather than
        # impossibilities. So: reload those exact weights, give them a FRESH LR
        # schedule (the old run ended on a decayed tail), run the LoRA gently at
        # 3e-5 while driving the token rows 5x faster, and train ~2 more epochs.
        # Everything else matches ks_cloud3 so the delta is attributable.
        "sm_lang": "kas", "name": "Kashmiri (warm-start from ks_cloud3, fast token rows)",
        "combined_manifest_dir": os.environ.get("KS_COMBINED_DIR", "ks_data"),
        "indicvoices_parquet_dir": os.environ.get(
            "KS_IVR_DIR", "ks_data/indicvoices_r/Kashmiri"),
        "train_cap": None,
        "init_adapter": os.environ.get(
            "KS_INIT_ADAPTER", "finetune_runs_seamless/ks_cloud3/adapter"),
        "learning_rate": 3e-5,
        "token_lr_mult": 5.0,
        "trainable_kas_token": True,
        "ks_extra_chars": True,
    },
    "doi_iv": {
        # DOGRI — the 8th border language, named in VANI's problem statement but
        # never fine-tuned or evaluated because no Dogri audio existed locally
        # (report 4.5). SeamlessM4T ships no __doi__/__dgo__ token, so this is
        # the SECOND use of the custom-token trick that made Kashmiri work:
        # __doi__ initialised from __hin__ and TRAINABLE (frozen init was the
        # bottleneck for ks). __hin__ rather than __pan__ deliberately — Dogri is
        # written in DEVANAGARI, and matching the script prior matters more than
        # the closer genetic tie to Punjabi, which SeamlessM4T only knows in
        # Gurmukhi.
        # Expected to be far easier than Kashmiri: Devanagari's working
        # orthography is well covered by the vocabulary (the ks <unk> defect
        # should not recur — VERIFY on real text before trusting that), and
        # Hindi gives a far stronger starting prior than Kashmiri ever had.
        # Data: IndicVoices-R Dogri (train + test), the same corpus family as ks.
        # Val = the IVR-R Dogri test split, so this run establishes the FIRST
        # Dogri baseline; there is no prior number to beat.
        "sm_lang": "doi", "name": "Dogri (IndicVoices-R, custom __doi__ token)",
        "custom_lang_token": {"token": "__doi__", "init_from": "__hin__"},
        "trainable_kas_token": True,          # generic: trains the custom row
        "indicvoices_parquet_dir": os.environ.get(
            "DOI_IVR_DIR", "doi_data/indicvoices_r/Dogri"),
        "train_cap": None,
        "lora_r": 128, "lora_alpha": 256,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    },
    "doi_iv2": {
        # Dogri continuation. doi_iv's validation loss improved at EVERY one of
        # its evaluations (2.095 -> ~1.23 by step 4400) and was still falling
        # when its linear schedule ran the learning rate down to zero — so it
        # stops because the SCHEDULE ended, not because the data was exhausted.
        # That is exactly the state ks_cloud was in when it early-stopped at
        # epoch 0.8; simply letting it converge (ks_cloud2) bought 3.84 pp, the
        # largest single gain of the campaign. So: warm-start from doi_iv's
        # weights with a FRESH schedule at half the original peak (5e-5) and
        # room to run. Same data, val split and rank, so the delta is
        # attributable to training length alone.
        "sm_lang": "doi", "name": "Dogri (continuation from doi_iv)",
        "custom_lang_token": {"token": "__doi__", "init_from": "__hin__"},
        "trainable_kas_token": True,
        "indicvoices_parquet_dir": os.environ.get(
            "DOI_IVR_DIR", "doi_data/indicvoices_r/Dogri"),
        "train_cap": None,
        "init_adapter": os.environ.get(
            "DOI_INIT_ADAPTER", "finetune_runs_seamless/doi_iv/adapter"),
        "learning_rate": 5e-5,
        "lora_r": 128, "lora_alpha": 256,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    },
    "ps_aug": {
        # Pashto attempt #5 — targets ps_bal2's robustness-gate failure (37.29
        # clean but 87.2 @ 0 dB vs Whisper 64.8). Same data and capacity as
        # ps_bal2; the ONLY change is noise-augmented training: each training
        # sample passes through the SAME degradation family the sweep tests
        # (scripts/eval/robustness_eval.degrade — clean 40%, bandpass 15%,
        # AWGN at 0/5/10/15 dB 30%, MP3 codec 10%, bandpass+AWGN5 5%).
        # Val stays clean FLEURS (comparability); the sweep is the judge.
        "fleurs": "ps_af", "sm_lang": "pbt", "name": "Pashto (bal, r32+MLP, noise-aug)",
        "cv_dataset": "SherwinDesouza/pashto-common-voice-20",
        "cv_cap": 10000, "fleurs_repeat": 8,
        "lora_r": 32, "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "augment": True,
    },
    "ps_aug2": {
        # Pashto attempt #6 (2026-07-25) — the WINNING ps_aug recipe (r=32+MLP,
        # noise-augmented) with SCALED, all-on-disk CV data: the pre-built
        # ps_combined pool (90,808 clips / 122 h = v20 validated 46,417 + CV22
        # 'other' quality-filtered 39,945 + CV22 new-clean 4,446) instead of
        # ps_aug's v20 cap-10k. FLEURS ps_af x8 anchor kept. cv_cap balances CV
        # vs FLEURS — ps_cv's 95%-CV dump drifted (42.47), ps_aug's cap-10k won
        # (36.91), so scale moderately (30k). Val = FLEURS ps_af (unchanged);
        # final eval = FLEURS ps_af test (compare_all_models) → comparable to
        # ps_aug 36.91%. Gate on the 5-condition degradation sweep, not clean WER.
        "fleurs": "ps_af", "sm_lang": "pbt", "name": "Pashto (r32+MLP, noise-aug, combined CV 122h)",
        "combined_ps_manifest_dir": r"E:\VANI\datasets\ps_combined",
        "cv_cap": 30000, "fleurs_repeat": 8,
        "lora_r": 32, "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "augment": True,
    },
    "ps_cloud": {
        # CLOUD high-capacity Pashto (sequential on the ks_cloud pod, 2026-07-26).
        # The WINNING ps_aug recipe untouched (v20 CV cap-10k + FLEURS ps_af x8 +
        # noise-aug; adding data proved harmful: ps_cv 42.47, ps_aug2 37.46) —
        # the ONLY change is capacity: r=128 (alpha 256), the lever the 8 GB
        # laptop couldn't lift. Honest expectation: ~34-36 (capacity is not the
        # known bottleneck; data/domain is) — this is the cheap falsification.
        # Data pulls straight from HF on the box (no prep script needed).
        # Val = FLEURS ps_af (unchanged ruler, comparable to ps_aug's 36.91).
        "fleurs": "ps_af", "sm_lang": "pbt", "name": "Pashto (CLOUD r128+MLP, noise-aug)",
        "cv_dataset": "SherwinDesouza/pashto-common-voice-20",
        "cv_cap": 10000, "fleurs_repeat": 8,
        "lora_r": 128, "lora_alpha": 256,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "augment": True,
    },
    "hi_iv": {
        # Experimental: Hindi with IndicVoices-R added (cap 20k) on top of FLEURS
        # hi_in (2,120). The deployed hi adapter (13.94) is FLEURS-only — this
        # probes whether domain-diverse data pushes further. Own run dir; the
        # deployed hi/ adapter is untouched.
        "fleurs": "hi_in", "sm_lang": "hin", "name": "Hindi (FLEURS + IndicVoices-R)",
        "indicvoices_config": "Hindi", "train_cap": 20000,
    },
    "ne_iv": {
        # Experimental: Nepali with IndicVoices-R added (cap 20k) on top of FLEURS
        # ne_np. Doubly fresh: the existing ne adapter still carries the
        # wrong-label training (only ps/hi were retrained after the label fix),
        # and zero-shot (28.46, deployed) has never been challenged with extra
        # data. Own run dir.
        "fleurs": "ne_np", "sm_lang": "npi", "name": "Nepali (FLEURS + IndicVoices-R)",
        "indicvoices_config": "Nepali", "train_cap": 20000,
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_fleurs(lang: str):
    from datasets import load_dataset, Audio
    cfg   = LANG_CFG[lang]
    token = os.environ.get("HF_TOKEN")
    cache = str(DATA_DIR / "fleurs")

    print(f"\n  [data] Loading FLEURS {cfg['fleurs']} ...")
    train_ds = load_dataset("google/fleurs", cfg["fleurs"], split="train",
                            token=token, cache_dir=cache)
    val_ds   = load_dataset("google/fleurs", cfg["fleurs"], split="validation",
                            token=token, cache_dir=cache)
    # Disable automatic audio decoding — we decode manually to avoid torchcodec dependency
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    val_ds   = val_ds.cast_column("audio",   Audio(decode=False))
    print(f"         Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_ds, val_ds


def load_fleurs_plus_cv(lang: str):
    """FLEURS + Common Voice Pashto, streamed in one pre-shuffled order.

    Both sources stay decode=False (bytes only) and are drawn through a
    generator-backed IterableDataset — same pattern as the ks IndicVoices run,
    avoiding the Windows Arrow-temp-file locking that binary-audio .map()
    caching hits at this row count. Val is FLEURS-only (unchanged ruler).
    The CV mirror's split names are inverted upstream (train=3.4k,
    validation=46.5k) — both splits are training data here; we never eval on CV.
    """
    import random
    from datasets import load_dataset, Audio, concatenate_datasets
    from datasets import IterableDataset as HFIterableDataset, Features, Value

    cfg = LANG_CFG[lang]
    fleurs_train, fleurs_val = load_fleurs(lang)

    token = os.environ.get("HF_TOKEN")
    print(f"\n  [data] Loading Common Voice ({cfg['cv_dataset']}) ...")
    parts = []
    for split in ("train", "validation"):
        ds = load_dataset(cfg["cv_dataset"], split=split, token=token,
                          cache_dir=str(DATA_DIR / "cv_ps"))
        ds = ds.cast_column("path", Audio(decode=False))
        before = len(ds)
        ds = ds.filter(lambda x: x["up_votes"] >= x["down_votes"]
                       and x["sentence"] and x["sentence"].strip())
        ds = ds.rename_column("path", "audio")
        ds = ds.rename_column("sentence", "transcription")
        ds = ds.select_columns(["audio", "transcription"])
        parts.append(ds)
        print(f"         CV {split}: {len(ds):,} kept ({before - len(ds):,} filtered)")
    cv_train = concatenate_datasets(parts)

    cv_cap = cfg.get("cv_cap")
    if cv_cap and len(cv_train) > cv_cap:
        cv_train = cv_train.shuffle(seed=42).select(range(cv_cap))
        print(f"         CV capped to {cv_cap:,}")

    feats = Features({
        "audio":         {"bytes": Value("binary"), "path": Value("string")},
        "transcription": Value("string"),
    })

    fleurs_repeat = cfg.get("fleurs_repeat", 1)

    def _gen(n_fleurs, n_cv, n_repeat):
        order = ([("f", i) for i in range(n_fleurs)] * n_repeat
                 + [("c", i) for i in range(n_cv)])
        random.Random(42).shuffle(order)
        for src, i in order:
            row = fleurs_train[i] if src == "f" else cv_train[i]
            audio = row["audio"]
            yield {
                "audio": {"bytes": audio.get("bytes"), "path": audio.get("path")},
                "transcription": row["transcription"],
            }

    train_ds = HFIterableDataset.from_generator(
        _gen, gen_kwargs={"n_fleurs": len(fleurs_train), "n_cv": len(cv_train),
                          "n_repeat": fleurs_repeat},
        features=feats,
    )
    total = len(fleurs_train) * fleurs_repeat + len(cv_train)
    print(f"\n  [data] Combined train: {total:,} samples "
          f"(FLEURS {len(fleurs_train):,} x{fleurs_repeat} + CV {len(cv_train):,}, "
          f"streaming, pre-shuffled)  Val: {len(fleurs_val)} (FLEURS only)")
    return train_ds, fleurs_val


def load_fleurs_plus_cv_manifest(lang: str):
    """FLEURS + the pre-built Pashto combined CV pool (scratchpad/build_ps_combined.py:
    v20 validated + CV22 'other' quality-filtered + CV22 new-clean splits). Same
    streaming/generator pattern as load_fleurs_plus_cv, but CV audio is pulled from the
    manifest's `path::row_idx` locators (v20 in place, CV22 from materialized shards)
    instead of load_dataset. Capped to cv_cap and held in memory. Val is FLEURS-only
    (unchanged ruler) so ps_aug2 stays comparable to ps_aug (36.91%)."""
    import random
    import pyarrow.parquet as pq
    from collections import defaultdict
    from datasets import IterableDataset as HFIterableDataset, Features, Value

    cfg = LANG_CFG[lang]
    fleurs_train, fleurs_val = load_fleurs(lang)

    comb = pathlib.Path(cfg["combined_ps_manifest_dir"])
    man  = pq.read_table(comb / "train_manifest.parquet").to_pydict()
    rows = list(zip(man["locator"], man["normalized"]))
    random.Random(42).shuffle(rows)
    cv_cap = cfg.get("cv_cap")
    if cv_cap and len(rows) > cv_cap:
        rows = rows[:cv_cap]
        print(f"  [data] CV pool capped to {cv_cap:,} of manifest")

    # Load the selected CV clips' audio into memory (grouped per source file).
    by_file = defaultdict(list)     # path -> [(row_idx, out_pos, text)]
    for pos, (loc, txt) in enumerate(rows):
        path, idx = loc.rsplit("::", 1)
        by_file[path].append((int(idx), pos, txt))
    cv_rows = [None] * len(rows)
    for path, items in by_file.items():
        names = pq.ParquetFile(path).schema_arrow.names
        acol  = "audio" if "audio" in names else ("path" if "path" in names else "audio_filepath")
        col   = pq.read_table(path, columns=[acol]).column(acol).to_pylist()
        for idx, pos, txt in items:
            if idx < len(col):
                a = col[idx]
                cv_rows[pos] = ({"bytes": a.get("bytes"), "path": a.get("path")}, txt)
        del col
    cv_rows = [r for r in cv_rows if r is not None]

    feats = Features({
        "audio":         {"bytes": Value("binary"), "path": Value("string")},
        "transcription": Value("string"),
    })
    fleurs_repeat = cfg.get("fleurs_repeat", 1)

    def _gen(n_fleurs, n_cv, n_repeat):
        order = ([("f", i) for i in range(n_fleurs)] * n_repeat
                 + [("c", i) for i in range(n_cv)])
        random.Random(42).shuffle(order)
        for src, i in order:
            if src == "f":
                row = fleurs_train[i]; audio = row["audio"]
                yield {"audio": {"bytes": audio.get("bytes"), "path": audio.get("path")},
                       "transcription": row["transcription"]}
            else:
                audio, txt = cv_rows[i]
                yield {"audio": audio, "transcription": txt}

    train_ds = HFIterableDataset.from_generator(
        _gen, gen_kwargs={"n_fleurs": len(fleurs_train), "n_cv": len(cv_rows),
                          "n_repeat": fleurs_repeat},
        features=feats,
    )
    total = len(fleurs_train) * fleurs_repeat + len(cv_rows)
    print(f"\n  [data] Combined train: {total:,} samples "
          f"(FLEURS {len(fleurs_train):,} x{fleurs_repeat} + CV {len(cv_rows):,} from manifest, "
          f"streaming)  Val: {len(fleurs_val)} (FLEURS only)")
    return train_ds, fleurs_val


def load_fleurs_plus_indicvoices(lang: str):
    """FLEURS + IndicVoices-R (hub-cache parquet, read directly via pyarrow —
    same battle-tested path as the ks run, no load_dataset for audio).
    FLEURS rows are interleaved into the IndicVoices stream at the ratio of the
    two set sizes (seeded RNG), then buffer-shuffled. Val is FLEURS-only."""
    import random
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from datasets import IterableDataset as HFIterableDataset, Features, Value

    cfg = LANG_CFG[lang]
    iv_config = cfg["indicvoices_config"]
    train_cap = cfg.get("train_cap", 20000)
    fleurs_train, fleurs_val = load_fleurs(lang)

    print(f"\n  [data] Resolving IndicVoices-R {iv_config} parquet (hub cache) ...")
    snap = pathlib.Path(snapshot_download(
        "ai4bharat/indicvoices_r", repo_type="dataset",
        allow_patterns=f"{iv_config}/*",
    ))
    iv_files = sorted((snap / iv_config).glob("train-*.parquet"))
    if not iv_files:
        raise FileNotFoundError(f"No {iv_config} train parquet under {snap}")

    feats = Features({
        "audio":         {"bytes": Value("binary"), "path": Value("string")},
        "transcription": Value("string"),
    })
    min_dur, max_dur = 2.0, 20.0

    def _gen(pq_files, max_iv, n_fleurs):
        rng = random.Random(42)
        fleurs_order = list(range(n_fleurs))
        rng.shuffle(fleurs_order)
        f_pos = 0
        p_fleurs = n_fleurs / max(1, max_iv)   # interleave ratio
        count = 0
        for pq_file in pq_files:
            if count >= max_iv:
                break
            t   = pq.read_table(pq_file, columns=["audio", "normalized", "duration"])
            raw = t.to_pydict()
            del t
            for audio, text, dur in zip(raw["audio"], raw["normalized"], raw["duration"]):
                if count >= max_iv:
                    break
                if dur is None or not (min_dur <= dur <= max_dur) or not text:
                    continue
                yield {"audio": audio, "transcription": text}
                count += 1
                if f_pos < n_fleurs and rng.random() < p_fleurs:
                    row = fleurs_train[fleurs_order[f_pos]]
                    f_pos += 1
                    a = row["audio"]
                    yield {"audio": {"bytes": a.get("bytes"), "path": a.get("path")},
                           "transcription": row["transcription"]}
            del raw
        # flush any FLEURS rows the ratio draw missed
        while f_pos < n_fleurs:
            row = fleurs_train[fleurs_order[f_pos]]
            f_pos += 1
            a = row["audio"]
            yield {"audio": {"bytes": a.get("bytes"), "path": a.get("path")},
                   "transcription": row["transcription"]}

    train_ds = HFIterableDataset.from_generator(
        _gen, gen_kwargs={"pq_files": iv_files, "max_iv": train_cap,
                          "n_fleurs": len(fleurs_train)},
        features=feats,
    ).shuffle(seed=42, buffer_size=3000)

    print(f"\n  [data] Combined train: ~{train_cap + len(fleurs_train):,} samples "
          f"(IndicVoices cap {train_cap:,} + FLEURS {len(fleurs_train):,}, streaming)"
          f"  Val: {len(fleurs_val)} (FLEURS only)")
    return train_ds, fleurs_val


def load_indicvoices_ks(cfg: dict):
    """Kashmiri from local IndicVoices-R parquet (same source that trained the
    Whisper ks model). Train is an IterableDataset (streaming — avoids the
    Windows Arrow-temp-file locking that binary-audio .map() caching hits);
    val (372 rows) is a regular in-memory Dataset. Yields 'transcription' from
    the `normalized` column so the FLEURS preprocess path applies unchanged."""
    import pyarrow.parquet as pq
    from datasets import Dataset, IterableDataset as HFIterableDataset, Features, Value

    parquet_dir = pathlib.Path(cfg["indicvoices_parquet_dir"])
    train_files = sorted(parquet_dir.glob("train-*.parquet"))
    test_files  = sorted(parquet_dir.glob("test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"No IndicVoices parquet at {parquet_dir}")
    min_dur, max_dur = 2.0, 20.0
    train_cap = cfg.get("train_cap", 20000)

    feats = Features({
        "audio":         {"bytes": Value("binary"), "path": Value("string")},
        "transcription": Value("string"),
    })

    def _gen(pq_files, max_samples):
        # max_samples None == use the whole split. Every caller before doi_iv
        # passed a number, so the None path was never exercised and raised
        # "'>=' not supported between int and NoneType" on the first shard.
        cap = float("inf") if max_samples is None else max_samples
        count = 0
        for pq_file in pq_files:
            if count >= cap:
                break
            t   = pq.read_table(pq_file, columns=["audio", "normalized", "duration"])
            raw = t.to_pydict()
            del t
            for audio, text, dur in zip(raw["audio"], raw["normalized"], raw["duration"]):
                if count >= cap:
                    break
                if dur is None or not (min_dur <= dur <= max_dur) or not text:
                    continue
                yield {"audio": audio, "transcription": text}
                count += 1
            del raw

    train_ds = HFIterableDataset.from_generator(
        _gen, gen_kwargs={"pq_files": train_files, "max_samples": train_cap},
        features=feats,
    ).shuffle(seed=42, buffer_size=3000)

    val_audio, val_text = [], []
    for pq_file in test_files:
        if len(val_text) >= 400:
            break
        t   = pq.read_table(pq_file, columns=["audio", "normalized", "duration"])
        raw = t.to_pydict()
        del t
        for audio, text, dur in zip(raw["audio"], raw["normalized"], raw["duration"]):
            if len(val_text) >= 400:
                break
            if dur is None or not (min_dur <= dur <= max_dur) or not text:
                continue
            val_audio.append(audio)
            val_text.append(text)
        del raw
    val_ds = Dataset.from_dict(
        {"audio": val_audio, "transcription": val_text}, features=feats,
    )
    print(f"\n  [data] IndicVoices-R KS train: {train_cap} samples (streaming)  Val: {len(val_ds)}")
    return train_ds, val_ds


def load_ks_combined(cfg: dict):
    """Kashmiri from the pre-built COMBINED corpus (scratchpad/build_ks_combined.py):
    humair025 IndicVoices + IndicVoices-R train + OpenSLR-122, filtered to dur>=2 s
    and deduped against the IVR-R TEST set. Train streams audio for the exact rows
    named in train_manifest.parquet — audio is pulled per source file via the
    `path::row_idx` locators, so the big embedded-audio sources are NOT copied.
    Val is the canonical IVR-R test split (identical to load_indicvoices_ks) so
    eval_loss/WER stay directly comparable to ks_max."""
    import random
    import pyarrow.parquet as pq
    from collections import defaultdict
    from datasets import Dataset, IterableDataset as HFIterableDataset, Features, Value

    comb = pathlib.Path(cfg["combined_manifest_dir"])
    man  = pq.read_table(comb / "train_manifest.parquet").to_pydict()
    min_dur, max_dur = 2.0, 20.0

    # Group manifest rows by their source parquet file (preserving the text),
    # dropping >max_dur (load_indicvoices_ks does the same; preprocessing also
    # hard-truncates at 20 s).
    by_file = defaultdict(list)      # path -> [(row_idx, text), ...]
    for loc, txt, dur in zip(man["locator"], man["normalized"], man["duration"]):
        if dur is None or dur > max_dur or not txt:
            continue
        path, idx = loc.rsplit("::", 1)
        by_file[path].append((int(idx), txt))
    files = list(by_file.keys())
    random.Random(42).shuffle(files)      # interleave sources so early steps mix
    n_train = sum(len(v) for v in by_file.values())

    feats = Features({
        "audio":         {"bytes": Value("binary"), "path": Value("string")},
        "transcription": Value("string"),
    })
    train_cap = cfg.get("train_cap")      # None -> use all

    def _gen(files, by_file, cap):
        count = 0
        for path in files:
            names = pq.ParquetFile(path).schema_arrow.names
            acol  = "audio" if "audio" in names else "audio_filepath"
            audio = pq.read_table(path, columns=[acol]).column(acol).to_pylist()
            for idx, txt in by_file[path]:
                if cap and count >= cap:
                    del audio
                    return
                if idx < len(audio):
                    yield {"audio": audio[idx], "transcription": txt}
                    count += 1
            del audio

    train_ds = HFIterableDataset.from_generator(
        _gen, gen_kwargs={"files": files, "by_file": dict(by_file), "cap": train_cap},
        features=feats,
    ).shuffle(seed=42, buffer_size=5000)

    # Val = canonical IVR-R test (identical construction to load_indicvoices_ks).
    ivr = pathlib.Path(cfg["indicvoices_parquet_dir"])
    val_audio, val_text = [], []
    for pq_file in sorted(ivr.glob("test-*.parquet")):
        if len(val_text) >= 400:
            break
        raw = pq.read_table(pq_file, columns=["audio", "normalized", "duration"]).to_pydict()
        for audio, text, dur in zip(raw["audio"], raw["normalized"], raw["duration"]):
            if len(val_text) >= 400:
                break
            if dur is None or not (min_dur <= dur <= max_dur) or not text:
                continue
            val_audio.append(audio)
            val_text.append(text)
    val_ds = Dataset.from_dict(
        {"audio": val_audio, "transcription": val_text}, features=feats,
    )

    cap_note = f"{train_cap:,}" if train_cap else f"all {n_train:,}"
    print(f"\n  [data] KS COMBINED train: {cap_note} clips (<=20 s) from {len(files)} "
          f"source files (streaming)  Val: {len(val_ds)}")
    return train_ds, val_ds


# Kashmiri characters ABSENT from the SeamlessM4T sentencepiece vocabulary —
# they encode to <unk>, so the model can neither read them in training targets
# nor ever emit them (verified 2026-07-27: U+0672 appears 370x in the 372-clip
# test references and 0x in ks_cloud's hypotheses). 96.9% of the 97k training
# sentences carry at least one, i.e. almost every target was corrupted, and
# ~7.4 pp of the 56.44% WER is this defect alone — a hard floor near 49%.
# Each maps to the closest in-vocab character used to initialise its embedding
# (same trick as __kas__ <- __urd__, which is what made Kashmiri work at all).
# Complete list: every character in the 97k-sentence ks_combined corpus that the
# SPM tokenizer maps to <unk> (20 distinct, 854,234 occurrences). Values are the
# closest in-vocab character, used to initialise the new embedding row.
KS_EXTRA_CHARS = {
    # ── letters (carry phonemic content; getting these right matters most) ──
    "ٲ": "ا",  # U+0672 ALEF WITH WAVY HAMZA ABOVE   90,480
    "ٮ": "ب",  # U+066E DOTLESS BEH                  64,763
    "ۄ": "و",  # U+06C4 WAW WITH RING                50,170
    "ۅ": "و",  # U+06C5 KIRGHIZ OE                    2,584
    "ؠ": "ی",  # U+0620 KASHMIRI YEH                  2,348
    # ── combining marks below the line ──
    "ٕ": "ء",  # U+0655 HAMZA BELOW                 316,046
    "ٖ": "ِ",  # U+0656 SUBSCRIPT ALEF               56,023
    "۪": "ِ",  # U+06EA EMPTY CENTRE LOW STOP        49,037
    "ۭ": "ِ",  # U+06ED SMALL LOW MEEM               15,104
    "ٟ": "ء",  # U+065F WAVY HAMZA BELOW                179
    # ── combining marks above the line ──
    "ٚ": "َ",  # U+065A VOWEL SIGN SMALL V ABOVE    162,929
    "ٗ": "ُ",  # U+0657 INVERTED DAMMA               44,510
    "٘": "ْ",  # U+0658 MARK NOON GHUNNA                 43
    "ٓ": "َ",  # U+0653 MADDAH ABOVE                      3
    "ٙ": "َ",  # U+0659 ZWARAKAY                          3
    # ── rare honorifics / signs (kept so round-trip is exact) ──
    "ؐ": "َ",  # U+0610 SALLALLAHOU ALAYHE WASSALLAM      5
    "ؒ": "َ",  # U+0612 RAHMATULLAH ALAYHE                2
    "ؑ": "َ",  # U+0611 ALAYHE ASSALLAM                   2
    "﷽": "ا",  # U+FDFD BISMILLAH LIGATURE                2
    "؎": "،",  # U+060E POETIC VERSE SIGN                 1
}


def add_ks_chars(processor, model):
    """Add the Kashmiri characters missing from the SPM vocab as real tokens,
    initialising each embedding from its closest in-vocab neighbour. Returns the
    list of new token ids so the caller can mark them TRAINABLE (they start as
    approximations and must learn their own acoustics/orthography).

    Only characters that actually resolve to <unk> are added, so re-running on an
    already-extended tokenizer is a no-op."""
    import torch as _torch
    tok = processor.tokenizer
    unk = tok.unk_token_id

    def _content_id(ch):
        """Last piece of ch's encoding — skips SPM's leading word-boundary marker."""
        ids = tok.encode(ch, add_special_tokens=False)
        return ids[-1] if ids else unk

    missing = [c for c in KS_EXTRA_CHARS
               if any(i == unk for i in tok.encode(c, add_special_tokens=False))]
    if not missing:
        print("  [ks-chars] all Kashmiri characters already in tokenizer")
        return []

    # capture init sources BEFORE extending the tokenizer (ids shift on resize)
    init_src = {c: _content_id(KS_EXTRA_CHARS[c]) for c in missing}
    tok.add_tokens(missing)                      # normal tokens: must compose into words
    print(f"  [ks-chars] added {len(missing)} chars to tokenizer (vocab {len(tok)}): "
          + " ".join(f"U+{ord(c):04X}" for c in missing))

    if model.get_input_embeddings().weight.shape[0] < len(tok):
        model.resize_token_embeddings(len(tok))

    new_ids = []
    with _torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings()
        # fallback for any char with no usable neighbour: the mean embedding is a
        # far more stable start than resize_token_embeddings' random init
        in_mean = in_emb[:len(tok) - len(missing)].mean(dim=0)
        for c in missing:
            nid, src = tok.convert_tokens_to_ids(c), init_src[c]
            assert nid != unk, f"failed to add U+{ord(c):04X}"
            if src != unk:
                in_emb[nid] = in_emb[src].clone()
                if out_emb is not None and out_emb.weight.data_ptr() != in_emb.data_ptr():
                    out_emb.weight[nid] = out_emb.weight[src].clone()
            else:
                in_emb[nid] = in_mean.clone()
            new_ids.append(nid)
    print(f"  [ks-chars] {len(new_ids)} embedding rows initialised from in-vocab neighbours")
    return new_ids


def add_lang_token(processor, model, token: str, init_from: str, tag: str = None):
    """Add a language token SeamlessM4T does not ship, initialised from the
    closest language it does. The embedding stays FROZEN under plain LoRA — the
    initialised prefix acts as a fixed conditioning vector, exactly as Whisper's
    <|ks|> did (74.02% WER) — unless the config marks it trainable, which is
    what finally made Kashmiri work. Also registers the code in the generation
    config so post-training generate(tgt_lang=...) resolves it.

    Kashmiri (__kas__ <- __urd__) was the first use; Dogri (__doi__ <- __hin__)
    is the second, which is why this is parameterised rather than hardcoded.
    Token ORDER matters: a deployed adapter's rows are positional, so never
    reorder or insert ahead of an existing custom token.
    """
    import torch as _torch
    tok = processor.tokenizer
    code = (tag or token.strip("_"))
    if tok.convert_tokens_to_ids(token) != tok.unk_token_id and token in tok.get_vocab():
        print(f"  [{code}] {token} already in tokenizer")
    else:
        tok.add_tokens([token], special_tokens=True)
        print(f"  [{code}] {token} added to tokenizer (vocab {len(tok)})")

    new_id = tok.convert_tokens_to_ids(token)
    src_id = tok.convert_tokens_to_ids(init_from)
    assert new_id != tok.unk_token_id, f"failed to add {token}"
    assert src_id != tok.unk_token_id, f"init source {init_from} not in vocabulary"

    if model.get_input_embeddings().weight.shape[0] < len(tok):
        model.resize_token_embeddings(len(tok))
    with _torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        in_emb[new_id] = in_emb[src_id].clone()
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != in_emb.data_ptr():
            out_emb.weight[new_id] = out_emb.weight[src_id].clone()
    print(f"  [{code}] embedding row {new_id} initialised from {init_from} ({src_id})")

    gc = model.generation_config
    lang_map = getattr(gc, "text_decoder_lang_to_code_id", None)
    if lang_map is not None:
        try:
            lang_map[code] = new_id
        except TypeError:
            pass  # non-dict mapping; eval must pass the bos id explicitly
    return new_id


def add_kas_token(processor, model):
    """Kashmiri's custom token. Thin wrapper kept so the deployed ks adapters
    (whose embedding rows are positional) keep byte-identical behaviour."""
    return add_lang_token(processor, model, "__kas__", "__urd__", tag="kas")


# ── Audio decoding ─────────────────────────────────────────────────────────────

def decode_audio(audio_dict: dict, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    arr = audio_dict.get("array")
    if arr is not None:
        arr = np.array(arr, dtype=np.float32)
        sr  = audio_dict.get("sampling_rate", target_sr)
    else:
        raw  = audio_dict.get("bytes")
        path = audio_dict.get("path")
        if raw:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        elif path:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
        else:
            return np.zeros(target_sr, dtype="float32")

    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype("float32")


# ── Preprocessing ──────────────────────────────────────────────────────────────

MAX_AUDIO_SEC = 20   # truncate long samples


def make_augment_fn():
    """Radio-channel training augmentation drawing from the SAME degradation
    family the robustness sweep evaluates (scripts/eval/robustness_eval.degrade),
    so robustness is trained on the distribution it is judged on. Seeded RNG for
    reproducibility. Menu: clean 40% / bandpass 15% / AWGN 0-15 dB 30% /
    MP3 codec 10% / bandpass+AWGN5 5%."""
    import random
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    from robustness_eval import degrade  # noqa: E402
    rng = random.Random(42)

    def augment(arr):
        r = rng.random()
        try:
            if r < 0.40:
                return arr
            if r < 0.55:
                return degrade(arr, 16000, "bandpass")
            if r < 0.85:
                snr = rng.choice([0, 5, 10, 15])
                return degrade(arr, 16000, f"awgn_{snr}")
            if r < 0.95:
                return degrade(arr, 16000, "codec_mp3")
            return degrade(degrade(arr, 16000, "bandpass"), 16000, "awgn_5")
        except Exception:
            return arr   # a failed ffmpeg round-trip must not kill training

    return augment


def make_preprocess_fn(processor, sm_lang: str, augment=None):
    """Returns a function that processes one FLEURS sample."""
    def fn(batch):
        arr = decode_audio(batch["audio"])
        arr = arr[: MAX_AUDIO_SEC * 16000]           # hard truncate
        if augment is not None:
            arr = augment(arr)

        # Audio features (speech encoder input)
        feat = processor.feature_extractor(
            arr, sampling_rate=16000, return_tensors="np"
        )

        # Text labels — MUST use text_target so the tokenizer emits target-mode
        # special tokens: prefix [eos, __lang__], suffix [eos], matching what
        # generate(tgt_lang=...) forces at inference. Passing the text as plain
        # `text` (as this script originally did) encodes SOURCE mode instead —
        # prefix [__eng__] — so the six 2026-07 FLEURS adapters were trained
        # against __eng__-prefixed labels. That mismatch likely contributed to
        # fine-tuning being a wash and to the S2TT chrF collapse.
        tok = processor.tokenizer(
            text_target=batch["transcription"],
            tgt_lang=sm_lang,
            return_tensors="np",
        )

        out = {
            "input_features": feat["input_features"][0],   # (T, C)
            "labels":         tok["input_ids"][0].tolist(),
        }
        if "attention_mask" in feat:
            out["attention_mask"] = feat["attention_mask"][0]  # (T,)
        return out

    return fn


# ── Data collator ──────────────────────────────────────────────────────────────

class SeamlessDataCollator:
    pad_label_id: int = -100

    def __init__(self, pad_feature_value: float = 0.0):
        self.pad_value = pad_feature_value

    def __call__(self, features: list) -> dict:
        # --- audio features ---
        feats = [np.array(f["input_features"]) for f in features]
        max_T = max(x.shape[0] for x in feats)
        C     = feats[0].shape[1]

        padded_feats = np.full((len(feats), max_T, C), self.pad_value, dtype=np.float32)
        attn_masks   = np.zeros((len(feats), max_T), dtype=np.int64)
        for i, x in enumerate(feats):
            t = x.shape[0]
            padded_feats[i, :t] = x
            attn_masks[i,  :t] = 1

        # honour pre-computed attention_mask if the feature extractor returned one
        if "attention_mask" in features[0]:
            for i, f in enumerate(features):
                m = np.array(f["attention_mask"])
                t = min(len(m), max_T)
                attn_masks[i, :t] = m[:t]

        # --- labels ---
        labels    = [f["labels"] for f in features]
        max_L     = max(len(l) for l in labels)
        pad_labels = np.full((len(labels), max_L), self.pad_label_id, dtype=np.int64)
        for i, l in enumerate(labels):
            pad_labels[i, :len(l)] = l

        return {
            "input_features": torch.tensor(padded_feats),
            "attention_mask": torch.tensor(attn_masks),
            "labels":         torch.tensor(pad_labels),
        }


# ── Training ───────────────────────────────────────────────────────────────────

def train(lang: str, args):
    from transformers import (
        AutoProcessor, SeamlessM4Tv2ForSpeechToText,
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    cfg        = LANG_CFG[lang]
    sm_lang    = cfg["sm_lang"]
    adapter_dir = RUNS_DIR / lang / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  SeamlessM4T v2 LoRA Fine-Tuning  |  {cfg['name']} ({lang})")
    print(f"  SM4T lang: {sm_lang}   Steps: {args.steps}")
    print(f"{'='*62}")

    # ── Load processor + model ────────────────────────────────────────────────
    print("\n  [model] Loading SeamlessM4Tv2ForSpeechToText ...")
    processor = AutoProcessor.from_pretrained(str(SEAMLESS_DIR))

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(SEAMLESS_DIR), torch_dtype=dtype,
    )
    model.enable_input_require_grads()   # required for gradient checkpointing + PEFT

    # Kashmiri: SeamlessM4T has no __kas__ — add it (embedding init from __urd__)
    # BEFORE the PEFT wrap so the resized embedding is part of the base model.
    kas_id = None
    ks_char_ids = []
    if sm_lang == "kas":
        kas_id = add_kas_token(processor, model)
    elif cfg.get("custom_lang_token"):
        # Any other language SeamlessM4T does not ship a token for (Dogri is the
        # first): {"token": "__doi__", "init_from": "__hin__"}
        _ct = cfg["custom_lang_token"]
        kas_id = add_lang_token(processor, model, _ct["token"], _ct["init_from"],
                                tag=sm_lang)
        # Repair the vocabulary itself: without these, ~31% of target WORDS
        # contain an <unk> and the model can never emit them (see KS_EXTRA_CHARS).
        if cfg.get("ks_extra_chars"):
            ks_char_ids = add_ks_chars(processor, model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_kwargs = {}
    trainable_ids = []
    if cfg.get("trainable_kas_token") and kas_id is not None:
        # Train ONLY the __kas__ row of text_decoder.embed_tokens (PEFT
        # TrainableTokens delta) — the frozen urd-init conditioning vector is
        # the suspected ks bottleneck. lm_head is unaffected (delta applies to
        # the input-embedding forward, which is what conditions generation).
        trainable_ids.append(kas_id)
        _tokname = (cfg.get("custom_lang_token") or {}).get("token", "__kas__")
        print(f"  [{sm_lang}] {_tokname} embedding row {kas_id} set TRAINABLE "
              f"(PEFT trainable_token_indices)")
    if ks_char_ids:
        # The new character rows start as copies of their nearest neighbour —
        # they only become useful once trained.
        trainable_ids += ks_char_ids
        print(f"  [ks-chars] {len(ks_char_ids)} character rows set TRAINABLE")
    if trainable_ids:
        lora_kwargs["trainable_token_indices"] = {"embed_tokens": trainable_ids}
    init_adapter = cfg.get("init_adapter")
    if init_adapter:
        # WARM START: continue from an existing adapter instead of random LoRA.
        # Its adapter_config.json supplies r/alpha/targets/trainable_token_indices,
        # so the config's own lora_* keys are ignored here by design — the point is
        # to resume those exact weights, with a fresh LR schedule (see cfg
        # learning_rate) rather than the decayed tail of the previous run.
        from peft import PeftModel
        p = pathlib.Path(init_adapter)
        if not (p / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"init_adapter has no weights: {p}")
        model = PeftModel.from_pretrained(model, str(p), is_trainable=True)
        print(f"  [warm-start] loaded adapter weights from {p}")
    else:
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=cfg.get("lora_targets", ["q_proj", "v_proj"]),
            lora_dropout=0.05,
            task_type=TaskType.SEQ_2_SEQ_LM,
            bias="none",
            **lora_kwargs,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    if cfg.get("combined_manifest_dir"):
        train_raw, val_raw = load_ks_combined(cfg)
    elif cfg.get("combined_ps_manifest_dir"):
        train_raw, val_raw = load_fleurs_plus_cv_manifest(lang)
    elif cfg.get("indicvoices_parquet_dir"):
        train_raw, val_raw = load_indicvoices_ks(cfg)
    elif cfg.get("cv_dataset"):
        train_raw, val_raw = load_fleurs_plus_cv(lang)
    elif cfg.get("indicvoices_config"):
        train_raw, val_raw = load_fleurs_plus_indicvoices(lang)
    else:
        train_raw, val_raw = load_fleurs(lang)

    # augmentation applies to TRAIN only; val stays clean (model selection on
    # the same clean ruler as every prior run)
    augment_fn = make_augment_fn() if cfg.get("augment") else None
    preprocess = make_preprocess_fn(processor, sm_lang, augment=augment_fn)
    preprocess_val = make_preprocess_fn(processor, sm_lang)
    if augment_fn is not None:
        print("  [data] Training-time radio augmentation ENABLED (val stays clean)")

    cache_base = RUNS_DIR / lang / "data"
    cache_base.mkdir(parents=True, exist_ok=True)

    from datasets import IterableDataset as HFIterableDataset
    if isinstance(train_raw, HFIterableDataset):
        # streaming: features computed lazily per step; no Arrow cache, no len()
        train_ds = train_raw.map(preprocess, remove_columns=["audio", "transcription"])
        print("\n  [data] Train: streaming (features on-the-fly)")
    else:
        print("\n  [data] Preprocessing train split ...")
        train_ds = train_raw.map(
            preprocess,
            remove_columns=train_raw.column_names,
            desc="train-preproc",
            num_proc=1,
            cache_file_name=str(cache_base / f"train_{sm_lang}.arrow"),
        )

    print("  [data] Preprocessing val split ...")
    val_ds = val_raw.map(
        preprocess_val,
        remove_columns=val_raw.column_names,
        desc="val-preproc",
        num_proc=1,
        cache_file_name=str(cache_base / f"val_{sm_lang}.arrow"),
    )

    if not isinstance(train_ds, HFIterableDataset):
        print(f"         Train: {len(train_ds)}  Val: {len(val_ds)}")

    collator = SeamlessDataCollator()

    # ── Training arguments ────────────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        max_steps=args.steps,
        per_device_train_batch_size=int(os.environ.get("VANI_TRAIN_BS", "1")),
        gradient_accumulation_steps=int(os.environ.get("VANI_GRAD_ACCUM", "8")),  # eff batch = BS*accum; cloud can raise VANI_TRAIN_BS on a big GPU
        per_device_eval_batch_size=1,
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        warmup_steps=50,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        remove_unused_columns=False,        # our collator expects named fields
    )

    # Optional: train the added-token embedding rows FASTER than the LoRA weights.
    # They are the least-converged parameters in a warm-started run — ks_cloud3's
    # 21 rows saw only 1.06 epochs and still account for ~4.9 pp of WER — so the
    # LoRA can be nudged gently while the rows keep learning.
    optimizers = (None, None)
    tok_mult = float(cfg.get("token_lr_mult", 1.0))
    if tok_mult != 1.0:
        base_lr = float(cfg.get("learning_rate", 1e-4))
        tok_p = [p for n, p in model.named_parameters()
                 if p.requires_grad and "trainable_tokens" in n]
        rest_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and "trainable_tokens" not in n]
        if not tok_p:
            raise RuntimeError("token_lr_mult set but no trainable_tokens parameters found")
        opt = torch.optim.AdamW(
            [{"params": rest_p, "lr": base_lr},
             {"params": tok_p,  "lr": base_lr * tok_mult}],
            weight_decay=training_args.weight_decay)
        optimizers = (opt, None)          # Trainer builds the scheduler on top
        print(f"  [lr] token rows {base_lr * tok_mult:.2e} ({tok_mult}x) | "
              f"LoRA {base_lr:.2e}  ({sum(p.numel() for p in tok_p):,} vs "
              f"{sum(p.numel() for p in rest_p):,} params)")

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=optimizers,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            # ks_cloud stopped at epoch 0.8 with the curve still falling; cloud
            # runs can afford more patience (VANI_ES_PATIENCE, default 3).
            early_stopping_patience=int(os.environ.get("VANI_ES_PATIENCE", "3")))],
    )

    # ── Resume logic (same fix as finetune_whisper.py) ────────────────────────
    if args.resume and adapter_dir.exists():
        if (adapter_dir / "trainer_state.json").exists():
            resume_from = str(adapter_dir)
        else:
            ckpts = sorted(
                [d for d in adapter_dir.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[1]),
            )
            resume_from = str(ckpts[-1]) if ckpts else None
    else:
        resume_from = None

    if resume_from:
        print(f"\n  [train] Resuming from {resume_from}")
    else:
        print(f"\n  [train] Starting training for {args.steps} steps ...")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\n  [OK] Adapter saved -> {adapter_dir}")
    print("       Run eval:  python scripts/eval/compare_all_models.py "
          f"--lang {lang} --skip-baseline")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SeamlessM4T v2 on FLEURS with LoRA"
    )
    parser.add_argument(
        "lang",
        choices=list(LANG_CFG.keys()) + ["all"],
        help="Language code (pa/ps/ur/ne/zh/hi/ks) or 'all' for the six FLEURS langs",
    )
    parser.add_argument("--steps",  type=int, default=1000, help="Max training steps")
    parser.add_argument("--resume", action="store_true",    help="Resume from checkpoint")
    args = parser.parse_args()

    if not SEAMLESS_DIR.exists():
        print(f"\n[ERROR] SeamlessM4T model not found at {SEAMLESS_DIR}")
        print("        It should already be at models/seamless-m4t-v2-large")
        sys.exit(1)

    # "all" = the six FLEURS languages; ks (custom token) and the extra-data
    # experiments run only when named explicitly
    EXPERIMENTS = {"ks", "ks_r16", "ks_max", "ks_max2", "ks_cloud", "ks_cloud2", "ks_cloud3", "ks_cloud4", "ps_cv", "ps_bal", "ps_bal2", "ps_aug", "ps_aug2", "ps_cloud", "hi_iv", "ne_iv", "doi_iv", "doi_iv2"}
    langs = [l for l in LANG_CFG if l not in EXPERIMENTS] if args.lang == "all" else [args.lang]
    for lang in langs:
        train(lang, args)


if __name__ == "__main__":
    main()

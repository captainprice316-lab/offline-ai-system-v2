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
SEAMLESS_DIR = MODELS_DIR / "seamless-m4t-v2-large"
RUNS_DIR     = ROOT / "finetune_runs_seamless"

LANG_CFG = {
    "pa": {"fleurs": "pa_in",       "sm_lang": "pan", "name": "Punjabi"},
    "ps": {"fleurs": "ps_af",       "sm_lang": "pbt", "name": "Pashto"},
    "ur": {"fleurs": "ur_pk",       "sm_lang": "urd", "name": "Urdu"},
    "ne": {"fleurs": "ne_np",       "sm_lang": "npi", "name": "Nepali"},
    "zh": {"fleurs": "cmn_hans_cn", "sm_lang": "cmn", "name": "Mandarin"},
    "hi": {"fleurs": "hi_in",       "sm_lang": "hin", "name": "Hindi"},
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


def make_preprocess_fn(processor, sm_lang: str):
    """Returns a function that processes one FLEURS sample."""
    def fn(batch):
        arr = decode_audio(batch["audio"])
        arr = arr[: MAX_AUDIO_SEC * 16000]           # hard truncate

        # Audio features (speech encoder input)
        feat = processor.feature_extractor(
            arr, sampling_rate=16000, return_tensors="np"
        )

        # Text labels — tokenise with tgt_lang = source language (ASR task)
        tok = processor.tokenizer(
            batch["transcription"],
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

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_raw, val_raw = load_fleurs(lang)

    preprocess = make_preprocess_fn(processor, sm_lang)

    cache_base = RUNS_DIR / lang / "data"
    cache_base.mkdir(parents=True, exist_ok=True)

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
        preprocess,
        remove_columns=val_raw.column_names,
        desc="val-preproc",
        num_proc=1,
        cache_file_name=str(cache_base / f"val_{sm_lang}.arrow"),
    )

    print(f"         Train: {len(train_ds)}  Val: {len(val_ds)}")

    collator = SeamlessDataCollator()

    # ── Training arguments ────────────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # effective batch = 8
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
        help="Language code (pa/ps/ur/ne/zh/hi) or 'all' for all six",
    )
    parser.add_argument("--steps",  type=int, default=1000, help="Max training steps")
    parser.add_argument("--resume", action="store_true",    help="Resume from checkpoint")
    args = parser.parse_args()

    if not SEAMLESS_DIR.exists():
        print(f"\n[ERROR] SeamlessM4T model not found at {SEAMLESS_DIR}")
        print("        It should already be at models/seamless-m4t-v2-large")
        sys.exit(1)

    langs = list(LANG_CFG.keys()) if args.lang == "all" else [args.lang]
    for lang in langs:
        train(lang, args)


if __name__ == "__main__":
    main()

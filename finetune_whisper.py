"""
finetune_whisper.py  --  LoRA fine-tune Whisper for VANI language models
=========================================================================
Fine-tunes language-specific Whisper models on FLEURS + (optionally)
Common Voice, then merges the LoRA adapter and exports a CT2 int8 model
ready for faster-whisper in VANI.

Dependencies already installed:
    peft, accelerate, datasets, jiwer

Usage:
    python finetune_whisper.py pa              # Punjabi (large-v3 base)
    python finetune_whisper.py ps              # Pashto  (medium base)
    python finetune_whisper.py pa --no-cv      # FLEURS only (no HF token needed)
    python finetune_whisper.py pa --steps 2000
    python finetune_whisper.py pa --resume     # continue from last checkpoint
    python finetune_whisper.py pa --skip-ct2   # save adapter only, convert later

Common Voice adds ~28h (pa) / ~54h (ps) on top of FLEURS (~12h each).
It requires a free HuggingFace account with accepted terms:
    1. Accept: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
    2. Run:    huggingface-cli login
Without a token, FLEURS alone is used -- still useful for language accuracy.

VRAM guide (RTX 5060 8 GB):
    pa  (large-v3,  LoRA r=8,  batch 2x8=16):  ~5.5 GB
    ps  (medium,    LoRA r=16, batch 4x4=16):  ~3.5 GB
"""

import sys
import os
import shutil
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project layout ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RUNS_DIR   = ROOT / "finetune_runs"
RUNS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Per-language config ────────────────────────────────────────────────────────
LANG_CONFIG = {
    "pa": {
        "hf_model":      "openai/whisper-large-v3",
        "whisper_lang":  "pa",
        "task":          "transcribe",
        "fleurs_config": "pa_in",
        "cv_config":     "pa-IN",
        "ct2_name":      "whisper-large-v3-pa-ct2",
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "batch_size":    2,
        "grad_accum":    1,   # no accumulation: ~19s/step vs 150s at accum=8
        "learning_rate": 5e-5,
        "warmup_steps":  50,
    },
    "ps": {
        "hf_model":      "Nasimbahar/pashto-ghag-whisper-medium-asr",
        "whisper_lang":  "ps",
        "task":          "transcribe",
        "fleurs_config": "ps_af",
        "cv_config":     "ps",
        "ct2_name":      "whisper-medium-pashto-ct2",
        "lora_r":        16,
        "lora_alpha":    32,
        "lora_dropout":  0.05,
        "batch_size":    4,
        "grad_accum":    1,   # no accumulation: medium model is ~4x faster than large-v3
        "learning_rate": 5e-5,
        "warmup_steps":  25,
    },
    "ur": {
        "hf_model":      "openai/whisper-large-v3",
        "whisper_lang":  "ur",
        "task":          "transcribe",
        "fleurs_config": "ur_pk",
        "cv_config":     "ur",
        "ct2_name":      "whisper-large-v3-ur-ct2",
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "batch_size":    2,
        "grad_accum":    1,
        "learning_rate": 5e-5,
        "warmup_steps":  50,
    },
    "ne": {
        "hf_model":      "openai/whisper-large-v3",
        "whisper_lang":  "ne",
        "task":          "transcribe",
        "fleurs_config": "ne_np",
        "cv_config":     "ne-NP",
        "ct2_name":      "whisper-large-v3-ne-ct2",
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "batch_size":    2,
        "grad_accum":    1,
        "learning_rate": 5e-5,
        "warmup_steps":  50,
    },
    "zh": {
        "hf_model":      "openai/whisper-large-v3",
        "whisper_lang":  "zh",
        "task":          "transcribe",
        "fleurs_config": "cmn_hans_cn",
        "cv_config":     "zh-CN",
        "ct2_name":      "whisper-large-v3-zh-ct2",
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "batch_size":    2,
        "grad_accum":    1,
        "learning_rate": 5e-5,
        "warmup_steps":  50,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ct2_converter() -> str:
    for subdir in ("Scripts", "bin"):
        for name in ("ct2-transformers-converter.exe", "ct2-transformers-converter"):
            candidate = ROOT / "venv" / subdir / name
            if candidate.exists():
                return str(candidate)
    return "ct2-transformers-converter"


def _hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


# ── Dataset loading ────────────────────────────────────────────────────────────

def _decode_audio(audio_info: dict, target_sr: int = 16000):
    """Decode audio from datasets raw dict using soundfile (no torchaudio/torchcodec)."""
    import io
    import numpy as np
    import soundfile as sf
    import librosa

    raw_bytes = audio_info.get("bytes")
    path      = audio_info.get("path")
    src_sr    = audio_info.get("sampling_rate", target_sr)

    if raw_bytes:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def load_datasets(lang: str, cfg: dict, use_cv: bool, token: Optional[str]) -> dict:
    from datasets import load_dataset, concatenate_datasets, Audio

    data_dir = RUNS_DIR / lang / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "validation": []}

    # FLEURS -- no token required
    print("\n  [data] Loading FLEURS ...")
    for split in ("train", "validation"):
        try:
            ds = load_dataset(
                "google/fleurs",
                cfg["fleurs_config"],
                split=split,
                cache_dir=str(data_dir / "fleurs"),
            )
            # Audio(decode=False) returns raw {bytes, path} dict without calling
            # torchaudio -- required because torchaudio 2.11+ needs torchcodec.
            ds = ds.cast_column("audio", Audio(decode=False))
            ds = ds.rename_column("transcription", "text")
            ds = ds.select_columns(["audio", "text"])
            splits[split].append(ds)
            print(f"         FLEURS {split}: {len(ds):,} samples")
        except Exception as e:
            print(f"         [WARN] FLEURS {split}: {e}")

    # Common Voice -- token required
    if use_cv:
        if not token:
            print("\n  [data] No HuggingFace token -- skipping Common Voice.")
            print("         Run: huggingface-cli login")
        else:
            print("\n  [data] Loading Common Voice 17 ...")
            for split in ("train", "validation"):
                try:
                    ds = load_dataset(
                        "mozilla-foundation/common_voice_17_0",
                        cfg["cv_config"],
                        split=split,
                        token=token,
                        cache_dir=str(data_dir / "cv"),
                    )
                    ds = ds.cast_column("audio", Audio(decode=False))
                    # Drop entries with more downvotes than upvotes
                    before = len(ds)
                    ds = ds.filter(lambda x: x["up_votes"] >= x["down_votes"])
                    ds = ds.rename_column("sentence", "text")
                    ds = ds.select_columns(["audio", "text"])
                    splits[split].append(ds)
                    print(f"         CV {split}: {len(ds):,} samples (filtered {before-len(ds):,} low-quality)")
                except Exception as e:
                    print(f"         [WARN] Common Voice {split}: {e}")

    if not splits["train"]:
        print("[ERROR] No training data loaded.")
        sys.exit(1)

    result = {}
    for split, ds_list in splits.items():
        if ds_list:
            result[split] = concatenate_datasets(ds_list).shuffle(seed=42)
            print(f"\n  [data] Total {split}: {len(result[split]):,} samples")

    return result


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        # generate() during eval runs without autocast; cast to match model dtype (float16)
        batch["input_features"] = batch["input_features"].to(torch.float16)

        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip leading BOS -- Whisper prepends it automatically during generate
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ── Model setup ────────────────────────────────────────────────────────────────

def setup_model(cfg: dict, run_dir: Path, resume: bool):
    import torch
    from transformers import WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel

    adapter_dir = run_dir / "adapter"

    if resume and (adapter_dir / "adapter_config.json").exists():
        print(f"\n  Resuming from {adapter_dir}")
        base  = WhisperForConditionalGeneration.from_pretrained(
            cfg["hf_model"], torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    else:
        print(f"\n  Loading base model: {cfg['hf_model']}")
        model = WhisperForConditionalGeneration.from_pretrained(
            cfg["hf_model"], torch_dtype=torch.float16,
        )
        lora_cfg = LoraConfig(
            # No task_type: creates base PeftModel (pass-through forward) instead of
            # PeftModelForSeq2SeqLM which injects input_ids=None into base_model kwargs.
            # In transformers 5.x, WhisperForConditionalGeneration.forward() has no
            # input_ids param, so it leaks through **kwargs to the decoder and causes
            # "got multiple values for input_ids". Pass-through avoids the injection.
            inference_mode=False,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    model.config.use_cache = False
    model.config.forced_decoder_ids = None   # loss is computed over all tokens
    model.config.suppress_tokens = []

    model.print_trainable_parameters()
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(lang: str, args: argparse.Namespace) -> Path:
    import torch
    from jiwer import wer as jiwer_wer
    from transformers import (
        WhisperProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        EarlyStoppingCallback,
    )

    cfg     = LANG_CONFIG[lang]
    run_dir = RUNS_DIR / lang
    run_dir.mkdir(exist_ok=True)
    adapter_dir = run_dir / "adapter"

    print(f"\n{'='*60}")
    print(f"  VANI Whisper LoRA Fine-Tune: {lang.upper()}")
    print(f"  Base model : {cfg['hf_model']}")
    print(f"  LoRA       : r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, target=q_proj+v_proj")
    print(f"  Steps      : {args.steps}  (save/eval every {args.save_steps})")
    print(f"  Eff. batch : {cfg['batch_size']} x {cfg['grad_accum']} = {cfg['batch_size']*cfg['grad_accum']}")
    print(f"  Output     : {run_dir}")
    print(f"{'='*60}")

    # Processor
    print("\n  Loading processor ...")
    try:
        processor = WhisperProcessor.from_pretrained(
            cfg["hf_model"], language=cfg["whisper_lang"], task=cfg["task"],
        )
    except Exception:
        # Community model may not have processor -- fall back to base arch
        base_arch = "openai/whisper-large-v3" if "large" in cfg["hf_model"] else "openai/whisper-medium"
        print(f"  [WARN] Processor not found in repo, using {base_arch}")
        processor = WhisperProcessor.from_pretrained(
            base_arch, language=cfg["whisper_lang"], task=cfg["task"],
        )

    # forced_decoder_ids for generation during eval (different from training config)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg["whisper_lang"], task=cfg["task"]
    )

    # Datasets
    token  = _hf_token()
    raw    = load_datasets(lang, cfg, not args.no_cv, token)

    print("\n  Preprocessing audio features ...")

    def prepare(batch):
        data = _decode_audio(batch["audio"], target_sr=16000)
        batch["input_features"] = processor.feature_extractor(
            data, sampling_rate=16000
        ).input_features[0]
        # Whisper decoder max length is 448; truncate to avoid shape errors
        batch["labels"] = processor.tokenizer(
            batch["text"], max_length=448, truncation=True
        ).input_ids
        return batch

    proc_kwargs = dict(
        function=prepare,
        remove_columns=["audio", "text"],
        num_proc=1,
        desc="features",
    )
    train_ds = raw["train"].map(**proc_kwargs)
    if "validation" in raw:
        eval_ds = raw["validation"].map(**proc_kwargs)
    else:
        eval_ds = train_ds.select(range(min(500, len(train_ds))))

    collator = DataCollator(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
    )

    # Metrics
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        # jiwer expects lists of strings
        score = 100.0 * jiwer_wer(label_str, pred_str)
        return {"wer": round(score, 2)}

    # Model
    model = setup_model(cfg, run_dir, args.resume)

    # Set forced_decoder_ids on generation config for eval generate() calls
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=args.steps,
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=max(1, cfg["batch_size"] // 2),
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        logging_steps=max(10, args.save_steps // 5),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=0,   # Windows: no fork-based multiprocessing
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    resume_from = str(adapter_dir) if args.resume and adapter_dir.exists() else None
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\n  [OK] Adapter saved -> {adapter_dir}")

    return adapter_dir


# ── Merge and CT2 conversion ───────────────────────────────────────────────────

def merge_and_convert(lang: str, adapter_dir: Path) -> bool:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    cfg        = LANG_CONFIG[lang]
    run_dir    = RUNS_DIR / lang
    merged_dir = run_dir / "merged"
    ct2_dir    = MODELS_DIR / cfg["ct2_name"]

    print(f"\n  [1/2] Merging LoRA adapter into base model ...")
    base   = WhisperForConditionalGeneration.from_pretrained(
        cfg["hf_model"], torch_dtype=torch.float16,
    )
    peft_m = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_m.merge_and_unload()
    merged.save_pretrained(str(merged_dir))

    processor = WhisperProcessor.from_pretrained(str(adapter_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"  [OK] Merged model -> {merged_dir}")

    print(f"\n  [2/2] Converting to CT2 int8 -> {ct2_dir}")
    if ct2_dir.exists():
        shutil.rmtree(ct2_dir)

    cmd = [
        _ct2_converter(),
        "--model",        str(merged_dir),
        "--output_dir",   str(ct2_dir),
        "--quantization", "int8",
        "--force",
    ]
    print(f"  Running: {' '.join(str(c) for c in cmd)}\n")
    ret = subprocess.run(cmd)

    if ret.returncode != 0:
        print(f"  [ERR] CT2 conversion failed (exit {ret.returncode})")
        return False

    # Verify output
    model_bin = ct2_dir / "model.bin"
    if not model_bin.exists():
        print(f"  [ERR] model.bin not found in {ct2_dir}")
        return False

    size_mb = model_bin.stat().st_size / 1048576
    print(f"\n  [OK] CT2 model saved -> {ct2_dir}")
    print(f"       model.bin: {size_mb:.0f} MB")

    # Write preprocessor_config.json so faster-whisper reads the correct
    # feature_size (n_mels). The CT2 converter omits this file, causing a
    # shape mismatch crash at runtime for large-v3 models (128 vs 80 bins).
    # WhisperProcessor.save_pretrained writes processor_config.json (nested),
    # so we extract the feature_extractor block and write the flat format.
    import json as _json
    proc_cfg_path = merged_dir / "processor_config.json"
    if proc_cfg_path.exists():
        proc_cfg = _json.loads(proc_cfg_path.read_text(encoding="utf-8"))
        feat_cfg = proc_cfg.get("feature_extractor", {})
        if feat_cfg:
            (ct2_dir / "preprocessor_config.json").write_text(
                _json.dumps(feat_cfg, indent=2), encoding="utf-8"
            )
            print(f"  [OK] preprocessor_config.json written -> {ct2_dir}")
    print(f"\n  Raw merged model kept at: {merged_dir}")
    print(f"  (Delete with: rmdir /s /q \"{merged_dir}\")")
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VANI Whisper LoRA fine-tuner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("lang",
        choices=list(LANG_CONFIG),
        help="Language code: pa (Punjabi) or ps (Pashto)")
    parser.add_argument("--steps",      type=int, default=1000,
        help="Training steps (default: 1000)")
    parser.add_argument("--save-steps", type=int, default=200,
        help="Checkpoint and eval every N steps (default: 200)")
    parser.add_argument("--no-cv",      action="store_true",
        help="Skip Common Voice, use FLEURS only (no HF token needed)")
    parser.add_argument("--resume",     action="store_true",
        help="Resume from last checkpoint in finetune_runs/<lang>/adapter/")
    parser.add_argument("--skip-ct2",   action="store_true",
        help="Save LoRA adapter only; skip merge + CT2 conversion")
    args = parser.parse_args()

    cfg = LANG_CONFIG[args.lang]

    print(f"\nVANI Whisper Fine-Tuner")
    print(f"Language : {args.lang.upper()}")
    print(f"Base     : {cfg['hf_model']}")
    print(f"CT2 out  : models/{cfg['ct2_name']}")

    adapter_dir = train(args.lang, args)

    if args.skip_ct2:
        print(f"\n[DONE] Adapter at: {adapter_dir}")
        print(f"       Convert later: python finetune_whisper.py {args.lang} --resume --steps 0")
    else:
        ok = merge_and_convert(args.lang, adapter_dir)
        if ok:
            print(f"\n{'='*60}")
            print(f"[DONE] Fine-tuned model ready:")
            print(f"       models/{cfg['ct2_name']}")
            print(f"\n       Restart VANI to load the updated model.")
            print(f"{'='*60}")
        else:
            print(f"\n[WARN] CT2 conversion failed. Adapter is at: {adapter_dir}")
            print(f"       Investigate the error and re-run with --resume --skip-ct2")


if __name__ == "__main__":
    main()

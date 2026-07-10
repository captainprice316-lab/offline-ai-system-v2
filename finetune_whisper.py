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
        "hf_model":              "openai/whisper-large-v3",
        "whisper_lang":          "pa",
        "task":                  "transcribe",
        "fleurs_config":         "pa_in",
        "cv_config":             "pa-IN",
        "indicvoices_config":    "Punjabi",
        "indicvoices_train_cap": 20000,   # v3: doubled from 10k (25.7k available)
        "ct2_name":              "whisper-large-v3-pa-ct2",
        "lora_r":                16,      # v3: doubled from 8 for Gurmukhi complexity
        "lora_alpha":            32,
        "lora_dropout":          0.05,
        "batch_size":            2,
        "grad_accum":            1,       # no accumulation: ~19s/step vs 150s at accum=8
        "learning_rate":         5e-5,
        "warmup_steps":          100,     # v3: longer warmup for r=16
        "max_grad_norm":         0.5,     # prevent fp16 gradient spikes (seen in zh)
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
        "hf_model":              "openai/whisper-large-v3",
        "whisper_lang":          "ne",
        "task":                  "transcribe",
        "fleurs_config":         "ne_np",
        "cv_config":             "ne-NP",
        "indicvoices_config":    "Nepali",
        "indicvoices_train_cap": 10000,
        "ct2_name":              "whisper-large-v3-ne-ct2",
        "lora_r":                8,
        "lora_alpha":            16,
        "lora_dropout":          0.05,
        "batch_size":            2,
        "grad_accum":            1,
        "learning_rate":         5e-5,
        "warmup_steps":          50,
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
    "hi": {
        "hf_model":      "openai/whisper-large-v3",
        "whisper_lang":  "hi",
        "task":          "transcribe",
        "fleurs_config": "hi_in",
        "cv_config":     "hi",
        "ct2_name":      "whisper-large-v3-hi-ct2",
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "batch_size":    2,
        "grad_accum":    1,
        "learning_rate": 5e-5,
        "warmup_steps":  50,
        "max_grad_norm": 0.5,   # prevent fp16 gradient spike (seen in zh at step ~820)
    },
    "ks": {
        # Base is the vocab-patched local model (run scripts/add_ks_token.py first)
        "hf_model":                "models/whisper-large-v3-ks-base",
        "whisper_lang":            "ks",   # custom token added by add_ks_token.py
        "task":                    "transcribe",
        "fleurs_config":           None,   # FLEURS has no Kashmiri config
        "cv_config":               None,
        # Local parquet files already downloaded to E:\VANI\datasets\hf_ks_temp
        "indicvoices_parquet_dir": r"E:\VANI\datasets\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r\snapshots\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri",
        "indicvoices_train_cap":   20000,
        "ct2_name":                "whisper-large-v3-ks-ct2",
        "lora_r":                  8,
        "lora_alpha":              16,
        "lora_dropout":            0.05,
        "batch_size":              2,
        "grad_accum":              1,
        "learning_rate":           5e-5,
        "warmup_steps":            50,
        "max_grad_norm":           0.5,
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


def _resolve_model(cfg: dict) -> str:
    """Return absolute path for local models; pass through HF hub IDs unchanged."""
    path = cfg["hf_model"]
    if "/" in path and not path.startswith("models/"):
        return path  # HF hub ID like "openai/whisper-large-v3"
    return str(ROOT / path)


def _setup_ks_prefix(processor):
    """
    Wire up the tokenizer post-processor to emit the Kashmiri prefix
    [<|startoftranscript|> <|ks|> <|transcribe|> <|notimestamps|>] on every
    tokenize() call, and return the corresponding forced_decoder_ids for eval.

    This bypasses WhisperTokenizer.prefix_tokens (which computes language IDs
    as fixed offsets from bos_id and cannot handle the new out-of-range <|ks|>
    token) by directly setting backend_tokenizer.post_processor.
    """
    from tokenizers import processors as hf_processors

    tok = processor.tokenizer
    bos_id          = tok.convert_tokens_to_ids("<|startoftranscript|>")
    ks_id           = tok.convert_tokens_to_ids("<|ks|>")
    transcribe_id   = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
    eos_id          = tok.eos_token_id
    eos_str         = tok.eos_token

    assert ks_id != tok.unk_token_id, (
        "<|ks|> not in tokenizer — run `python scripts/add_ks_token.py` first"
    )

    bos_str = "<|startoftranscript|>"
    ks_str  = "<|ks|>"
    tr_str  = "<|transcribe|>"
    nts_str = "<|notimestamps|>"
    tmpl    = f"{bos_str}:0 {ks_str}:0 {tr_str}:0 {nts_str}:0"

    tok.backend_tokenizer.post_processor = hf_processors.TemplateProcessing(
        single=f"{tmpl} $A:0 {eos_str}:0",
        pair=f"{tmpl} $A:0 $B:1 {eos_str}:1",
        special_tokens=[
            (eos_str, eos_id),
            (bos_str, bos_id),
            (ks_str,  ks_id),
            (tr_str,  transcribe_id),
            (nts_str, notimestamps_id),
        ],
    )

    print(f"  <|ks|> token ID : {ks_id}")
    print(f"  Label prefix    : [<|startoftranscript|>({bos_id}), <|ks|>({ks_id}), "
          f"<|transcribe|>({transcribe_id}), <|notimestamps|>({notimestamps_id})]")

    # forced_decoder_ids for eval generate(): positions 1,2,3 after bos
    forced_decoder_ids = [(1, ks_id), (2, transcribe_id), (3, notimestamps_id)]
    return forced_decoder_ids


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

    # ── IndicVoices-R local parquet (Kashmiri) ───────────────────────────────────
    # Uses IterableDataset (streaming) for train to avoid Windows file-locking
    # that occurs when HF datasets tries to finalize Arrow shard temp files.
    # Val is small enough (372 samples) to load into a regular in-memory Dataset.
    if cfg.get("indicvoices_parquet_dir"):
        import pyarrow.parquet as pq
        from datasets import Dataset, IterableDataset as HFIterableDataset, Features, Value

        parquet_dir = Path(cfg["indicvoices_parquet_dir"])
        train_files = sorted(parquet_dir.glob("train-*.parquet"))
        test_files  = sorted(parquet_dir.glob("test-*.parquet"))
        min_dur, max_dur = 2.0, 20.0
        train_cap = cfg.get("indicvoices_train_cap", 20000)

        audio_features = Features({
            "audio": {"bytes": Value("binary"), "path": Value("string")},
            "text":  Value("string"),
        })

        def _ks_gen(pq_files, max_samples):
            count = 0
            for pq_file in pq_files:
                if count >= max_samples:
                    break
                try:
                    t   = pq.read_table(pq_file, columns=["audio", "normalized", "duration"])
                    raw = t.to_pydict()
                    del t
                    for audio, text, dur in zip(raw["audio"], raw["normalized"], raw["duration"]):
                        if count >= max_samples:
                            break
                        if dur is None or not (min_dur <= dur <= max_dur) or not text:
                            continue
                        yield {"audio": audio, "text": text}
                        count += 1
                    del raw
                except Exception as gen_e:
                    print(f"         [WARN] {Path(pq_file).name}: {gen_e}")

        # Train: IterableDataset — no Arrow temp files, no Windows file locks
        if train_files:
            try:
                train_iterable = HFIterableDataset.from_generator(
                    _ks_gen,
                    gen_kwargs={"pq_files": train_files, "max_samples": train_cap},
                    features=audio_features,
                ).shuffle(seed=42, buffer_size=3000)
                splits["train"].append(train_iterable)
                print(f"\n  [data] IndicVoices-R KS train: {train_cap} samples (streaming, no cache)")
            except Exception as e:
                print(f"         [WARN] KS train IterableDataset: {e}")

        # Validation: 372 samples fit in RAM — use regular Dataset
        if test_files:
            try:
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
                    {"audio": val_audio, "text": val_text},
                    features=audio_features,
                )
                splits["validation"].append(val_ds)
                print(f"  IndicVoices-R KS val: {len(val_ds)} samples")
            except Exception as e:
                print(f"         [WARN] KS val Dataset: {e}")

    # ── Custom dataset (e.g. IndicVoices for Kashmiri) ────────────────────────
    if cfg.get("custom_dataset"):
        print(f"\n  [data] Loading custom dataset: {cfg['custom_dataset']} ...")
        for split in ("train", "validation"):
            try:
                ds = load_dataset(
                    cfg["custom_dataset"],
                    split=split,
                    token=token,
                    cache_dir=str(data_dir / "custom"),
                )
                # audio is in audio_filepath (Audio feature); decode=False avoids torchcodec
                ds = ds.cast_column("audio_filepath", Audio(decode=False))
                # filter by duration (2–20 s), drop corrupt entries
                before = len(ds)
                ds = ds.filter(lambda x: x["duration"] is not None and 2.0 <= x["duration"] <= 20.0)
                print(f"         duration filter: {before:,} -> {len(ds):,} samples")
                # keep only needed columns, normalise names to match prepare()
                # drop existing 'text' column then use 'normalized' (cleaner for ASR)
                ds = ds.remove_columns(["text"])
                ds = ds.rename_column("audio_filepath", "audio")
                ds = ds.rename_column("normalized", "text")
                ds = ds.select_columns(["audio", "text"])
                # subsample to match FLEURS-scale sizes (fast eval, reasonable train time)
                train_cap = cfg.get("train_samples", 3000)
                if split == "train" and len(ds) > train_cap:
                    ds = ds.shuffle(seed=42).select(range(train_cap))
                elif split == "validation" and len(ds) > 300:
                    ds = ds.shuffle(seed=42).select(range(300))
                splits[split].append(ds)
                print(f"         custom {split}: {len(ds):,} samples")
            except Exception as e:
                print(f"         [WARN] custom {split}: {e}")

    # ── FLEURS -- no token required ───────────────────────────────────────────
    if cfg.get("fleurs_config"):
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

    # ── IndicVoices-R (ai4bharat/indicvoices_r) ───────────────────────────────
    if cfg.get("indicvoices_config"):
        iv_lang = cfg["indicvoices_config"]
        print(f"\n  [data] Loading indicvoices_r ({iv_lang}) ...")
        # dataset only has train + test splits; use test as extra validation
        for hf_split, out_split in [("train", "train"), ("test", "validation")]:
            try:
                ds = load_dataset(
                    "ai4bharat/indicvoices_r",
                    iv_lang,
                    split=hf_split,
                    token=token,
                    cache_dir=str(data_dir / "indicvoices_r"),
                )
                ds = ds.cast_column("audio", Audio(decode=False))
                before = len(ds)
                ds = ds.filter(
                    lambda x: x["duration"] is not None and 2.0 <= float(x["duration"]) <= 20.0
                )
                print(f"         duration filter: {before:,} -> {len(ds):,}")
                # drop existing text column; use normalized (cleaner for ASR)
                ds = ds.remove_columns([c for c in ds.column_names if c not in ("audio", "normalized")])
                ds = ds.rename_column("normalized", "text")
                cap = cfg.get("indicvoices_train_cap", 10000)
                if hf_split == "train" and len(ds) > cap:
                    ds = ds.shuffle(seed=42).select(range(cap))
                splits[out_split].append(ds)
                print(f"         indicvoices_r {out_split}: {len(ds):,} samples")
            except Exception as e:
                print(f"         [WARN] indicvoices_r {hf_split}: {e}")

    # ── Common Voice -- token required ────────────────────────────────────────
    if use_cv and cfg.get("cv_config"):
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

    from datasets import IterableDataset as HFIterableDataset

    result = {}
    for split, ds_list in splits.items():
        if not ds_list:
            continue
        if len(ds_list) == 1:
            ds = ds_list[0]
        else:
            ds = concatenate_datasets(ds_list)
        # IterableDataset handles its own shuffling (buffer_size already set);
        # only shuffle regular datasets here.
        if not isinstance(ds, HFIterableDataset):
            ds = ds.shuffle(seed=42)
        result[split] = ds
        n_str = f"{len(ds):,}" if hasattr(ds, "__len__") else "streaming"
        print(f"\n  [data] Total {split}: {n_str} samples")

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

    adapter_dir   = run_dir / "adapter"
    hf_model_path = _resolve_model(cfg)

    if resume and (adapter_dir / "adapter_config.json").exists():
        print(f"\n  Resuming from {adapter_dir}")
        base  = WhisperForConditionalGeneration.from_pretrained(
            hf_model_path, torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    else:
        print(f"\n  Loading base model: {hf_model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(
            hf_model_path, torch_dtype=torch.float16,
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
        TrainerCallback,
    )

    class EmptyCacheCallback(TrainerCallback):
        """Free CUDA memory before each eval to avoid OOM from fragmentation
        on small-VRAM GPUs (8 GB RTX 5060). Added after PA v3 OOM'd at step 2400."""
        def on_evaluate(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
        def on_step_end(self, args, state, control, **kwargs):
            # clear right before the eval step fires
            if args.eval_steps and state.global_step % args.eval_steps == 0:
                torch.cuda.empty_cache()

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
    hf_model_path = _resolve_model(cfg)
    whisper_lang  = cfg.get("whisper_lang")   # None = no forced language token

    if lang == "ks":
        # Kashmiri uses a custom <|ks|> token added by scripts/add_ks_token.py.
        # WhisperTokenizer.prefix_tokens computes language IDs as fixed offsets
        # from bos_id (positions 50259-50358) so it cannot handle a new token at
        # ID 51866.  We load the processor without a language arg and set up the
        # backend post-processor manually via _setup_ks_prefix().
        processor = WhisperProcessor.from_pretrained(hf_model_path)
        forced_decoder_ids = _setup_ks_prefix(processor)
    else:
        try:
            processor = WhisperProcessor.from_pretrained(
                hf_model_path, language=whisper_lang, task=cfg["task"],
            )
        except Exception:
            # Community model may not have processor -- fall back to base arch
            base_arch = "openai/whisper-large-v3" if "large" in cfg["hf_model"] else "openai/whisper-medium"
            print(f"  [WARN] Processor not found in repo, using {base_arch}")
            processor = WhisperProcessor.from_pretrained(
                base_arch, language=whisper_lang, task=cfg["task"],
            )

        # forced_decoder_ids for generation during eval (None when whisper_lang unset)
        if whisper_lang:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=whisper_lang, task=cfg["task"]
            )
        else:
            forced_decoder_ids = None
            print("  [INFO] No forced language token — model will auto-detect language from audio")

    # Datasets
    token  = _hf_token()
    raw    = load_datasets(lang, cfg, not args.no_cv, token)

    print("\n  Preprocessing audio features ...")

    from datasets import IterableDataset as HFIterableDataset
    is_iterable_train = isinstance(raw.get("train"), HFIterableDataset)

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
    # IterableDataset.map() doesn't accept num_proc or desc
    iterable_proc_kwargs = dict(function=prepare, remove_columns=["audio", "text"])

    # map() writes its Arrow feature cache next to the source unless cache_file_name is
    # explicit; keep it on E: alongside the datasets rather than filling the OS drive.
    _map_cache = Path("E:/VANI/datasets/ds_map_cache")
    _map_cache.mkdir(parents=True, exist_ok=True)

    if is_iterable_train:
        # Lazy map — prepare() runs per-batch during training (no upfront cache)
        train_ds = raw["train"].map(**iterable_proc_kwargs)
        print("  [INFO] Train dataset: streaming (IterableDataset), features computed on-the-fly")
    else:
        train_ds = raw["train"].map(
            **proc_kwargs,
            cache_file_name=str(_map_cache / f"{lang}_train_features.arrow"),
        )

    if args.no_eval:
        eval_ds = None
    elif "validation" in raw:
        eval_ds = raw["validation"].map(
            **proc_kwargs,
            cache_file_name=str(_map_cache / f"{lang}_eval_features.arrow"),
        )
    elif not is_iterable_train:
        eval_ds = train_ds.select(range(min(500, len(train_ds))))
    else:
        eval_ds = None  # no eval without a validation split

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
    model.generation_config.num_beams = 1   # greedy eval decoding (low VRAM)

    has_eval = eval_ds is not None
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=args.steps,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps" if has_eval else "no",
        per_device_eval_batch_size=1,          # smallest eval batch (8 GB VRAM)
        predict_with_generate=has_eval,
        generation_max_length=225,
        generation_num_beams=1,                # greedy eval decoding — lowers eval VRAM,
                                               # prevents the OOM that halted PA v3 at step 2400
        eval_accumulation_steps=1,             # offload eval preds to CPU each step
        save_steps=args.save_steps,
        eval_steps=args.save_steps if has_eval else None,
        logging_steps=max(10, args.save_steps // 5),
        report_to="none",
        load_best_model_at_end=has_eval,
        metric_for_best_model="wer" if has_eval else None,
        greater_is_better=False if has_eval else None,
        push_to_hub=False,
        dataloader_num_workers=0,   # Windows: no fork-based multiprocessing
        remove_unused_columns=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"]    = eval_ds
        trainer_kwargs["compute_metrics"] = compute_metrics
        trainer_kwargs["callbacks"]       = [EarlyStoppingCallback(early_stopping_patience=3),
                                             EmptyCacheCallback()]

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    if args.resume and adapter_dir.exists():
        if (adapter_dir / "trainer_state.json").exists():
            resume_from = str(adapter_dir)
        else:
            ckpts = sorted(
                [d for d in adapter_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[1]),
            )
            resume_from = str(ckpts[-1]) if ckpts else str(adapter_dir)
    else:
        resume_from = None
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
        _resolve_model(cfg), torch_dtype=torch.float16,
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

    # Copy tokenizer.json — ct2-transformers-converter does not include it,
    # but faster-whisper needs it to look up language/task token IDs correctly.
    # Without it, faster-whisper falls back to openai/whisper-tiny whose token
    # ordering differs from large-v3 (transcribe=50359 vs 50360), causing the
    # model to translate instead of transcribe for every non-English language.
    import json as _json
    tok_src = merged_dir / "tokenizer.json"
    if tok_src.exists():
        import shutil as _shutil
        _shutil.copy2(str(tok_src), str(ct2_dir / "tokenizer.json"))
        print(f"  [OK] tokenizer.json copied -> {ct2_dir}")
    else:
        print(f"  [WARN] tokenizer.json not found in {merged_dir}; faster-whisper may use wrong token IDs")

    # Write preprocessor_config.json so faster-whisper reads the correct
    # feature_size (n_mels). The CT2 converter omits this file, causing a
    # shape mismatch crash at runtime for large-v3 models (128 vs 80 bins).
    # WhisperProcessor.save_pretrained writes processor_config.json (nested),
    # so we extract the feature_extractor block and write the flat format.
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
        help="Language code: pa, ps, ur, ne, zh, hi, ks")
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
    parser.add_argument("--no-eval",    action="store_true",
        help="Disable evaluation during training (speeds up runs with slow generate())")
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

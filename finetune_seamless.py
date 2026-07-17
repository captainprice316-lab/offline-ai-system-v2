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

    feats = Features({
        "audio":         {"bytes": Value("binary"), "path": Value("string")},
        "transcription": Value("string"),
    })

    def _gen(n_fleurs, n_cv):
        order = [("f", i) for i in range(n_fleurs)] + [("c", i) for i in range(n_cv)]
        random.Random(42).shuffle(order)
        for src, i in order:
            row = fleurs_train[i] if src == "f" else cv_train[i]
            audio = row["audio"]
            yield {
                "audio": {"bytes": audio.get("bytes"), "path": audio.get("path")},
                "transcription": row["transcription"],
            }

    train_ds = HFIterableDataset.from_generator(
        _gen, gen_kwargs={"n_fleurs": len(fleurs_train), "n_cv": len(cv_train)},
        features=feats,
    )
    total = len(fleurs_train) + len(cv_train)
    print(f"\n  [data] Combined train: {total:,} samples "
          f"(FLEURS {len(fleurs_train):,} + CV {len(cv_train):,}, streaming, pre-shuffled)"
          f"  Val: {len(fleurs_val)} (FLEURS only)")
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
        count = 0
        for pq_file in pq_files:
            if count >= max_samples:
                break
            t   = pq.read_table(pq_file, columns=["audio", "normalized", "duration"])
            raw = t.to_pydict()
            del t
            for audio, text, dur in zip(raw["audio"], raw["normalized"], raw["duration"]):
                if count >= max_samples:
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


def add_kas_token(processor, model):
    """Add __kas__ to the tokenizer and model embeddings, initialised from
    __urd__. Embeddings stay FROZEN under LoRA — the urd-initialised prefix is
    a fixed conditioning vector, exactly as Whisper's <|ks|> was (74.02% WER).
    Also registers kas in the generation config so post-training generate()
    with tgt_lang='kas' works."""
    import torch as _torch
    tok = processor.tokenizer
    if tok.convert_tokens_to_ids("__kas__") != tok.unk_token_id and "__kas__" in tok.get_vocab():
        print("  [kas] __kas__ already in tokenizer")
    else:
        tok.add_tokens(["__kas__"], special_tokens=True)
        print(f"  [kas] __kas__ added to tokenizer (vocab {len(tok)})")

    kas_id = tok.convert_tokens_to_ids("__kas__")
    urd_id = tok.convert_tokens_to_ids("__urd__")
    assert kas_id != tok.unk_token_id and urd_id != tok.unk_token_id

    if model.get_input_embeddings().weight.shape[0] < len(tok):
        model.resize_token_embeddings(len(tok))
    with _torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        in_emb[kas_id] = in_emb[urd_id].clone()
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != in_emb.data_ptr():
            out_emb.weight[kas_id] = out_emb.weight[urd_id].clone()
    print(f"  [kas] embedding row {kas_id} initialised from __urd__ ({urd_id})")

    gc = model.generation_config
    lang_map = getattr(gc, "text_decoder_lang_to_code_id", None)
    if lang_map is not None:
        try:
            lang_map["kas"] = kas_id
        except TypeError:
            pass  # non-dict mapping; eval must pass the bos id explicitly
    return kas_id


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
    if sm_lang == "kas":
        add_kas_token(processor, model)

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
    if cfg.get("indicvoices_parquet_dir"):
        train_raw, val_raw = load_indicvoices_ks(cfg)
    elif cfg.get("cv_dataset"):
        train_raw, val_raw = load_fleurs_plus_cv(lang)
    elif cfg.get("indicvoices_config"):
        train_raw, val_raw = load_fleurs_plus_indicvoices(lang)
    else:
        train_raw, val_raw = load_fleurs(lang)

    preprocess = make_preprocess_fn(processor, sm_lang)

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
        preprocess,
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
    EXPERIMENTS = {"ks", "ps_cv", "hi_iv", "ne_iv"}
    langs = [l for l in LANG_CFG if l not in EXPERIMENTS] if args.lang == "all" else [args.lang]
    for lang in langs:
        train(lang, args)


if __name__ == "__main__":
    main()

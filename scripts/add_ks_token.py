#!/usr/bin/env python3
"""
add_ks_token.py — Patch whisper-large-v3 to add a <|ks|> (Kashmiri) language token.

Whisper has no native Kashmiri token. This script:
  1. Adds <|ks|> to the tokenizer's additional_special_tokens
  2. Expands the model's embedding matrix (embed_tokens + proj_out) to match
  3. Initialises the new <|ks|> row from <|ur|> (Urdu — same Nastaliq script)
  4. Saves the patched model + processor to models/whisper-large-v3-ks-base/

Run once before fine-tuning:
    python scripts/add_ks_token.py

Then fine-tune with:
    python finetune_whisper.py ks
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT    = Path(__file__).resolve().parents[1]
SRC     = "openai/whisper-large-v3"
OUT_DIR = ROOT / "models" / "whisper-large-v3-ks-base"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS_TOKEN = "<|ks|>"

print(f"Source : {SRC}")
print(f"Output : {OUT_DIR}")
print()

# ── Load processor ─────────────────────────────────────────────────────────────
print("Loading processor ...")
processor = WhisperProcessor.from_pretrained(SRC)
tokenizer = processor.tokenizer

# ── Check / add <|ks|> ────────────────────────────────────────────────────────
vocab = tokenizer.get_vocab()
if KS_TOKEN in vocab:
    ks_id = vocab[KS_TOKEN]
    print(f"<|ks|> already present at ID {ks_id}.")
    model_exists = OUT_DIR / "model.safetensors"
    if model_exists.exists():
        print(f"Patched model already at {OUT_DIR} — nothing to do.")
        sys.exit(0)
    print("Model not saved yet — will proceed with saved processor but reload model.")
    need_token = False
else:
    ur_id = tokenizer.convert_tokens_to_ids("<|ur|>")
    print(f"<|ur|> ID      : {ur_id}")
    print(f"Current vocab  : {len(tokenizer)}")

    n_added = tokenizer.add_tokens([KS_TOKEN], special_tokens=True)
    ks_id   = tokenizer.convert_tokens_to_ids(KS_TOKEN)
    print(f"\n<|ks|> added   : ID {ks_id}  (vocab now {len(tokenizer)},  {n_added} token added)")
    need_token = True

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\nLoading model from {SRC} ...")
model = WhisperForConditionalGeneration.from_pretrained(SRC, torch_dtype=torch.float16)
old_vocab = model.model.decoder.embed_tokens.weight.shape[0]
print(f"Embedding matrix before: {old_vocab}")

model.resize_token_embeddings(len(tokenizer))
new_vocab = model.model.decoder.embed_tokens.weight.shape[0]
print(f"Embedding matrix after : {new_vocab}")

# ── Init new embedding from <|ur|> ─────────────────────────────────────────────
if need_token:
    ur_id = tokenizer.convert_tokens_to_ids("<|ur|>")
    with torch.no_grad():
        ur_emb = model.model.decoder.embed_tokens.weight[ur_id].clone()
        model.model.decoder.embed_tokens.weight[ks_id] = ur_emb
        print(f"\nembed_tokens[{ks_id}] <- embed_tokens[{ur_id}]  (<|ks|> <- <|ur|>)")

        # proj_out is Whisper's output projection (may or may not share storage with embed_tokens)
        proj_out = model.get_output_embeddings()
        if proj_out is not None and proj_out.weight.data_ptr() != model.model.decoder.embed_tokens.weight.data_ptr():
            ur_proj = proj_out.weight[ur_id].clone()
            proj_out.weight[ks_id] = ur_proj
            print(f"proj_out.weight[{ks_id}] <- proj_out.weight[{ur_id}]  (not tied, copied separately)")
        else:
            print("proj_out is tied to embed_tokens — no separate copy needed")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving patched model -> {OUT_DIR}")
model.save_pretrained(str(OUT_DIR))
processor.save_pretrained(str(OUT_DIR))
print("[OK] Saved.")

# ── Sanity checks ──────────────────────────────────────────────────────────────
print("\n--- Sanity checks ---")
check_proc = WhisperProcessor.from_pretrained(str(OUT_DIR))
check_tok  = check_proc.tokenizer

ks_id_check = check_tok.convert_tokens_to_ids(KS_TOKEN)
ur_id_check = check_tok.convert_tokens_to_ids("<|ur|>")
unk_id      = check_tok.unk_token_id

print(f"  Vocab size : {len(check_tok)}")
print(f"  <|ks|> ID  : {ks_id_check}")
print(f"  <|ur|> ID  : {ur_id_check}")
print(f"  unk_token  : {unk_id}")

assert ks_id_check != unk_id,  "<|ks|> resolved to <unk> — token not added!"
assert ks_id_check != ur_id_check, "<|ks|> and <|ur|> have the same ID — something went wrong"

encoded = check_tok.encode(KS_TOKEN, add_special_tokens=False)
decoded = check_tok.decode(encoded)
print(f"  encode('{KS_TOKEN}') -> {encoded}")
print(f"  decode({encoded})    -> '{decoded}'")
assert decoded.strip() == KS_TOKEN, f"Round-trip failed: got '{decoded}'"

print(f"\n[OK] Patched model ready.")
print(f"     <|ks|> = token ID {ks_id_check}")
print(f"     Location: {OUT_DIR}")
print(f"\nNext step:")
print(f"     python finetune_whisper.py ks --steps 3000")

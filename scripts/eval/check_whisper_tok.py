import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from transformers import WhisperProcessor, WhisperTokenizerFast
tok = WhisperProcessor.from_pretrained("openai/whisper-large-v3").tokenizer
print("Type:", type(tok).__name__)
print("Has additional_special_tokens:", hasattr(tok, "additional_special_tokens"))

# Try different ways to add a token
print("\nTrying add_tokens with special_tokens=True:")
n = tok.add_tokens(["<|ks|>"], special_tokens=True)
print(f"  n added: {n}")
ks_id = tok.convert_tokens_to_ids("<|ks|>")
print(f"  <|ks|> ID: {ks_id}")
print(f"  Vocab size now: {len(tok)}")

# Verify round-trip
enc = tok.encode("<|ks|>", add_special_tokens=False)
dec = tok.decode(enc)
print(f"  encode: {enc}")
print(f"  decode: '{dec}'")

# -*- coding: utf-8 -*-
"""gen_nbest_ks.py — dump n-best hypotheses (+ acoustic scores) for Kashmiri.

Step 2 of LM rescoring. The deployed evaluation decodes GREEDILY (num_beams=1),
so beam search alone shifts the baseline; we therefore persist the full n-best
list and its scores, letting the rescoring step report three numbers:

    greedy (deployed)  |  beam 1-best  |  beam + LM rescoring

which separates "beam search helped" from "the LM helped".

Decode settings are copied from eval_ks_seamless.py so the comparison is
apples-to-apples: min_new_tokens from --min-tok-per-sec, no_repeat_ngram_size.

Output: eval_data/ks_cloud3_nbest.jsonl — one row per clip:
    {"idx", "ref", "cands":[{"text", "am": <raw sum log-prob>, "ntok"}...]}

Usage:
    python scripts/lm/gen_nbest_ks.py --beams 8 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

BASE_DIR    = ROOT / "models" / "seamless-m4t-v2-large"
ADAPTER_DIR = ROOT / "finetune_runs_seamless" / "ks_cloud3" / "adapter"
OUT_JSONL   = ROOT / "eval_data" / "ks_cloud3_nbest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-tok-per-sec", type=float, default=2.5)
    ap.add_argument("--min-tok-cap", type=int, default=180)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    ap.add_argument("--out", default=str(OUT_JSONL))
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel
    from eval_ks_seamless import load_indicvoices_ks, LANG_CFG, decode_audio

    adapter = pathlib.Path(args.adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {adapter} on {device}, beams={args.beams}")

    processor = AutoProcessor.from_pretrained(str(adapter))
    tok = processor.tokenizer
    kas_id = tok.convert_tokens_to_ids("__kas__")
    assert kas_id != tok.unk_token_id, "__kas__ missing from adapter tokenizer"

    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    if base.get_input_embeddings().weight.shape[0] < len(tok):
        base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()

    gc = model.generation_config
    lang_map = getattr(gc, "text_decoder_lang_to_code_id", None)
    if lang_map is not None:
        try:
            lang_map["kas"] = kas_id
        except TypeError:
            pass

    def nbest(arr):
        gen_kwargs = {}
        if args.min_tok_per_sec > 0:
            dur = len(arr) / 16000.0
            gen_kwargs["min_new_tokens"] = min(
                args.min_tok_cap, max(5, int(dur * args.min_tok_per_sec)))
        if args.no_repeat_ngram > 0:
            gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram
        feat = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            out = model.generate(**feat, tgt_lang="kas",
                                 num_beams=args.beams,
                                 num_return_sequences=args.beams,
                                 length_penalty=1.0,
                                 max_new_tokens=200,
                                 output_scores=True,
                                 return_dict_in_generate=True,
                                 **gen_kwargs)
        cands = []
        for seq, sc in zip(out.sequences, out.sequences_scores.tolist()):
            ntok = int((seq != tok.pad_token_id).sum().item())
            # HF returns sum_logprob / (len ** length_penalty); recover the raw sum
            # so the acoustic term is on the same (unnormalised) footing as the LM.
            cands.append({"text": processor.decode(seq, skip_special_tokens=True).strip(),
                          "am": sc * ntok, "ntok": ntok})
        return cands

    _, val_ds = load_indicvoices_ks(LANG_CFG["ks"])
    n = len(val_ds) if args.limit is None else min(args.limit, len(val_ds))
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(exist_ok=True)
    print(f"[gen] {n} clips -> {outp}")
    with open(outp, "w", encoding="utf-8") as fh:
        for i in range(n):
            s = val_ds[i]
            cands = nbest(decode_audio(s["audio"]))
            fh.write(json.dumps({"idx": i, "ref": s["transcription"], "cands": cands},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"       {i+1}/{n}", flush=True)
    print(f"[done] wrote {outp}")


if __name__ == "__main__":
    main()

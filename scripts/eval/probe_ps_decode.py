"""Decode-parameter probe for the ps_bal SeamlessM4T adapter (no retraining).

ps_bal scored 39.72 WER greedy (n=100 held-out) vs Whisper-medium's 38.55 —
a 1.17 pp gap. Beams/length have never been tuned for Pashto ASR; this probes
whether decode settings alone bridge it, on the SAME 100 FLEURS test samples
and the SAME normaliser as the eval, so numbers are directly comparable.

  A  greedy               (sanity check — must reproduce ~39.72)
  B  beam 5
  C  beam 5, length_penalty 1.2
  D  beam 5, length_penalty 0.8

Usage:  python scripts/eval/probe_ps_decode.py [--samples 100]
"""
import argparse
import io
import json
import pathlib
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_seamless_ft import (  # noqa: E402
    LANG_CFG, SEAMLESS_DIR, RUNS_DIR, load_test_samples, decode_audio,
)
from text_norm import compute_wer, compute_cer  # noqa: E402

ADAPTER_NAME = "ps_bal"   # overridden by --adapter

CONDITIONS = {
    "A_greedy":   dict(num_beams=1),
    "B_beam5":    dict(num_beams=5),
    "C_beam5_lp12": dict(num_beams=5, length_penalty=1.2),
    "D_beam5_lp08": dict(num_beams=5, length_penalty=0.8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--adapter", default=ADAPTER_NAME,
                    help="finetune_runs_seamless subdir (e.g. ps_bal2)")
    args = ap.parse_args()

    adapter_dir = RUNS_DIR / args.adapter / "adapter"
    out_json    = ROOT / "docs" / f"{args.adapter}_decode_probe.json"

    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm_lang = LANG_CFG["ps_bal"]["sm_lang"]

    print(f"[model] loading {args.adapter} adapter on {device} ...")
    processor = AutoProcessor.from_pretrained(str(SEAMLESS_DIR))
    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(SEAMLESS_DIR),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model = model.to(device).eval()

    print(f"[data] loading {args.samples} FLEURS ps test samples (same seed as eval) ...")
    samples = load_test_samples(LANG_CFG["ps_bal"]["fleurs"], args.samples)
    refs = [s["transcription"] for s in samples]
    arrs = [decode_audio(s["audio"]) for s in samples]

    results = {}
    for name, cond in CONDITIONS.items():
        preds = []
        t0 = time.time()
        for i, arr in enumerate(arrs):
            inputs = processor(audio=arr, return_tensors="pt",
                               sampling_rate=16000, src_lang=sm_lang).to(device)
            with torch.no_grad():
                toks = model.generate(**inputs, tgt_lang=sm_lang, **cond)
            preds.append(processor.decode(toks[0], skip_special_tokens=True).strip())
            if (i + 1) % 25 == 0:
                print(f"  [{name}] {i+1}/{len(arrs)}")
        results[name] = {
            "wer": compute_wer(preds, refs, "ps"),
            "cer": compute_cer(preds, refs, "ps"),
            "time": round(time.time() - t0, 1),
            "n": len(preds),
        }
        print(f"[done] {name}: WER {results[name]['wer']}%  CER {results[name]['cer']}%")

    results["_reference"] = {
        "adapter": args.adapter,
        "ps_bal_greedy_eval": 39.72,
        "ps_bal_best_decode": 38.88,
        "whisper_medium_deployed": 38.55,
        "note": "Same 100 FLEURS ps_af test samples and text_norm normaliser as "
                "eval_seamless_ft. If any condition beats 38.55, run the "
                "5-condition degradation sweep before any routing change.",
    }
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()

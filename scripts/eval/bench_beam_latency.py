# -*- coding: utf-8 -*-
"""bench_beam_latency.py — what does beam search actually cost VANI?

Beam-8 decoding is worth 2.88 pp WER on Kashmiri (50.26 -> 47.37) but multiplies
decode work. This times the ASR generate() call — the only stage beam search
changes — at several beam widths on real test audio, using the deployed adapter
and the production decode kwargs from src/seamless_asr.py::_gen_kwargs.

Reports per-clip latency and REAL-TIME FACTOR (decode seconds per second of
audio). RTF < 1 means the stage keeps up with live radio; the intercept pipeline
runs several stages, so the ASR share is what matters, not the absolute number.

Usage:
    python scripts/eval/bench_beam_latency.py --clips 20 --beams 1 2 4 8
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

BASE_DIR = ROOT / "models" / "seamless-m4t-v2-large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--adapter-dir",
                    default=str(ROOT / "finetune_runs_seamless" / "ks_cloud3" / "adapter"))
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel
    from eval_ks_seamless import load_indicvoices_ks, LANG_CFG, decode_audio

    adapter = pathlib.Path(args.adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(str(adapter))
    tok = processor.tokenizer
    kas_id = tok.convert_tokens_to_ids("__kas__")

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

    _, val_ds = load_indicvoices_ks(LANG_CFG["ks"])
    clips = []
    for i in range(min(args.clips, len(val_ds))):
        arr = decode_audio(val_ds[i]["audio"])
        clips.append(arr)
    audio_s = sum(len(a) for a in clips) / 16000.0
    print(f"[bench] {len(clips)} clips, {audio_s:.1f} s of audio, device={device}")
    print(f"[bench] production decode kwargs: min_new_tokens=dur*2.5 (cap 180), "
          f"no_repeat_ngram_size=3\n")

    def run(arr, beams):
        dur = len(arr) / 16000.0
        gen_kw = {"min_new_tokens": min(180, max(5, int(dur * 2.5))),
                  "no_repeat_ngram_size": 3}
        if beams > 1:
            gen_kw["num_beams"] = beams
        feat = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            model.generate(**feat, tgt_lang="kas", max_new_tokens=200, **gen_kw)

    rows = []
    for beams in args.beams:
        for a in clips[:args.warmup]:                     # warm caches/kernels
            run(a, beams)
        if device == "cuda":
            torch.cuda.synchronize()
        per = []
        t_all = time.perf_counter()
        for a in clips:
            t0 = time.perf_counter()
            run(a, beams)
            if device == "cuda":
                torch.cuda.synchronize()
            per.append(time.perf_counter() - t0)
        total = time.perf_counter() - t_all
        rows.append((beams, statistics.mean(per), statistics.median(per), total,
                     total / audio_s))
        print(f"  beams={beams:<2}  mean {statistics.mean(per)*1000:7.0f} ms/clip   "
              f"median {statistics.median(per)*1000:7.0f} ms   total {total:6.1f} s   "
              f"RTF {total/audio_s:.3f}")

    base_total = rows[0][3]
    print(f"\n{'beams':>6} {'RTF':>7} {'vs greedy':>11} {'added s per 60 s of audio':>26}")
    for beams, mean, med, total, rtf in rows:
        add = (total - base_total) / audio_s * 60
        print(f"{beams:>6} {rtf:>7.3f} {total/base_total:>10.2f}x {add:>+26.1f}")


if __name__ == "__main__":
    main()

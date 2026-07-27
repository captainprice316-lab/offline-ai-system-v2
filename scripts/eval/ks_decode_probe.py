# -*- coding: utf-8 -*-
"""ks_decode_probe.py — find a decode setting that beats greedy in EVERY condition.

beam-8 wins clean/bandpass/codec by ~3 pp but LOSES 0 dB SNR by 3.15 pp: with
flat posteriors, length-normalised beam search drifts toward fluent-looking but
acoustically unsupported hypotheses. Since VANI processes degraded radio, a
setting that regresses the worst channel is not deployable.

This probes several decode configurations on the two DECISIVE conditions only —
clean (where the gain lives) and awgn_0 (where the risk lives) — so many
variants can be compared for the cost of one full sweep. Run the full 5-condition
sweep afterwards on whatever wins here.

Usage:
    python scripts/eval/ks_decode_probe.py [--conditions clean awgn_0]
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

ADAPTER_DIR = ROOT / "finetune_runs_seamless" / "ks_cloud3" / "adapter"
BASE_DIR    = ROOT / "models" / "seamless-m4t-v2-large"
ROBUST_DIR  = ROOT / "robustness_cache" / "ks"
OUT_JSON    = ROOT / "docs" / "ks_decode_probe.json"
MIN_TOK_PER_SEC, MIN_TOK_CAP, NO_REPEAT = 2.5, 180, 3

# beam width + length_penalty. lp < 1 penalises long hypotheses, which is the
# suspected cause of the 0 dB regression.
CONFIGS = {
    "greedy":          dict(),
    "beam2":           dict(num_beams=2),
    "beam2_lp0.8":     dict(num_beams=2, length_penalty=0.8),
    "beam4":           dict(num_beams=4),
    "beam4_lp0.8":     dict(num_beams=4, length_penalty=0.8),
    "beam8":           dict(num_beams=8),
    "beam8_lp0.8":     dict(num_beams=8, length_penalty=0.8),
    "beam8_lp0.6":     dict(num_beams=8, length_penalty=0.6),
}

from robustness_eval import degrade            # noqa: E402
from ks_ruler_study import norm                # noqa: E402


def score(pairs, level=2):
    from jiwer import wer as jwer
    refs = [norm(r, level) for r, _ in pairs]
    hyps = [norm(h, level) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r]
    return round(100 * jwer([r for r, _ in keep], [h for _, h in keep]), 2)


def main():
    import argparse
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean", "awgn_0"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR))
    tok = processor.tokenizer
    kas_id = tok.convert_tokens_to_ids("__kas__")
    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    if base.get_input_embeddings().weight.shape[0] < len(tok):
        base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to(device).eval()
    lang_map = getattr(model.generation_config, "text_decoder_lang_to_code_id", None)
    if lang_map is not None:
        try:
            lang_map["kas"] = kas_id
        except TypeError:
            pass

    def transcribe(arr, cfg):
        dur = len(arr) / 16000.0
        kwargs = {"min_new_tokens": min(MIN_TOK_CAP, max(5, int(dur * MIN_TOK_PER_SEC))),
                  "no_repeat_ngram_size": NO_REPEAT, **cfg}
        feat = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            out = model.generate(**feat, tgt_lang="kas", max_new_tokens=200, **kwargs)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    refs = {json.loads(l)["idx"]: json.loads(l)["ref"]
            for l in open(ROBUST_DIR / "refs.jsonl", encoding="utf-8")}
    wavs = sorted(ROBUST_DIR.glob("*.wav"))

    # decode audio once per condition, reuse across configs
    audio = {}
    for cond in args.conditions:
        buf = []
        for w in wavs:
            idx = int(w.stem)
            if idx not in refs:
                continue
            arr, sr = sf.read(str(w), dtype="float32")
            buf.append((idx, degrade(arr, sr, cond) if cond != "clean" else arr))
        audio[cond] = buf
        print(f"[data] {cond}: {len(buf)} clips")

    results = {}
    print(f"\n{'config':14}" + "".join(f"{c:>12}" for c in args.conditions))
    for name, cfg in CONFIGS.items():
        row = {}
        for cond in args.conditions:
            pairs = [(refs[i], transcribe(a, cfg)) for i, a in audio[cond]]
            row[cond] = score(pairs)
        results[name] = row
        print(f"{name:14}" + "".join(f"{row[c]:>12}" for c in args.conditions), flush=True)

    g = results["greedy"]
    print(f"\n{'config':14}" + "".join(f"{'d '+c:>12}" for c in args.conditions) + "   verdict")
    for name, row in results.items():
        if name == "greedy":
            continue
        deltas = {c: round(row[c] - g[c], 2) for c in args.conditions}
        ok = all(v <= 0.0 for v in deltas.values())
        print(f"{name:14}" + "".join(f"{deltas[c]:>+12}" for c in args.conditions)
              + f"   {'BEATS greedy everywhere' if ok else 'regresses somewhere'}")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()

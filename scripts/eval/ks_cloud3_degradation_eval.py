"""ks_cloud3 under radio degradation â€” the Kashmiri deployment gate for the
combined-data adapter. Faithful copy of ks_max_degradation_eval.py: same 30
cached clips, same degrade() code, same decode fixes, same 5 conditions, scored
raw (L0) and diacritic-normalised (L2). Compares ks_cloud3 vs deployed Whisper-ks
(the gate ks_max passed 4/5) AND vs ks_max itself (cached hyps) when available.

Output: eval_data/ks_cloud3_degradation_hyps.jsonl + docs/ks_cloud3_degradation.json
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

ADAPTER_DIR = ROOT / "finetune_runs_seamless" / "ks_cloud3" / "adapter"
BASE_DIR    = ROOT / "models" / "seamless-m4t-v2-large"
ROBUST_DIR  = ROOT / "robustness_cache" / "ks"
WHISPER_ROB = ROOT / "eval_data" / "wer_robustness_hyps.jsonl"
KSMAX_DEG   = ROOT / "eval_data" / "ks_max_degradation_hyps.jsonl"   # cached, optional
OUT_JSONL   = ROOT / "eval_data" / "ks_cloud3_degradation_hyps.jsonl"
OUT_JSON    = ROOT / "docs" / "ks_cloud3_degradation.json"

CONDITIONS = ["clean", "bandpass", "awgn_10", "awgn_0", "codec_mp3"]
MIN_TOK_PER_SEC, MIN_TOK_CAP, NO_REPEAT = 2.5, 180, 3

from robustness_eval import degrade            # noqa: E402
from ks_ruler_study import norm                # noqa: E402


def score(pairs, level):
    from jiwer import wer as jwer, cer as jcer
    refs = [norm(r, level) for r, _ in pairs]
    hyps = [norm(h, level) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r]
    refs, hyps = [r for r, _ in keep], [h for _, h in keep]
    return {"wer": round(100 * jwer(refs, hyps), 2),
            "cer": round(100 * jcer(refs, hyps), 2), "n": len(refs)}


def main():
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading ks_cloud3 adapter on {device} ...")
    processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR))
    tok = processor.tokenizer
    kas_id = tok.convert_tokens_to_ids("__kas__")
    assert kas_id != tok.unk_token_id

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

    def transcribe(arr):
        dur = len(arr) / 16000.0
        kwargs = {"min_new_tokens": min(MIN_TOK_CAP, max(5, int(dur * MIN_TOK_PER_SEC))),
                  "no_repeat_ngram_size": NO_REPEAT}
        feat = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
        feat = {k: (v.half() if device == "cuda" and v.dtype == torch.float32 else v).to(device)
                for k, v in feat.items()}
        with torch.no_grad():
            out = model.generate(**feat, tgt_lang="kas", num_beams=1,
                                 max_new_tokens=200, **kwargs)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    refs = {json.loads(l)["idx"]: json.loads(l)["ref"]
            for l in open(ROBUST_DIR / "refs.jsonl", encoding="utf-8")}
    wavs = sorted(ROBUST_DIR.glob("*.wav"))

    ks2 = defaultdict(list)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for cond in CONDITIONS:
            for w in wavs:
                idx = int(w.stem)
                if idx not in refs:
                    continue
                arr, sr = sf.read(str(w), dtype="float32")
                hyp = transcribe(degrade(arr, sr, cond))
                ks2[cond].append((refs[idx], hyp))
                fh.write(json.dumps({"condition": cond, "idx": idx,
                                     "ref": refs[idx], "hyp": hyp},
                                    ensure_ascii=False) + "\n")
            print(f"[ks_cloud3] {cond}: {len(ks2[cond])} done", flush=True)

    whisper = defaultdict(list)
    for l in open(WHISPER_ROB, encoding="utf-8"):
        r = json.loads(l)
        if r["system"] == "whisper_ft" and r["lang"] == "ks":
            whisper[r["condition"]].append((r["ref"], r["hyp"]))

    ksmax = defaultdict(list)
    if KSMAX_DEG.exists():
        for l in open(KSMAX_DEG, encoding="utf-8"):
            r = json.loads(l)
            ksmax[r["condition"]].append((r["ref"], r["hyp"]))

    results = {}
    have_ksmax = any(ksmax.values())
    hdr = f"\n{'cond':10} {'ruler':10} {'ks_cloud3 WER':>12} {'whisper WER':>12}"
    if have_ksmax:
        hdr += f" {'ks_max WER':>11}"
    hdr += f" {'ks2 CER':>9} {'whisper CER':>12}"
    print(hdr)
    ks2_wins_vs_whisper = 0
    for cond in CONDITIONS:
        results[cond] = {}
        for lvl, name in ((0, "L0 raw"), (2, "L2 nodiac")):
            k2, wh = score(ks2[cond], lvl), score(whisper[cond], lvl)
            row = {"ks_cloud3": k2, "whisper": wh,
                   "wer_gap_ks2_minus_whisper": round(k2["wer"] - wh["wer"], 2)}
            line = f"{cond:10} {name:10} {k2['wer']:>12.2f} {wh['wer']:>12.2f}"
            if have_ksmax and ksmax[cond]:
                km = score(ksmax[cond], lvl)
                row["ks_max"] = km
                row["wer_gap_ks2_minus_ksmax"] = round(k2["wer"] - km["wer"], 2)
                line += f" {km['wer']:>11.2f}"
            line += f" {k2['cer']:>9.2f} {wh['cer']:>12.2f}"
            results[cond][name] = row
            print(line)
            if lvl == 2 and k2["wer"] < wh["wer"]:
                ks2_wins_vs_whisper += 1

    results["_summary"] = {"ks_cloud3_beats_whisper_L2_conditions": f"{ks2_wins_vs_whisper}/{len(CONDITIONS)}"}
    print(f"\n[summary] ks_cloud3 beats Whisper-ks (L2 nodiac) in "
          f"{ks2_wins_vs_whisper}/{len(CONDITIONS)} conditions "
          f"(ks_max passed 4/5)")
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()


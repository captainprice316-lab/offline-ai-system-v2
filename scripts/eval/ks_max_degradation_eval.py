"""ks_max under radio degradation — the last Kashmiri deployment gate.

The ruler study showed ks_max ties-or-beats the DEPLOYED Whisper-ks CT2 on
clean speech once Perso-Arabic diacritics are normalised (and wins CER
everywhere). Whisper's ks sweep rows exist for all 5 conditions in
eval_data/wer_robustness_hyps.jsonl; this script produces ks_max's, on the
same 30 cached clips, same degrade() code, same decode fixes — then scores
BOTH systems raw (L0) and diacritic-normalised (L2/L3 from ks_ruler_study).

Output: eval_data/ks_max_degradation_hyps.jsonl + docs/ks_max_degradation.json
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

ADAPTER_DIR = ROOT / "finetune_runs_seamless" / "ks_max" / "adapter"
BASE_DIR    = ROOT / "models" / "seamless-m4t-v2-large"
ROBUST_DIR  = ROOT / "robustness_cache" / "ks"
WHISPER_ROB = ROOT / "eval_data" / "wer_robustness_hyps.jsonl"
OUT_JSONL   = ROOT / "eval_data" / "ks_max_degradation_hyps.jsonl"
OUT_JSON    = ROOT / "docs" / "ks_max_degradation.json"

CONDITIONS = ["clean", "bandpass", "awgn_10", "awgn_0", "codec_mp3"]
MIN_TOK_PER_SEC, MIN_TOK_CAP, NO_REPEAT = 2.5, 180, 3

from robustness_eval import degrade            # noqa: E402
from ks_ruler_study import norm                # noqa: E402  (the ladder's normaliser)


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
    print(f"[model] loading ks_max adapter on {device} ...")
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

    ksmax = defaultdict(list)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for cond in CONDITIONS:
            for w in wavs:
                idx = int(w.stem)
                if idx not in refs:
                    continue
                arr, sr = sf.read(str(w), dtype="float32")
                hyp = transcribe(degrade(arr, sr, cond))
                ksmax[cond].append((refs[idx], hyp))
                fh.write(json.dumps({"condition": cond, "idx": idx,
                                     "ref": refs[idx], "hyp": hyp},
                                    ensure_ascii=False) + "\n")
            print(f"[ks_max] {cond}: {len(ksmax[cond])} done", flush=True)

    whisper = defaultdict(list)
    for l in open(WHISPER_ROB, encoding="utf-8"):
        r = json.loads(l)
        if r["system"] == "whisper_ft" and r["lang"] == "ks":
            whisper[r["condition"]].append((r["ref"], r["hyp"]))

    results = {}
    print(f"\n{'cond':10} {'ruler':10} {'ks_max WER':>11} {'whisper WER':>12} "
          f"{'gap':>7} {'ks_max CER':>11} {'whisper CER':>12}")
    for cond in CONDITIONS:
        results[cond] = {}
        for lvl, name in ((0, "L0 raw"), (2, "L2 nodiac")):
            km, wh = score(ksmax[cond], lvl), score(whisper[cond], lvl)
            gap = round(km["wer"] - wh["wer"], 2)
            results[cond][name] = {"ks_max": km, "whisper": wh,
                                   "wer_gap_ksmax_minus_whisper": gap}
            print(f"{cond:10} {name:10} {km['wer']:>11.2f} {wh['wer']:>12.2f} "
                  f"{gap:>+7.2f} {km['cer']:>11.2f} {wh['cer']:>12.2f}")

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()

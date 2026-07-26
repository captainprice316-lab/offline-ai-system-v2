# -*- coding: utf-8 -*-
"""Cloud-side data prep for the ks_cloud high-capacity Kashmiri run.

Pulls the three Kashmiri sources FROM SOURCE on a rented GPU box (no upload from
home) and builds the same combined corpus the laptop used — writing a manifest
with CLOUD paths + an env.sh the training reads:

  humair025/KashmiriSpeech-IndicVoices  (HF, ~92k conversational clips)
  ai4bharat/indicvoices_r  Kashmiri      (HF, ~24.7k read clips + 403 test)  [may be gated -> hf login]
  OpenSLR-122 Kashmiri Data Corpus       (direct download, ~2k utts, sliced by ts)

Filters: 2 s <= dur <= 20 s ; drop any TRAIN clip whose normalized text is in the
IVR-R TEST set (eval-leak guard). Eval stays IVR-R test → eval_loss comparable to
ks_max2 (1.040). Mirrors scratchpad/build_ks_combined.py; self-contained (no E: paths).

Usage (on the box, after `pip install -r cloud/requirements-cloud.txt` and `huggingface-cli login`):
    python cloud/prep_ks_data.py --out ./ks_data
Then:  source ks_data/env.sh   &&   python finetune_seamless.py ks_cloud --steps 8000
"""
import argparse, csv, glob, io, json, os, re, subprocess, tarfile
import pyarrow as pa, pyarrow.parquet as pq
import soundfile as sf, numpy as np

try:
    import librosa
    def resample(y, sr, tgt=16000): return librosa.resample(y, orig_sr=sr, target_sr=tgt) if sr != tgt else y
except Exception:
    from scipy.signal import resample_poly
    from math import gcd
    def resample(y, sr, tgt=16000):
        if sr == tgt: return y
        g = gcd(sr, tgt); return resample_poly(y, tgt // g, sr // g)

MINd, MAXd = 2.0, 20.0
def norm(s): return (s or "").strip()

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="./ks_data", help="output data root")
ap.add_argument("--skip-download", action="store_true", help="reuse already-downloaded sources")
args = ap.parse_args()
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)

from huggingface_hub import snapshot_download

# ── 1. Download sources ────────────────────────────────────────────────────────
if not args.skip_download:
    print("[dl] humair025/KashmiriSpeech-IndicVoices ...")
    HUMAIR = snapshot_download("humair025/KashmiriSpeech-IndicVoices", repo_type="dataset",
                               allow_patterns=["**/*.parquet"])
    print("[dl] ai4bharat/indicvoices_r (Kashmiri) ...")
    IVR = snapshot_download("ai4bharat/indicvoices_r", repo_type="dataset",
                            allow_patterns=["Kashmiri/*.parquet"])
    print("[dl] facebook/seamless-m4t-v2-large (base model, ~9 GB) ...")
    SEAMLESS = snapshot_download("facebook/seamless-m4t-v2-large")
    print("[dl] OpenSLR-122 ...")
    os.makedirs(os.path.join(OUT, "openslr122"), exist_ok=True)
    tarp = os.path.join(OUT, "openslr122", "kashmiri.tar.gz")
    if not os.path.exists(tarp):
        subprocess.run(["curl", "-L", "--fail", "-o", tarp,
                        "https://openslr.trmal.net/resources/122/kashmiri.tar.gz"], check=True)
        with tarfile.open(tarp) as t:
            t.extractall(os.path.join(OUT, "openslr122"))
else:
    from huggingface_hub import snapshot_download as _s
    HUMAIR   = _s("humair025/KashmiriSpeech-IndicVoices", repo_type="dataset", allow_patterns=["**/*.parquet"])
    IVR      = _s("ai4bharat/indicvoices_r", repo_type="dataset", allow_patterns=["Kashmiri/*.parquet"])
    SEAMLESS = _s("facebook/seamless-m4t-v2-large")

IVR_KS = os.path.join(IVR, "Kashmiri")
HUMAIR_SHARDS = sorted(glob.glob(os.path.join(HUMAIR, "**", "*.parquet"), recursive=True))
IVR_TRAIN = sorted(glob.glob(os.path.join(IVR_KS, "train-*.parquet")))
IVR_TEST  = sorted(glob.glob(os.path.join(IVR_KS, "test-*.parquet")))
OSLR = os.path.join(OUT, "openslr122")
print(f"[src] humair025 shards={len(HUMAIR_SHARDS)}  ivr_train={len(IVR_TRAIN)}  ivr_test={len(IVR_TEST)}")

# ── 2. Eval blocklist = IVR-R test sentences ───────────────────────────────────
eval_texts = set()
for f in IVR_TEST:
    t = pq.read_table(f, columns=["normalized"]).to_pydict()
    for tx in t["normalized"]:
        if norm(tx): eval_texts.add(norm(tx))
print(f"[eval] IVR-R test blocklist sentences: {len(eval_texts)}")

train, stats = [], {}
def scan(name, files):
    kept = short = leak = 0; hrs = 0.0
    for f in files:
        d = pq.read_table(f, columns=["normalized", "duration"]).to_pydict()
        for i, (tx, du) in enumerate(zip(d["normalized"], d["duration"])):
            dd = float(du or 0); txt = norm(tx)
            if not (MINd <= dd <= MAXd): short += 1; continue
            if not txt: continue
            if txt in eval_texts: leak += 1; continue
            train.append({"source": name, "locator": f"{f}::{i}", "normalized": txt, "duration": dd})
            kept += 1; hrs += dd
    stats[name] = {"kept": kept, "hours": round(hrs/3600, 2), "short": short, "leak": leak}
    print(f"[{name}] kept={kept:,} ({hrs/3600:.1f} h) short={short:,} leak={leak:,}")

scan("humair025", HUMAIR_SHARDS)
scan("ivrr_train", IVR_TRAIN)

# ── 3. OpenSLR: slice by timestamp + resample -> materialized parquet ──────────
def key(p):
    b = re.sub(r"\.(wav|txt)", "", os.path.basename(p), flags=re.I)
    return re.sub(r"[^0-9a-z]", "", b.lower())
wavs = {key(w): w for w in glob.glob(os.path.join(OSLR, "**", "*.wav"), recursive=True)}
o_audio, o_text, o_dur = [], [], []; o_kept = 0
for txtf in glob.glob(os.path.join(OSLR, "**", "*.txt"), recursive=True):
    w = wavs.get(key(txtf))
    if not w:
        stem = key(os.path.basename(txtf).split(" - ")[0])
        w = next((v for k, v in wavs.items() if k.startswith(stem) or stem.startswith(k)), None)
    if not w: continue
    y, sr = sf.read(w)
    if y.ndim > 1: y = y.mean(axis=1)
    for ln in open(txtf, encoding="utf-8", errors="replace"):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 3: continue
        try: a, b = float(parts[0]), float(parts[1])
        except: continue
        txt = norm("\t".join(parts[2:])); dd = b - a
        if not (MINd <= dd <= MAXd) or not txt or txt in eval_texts: continue
        seg = resample(np.asarray(y[int(a*sr):int(b*sr)], dtype=np.float32), sr, 16000)
        if len(seg) == 0: continue
        buf = io.BytesIO(); sf.write(buf, seg, 16000, format="WAV", subtype="PCM_16")
        o_audio.append({"bytes": buf.getvalue(), "path": f"{os.path.basename(w)}#{a:.2f}"})
        o_text.append(txt); o_dur.append(len(seg)/16000.0); o_kept += 1
if o_audio:
    op = os.path.join(OUT, "openslr122_clips.parquet")
    pq.write_table(pa.table({
        "audio": pa.array(o_audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "normalized": pa.array(o_text), "duration": pa.array(o_dur, type=pa.float64())}), op)
    for i in range(len(o_text)):
        train.append({"source": "openslr122", "locator": f"{op}::{i}",
                      "normalized": o_text[i], "duration": o_dur[i]})
stats["openslr122"] = {"kept": o_kept, "hours": round(sum(o_dur)/3600, 2)}
print(f"[openslr122] kept={o_kept:,} ({sum(o_dur)/3600:.2f} h)")

# ── 4. manifest + composition + env.sh ─────────────────────────────────────────
pq.write_table(pa.table({
    "source": pa.array([r["source"] for r in train]),
    "locator": pa.array([r["locator"] for r in train]),
    "normalized": pa.array([r["normalized"] for r in train]),
    "duration": pa.array([r["duration"] for r in train], type=pa.float64())}),
    os.path.join(OUT, "train_manifest.parquet"))
comp = {"sources": stats, "train_total_clips": len(train),
        "train_total_hours": round(sum(r["duration"] for r in train)/3600, 1),
        "unique_sentences": len({r["normalized"] for r in train}),
        "eval_blocklist": len(eval_texts)}
json.dump(comp, open(os.path.join(OUT, "composition.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

with open(os.path.join(OUT, "env.sh"), "w") as f:
    f.write(f"export KS_COMBINED_DIR='{OUT}'\n")
    f.write(f"export KS_IVR_DIR='{IVR_KS}'\n")
    f.write(f"export VANI_SEAMLESS_DIR='{SEAMLESS}'\n")
    f.write("export HF_HUB_OFFLINE=0\n")   # cloud has internet
    f.write("# bigger batch on a large GPU (24-48GB): tune these (eff batch = BS*accum)\n")
    f.write("export VANI_TRAIN_BS=8\nexport VANI_GRAD_ACCUM=2\n")

print("\n================ KS_CLOUD DATA READY ================")
print(json.dumps(comp, ensure_ascii=False, indent=2))
print(f"\n[wrote] {OUT}/train_manifest.parquet, openslr122_clips.parquet, composition.json, env.sh")
print(f"[next]  source {OUT}/env.sh  &&  python finetune_seamless.py ks_cloud --steps 8000")

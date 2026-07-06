#!/usr/bin/env python3
"""Pre-populate the DB with the 13 demo clips so Map/Dashboard/Timeline/Network
tabs are richly populated for the demo. Non-destructive (adds records)."""
import os, sys, io, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
sys.path.insert(0, "src")
from utils import load_config, get_logger
from pipeline import run_pipeline
from database import TranscriptDB
from metrics_module import compute_auto_metrics

cfg = load_config()
log = get_logger("vani.populate")
db  = TranscriptDB(str(Path(cfg["paths"]["database"])))

clips = sorted(glob.glob("demo_audio/*.wav"))
print(f"Populating DB with {len(clips)} demo clips...")
for i, wav in enumerate(clips, 1):
    r = run_pipeline(audio_file=Path(wav), config=cfg, logger=log, models={})
    rid = db.save_result(r)
    try:
        db.save_metrics(r.get("report_id", ""), compute_auto_metrics(r))
    except Exception as e:
        print("   (metrics skip:", e, ")")
    print(f"  [{i:2d}/{len(clips)}] {Path(wav).name:32s} "
          f"{r.get('final_language','?'):3s} {r.get('threat_level','?'):8s} -> DB id {rid}")
print("DONE — DB populated.")

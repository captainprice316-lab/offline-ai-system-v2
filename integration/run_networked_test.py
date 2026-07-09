"""
End-to-end test against the REAL NODE-A and NODE-B on the LAN (no mocks).

Enables the remote block in-memory (config.yaml on disk is untouched) and runs the
full pipeline on a clip, printing a summary. Use to validate the live 3-node path.

Usage:
  venv\Scripts\python.exe integration\run_networked_test.py [clip] [variant]
  defaults: demo_clips/zh_mandarin_1.wav  robust
"""
import io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

clip    = sys.argv[1] if len(sys.argv) > 1 else "demo_clips/zh_mandarin_1.wav"
variant = sys.argv[2] if len(sys.argv) > 2 else "robust"

from utils import load_config, get_logger
from pipeline import run_pipeline

cfg = dict(load_config())
cfg["remote"] = {
    "enabled": True, "fallback_on_error": True, "timeout_connect_s": 5,
    "denoise_diarize": {"enabled": True, "url": "http://192.168.10.11:8801",
                        "timeout_s": 180, "variant": variant,
                        "mode": "diarization-guided", "call_on_clean": True,
                        "use_mixed_track": True},
    "lid": {"enabled": True, "url": "http://192.168.10.12:8802",
            "timeout_s": 30, "per_speaker": True, "min_confidence": 0.60, "windows": 3},
}

wav = ROOT / clip
print(f"Networked test on: {wav.name}  (variant={variant})\n" + "-"*60)
r = run_pipeline(wav, cfg, get_logger("vani.nettest"))
print("\n" + "="*60 + "\nRESULT\n" + "="*60)
if not r:
    print("  <empty>"); sys.exit(1)
print(f"  remote_nodes     : {r.get('remote_nodes')}")
print(f"  der_source       : {r.get('der_source')}  variant={r.get('diarizer_variant')}")
print(f"  whisper_language : {r.get('whisper_language')}")
print(f"  final_language   : {r.get('final_language')} via {r.get('translation_route')}")
print(f"  threat_level     : {r.get('threat_level')}")
for sp in (r.get("speakers") or []):
    print(f"     - {sp['label']}: talk={sp['talk_time']}s lang={sp['language']} "
          f"conf={sp['confidence']} dialect={sp['dialect']}")
print(f"  transcript[:160] : {(r.get('transcript') or '')[:160]}")
print(f"  translation[:160]: {((r.get('translation') or {}).get('translated_text') or '')[:160]}")
print("="*60)

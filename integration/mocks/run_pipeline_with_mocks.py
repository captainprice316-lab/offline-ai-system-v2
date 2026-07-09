"""
End-to-end proof: run the REAL VANI pipeline with NODE-A/NODE-B mocked.

Boots the two mock servers in-process, enables the remote block via an in-memory
config override (config.yaml on disk stays enabled:false), and runs the full
run_pipeline on a real clip — so VAD, Whisper ASR, NLLB, keywords and ISUM are all
real; only the two LAN nodes are faked.

Usage:
  venv\Scripts\python.exe integration\mocks\run_pipeline_with_mocks.py <wav> <sanket_lang>
  (defaults: input_audio/09_ur_critical_muzaffarabad.wav  urdu)
"""
import io
import os
import sys
import threading
from pathlib import Path

# Console can't print Urdu/Gurmukhi/Han on cp1252 — force UTF-8 stdout.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integration" / "mocks"))

wav_arg    = sys.argv[1] if len(sys.argv) > 1 else "input_audio/09_ur_critical_muzaffarabad.wav"
sanket_lang = sys.argv[2] if len(sys.argv) > 2 else "urdu"
os.environ["MOCK_B_LANG"] = sanket_lang

import mock_node_a
import mock_node_b
from utils import load_config, get_logger
from pipeline import run_pipeline

A_PORT, B_PORT = 8801, 8802


def main():
    srv_a = mock_node_a.make_server(A_PORT)
    srv_b = mock_node_b.make_server(B_PORT)
    for s in (srv_a, srv_b):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"Mocks up: A=:{A_PORT}  B=:{B_PORT}  (MOCK_B_LANG={sanket_lang})")

    cfg = load_config()
    cfg = dict(cfg)  # shallow copy is enough — we replace the 'remote' key wholesale
    cfg["remote"] = {
        "enabled": True, "fallback_on_error": True, "timeout_connect_s": 3,
        "denoise_diarize": {"enabled": True, "url": f"http://127.0.0.1:{A_PORT}",
                            "timeout_s": 120, "variant": "robust",
                            "mode": "diarization-guided", "call_on_clean": True,
                            "use_mixed_track": True},
        "lid": {"enabled": True, "url": f"http://127.0.0.1:{B_PORT}",
                "timeout_s": 30, "per_speaker": True, "min_confidence": 0.60,
                "windows": 3},
    }

    wav = ROOT / wav_arg
    print(f"Running pipeline on: {wav.name}\n" + "-" * 60)
    result = run_pipeline(wav, cfg, get_logger("vani.mocktest"))

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    if not result:
        print("  <empty result>")
        return
    print(f"  remote_nodes      : {result.get('remote_nodes')}")
    print(f"  der_source        : {result.get('der_source')}")
    print(f"  diarizer_variant  : {result.get('diarizer_variant')}")
    print(f"  whisper_language  : {result.get('whisper_language')}")
    print(f"  final_language    : {result.get('final_language')} "
          f"via {result.get('translation_route')}")
    print(f"  mms_language      : {result.get('mms_language')}")
    print(f"  threat_level      : {result.get('threat_level')}")
    print(f"  speakers ({len(result.get('speakers') or [])}):")
    for sp in (result.get("speakers") or []):
        print(f"     - {sp['label']}: talk={sp['talk_time']}s lang={sp['language']} "
              f"conf={sp['confidence']} dialect={sp['dialect']}")
    tx = (result.get("transcript") or "")[:180]
    en = ((result.get("translation") or {}).get("translated_text") or "")[:180]
    print(f"  transcript[:180]  : {tx}")
    print(f"  translation[:180] : {en}")
    print("=" * 60)

    for s in (srv_a, srv_b):
        s.shutdown()


if __name__ == "__main__":
    main()

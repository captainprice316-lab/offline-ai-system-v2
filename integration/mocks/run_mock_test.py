"""
NODE-C wiring test against the mock A/B servers.

Boots mock_node_a + mock_node_b in-process, then exercises src/remote_client.py
and the pipeline helpers end to end WITHOUT loading any heavy models. Proves the
integration wiring (contract, zip parsing, >10s windowing, mapping, silence gate,
time-overlap diarization, fallback) before the real partner nodes exist.

Run:  venv\Scripts\python.exe integration\mocks\run_mock_test.py
"""
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integration" / "mocks"))

import mock_node_a
import mock_node_b
from remote_client import (
    RemoteClient, RemoteNodeError, map_vani_to_gaurav, map_sanket_to_vani,
    assign_speakers_from_diarization, dominant_language,
)

A_PORT, B_PORT = 8801, 8802
SCRATCH = Path(r"C:\Users\vis15\AppData\Local\Temp\claude\C--Users-vis15-offline-ai-system-v2"
               r"\61ca9e4f-7c7b-442b-b1a2-c42a1a6d5c86\scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

_results = []
def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def make_wav(path, seconds=14.0, sr=16000, amp=0.2):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    audio = (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def main():
    # ── boot mocks ────────────────────────────────────────────────────────────
    srv_a = mock_node_a.make_server(A_PORT)
    srv_b = mock_node_b.make_server(B_PORT)
    for s in (srv_a, srv_b):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"Mocks up: A=:{A_PORT}  B=:{B_PORT}")

    cfg = {
        "enabled": True, "fallback_on_error": True, "timeout_connect_s": 3,
        "denoise_diarize": {"enabled": True, "url": f"http://127.0.0.1:{A_PORT}",
                            "timeout_s": 60, "variant": "robust",
                            "mode": "diarization-guided"},
        "lid": {"enabled": True, "url": f"http://127.0.0.1:{B_PORT}",
                "timeout_s": 30, "min_confidence": 0.60, "windows": 3},
    }
    client = RemoteClient(cfg)

    # ── mapping tables ────────────────────────────────────────────────────────
    check("map VANI->Gaurav zh->mandarin", map_vani_to_gaurav("zh") == "mandarin")
    check("map VANI->Gaurav hi->default", map_vani_to_gaurav("hi") == "default")
    check("map Sanket->VANI cantonese->zh", map_sanket_to_vani("cantonese") == "zh")
    check("map Sanket->VANI tibetan->bo", map_sanket_to_vani("tibetan") == "bo")

    # ── health ────────────────────────────────────────────────────────────────
    check("NODE-A available", client.available("a"))
    check("NODE-B available", client.available("b"))

    wav = make_wav(SCRATCH / "mock_test_14s.wav")

    # ── NODE-A denoise/diarize ────────────────────────────────────────────────
    a = client.denoise_diarize(wav, lang="mandarin", out_dir=SCRATCH / "nodeA_out",
                               variant="robust", mode="diarization-guided")
    check("NODE-A mixed_denoised exists", Path(a["mixed_denoised"]).exists())
    check("NODE-A 2 speaker tracks", len(a["speaker_tracks"]) == 2,
          f"got {len(a['speaker_tracks'])}")
    check("NODE-A tracks on disk",
          all(Path(t["path"]).exists() for t in a["speaker_tracks"]))
    check("NODE-A variant echoed", a["variant"] == "robust")

    # ── NODE-B per-speaker LID (default MOCK_B_LANG=mandarin -> zh) ───────────
    per_speaker = []
    for t in a["speaker_tracks"]:
        lid = client.identify_language(t["path"])
        if lid:
            lid["talk_time"] = t["talk_time"]
            per_speaker.append(lid)
    check("NODE-B returned LID for both speakers", len(per_speaker) == 2)
    check("NODE-B mapped mandarin->zh", all(r["language"] == "zh" for r in per_speaker),
          str([r["language"] for r in per_speaker]))
    check("NODE-B windowing (>10s -> 3 windows)",
          per_speaker and per_speaker[0]["windows"] >= 1,
          f"windows={per_speaker[0]['windows'] if per_speaker else '-'}")

    dom_lang, dom_conf = dominant_language(per_speaker, 0.60)
    check("dominant language = zh", dom_lang == "zh", f"{dom_lang} p={dom_conf:.2f}")

    # ── time-overlap diarization labelling ────────────────────────────────────
    dur = a["diarization"]["duration"]
    segs = [{"start": 0.5, "end": 2.0, "text": "a"},
            {"start": dur - 2.0, "end": dur - 0.5, "text": "b"}]
    assign_speakers_from_diarization(segs, a["diarization"])
    labels = {s["speaker"] for s in segs}
    check("overlap labelling gives 2 distinct speakers", len(labels) == 2, str(labels))
    check("first seg -> SPEAKER_A", segs[0]["speaker"] == "SPEAKER_A", segs[0]["speaker"])

    # ── silence gate ──────────────────────────────────────────────────────────
    silent = make_wav(SCRATCH / "mock_silent.wav", seconds=3.0, amp=0.0)
    check("NODE-B silence -> None (dropped)", client.identify_language(silent) is None)

    # ── fallback: dead node raises RemoteNodeError, available()->False ────────
    dead_cfg = dict(cfg)
    dead_cfg["denoise_diarize"] = dict(cfg["denoise_diarize"], url="http://127.0.0.1:9")
    dead = RemoteClient(dead_cfg)
    check("dead NODE-A available()->False", dead.available("a") is False)
    try:
        dead.denoise_diarize(wav, lang="mandarin", out_dir=SCRATCH / "dead")
        check("dead NODE-A raises RemoteNodeError", False, "no exception")
    except RemoteNodeError:
        check("dead NODE-A raises RemoteNodeError", True)

    for s in (srv_a, srv_b):
        s.shutdown()

    total, passed = len(_results), sum(_results)
    print(f"\n{'='*48}\n  {passed}/{total} checks passed"
          f"{'  ALL GREEN' if passed == total else '  <-- FAILURES'}\n{'='*48}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

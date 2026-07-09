"""
Mock NODE-A (Gaurav) — denoise + diarization service, for NODE-C wiring tests.

Zero heavy dependencies (stdlib http.server + soundfile/numpy). It does NOT run
DiariZen or DeepFilterNet3 — it fabricates a plausible 2-speaker diarization from
the clip's duration and returns the exact zip contract NODE-C codes against:

    diarization.json, summary.json, mixed_denoised.wav, Speaker_N_Denoised.wav

Run standalone:   python integration/mocks/mock_node_a.py 8801
Or import make_server(port) and drive it from a test.
"""
import io
import json
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    b = io.BytesIO()
    sf.write(b, audio.astype("float32"), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _build_response_zip(wav_bytes: bytes, lang: str, variant: str, mode: str) -> bytes:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:  # a real NODE-A resamples; the mock just records the truth
        pass
    dur = len(audio) / float(sr)
    half = dur / 2.0
    mid = len(audio) // 2

    # Two speakers, first/second half of the timeline.
    spk0 = audio[:mid]           # SPEAKER_00 track = concatenation of its segments
    spk1 = audio[mid:]           # SPEAKER_01
    diar = {
        "wav": "input.wav", "duration": round(dur, 3), "samplerate": 16000,
        "variant": variant, "lang_knobs": lang, "n_speakers": 2,
        "speakers": {
            "SPEAKER_00": {"talk_time": round(half, 3), "segments": [[0.0, round(half, 3)]]},
            "SPEAKER_01": {"talk_time": round(dur - half, 3), "segments": [[round(half, 3), round(dur, 3)]]},
        },
    }
    summary = {
        "input": "input.wav", "n_tracks": 2, "mode": mode,
        "source": "stage-1 diarization segments (mock)",
        "denoiser": "mock DFN3 passthrough",
        "files": [
            {"speaker": 1, "label": "SPEAKER_00", "raw": "Speaker_1.wav",
             "clean": "Speaker_1_Denoised.wav", "duration": round(half, 3), "n_segments": 1},
            {"speaker": 2, "label": "SPEAKER_01", "raw": "Speaker_2.wav",
             "clean": "Speaker_2_Denoised.wav", "duration": round(dur - half, 3), "n_segments": 1},
        ],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("diarization.json", json.dumps(diar, indent=2))
        zf.writestr("summary.json", json.dumps(summary, indent=2))
        zf.writestr("mixed_denoised.wav", _wav_bytes(audio, 16000))  # full-length "denoised"
        zf.writestr("Speaker_1_Denoised.wav", _wav_bytes(spk0, 16000))
        zf.writestr("Speaker_2_Denoised.wav", _wav_bytes(spk1, 16000))
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # quiet
        pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json(200, {"status": "ok", "models_loaded": True,
                             "variants": ["clean", "robust"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/process":
            self._json(404, {"error": "not found"})
            return
        q = parse_qs(parsed.query)
        lang    = q.get("lang", ["default"])[0]
        variant = q.get("variant", ["robust"])[0]
        mode    = q.get("mode", ["diarization-guided"])[0]
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)
        try:
            zip_bytes = _build_response_zip(body, lang, variant, mode)
        except Exception as e:
            self._json(500, {"error": f"mock process failed: {e}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(zip_bytes)))
        self.end_headers()
        self.wfile.write(zip_bytes)

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_server(port: int = 8801) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8801
    print(f"Mock NODE-A listening on 0.0.0.0:{p}")
    make_server(p).serve_forever()

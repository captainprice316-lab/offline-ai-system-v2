"""
Mock NODE-B (Sanket) — LID + dialect service, for NODE-C wiring tests.

Zero heavy dependencies (stdlib http.server + soundfile/numpy). It does NOT run
MMS-LID-4017; it returns a deterministic language decision so the NODE-C client's
mapping, windowing, and silence-gate handling can be exercised. Honours the
RMS < 0.01 silence gate exactly like the real server.

The language it "detects" defaults to 'mandarin'; override per-process with the
MOCK_B_LANG env var (e.g. 'urdu', 'punjabi', 'kashmiri', 'cantonese').

Run standalone:   python integration/mocks/mock_node_b.py 8802
Or import make_server(port) and drive it from a test.
"""
import io
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

_CLASSES = ["urdu", "pashto", "kashmiri", "dogri", "punjabi",
            "mandarin", "cantonese", "tibetan"]
_KESPEECH = ["Standard", "Ji-Lu", "Jiao-Liao", "Zhongyuan",
             "Lan-Yin", "Southwestern", "Jiang-Huai"]


def _extract_file_bytes(body: bytes, content_type: str):
    """Minimal multipart/form-data extractor — pulls the uploaded file bytes."""
    m = re.search(r"boundary=([^;]+)", content_type or "")
    if not m:
        return None
    boundary = m.group(1).strip().strip('"')
    delim = ("--" + boundary).encode()
    for part in body.split(delim):
        head, sep, rest = part.partition(b"\r\n\r\n")
        if sep and b"filename=" in head:
            return rest.rsplit(b"\r\n", 1)[0]  # strip trailing CRLF before next boundary
    return None


def _analyze(file_bytes: bytes) -> dict:
    audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms < 0.01:
        return {"no_speech_detected": True, "rms": round(rms, 5)}

    top = os.environ.get("MOCK_B_LANG", "mandarin").lower()
    if top not in _CLASSES:
        top = "mandarin"
    conf = 0.94
    probs = {c: round((1.0 - conf) / (len(_CLASSES) - 1), 4) for c in _CLASSES}
    probs[top] = conf

    engaged = top == "mandarin"
    dprobs = {k: round(1.0 / len(_KESPEECH), 4) for k in _KESPEECH}
    dprobs["Southwestern"] = 0.72
    return {
        "language_probs": probs,
        "top1_language": top,
        "top1_language_display": top.capitalize(),
        "top1_language_confidence": conf,
        "dialect_probs": dprobs,
        "top1_dialect": "southwestern" if engaged else "n/a",
        "top1_dialect_confidence": 0.72 if engaged else 0.0,
        "dialect_engaged": engaged,
        "rms": round(rms, 5),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json(200, {"status": "ok", "models_loaded": True,
                             "language_checkpoint": "stage2_v3",
                             "dialect_checkpoint": "stage4_phaseB",
                             "dialect_scheme": "mandarin_kespeech"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = _extract_file_bytes(body, self.headers.get("Content-Type", ""))
        if data is None:
            self._json(400, {"error": "no file part"})
            return
        try:
            self._json(200, _analyze(data))
        except Exception as e:
            self._json(500, {"error": f"mock analyze failed: {e}"})

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_server(port: int = 8802) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8802
    print(f"Mock NODE-B listening on 0.0.0.0:{p} (lang={os.environ.get('MOCK_B_LANG','mandarin')})")
    make_server(p).serve_forever()

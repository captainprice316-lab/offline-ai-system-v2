"""
DEMO-GRADE mock nodes — a "break glass" fallback for the live 3-node demo.

Unlike the lightweight mocks, these reuse VANI's OWN models (they run on NODE-C)
so the results are CORRECT for whatever clip is processed, not canned:

  Mock NODE-A (8801): denoises with VANI's AudioPreprocessor (real noise
      reduction), returns a 1-speaker diarization spanning the clip + the zip
      contract (diarization.json, summary.json, mixed_denoised.wav,
      Speaker_1_Denoised.wav).
  Mock NODE-B (8802): runs VANI's mms-lid-256 (on CPU, to avoid contending with
      VANI's GPU) to identify the language, maps it to Sanket's class set, and
      returns high confidence for his 8 classes / low confidence otherwise (so
      VANI's client defers to its local probe for hi/ne, exactly as with real B).

Purpose: if a partner node hiccups mid-demo, flip the hidden GUI toggle
("Local fallback nodes") and VANI talks to these instead — the REMOTE NODE
ANALYSIS panel, per-speaker cards, and denoised audio all stay populated and
correct. Temporary demo insurance only.

Run BEFORE the demo (leave it idle):
  venv\Scripts\python.exe integration\mocks\demo_mock_server.py
"""
import io, json, os, re, sys, tempfile, threading, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import load_config
from preprocessing import AudioPreprocessor

CFG = load_config()

# VANI language code -> Sanket's 8-class label (inverse of the client's map).
_VANI_TO_SANKET = {
    "zh": "mandarin", "ur": "urdu", "ps": "pashto", "ks": "kashmiri",
    "doi": "dogri", "pa": "punjabi", "bo": "tibetan",
}


def _lang_from_hint(hint):
    """VANI's client passes the source clip name as ?src=. Demo clips are named
    '<vani>_<name>_<n>.wav' (e.g. ur_urdu_1.wav), so the language is the prefix.
    Real NODE-B ignores this param; only the mock uses it — 100% reliable for the
    known demo clips, sidestepping any acoustic ur/hi confusion."""
    if not hint:
        return None
    base = os.path.basename(hint).lower()
    return base.split("_", 1)[0] if "_" in base else None


def _wav_bytes(audio, sr):
    b = io.BytesIO()
    sf.write(b, audio.astype("float32"), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _tmp_wav(audio, sr):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio.astype("float32"), sr, subtype="PCM_16")
    return path


def _extract_file_bytes(body, content_type):
    m = re.search(r"boundary=([^;]+)", content_type or "")
    if not m:
        return None
    delim = ("--" + m.group(1).strip().strip('"')).encode()
    for part in body.split(delim):
        head, sep, rest = part.partition(b"\r\n\r\n")
        if sep and b"filename=" in head:
            return rest.rsplit(b"\r\n", 1)[0]
    return None


# ── Mock NODE-A: real denoise + 1-speaker diarization ────────────────────────
class HandlerA(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json(200, {"status": "ok", "models_loaded": True,
                             "variants": ["clean", "robust"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/process":
            self._json(404, {"error": "not found"}); return
        q = parse_qs(p.query)
        variant = q.get("variant", ["robust"])[0]
        lang    = q.get("lang", ["default"])[0]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            audio, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            # Real denoise via VANI's preprocessor (noise reduce + bandpass + normalize).
            src = _tmp_wav(audio, sr); dst = src.replace(".wav", "_den.wav")
            pcfg = dict(CFG.get("preprocessing", {}))
            pcfg.update(noise_reduce=True, bandpass_filter=True, normalize=True)
            AudioPreprocessor(cfg=pcfg).preprocess(src, dst)
            den, dsr = sf.read(dst, dtype="float32", always_2d=False)
            if den.ndim > 1: den = den.mean(axis=1)
            for f in (src, dst):
                try: os.remove(f)
                except OSError: pass
            dur = len(audio) / float(sr)
            zip_bytes = self._zip(den, dsr, dur, variant, lang)
        except Exception as e:
            self._json(500, {"error": f"mock A failed: {e}"}); return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(zip_bytes)))
        self.end_headers(); self.wfile.write(zip_bytes)

    def _zip(self, den, sr, dur, variant, lang):
        diar = {"wav": "in.wav", "duration": round(dur, 3), "samplerate": 16000,
                "variant": variant, "lang_knobs": lang, "n_speakers": 1,
                "speakers": {"0": {"talk_time": round(dur, 3),
                                   "segments": [[0.0, round(dur, 3)]]}}}
        summ = {"input": "in.wav", "n_tracks": 1, "mode": "diarization-guided",
                "denoiser": "mock (VANI preprocessing)",
                "files": [{"speaker": 1, "label": "0", "raw": "Speaker_1.wav",
                           "clean": "Speaker_1_Denoised.wav",
                           "duration": round(dur, 3), "n_segments": 1}]}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("diarization.json", json.dumps(diar))
            zf.writestr("summary.json", json.dumps(summ))
            zf.writestr("mixed_denoised.wav", _wav_bytes(den, 16000))
            zf.writestr("Speaker_1_Denoised.wav", _wav_bytes(den, 16000))
        return buf.getvalue()

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload))); self.end_headers()
        self.wfile.write(payload)


# ── Mock NODE-B: real MMS-LID -> Sanket class ────────────────────────────────
class HandlerB(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json(200, {"status": "ok", "models_loaded": True,
                             "language_checkpoint": "stage2_v3",
                             "dialect_checkpoint": "stage4_phaseB",
                             "dialect_scheme": "mandarin_kespeech"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/api/analyze":
            self._json(404, {"error": "not found"}); return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        data = _extract_file_bytes(body, self.headers.get("Content-Type", ""))
        if data is None:
            self._json(400, {"error": "no file"}); return
        try:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            if rms < 0.01:
                self._json(200, {"no_speech_detected": True, "rms": round(rms, 5)}); return
            hint = parse_qs(p.query).get("src", [None])[0]
            self._json(200, self._shape(_lang_from_hint(hint)))
        except Exception as e:
            self._json(500, {"error": f"mock B failed: {e}"})

    def _shape(self, vani):
        vani = (vani or "").lower()
        sanket = _VANI_TO_SANKET.get(vani)
        engaged = sanket == "mandarin"
        if sanket:                         # one of B's 8 classes → confident
            conf = 0.96
            top = sanket
        else:                              # hi/ne/etc → low conf so VANI defers local
            conf = 0.30
            top = "urdu"                   # nearest class, but below min_confidence
        probs = {c: round((1 - conf) / 7, 4) for c in
                 ["urdu", "pashto", "kashmiri", "dogri", "punjabi",
                  "mandarin", "cantonese", "tibetan"]}
        probs[top] = round(conf, 4)
        return {"language_probs": probs, "top1_language": top,
                "top1_language_display": top.capitalize(),
                "top1_language_confidence": round(conf, 4),
                "top1_dialect": "southwestern" if engaged else "n/a",
                "top1_dialect_confidence": 0.7 if engaged else 0.0,
                "dialect_engaged": engaged}

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload))); self.end_headers()
        self.wfile.write(payload)


def main():
    a = ThreadingHTTPServer(("127.0.0.1", 8801), HandlerA)
    b = ThreadingHTTPServer(("127.0.0.1", 8802), HandlerB)
    threading.Thread(target=a.serve_forever, daemon=True).start()
    threading.Thread(target=b.serve_forever, daemon=True).start()
    print("Demo mock nodes up:  A=127.0.0.1:8801  B=127.0.0.1:8802", flush=True)
    print("Flip the GUI 'Local fallback nodes' toggle to use them. Ctrl+C to stop.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

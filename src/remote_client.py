"""
src/remote_client.py – NODE-C client for the 3-node LAN integration
--------------------------------------------------------------------
Talks to two stateless HTTP services on an isolated LAN:

  • NODE-A (Gaurav)  – denoise + diarization    POST /process  -> zip
  • NODE-B (Sanket)  – LID + Mandarin dialect   POST /api/analyze (multipart)

Design contract (frozen in integration/NODE_A_GAURAV_TASKS.md and
integration/NODE_B_SANKET_TASKS.md):
  - Every method has explicit connect/read timeouts and one retry.
  - Any transport/HTTP/parse failure raises RemoteNodeError; the pipeline
    catches it and falls back to the existing local stage. Remote work is
    therefore strictly additive — it can never regress the local demo path.
  - All audio is 16 kHz mono end to end.

This module has no torch / transformers dependency — only requests, soundfile,
numpy, and the stdlib. It is safe to import even when remote mode is off.
"""

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    import requests
except ImportError:  # requests is in the venv, but never hard-crash on import
    requests = None


# ── Language-code mapping tables (integration plan §4) ───────────────────────

# VANI language code -> Gaurav's per-language clustering knob.
# Anything not listed maps to "default" (he has no tuned operating point for it).
_VANI_TO_GAURAV = {
    "zh": "mandarin",
    "ur": "urdu",
    "pa": "punjabi",
    "ps": "pashto",
}

# Sanket's 8 output classes -> VANI ASR/translation code.
# NOTE the asymmetry: hi/ne are NOT in his set, so his answer is only ever
# *accepted* when it maps here AND clears the confidence gate; otherwise the
# local MMS-LID vote (which covers a wider label set) carries the decision.
_SANKET_TO_VANI = {
    "urdu":      "ur",
    "pashto":    "ps",
    "kashmiri":  "ks",
    "dogri":     "doi",
    "punjabi":   "pa",
    "mandarin":  "zh",
    "cantonese": "zh",   # no yue model — falls back to the zh model
    "tibetan":   "bo",
}


def map_vani_to_gaurav(lang: Optional[str]) -> str:
    """VANI code -> Gaurav language tag. Unknown/None -> 'default'."""
    return _VANI_TO_GAURAV.get((lang or "").lower(), "default")


def map_sanket_to_vani(lang: Optional[str]) -> Optional[str]:
    """Sanket class -> VANI code. Returns None if not actionable by VANI."""
    return _SANKET_TO_VANI.get((lang or "").lower())


class RemoteNodeError(Exception):
    """Raised on any failure talking to a remote node (transport/HTTP/parse)."""


# NODE-B (Sanket) does NO input normalization and gates on RMS < 0.01, with its
# gate calibrated to "real speech ~0.07-0.11". Upstream/denoised audio can be much
# quieter (FLEURS clips run RMS ~0.006), which would spuriously trip the silence
# gate. Per the integration plan (§5.7 / Sanket open Q#2), NODE-C normalises audio
# before forwarding to B. We RMS-normalise toward the low end of B's speech range,
# capping the gain so a genuinely silent track is NOT amplified into false speech.
_B_TARGET_RMS = 0.08
_B_MAX_GAIN   = 20.0


def _normalize_for_lid(chunk: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
    if rms <= 1e-5:
        return chunk  # true silence — leave it so B's gate still fires
    gain = min(_B_TARGET_RMS / rms, _B_MAX_GAIN)
    return np.clip(chunk * gain, -1.0, 1.0).astype("float32")


# ── Client ───────────────────────────────────────────────────────────────────

class RemoteClient:
    """
    Thin HTTP client for NODE-A and NODE-B. Construct from the ``remote:`` block
    of config.yaml. All methods raise RemoteNodeError on failure so the caller
    can fall back locally.
    """

    def __init__(self, remote_cfg: dict, logger=None):
        self.cfg     = remote_cfg or {}
        self.logger  = logger
        self.connect_timeout = float(self.cfg.get("timeout_connect_s", 5))
        self._a = self.cfg.get("denoise_diarize", {}) or {}
        self._b = self.cfg.get("lid", {}) or {}
        if requests is None:
            self._log("warning", "requests not importable — remote mode disabled")

    # -- small helpers --------------------------------------------------------

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(f"  [remote] {msg}")

    def _post(self, url, *, read_timeout, retries=1, **kwargs):
        """POST with connect/read timeouts and one retry on transport error."""
        if requests is None:
            raise RemoteNodeError("requests library not available")
        last = None
        for attempt in range(retries + 1):
            try:
                return requests.post(
                    url, timeout=(self.connect_timeout, read_timeout), **kwargs
                )
            except requests.RequestException as e:  # ConnectionError/Timeout/etc.
                last = e
                if attempt < retries:
                    self._log("warning", f"POST {url} failed ({e}); retrying")
                    time.sleep(0.5)
        raise RemoteNodeError(f"POST {url} failed after {retries + 1} tries: {last}")

    # -- NODE-A / NODE-B health ----------------------------------------------

    def health(self, which: str, timeout: float = None) -> dict:
        """which in {'a','b'}. Returns the parsed /health JSON or raises.
        `timeout` overrides the connect timeout — pass a small value (e.g. 1.5)
        for a fast startup probe so a standalone box with unreachable LAN nodes
        doesn't hang."""
        base = (self._a if which == "a" else self._b).get("url", "")
        if not base:
            raise RemoteNodeError(f"no url configured for node {which}")
        if requests is None:
            raise RemoteNodeError("requests library not available")
        ct  = self.connect_timeout if timeout is None else timeout
        url = base.rstrip("/") + "/health"
        try:
            r = requests.get(url, timeout=(ct, 5))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise RemoteNodeError(f"health {url}: {e}")

    def available(self, which: str, timeout: float = None) -> bool:
        """True if the node answers /health with models_loaded. Never raises."""
        try:
            h = self.health(which, timeout=timeout)
            return bool(h.get("models_loaded", True))
        except RemoteNodeError:
            return False

    # -- NODE-A: denoise + diarize -------------------------------------------

    def denoise_diarize(self, wav_path, lang: str, out_dir,
                        variant: str = "robust", mode: str = "diarization-guided") -> dict:
        """
        POST a WAV to NODE-A, unpack the returned zip into ``out_dir``.

        Returns a dict:
          {
            "diarization": {...},          # parsed diarization.json
            "summary": {...},              # parsed summary.json
            "mixed_denoised": <Path>,      # full-length denoised track
            "speaker_tracks": [            # one per detected speaker
                {"label": "SPEAKER_00", "path": <Path>, "talk_time": 12.35}, ...
            ],
            "variant": "robust",
          }

        Raises RemoteNodeError on any failure (caller falls back to local denoise).
        """
        base = self._a.get("url", "")
        if not base:
            raise RemoteNodeError("NODE-A url not configured")
        url = base.rstrip("/") + "/process"
        wav_bytes = Path(wav_path).read_bytes()
        read_timeout = float(self._a.get("timeout_s", 180))

        r = self._post(
            url, read_timeout=read_timeout,
            params={"lang": lang, "variant": variant, "mode": mode},
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        if r.status_code == 503:
            raise RemoteNodeError("NODE-A busy (503) — GPU lock held")
        try:
            r.raise_for_status()
        except Exception as e:
            raise RemoteNodeError(f"NODE-A /process HTTP {r.status_code}: {e}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(out_dir)
                names = zf.namelist()
        except Exception as e:
            raise RemoteNodeError(f"NODE-A response was not a valid zip: {e}")

        def _find(fname):
            # match by basename so nested/prefixed paths in the zip still resolve
            for n in names:
                if Path(n).name == fname:
                    return out_dir / n
            return None

        diar_p = _find("diarization.json")
        summ_p = _find("summary.json")
        if not diar_p or not diar_p.exists():
            raise RemoteNodeError("NODE-A zip missing diarization.json")
        diar = json.loads(diar_p.read_text(encoding="utf-8"))
        summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p and summ_p.exists() else {}

        mixed_p = _find("mixed_denoised.wav")
        if not mixed_p or not mixed_p.exists():
            raise RemoteNodeError("NODE-A zip missing mixed_denoised.wav")

        # Build the per-speaker track list from summary.json's authoritative
        # 'files' mapping (fall back to diarization speaker keys if absent).
        speaker_tracks = []
        files = summ.get("files") or []
        if files:
            for entry in files:
                label = entry.get("label")
                clean = entry.get("clean")
                tp = _find(Path(clean).name) if clean else None
                if tp and tp.exists():
                    tt = float(diar.get("speakers", {}).get(label, {}).get("talk_time", 0.0))
                    speaker_tracks.append({"label": label, "path": tp, "talk_time": tt})
        else:
            for label, info in (diar.get("speakers") or {}).items():
                speaker_tracks.append({"label": label, "path": None,
                                       "talk_time": float(info.get("talk_time", 0.0))})

        self._log("info", f"NODE-A ok: {diar.get('n_speakers', '?')} speaker(s), "
                          f"variant={diar.get('variant', variant)}")
        return {
            "diarization":    diar,
            "summary":        summ,
            "mixed_denoised": mixed_p,
            "speaker_tracks": speaker_tracks,
            "variant":        diar.get("variant", variant),
        }

    # -- NODE-B: per-track language ID ---------------------------------------

    def identify_language(self, wav_path) -> Optional[dict]:
        """
        POST one speaker track to NODE-B. Handles tracks longer than 10 s by
        sending up to ``windows`` non-overlapping 10 s windows and averaging the
        returned language probability vectors.

        Returns a dict in VANI terms:
          {"language": "<vani code or None>", "confidence": float,
           "raw_language": "<sanket class>", "dialect": str|None,
           "dialect_confidence": float, "windows": int}
        Returns None if NODE-B reported no_speech_detected on every window.
        Raises RemoteNodeError on transport/HTTP failure.
        """
        base = self._b.get("url", "")
        if not base:
            raise RemoteNodeError("NODE-B url not configured")
        url = base.rstrip("/") + "/api/analyze"
        read_timeout = float(self._b.get("timeout_s", 30))
        n_windows = int(self._b.get("windows", 3))

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as e:
            raise RemoteNodeError(f"could not read {wav_path} for NODE-B: {e}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        win = 10 * sr
        if len(audio) <= win:
            slices = [audio]
        else:
            starts = np.linspace(0, len(audio) - win, num=n_windows, dtype=int)
            slices = [audio[s:s + win] for s in starts]

        prob_vectors = []
        best = {"language": None, "confidence": 0.0, "dialect": None,
                "dialect_confidence": 0.0}
        speech_seen = False

        for chunk in slices:
            chunk = _normalize_for_lid(chunk)   # match B's expected gain convention
            buf = io.BytesIO()
            sf.write(buf, chunk, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            r = self._post(
                url, read_timeout=read_timeout,
                files={"file": ("speaker.wav", buf.read(), "audio/wav")},
            )
            try:
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                raise RemoteNodeError(f"NODE-B /api/analyze: {e}")

            if data.get("no_speech_detected"):
                continue
            speech_seen = True

            probs = data.get("language_probs") or {}
            if probs:
                prob_vectors.append(probs)

            conf = float(data.get("top1_language_confidence", 0.0))
            if conf >= best["confidence"]:
                # NODE-B's dialect head always runs, but is only meaningful when the
                # language is Mandarin (dialect_engaged). Drop it otherwise so the GUI
                # never shows a spurious dialect for non-Mandarin speakers.
                _engaged = bool(data.get("dialect_engaged"))
                best = {
                    "language":           data.get("top1_language"),
                    "confidence":         conf,
                    "dialect":            data.get("top1_dialect") if _engaged else None,
                    "dialect_confidence": float(data.get("top1_dialect_confidence", 0.0)) if _engaged else 0.0,
                }

        if not speech_seen:
            return None

        # Average probability vectors across windows for a stabler top-1.
        raw_lang = best["language"]
        conf     = best["confidence"]
        if prob_vectors:
            keys = prob_vectors[0].keys()
            avg  = {k: float(np.mean([pv.get(k, 0.0) for pv in prob_vectors])) for k in keys}
            raw_lang = max(avg, key=avg.get)
            conf     = avg[raw_lang]

        return {
            "raw_language":       raw_lang,
            "language":           map_sanket_to_vani(raw_lang),
            "confidence":         conf,
            "dialect":            best["dialect"],
            "dialect_confidence": best["dialect_confidence"],
            "windows":            len(slices),
        }


# ── Pipeline-side helpers (pure functions, no HTTP) ──────────────────────────

_SPEAKER_LABELS = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_D"]


def _canonical_speaker(label: str, order: dict) -> str:
    """Map Gaurav's SPEAKER_00/01/... to VANI's SPEAKER_A/B/... (stable order)."""
    if label not in order:
        order[label] = len(order)
    idx = order[label]
    return _SPEAKER_LABELS[idx] if idx < len(_SPEAKER_LABELS) else f"SPEAKER_{idx}"


def assign_speakers_from_diarization(segments: list, diarization: dict) -> list:
    """
    Label ASR ``segments`` (each with 'start'/'end' seconds) by max time-overlap
    against NODE-A's diarization.json speaker segments. In-place; also returns it.

    Replaces the local diarize_module when NODE-A served the run.
    """
    speakers = (diarization or {}).get("speakers") or {}
    # Flatten to (start, end, label) intervals.
    intervals = []
    for label, info in speakers.items():
        for seg in info.get("segments", []):
            if len(seg) >= 2:
                intervals.append((float(seg[0]), float(seg[1]), label))

    if not intervals:
        for s in segments:
            s["speaker"] = "SPEAKER_A"
        return segments

    order: dict = {}
    for s in segments:
        a, b = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        best_label, best_ov = None, 0.0
        for (x, y, label) in intervals:
            ov = max(0.0, min(b, y) - max(a, x))
            if ov > best_ov:
                best_ov, best_label = ov, label
        s["speaker"] = _canonical_speaker(best_label, order) if best_label else "SPEAKER_A"
    return segments


def resolve_remote_mode(base_remote: dict, mode: str, health: dict = None) -> dict:
    """
    Produce the EFFECTIVE remote config for one pipeline run from three inputs:

      base_remote : the config.yaml `remote:` block (URLs, timeouts, per-node knobs)
      mode        : operator selection — 'auto' | 'standalone' | 'networked'
      health      : cached probe {'A': bool, 'B': bool} or None if not yet probed

    Modes:
      standalone  → remote fully off; the pipeline runs 100% locally (no network).
      networked   → trust the config and use the nodes; if a probe result is known,
                    disable an unreachable node so files don't pay its connect
                    timeout (a reconnected node returns after a re-probe).
      auto        → use only nodes the last probe found reachable; if nothing has
                    been probed yet, behave standalone (no latency) until a probe
                    runs; disable remote entirely if neither node is up.

    This is what makes one build run cleanly both standalone and on the LAN.
    """
    import copy
    r = copy.deepcopy(base_remote or {})
    mode = (mode or "auto").lower()
    dd  = r.setdefault("denoise_diarize", {})
    lid = r.setdefault("lid", {})

    if mode == "standalone":
        r["enabled"] = False
        return r

    a_up = (health or {}).get("A")
    b_up = (health or {}).get("B")

    if mode == "networked":
        r["enabled"] = True
        if health is not None:                 # skip a node we know is down
            if a_up is False: dd["enabled"] = False
            if b_up is False: lid["enabled"] = False
        return r

    # auto
    if health is None:                         # not probed yet → no-latency local
        r["enabled"] = False
        return r
    dd["enabled"]  = bool(a_up) and dd.get("enabled", True)
    lid["enabled"] = bool(b_up) and lid.get("enabled", True)
    r["enabled"]   = bool(a_up or b_up)
    return r


def dominant_language(per_speaker: list, min_confidence: float = 0.60):
    """
    Given a list of per-speaker LID dicts (from identify_language) each optionally
    carrying a 'talk_time', pick the dominant actionable VANI language weighted by
    talk time. Returns (lang_code, confidence) or (None, 0.0) if none qualify.
    """
    weights: dict = {}
    conf_of: dict = {}
    for r in per_speaker:
        if not r:
            continue
        lang = r.get("language")
        conf = float(r.get("confidence", 0.0))
        if not lang or conf < min_confidence:
            continue
        w = float(r.get("talk_time", 0.0)) or 1.0
        weights[lang] = weights.get(lang, 0.0) + w
        conf_of[lang] = max(conf_of.get(lang, 0.0), conf)
    if not weights:
        return None, 0.0
    lang = max(weights, key=weights.get)
    return lang, conf_of[lang]

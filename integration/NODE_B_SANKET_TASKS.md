# NODE-B Integration Tasks — LID + Dialect Service

**For:** Sanket's Claude Code (the VANI LID module — MMS-LID-4017 + LoRA language/dialect model).
**From:** the VANI integration team (NODE-C).
**Goal:** make your existing `demo/server.py` FastAPI app reachable from another machine on an
isolated LAN, add a health check, and freeze the checkpoint pair. **Target: working tonight.**

Good news: **your `POST /api/analyze` endpoint is already the integration point.** You are not
building anything new — you are making three small changes so a second machine can call the endpoint
you already have. No model, inference, or accuracy changes.

---

## 0. The one-paragraph summary

Bind `demo/server.py` to `0.0.0.0:8802` instead of localhost. Add a `GET /health` route. Freeze the
loaded checkpoint pair to **v1 language (`stage2_v3`) + v1 dialect (`stage4_phaseB`)** — the demo
default and the only internally-consistent pair. Keep everything else exactly as-is (the 10 s
truncation and the RMS silence gate both stay — the caller handles them). That's the whole job.

---

## 1. Network setup (do this first — ~15 min)

Isolated LAN: no internet, no gateway, no DNS. Three machines on one switch.

- Set this machine's **static IPv4** to `192.168.10.12`, mask `255.255.255.0`, **gateway blank**,
  DNS blank.
- Add a **Windows Firewall inbound rule**: allow TCP port **8802**, scoped to remote address
  `192.168.10.0/24`. Profile: Private (and Domain if applicable).
- The caller machine is `192.168.10.13`; it will run `curl http://192.168.10.12:8802/health` and
  expect a 200.

You already have `fastapi` 0.136.3 and `uvicorn` 0.49.0 — **no new dependencies needed.**

---

## 2. The three changes

### Change 1 — bind to `0.0.0.0:8802`

Wherever `demo/server.py` is launched (its `uvicorn.run(...)` call, or the command you use), bind the
host to `0.0.0.0` and the port to `8802`:

```python
uvicorn.run(app, host="0.0.0.0", port=8802)     # was 127.0.0.1 / localhost
```

or from the CLI: `uvicorn demo.server:app --host 0.0.0.0 --port 8802`. Nothing else about the app
changes.

### Change 2 — add `GET /health`

Add a lightweight route that reports the loaded checkpoint pair so the caller can light a
"NODE-B online" badge and verify it's talking to the right build:

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": True,
        "language_checkpoint": "stage2_v3",       # v1 language head
        "dialect_checkpoint": "stage4_phaseB",    # v1 dialect head
        "dialect_scheme": "mandarin_kespeech",
    }
```

Return the actual checkpoint identifiers you have loaded — the point is that NODE-C can confirm the
frozen pair (Change 3) is really what's running.

### Change 3 — freeze the checkpoint pair (a decision, not much code)

Load **v1 language (`stage2_v3`) + v1 dialect (`stage4_phaseB`)** — the pair your demo already loads
by default. **Do not** pair the v2 language checkpoint with the v1 dialect head: your own docstring
documents that this collapses dialect F1 to near-zero. This is the only combination validated as
internally consistent, so it is the one that ships tonight.

Consequence for reporting: the numbers to display/quote for this integration are the **v1** figures
(language macro-F1 **0.9764**, dialect macro-F1 **0.601**), matching the weights actually running —
not the v2 figures.

Also make sure the call path goes through your **HTTP endpoint / `demo/inference.py`'s hardcoded
`dialect_scheme = "mandarin_kespeech"`**, i.e. do not expose a path that reads the raw
`VANIModel` `punjabi_ldcil` logits — that head is architecturally present but never trained, and its
random-weight output is indistinguishable by shape from the real head. Your existing `/api/analyze`
already avoids this; just don't add a new path that bypasses it.

---

## 3. The contract NODE-C codes against (your existing endpoint — confirm it matches)

### `POST /api/analyze` — multipart `file` upload

NODE-C sends one **denoised, single-speaker WAV, 16 kHz mono** per request (it will call you once per
detected speaker). You already resample/downmix/truncate internally, so no change is needed on your
side. The response fields NODE-C reads:

```json
{
  "top1_language": "mandarin",
  "top1_language_confidence": 0.97,
  "top1_dialect": "southwestern",
  "top1_dialect_confidence": 0.72,
  "dialect_engaged": true,
  "language_probs": { "...": "all 8 classes" },
  "dialect_probs":  { "...": "7 KeSpeech classes" }
}
```

NODE-C primarily uses `top1_language` + `top1_language_confidence` (and, when the language is
Mandarin, `top1_dialect`). Please keep these field names stable. `language_probs` is a bonus —
NODE-C can average probability vectors across multiple 10 s windows of a long track, so exposing the
full distribution is useful but not required.

### The silence path stays — just keep returning it explicitly

Your RMS < 0.01 near-silence gate stays as-is. When it fires, keep returning:

```json
{ "no_speech_detected": true, "rms": 0.004 }
```

NODE-C treats this as a **legitimate response**, not an error: it simply drops that speaker track
from the language vote (it's usually a diarization artefact). So please make sure this exact shape
(a JSON 200 with `"no_speech_detected": true`) is what comes back on silent input, rather than an
HTTP error.

### The 10 s truncation stays

Do **not** add chunking. NODE-C handles clips longer than 10 s on its side: for a long speaker track
it sends up to 3 non-overlapping 10 s windows and averages the returned `language_probs`. That's
zero lines of work for you.

---

## 4. Decisions NODE-C needs from you (reply with these — they don't block you starting)

1. **Confirm the shipping checkpoint pair** is v1 language (`stage2_v3`) + v1 dialect
   (`stage4_phaseB`), and that `/health` reports it. (This is the plan of record; just confirm.)
2. Confirm the response field names in §3 are exactly what `/api/analyze` returns today (if any
   name differs, tell NODE-C the real name and it will map it — but stable names are easier).
3. Confirm the silence response is `{"no_speech_detected": true, ...}` with HTTP 200.

---

## 5. Definition of done (self-check before you hand off)

- [ ] Static IP `192.168.10.12`, firewall allows inbound TCP 8802 from `192.168.10.0/24`.
- [ ] Server binds `0.0.0.0:8802`.
- [ ] `curl http://127.0.0.1:8802/health` → 200 with the frozen checkpoint pair.
- [ ] `POST /api/analyze` with a 16 kHz mono WAV returns `top1_language` + confidence (+ dialect
      when Mandarin).
- [ ] A near-silent WAV returns `{"no_speech_detected": true, ...}` at HTTP 200 (not a 4xx/5xx).
- [ ] The running weights are the v1/v1 pair (0.9764 / 0.601), confirmed via `/health`.

Self-test once running (from this machine):

```
curl -s http://127.0.0.1:8802/health
curl -s -X POST "http://127.0.0.1:8802/api/analyze" -F "file=@some_speaker_clip_16k.wav"
```

Ping NODE-C (`192.168.10.13`) when `/health` is green and they'll run the same `curl` across the LAN
and start diffing your `top1_language` against their local LID on known clips.

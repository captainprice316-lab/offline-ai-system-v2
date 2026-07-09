# NODE-A Integration Tasks — Diarization + Denoising Service

**For:** Gaurav's Claude Code (the DiariZen + DeepFilterNet3 module).
**From:** the VANI integration team (NODE-C).
**Goal:** turn your existing command-line diarize + denoise scripts into a small, always-on HTTP
service so a second machine on an isolated LAN can send it a WAV and get back diarization +
denoised audio. **Target: working tonight.**

You do **not** need to know anything about VANI to do this. This document is the complete contract.
Everything your module already does stays the same — you are adding one network wrapper and one
small output file. Nothing about your models, checkpoints, or accuracy changes.

---

## 0. The one-paragraph summary

Wrap `stage_diarize.py` + `stage_extract_denoise.py` in a single FastAPI file, `server_a.py`, that
(1) loads your models **once** at startup and holds them in memory, (2) serves `GET /health` and
`POST /process`, (3) serializes GPU work behind a lock, and (4) emits one **new** output file,
`mixed_denoised.wav` — a single full-length denoised track on the original wall-clock timeline. Bind
it to `0.0.0.0:8801`. That's the whole job.

---

## 1. Network setup (do this first — ~15 min)

This is an isolated LAN: no internet, no gateway, no DNS. Three machines on one switch.

- Set this machine's **static IPv4** to `192.168.10.11`, mask `255.255.255.0`, **gateway blank**,
  DNS blank.
- Add a **Windows Firewall inbound rule**: allow TCP port **8801**, scoped to remote address
  `192.168.10.0/24` (LAN only). Profile: Private (and Domain if applicable).
- Confirm the caller machine (`192.168.10.13`) can reach you: after the server is up, they will run
  `curl http://192.168.10.11:8801/health` and expect a 200. You can test locally first with
  `curl http://127.0.0.1:8801/health`.

Install these while you still have internet (do it now, before going offline):

```
pip install fastapi uvicorn python-multipart
```

If staging those wheels is a problem, the same contract can be served from Python's stdlib
`http.server.ThreadingHTTPServer` with **zero new dependencies** — the `/process` body is a raw
octet-stream precisely so multipart parsing is never required. Prefer FastAPI if you can install it;
fall back to `http.server` only if you can't.

---

## 2. The HTTP contract (this is the interface NODE-C codes against — match it exactly)

### `GET /health`

Returns 200 with JSON as soon as models are loaded and ready:

```json
{ "status": "ok", "models_loaded": true, "variants": ["clean", "robust"] }
```

Return `"models_loaded": false` (still 200) if you respond before load finishes. NODE-C polls this
to light a "NODE-A online" badge and to decide whether to route to you at all.

### `POST /process`

**Query parameters:**

| Param | Values | Meaning |
|---|---|---|
| `lang` | `mandarin` \| `urdu` \| `punjabi` \| `pashto` \| `default` | Language tag for your per-language clustering knobs. **Always honor it. `"default"` is not a safe no-op** — set your `lang_knobs` from this exactly as your stage scripts already do. NODE-C will send a real tag whenever it has one. |
| `variant` | `clean` \| `robust` | Which DiariZen checkpoint. **These are not interchangeable** (your handover §9): `robust` (real-DMR, ~32% DER) is *worse* on clean/synthetic audio; `clean` (~12.7% DER) is worse on real radio. Use exactly the one requested. |
| `mode` | `diarization-guided` \| `blind` | Default `diarization-guided`. `blind` is your MossFormer2 2-track mode; NODE-C will normally send `diarization-guided`. |

**Request body:** raw WAV bytes, `Content-Type: audio/wav`. The audio is **16 kHz mono**. You may
resample internally as you already do, but **emit your outputs at 16 kHz mono** (see §4).

**Response 200:** `application/zip` (`Content-Type: application/zip`) containing exactly:

```
diarization.json            # your existing schema, unchanged
summary.json                # your existing schema, unchanged
mixed_denoised.wav          # NEW — see §3. 16 kHz mono, full original length
Speaker_1_Denoised.wav      # your existing per-speaker denoised tracks, 16 kHz mono
Speaker_2_Denoised.wav
...                         # one per detected speaker, N adapts
```

You do **not** need to include the raw (non-denoised) `Speaker_N.wav` or `speakers.rttm` — NODE-C
doesn't consume them. Including them is harmless if it's easier to just zip the whole output folder.

**Response 503:** return this (empty or a short JSON body) when the GPU lock is already held by
another in-flight request. NODE-C will treat 503 as "busy, fall back locally," **not** as a crash.
Do **not** let a second concurrent request OOM the card — return 503 instead.

**Response 4xx/5xx on real errors:** any decode failure or exception → non-200. NODE-C has a local
fallback, so a clean error is always better than a hang. Keep responses fast to fail.

---

## 3. The one new output: `mixed_denoised.wav` (this is the only real code addition)

**Why NODE-C needs it:** your current per-speaker denoised tracks are *concatenations* of each
speaker's segments (your `summary.json` shows e.g. `duration: 11.77` for a speaker whose segments
sum to 12.35 s), so their internal timestamps are no longer wall-clock. NODE-C's entire downstream
(chunker, transcription offsets, keyword timing, timeline view, audio player) is built around **one
full-length audio stream on one wall-clock timeline**. So it needs a single denoised file that is
the same length and timeline as the input, with each speaker's denoised segments placed back where
they occurred.

**Where:** add this at the end of `stage_extract_denoise.py`, after per-speaker denoising, using the
metadata you already produce.

```python
import numpy as np, soundfile as sf

# duration + samplerate come from diarization.json; sr should be 16000 here
n   = int(round(duration * samplerate))
mix = np.zeros(n, dtype=np.float32)

for entry in summary["files"]:                       # authoritative speaker→file mapping
    den, sr = sf.read(entry["clean"], dtype="float32")   # Speaker_k_Denoised.wav
    if den.ndim > 1:                                  # safety: force mono
        den = den.mean(axis=1)

    seg_lens = []
    cursor = 0
    for (start, end) in diarization["speakers"][entry["label"]]["segments"]:
        k = int(round((end - start) * sr))
        seg = den[cursor:cursor + k]                  # DFN3 preserves length
        s0 = int(round(start * sr))
        m  = min(len(seg), n - s0)
        mix[s0:s0 + m] += seg[:m]                     # += so overlapped speech sums
        cursor += k
        seg_lens.append(k)

    # sanity check: the denoised track should equal the sum of its segment lengths
    if abs(len(den) - sum(seg_lens)) > 2:             # ±2 samples for 48k resample round-trip
        print(f"##WARN mixed_denoised: {entry['clean']} length {len(den)} "
              f"!= sum(segments) {sum(seg_lens)} — timeline may be misaligned")

np.clip(mix, -1.0, 1.0, out=mix)
sf.write("mixed_denoised.wav", mix, 16000)            # 16 kHz mono
```

**Assumptions this relies on** (both should already hold in your pipeline — assert, don't assume
silently): each denoised track is the *chronological* concatenation of that speaker's segments, and
DeepFilterNet3 preserves length. The `abs(...) > 2` check above logs a warning rather than producing
a silently misaligned file.

**Fallback if this is genuinely hard:** run a single DFN3 pass over the *original mixed input* and
emit that as `mixed_denoised.wav`. NODE-C works either way — but the per-speaker reconstruction above
is strongly preferred, because a single pass over the mix loses the speaker-wise denoising benefit
that is the main value your module adds. Try the reconstruction first.

---

## 4. Three things the wrapper must do (priority order — #1 is the biggest win)

1. **Load models once, hold them.** Your reference scripts reload DiariZen + DFN3 on every call —
   that's 30–60 s + 20–40 s of pure model-load per clip (your handover §9), which dominates runtime
   on 10–45 s audio. Load **both** DiariZen checkpoints (`clean` and `robust`) and the DFN3 `ep26`
   checkpoint **at process startup** and keep them resident. This turns a ~90 s call into a ~5–10 s
   call and is the single most important part of this task. Pick the checkpoint per request from the
   `variant` param; don't reload.

2. **Serialize GPU work.** One DiariZen instance sits at ~6 GB of 8 GB VRAM at 97–99% util (your
   §9) — there is no headroom for a second concurrent job. Guard all GPU work with a single
   `threading.Lock`, run `uvicorn server_a:app --host 0.0.0.0 --port 8801 --workers 1`, and return
   **503** if the lock is already held rather than starting a second job. NODE-C processes one clip
   at a time, so contention should be rare, but the guard must exist.

3. **Emit `mixed_denoised.wav`** (§3).

Keep everything **16 kHz mono** on the way out (input is 16 kHz; resample back to 16 kHz on output).
This matters because NODE-C forwards your `Speaker_N_Denoised.wav` to a third machine that expects
16 kHz — emitting a different rate would cause a second resample there.

---

## 5. Decisions NODE-C needs from you (reply with these — they don't block you starting)

1. **Default `variant`** — are tonight's demo clips real DMR/radio recordings (→ `robust`) or
   clean/synthetic-noise (→ `clean`)? NODE-C will send it per request regardless, but it needs a
   sensible default. Your call.
2. **Confirm `mixed_denoised.wav` reconstruction works** on at least one multi-speaker clip (the
   `##WARN` line above should not fire).
3. Confirm the exact filenames in the zip match §2 (or tell NODE-C the real names — it reads
   `summary.json`'s `files` array as authoritative, so non-standard names are fine as long as
   `summary.json` points at them).

---

## 6. Definition of done (self-check before you hand off)

- [ ] Static IP `192.168.10.11`, firewall allows inbound TCP 8801 from `192.168.10.0/24`.
- [ ] `server_a.py` loads DiariZen (both variants) + DFN3 once at startup.
- [ ] `curl http://127.0.0.1:8801/health` → `{"status":"ok","models_loaded":true,...}`.
- [ ] `POST /process` with a real WAV returns a zip containing `diarization.json`, `summary.json`,
      `mixed_denoised.wav`, and `Speaker_N_Denoised.wav` ×N — all 16 kHz mono.
- [ ] A second concurrent `POST /process` returns **503**, not an OOM/crash.
- [ ] `mixed_denoised.wav` is the full input length and audibly correct (each speaker in the right
      place on the timeline).
- [ ] Warm call latency is single-digit-to-low-teens seconds, not ~90 s (proves models are held).

Self-test once running (from this machine):

```
curl -s http://127.0.0.1:8801/health
curl -s -X POST "http://127.0.0.1:8801/process?lang=mandarin&variant=clean&mode=diarization-guided" \
     -H "Content-Type: audio/wav" --data-binary @some_test_clip.wav -o out.zip
# then unzip out.zip and confirm mixed_denoised.wav plays back full-length
```

That's everything. Ping NODE-C (`192.168.10.13`) when `/health` is green and they'll run the same
`curl` across the LAN.

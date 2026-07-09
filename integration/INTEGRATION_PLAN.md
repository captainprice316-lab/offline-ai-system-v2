# VANI 3-Node Integration Plan

**Goal:** run Gaurav's denoise+diarization module, Sanket's LID/dialect module, and the existing
VANI system on three separate machines connected by an isolated LAN (no internet, no gateway),
with the final demo and all output presented through VANI's existing Streamlit GUI.

**Constraint given:** easiest, lowest-cost implementation with the fewest changes.

Written 2026-07-09 against `integration/gaurav's module/integration.md`,
`integration/sanket's module/VANI_LID_INTEGRATION_CONTEXT.md`, and this repo's `src/pipeline.py`.

---

## 1. Topology

Three nodes, one unmanaged gigabit switch, static IPs, `/24`, **no default gateway, no DNS**.

| Node | Owner | Hardware | Role | Port |
|---|---|---|---|---|
| **NODE-A** | Gaurav | Win10, Quadro P4000 8 GB | Denoise + diarization service | 8801 |
| **NODE-B** | Sanket | RTX 4050 6 GB | LID + Mandarin dialect service | 8802 |
| **NODE-C** | you | ROG G16, RTX 5060 8 GB | VANI pipeline + Streamlit GUI + **orchestrator** | 8501 |

Suggested addressing: `192.168.10.11` (A), `.12` (B), `.13` (C), mask `255.255.255.0`,
gateway blank.

**NODE-C is the only orchestrator.** A and B are stateless request/response HTTP services that know
nothing about each other or about VANI. No shared filesystem, no message broker, no queue, no
container runtime. This matches Gaurav's explicit instruction not to assume a shared filesystem
between his module and its consumers.

Ports 8801/8802 need a Windows Firewall inbound TCP allow rule scoped to `192.168.10.0/24`.
Ollama (11434) and Streamlit (8501) stay bound to localhost on C — they are not part of the LAN
contract.

---

## 2. Data flow for one intercept

```
  Analyst uploads clip.wav in Streamlit                          [NODE-C]
        │
   1. VAD (local, unchanged)                                     [NODE-C]
        │
   2. MMS-LID-256 coarse probe on the VAD'd audio                [NODE-C]
        │   → coarse language hint (pa / ur / ps / zh / ...)
        │
   3. POST /process  { wav, lang=<hint>, variant=robust }        [NODE-C → NODE-A]
        │   ← zip: diarization.json, summary.json,
        │          mixed_denoised.wav, Speaker_N_Denoised.wav ×N
        │
   4. POST /api/analyze  { Speaker_N_Denoised.wav }  × N         [NODE-C → NODE-B]
        │   ← per-speaker { top1_language, confidence, dialect }
        │
   5. Chunking + ASR on mixed_denoised.wav, using the            [NODE-C]
        │   dominant speaker's language to pick the Whisper model
        │
   6. Speaker labels attached to ASR segments by time-overlap    [NODE-C]
        │   against A's diarization.json  (replaces diarize_module)
        │
   7. Language vote — B's result enters LanguageRouter as the    [NODE-C]
        │   audio vote; local MMS-LID is the fallback vote
        │
   8. Translation → Keywords → ISUM → SQLite → GUI               [NODE-C]
            (all completely unchanged)
```

### Why the local MMS-LID probe stays in step 2

There is a genuine ordering conflict between the two modules:

- Gaurav's diarizer **needs a language tag up front** — each language has its own tuned clustering
  operating point, and leaving it as `"default"` measurably hurts accuracy (his §7, §10: one
  observed case found 2 speakers instead of the correct 4). It does no language ID of its own.
- Sanket's LID is **trained on a post-denoiser distribution** (his §3.3: augmentation explicitly
  excludes reverb, models a denoised telephone-band signal), so it wants to run *after* Gaurav.

VANI already resolves this for free: it has a local `mms-lid-256` probe that currently runs at
stage 3.5. Moving that call earlier (onto `vad_out` instead of `pre_out`) costs nothing — it is a
call the pipeline already makes — and gives Gaurav his language tag. Sanket's model then runs on
clean per-speaker tracks and produces the authoritative answer. Neither of them needs to change to
accommodate the other.

---

## 3. What changes on each node

### NODE-A (Gaurav) — one wrapper + one 10-line output addition

**Everything about his existing stage scripts stays as-is.** He adds a single file, `server_a.py`,
that wraps `stage_diarize.py` + `stage_extract_denoise.py`:

```python
GET  /health   → {"status":"ok", "models_loaded": true, "variants": ["clean","robust"]}
POST /process?lang=punjabi&variant=robust&mode=diarization-guided
     body:  raw WAV bytes, Content-Type: audio/wav
     200 →  application/zip { diarization.json, summary.json,
                              mixed_denoised.wav, Speaker_1_Denoised.wav, ... }
     503 →  busy (GPU lock held)
```

Three things the wrapper must do, in priority order:

1. **Load both DiariZen checkpoints and the DFN3 `ep26` checkpoint once at process start and hold
   them.** His reference scripts reload models on every invocation, which costs 30–60 s
   (diarization) + 20–40 s (denoise) per clip and dominates runtime on 10–45 s audio. Persisting
   the models turns a ~90 s call into a ~5–10 s call. This is the single biggest win in the entire
   integration and it is a property of the wrapper, not of his models.
2. **Serialize GPU work behind a `threading.Lock`, and run uvicorn with `--workers 1`.** A single
   DiariZen instance sits at ~6 GB of his 8 GB card at 97–99 % utilisation — there is no headroom
   for a second concurrent job. Return `503` if the lock is held rather than OOM-ing.
3. **Emit one new output file, `mixed_denoised.wav`.**

Why `mixed_denoised.wav` matters: VANI's entire downstream — chunker, ASR segment offsets, keyword
segment mapping, timeline tab, DB segment rows, GUI audio player — is built around **one
full-length audio stream on one wall-clock timeline**. His current outputs are per-speaker
*concatenated* tracks (`summary.json` shows `duration: 11.77` for a speaker whose segments sum to
12.35 s), so their internal timestamps are no longer wall-clock. Handing VANI N concatenated files
would require rewriting timestamp handling across five modules.

Reconstructing a full-length denoised mix is ~10 lines at the end of `stage_extract_denoise.py`,
after per-speaker denoising:

```python
import numpy as np, soundfile as sf

n = int(round(duration * samplerate))          # from diarization.json
mix = np.zeros(n, dtype=np.float32)

for entry in summary["files"]:                 # authoritative speaker → file mapping
    den, sr = sf.read(entry["clean"], dtype="float32")
    cursor = 0
    for (start, end) in diarization["speakers"][entry["label"]]["segments"]:
        k   = int(round((end - start) * sr))
        seg = den[cursor:cursor + k]           # DFN3 preserves length
        s0  = int(round(start * sr))
        m   = min(len(seg), n - s0)
        mix[s0:s0 + m] += seg[:m]              # += so overlapped speech sums
        cursor += k

np.clip(mix, -1.0, 1.0, out=mix)
sf.write("mixed_denoised.wav", mix, sr)
```

This assumes each denoised track is the chronological concatenation of that speaker's segments and
that DeepFilterNet3 preserves length. Both should hold, but he should assert
`len(den) == sum(segment_lengths)` (allow ±2 samples for the internal 48 kHz resample round-trip)
and log a warning rather than silently mis-aligning.

*If he pushes back on this:* the fallback is a single extra DFN3 pass over the original mixed input
(one call, ~1–2 s warm), emitted as `mixed_denoised.wav`. VANI's downstream then works unchanged,
but the ASR path loses the benefit of speaker-wise denoising — which is the main value his module
adds. Prefer the reconstruction.

**Dependencies to install while NODE-A still has internet:** `fastapi`, `uvicorn`,
`python-multipart`. If staging those wheels isn't practical, the same contract can be served from
`http.server.ThreadingHTTPServer` in ~80 lines with zero new dependencies — the endpoint takes a
raw octet-stream body precisely so that multipart parsing is never required.

### NODE-B (Sanket) — three small changes, no new code paths

His `demo/server.py` FastAPI app **is already the integration point.** `POST /api/analyze` with a
multipart `file` upload does exactly what VANI needs, and it goes through `demo/inference.py`,
which hardcodes `dialect_scheme = "mandarin_kespeech"` and therefore never touches the untrained
`punjabi_ldcil` head. That trap (his §4, §7.6) is avoided *for free* by calling the HTTP endpoint
instead of `VANIModel` directly. Do not bypass it.

Changes:

1. **Bind `0.0.0.0:8802`** instead of localhost.
2. **Add `GET /health`** returning the loaded checkpoint pair.
3. **Freeze the checkpoint pairing at v1 language (`stage2_v3`) + v1 dialect (`stage4_phaseB`)** —
   the demo default, and the only combination validated as internally consistent. Pairing the v2
   language checkpoint with the v1 dialect head collapses dialect F1 to near-zero, per his own
   docstring. The GUI must then display 0.9764 / 0.601 as the language/dialect numbers, not the v2
   figures. This is a decision, not a code change; it needs to be made explicitly and once.

Nothing else on his side changes. In particular:

- **The 10 s truncation stays.** VANI handles it client-side: for a speaker track longer than 10 s,
  send up to 3 non-overlapping 10 s windows and average the returned probability vectors. That is
  ~10 lines in VANI's client and zero lines on his side.
- **His RMS < 0.01 silence gate stays.** VANI must handle `{"no_speech_detected": true}` as a
  legitimate response — the correct action is to drop that speaker track from the LID vote (it will
  typically be a diarization artefact), not to treat it as an error.

### NODE-C (VANI) — the only place with real work

**New file `src/remote_client.py`** (~150 lines): `health()`, `denoise_diarize()`,
`identify_language()`. Uses `requests` (already installed), explicit connect/read timeouts, one
retry, and raises a typed `RemoteNodeError` on failure. Also holds the language-code mapping tables
(§4 below).

**New `remote:` block in `config.yaml`:**

```yaml
remote:
  enabled: true                # master switch — false restores exact current behaviour
  fallback_on_error: true      # any remote failure → run the existing local stage instead
  timeout_connect_s: 5

  denoise_diarize:             # NODE-A
    enabled: true
    url: http://192.168.10.11:8801
    timeout_s: 180
    variant: robust            # clean | robust — NOT interchangeable, see §5
    mode: diarization-guided   # | blind  (blind is hard-capped at 2 speakers)
    call_on_clean: false       # skip the round trip when VANI's SNR gate says clean
    use_mixed_track: true      # Phase 1. false → per-speaker ASR (Phase 2)

  lid:                         # NODE-B
    enabled: true
    url: http://192.168.10.12:8802
    timeout_s: 30
    per_speaker: true
    min_confidence: 0.60       # below this, defer to local MMS-LID
    windows: 3                 # >10 s tracks: N windows, client-side prob averaging
```

**Modified `src/pipeline.py`** (~120 lines net). The existing stage structure survives intact:

| Stage | Change |
|---|---|
| 1 VAD | unchanged |
| — | MMS-LID coarse probe **moved here** from 3.5, now runs on `vad_out` |
| 2 Preprocessing | if NODE-A used: skip denoise/bandpass, set `pre_out = mixed_denoised.wav`. **Keep `normalize: true`** — it costs nothing and neutralises any gain-convention mismatch from DFN3 (Sanket's open question #2). |
| 3 Chunking | unchanged |
| 3.5 Pre-ASR probe | if NODE-B used: per-speaker LID → dominant speaker's language selects the Whisper model. Else existing local probe. |
| 4 ASR | unchanged |
| 4.5 Diarization | if NODE-A used: label segments by time-overlap against `diarization.json`. Else existing `diarize_module`. |
| 5 Language ID | B's result feeds `LanguageRouter` as `mms_lang`/`mms_conf`. **`LanguageRouter` itself does not change.** |
| 6–8 Translation / Keywords / ISUM | unchanged |

The `result` dict gains `remote_nodes` (which nodes served this run), `speakers[]` (per-speaker
label, talk time, denoised track path, language, confidence, dialect), `diarizer_variant`, and
`der_source`. Everything already in the dict stays.

**Modified `app.py`** (~80 lines, purely additive): a per-speaker card panel (language, dialect,
confidence, an `st.audio` player for each `Speaker_N_Denoised.wav`) and two node-health badges in
the sidebar. No existing tab changes.

**Every remote path is behind a feature flag with `fallback_on_error: true`.** If NODE-A is
unplugged mid-demo, VANI silently runs its existing local denoise + `diarize_module` and the demo
continues. If NODE-B is unplugged, the local MMS-LID vote carries the language decision, exactly as
it does today. This makes the integration strictly additive and means **the currently-working demo
path cannot be regressed by any of this work.**

---

## 4. Language code mapping (get this right or the demo silently degrades)

**VANI → NODE-A** (his tuned per-language clustering knobs):

| VANI | Gaurav |
|---|---|
| `zh` | `mandarin` |
| `ur` | `urdu` |
| `pa` | `punjabi` |
| `ps` | `pashto` |
| anything else (`hi`, `ne`, `ks`, `doi`, …) | `default` — **log a warning**, he has no tuned operating point |

**NODE-B → VANI** (his 8 classes → VANI's ASR/translation codes):

| Sanket | VANI | Has a fine-tuned VANI ASR model? |
|---|---|---|
| `urdu` | `ur` | yes |
| `pashto` | `ps` | yes |
| `kashmiri` | `ks` | yes |
| `dogri` | `doi` | no — default Whisper + IndicTrans2 |
| `punjabi` | `pa` | yes (routed to SeamlessM4T) |
| `mandarin` | `zh` | yes |
| `cantonese` | `zh` | no `yue` model — falls back to the `zh` model |
| `tibetan` | `bo` | no — default Whisper |

**Critical asymmetry: `hi` and `ne` are not in Sanket's 8-class set**, but VANI supports both and
has fine-tuned Whisper models for both. Therefore:

- Accept B's answer as the audio vote when `confidence ≥ min_confidence` **and** the class maps to
  something VANI can act on.
- Otherwise fall back to the local MMS-LID result, which covers a much wider label set.
- Never delete the local MMS-LID path. It is the safety net for Hindi, Nepali, and for the case
  where NODE-B is down.

---

## 5. Correctness items that will bite if ignored

1. **Model persistence on NODE-A is not optional** (§3). Without it every clip costs ~90 s.
2. **Serialize NODE-A.** 6 GB of 8 GB VRAM at 97–99 % util; two concurrent jobs will OOM.
3. **`variant: clean` and `variant: robust` are not interchangeable.** The robust (real-DMR)
   checkpoint is *measurably worse* on clean/synthetic audio, confirmed by his A/B testing. Pick
   per clip source: real radio recordings → `robust` (32.02 % DER), clean/synthetic → `clean`
   (12.74 % DER). Expose it per request; don't hard-code it into VANI.
4. **Always send NODE-A a real language tag.** `"default"` is not a safe no-op.
5. **Never call Sanket's `VANIModel` directly** — the `punjabi_ldcil` head is architecturally
   present, never trained, and its random-weight logits are indistinguishable from the real
   `mandarin_kespeech` head's by shape or file. Go through his HTTP endpoint.
6. **Keep everything 16 kHz mono end to end.** VANI resamples to 16 k at VAD, A resamples back to
   the input rate on output, B expects 16 k. Held at 16 k throughout, nothing is ever resampled
   twice — which closes Sanket's open questions #1 and #4 (his model never sees reverberant or
   double-resampled audio).
7. **Normalise A's output before forwarding to B.** VANI's `AudioPreprocessor` already does this;
   just don't turn it off in remote mode. Closes Sanket's open question #2 (unvalidated gain
   convention across the denoiser boundary).
8. **Pashto will be the weakest end-to-end language.** Gaurav's diarizer is worst there
   (27.05 % DER, an unsolved speaker-overcounting problem needing more training data, not tuning),
   and VANI's own robustness eval already shows Pashto at 53–87 % WER with MMS-LID as the critical
   dependency. Choose demo clips accordingly and don't lead with Pashto.
9. **Licensing.** DiariZen's weights are CC BY-NC 4.0, Sanket's MMS-LID-4017 backbone is
   CC-BY-NC-4.0, and VANI's own `mms-lid-256` is in the same family. The integrated system is
   therefore **research / non-commercial only**. This is not a blocker for the demo, but it is a
   hard blocker for productization and should be raised now rather than discovered later.
10. **No speaker re-ID across files** in either module. VANI's `cross_file_reid` is already
    disabled (MFCC cosine gave a 77 % false-match rate). Note that Gaurav's DFN3 checkpoint was
    specifically selected because it *preserves* speaker identity (15.62 % EER, better than the raw
    signal at 18.75 %) — so running ECAPA embeddings on his denoised per-speaker tracks is the
    natural way to finally re-enable cross-file re-ID. Out of scope for this integration; worth
    recording as the obvious next capability.

---

## 6. Cost

**Zero.** No new hardware, no broker, no container runtime, no cloud.

| Node | New software |
|---|---|
| A | `fastapi`, `uvicorn`, `python-multipart` — install while still online. (Or zero deps via `http.server`.) |
| B | none — already has `fastapi` 0.136.3 / `uvicorn` 0.49.0 |
| C | none — `requests` already present in the venv |
| Network | one unmanaged gigabit switch + 3 cables |

Payload sizes are trivial: a 45 s 16 kHz mono WAV is ~1.4 MB; a 4-speaker response zip is under
6 MB. Transfer time on gigabit is negligible against a 5–10 s inference.

**Expected warm latency**, against VANI's current 33 s warm baseline: `+5–10 s` for NODE-A (minus
~3–4 s of local denoise+diarization that it replaces), `+0.5–1.5 s` for NODE-B (minus the ~1–2 s
local MMS probe it replaces). **Estimate ~38–42 s warm end-to-end.** Acceptable for the demo; the
Phase B speedup work in `project_pipeline_speedup_plan` is orthogonal and still applies.

---

## 7. Phasing

| Phase | Where | Effort | Deliverable |
|---|---|---|---|
| 0 | all | ½ day | Static IPs, firewall rules, `GET /health` on A and B, `curl` from C proves reachability |
| 1 | A | 1 day | `server_a.py`: persistent models, GPU lock, `/process` → zip, `mixed_denoised.wav` |
| 2 | B | ½ day | Bind `0.0.0.0`, `/health`, checkpoint pair frozen |
| 3 | C | 1–2 days | `remote_client.py`, `remote:` config block, `pipeline.py` wiring — both flags default **off**, then enabled one at a time |
| 4 | C | 1 day | `app.py` per-speaker panel + node health badges |
| 5 | all | ½ day | End-to-end rehearsal on the demo clips across all 4 languages; measure wall time; choose `variant` per clip |

**~4–5 working days, and roughly 70 % of it is on NODE-C.** Phases 1 and 2 are independent and can
run in parallel with Phase 3, since VANI's flags default to off.

Order of enabling in Phase 3 matters: turn on `remote.lid` first (small payload, fast, easy to
diff against the local MMS-LID answer on known clips), then `remote.denoise_diarize`.

---

## 8. Alternatives considered and rejected

| Option | Why not |
|---|---|
| **Shared SMB folder + folder watching** | Fewer lines than HTTP, but no request/response, no health check, no clean completion signal, and Gaurav's handover explicitly says not to assume a shared filesystem. The stdlib `http.server` fallback is simpler *and* better. |
| **Message queue (Redis / RabbitMQ / ZeroMQ)** | Needs a broker process, an extra install on an offline LAN, and an extra failure mode — for a workload of one analyst processing one clip at a time. |
| **gRPC** | `protoc`, codegen, new dependencies on all three nodes. No benefit at this scale. |
| **Consolidate all three modules onto one machine** | Hard blockers: torch 2.1.1+cu121 (A) vs 2.5.1+cu121 (B) vs VANI's own stack; and DiariZen alone needs ~6 GB of VRAM while VANI already runs Whisper + SeamlessM4T + NLLB + Ollama on an 8 GB card. The three-machine split isn't only a constraint here — it's the right architecture. |
| **Invoke A and B as remote CLI subprocesses (SSH / PsExec)** | Reintroduces the 50–100 s per-clip model reload that the whole design is trying to eliminate. |
| **Per-speaker ASR in Phase 1** | Architecturally the right end state (Gaurav's §7 explicitly recommends running STT on `Speaker_N_Denoised.wav`, not the mixed input) and it should be **Phase 2**. But it touches segment-timestamp handling in the chunker, ASR loop, keyword mapper, timeline tab, and DB writer. `mixed_denoised.wav` delivers most of the denoising benefit for a fraction of the change surface. |

---

## 9. Open decisions needed before Phase 1

1. **NODE-B checkpoint pair** — confirm v1 language + v1 dialect ships (see §3). Sanket's call.
2. **Default `variant`** for NODE-A — depends on whether the demo clips are real DMR recordings or
   synthetic-noise. Gaurav's call; make it per-request regardless.
3. **Demo clip set** — must be languages all three modules cover well. The safe intersection is
   **Punjabi (1.38 % DER), Mandarin (7.87 % DER), Urdu (14.67 % DER)**. Pashto is supported by all
   three but is the weakest link in two of them (§5.8).
4. **Whether Phase 2 (per-speaker ASR) is in scope** for this demo or deferred. Recommendation:
   defer — `mixed_denoised.wav` is enough to show the integration working end to end.

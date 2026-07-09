# VANI 3-Node Integration — Process Overview

**What this is:** the end-to-end walkthrough of how a single radio intercept flows through the three
machines — **NODE-A** (denoise + diarization), **NODE-B** (language + dialect ID), and **NODE-C**
(VANI: VAD, ASR, translation, summary, GUI, and the sole orchestrator). Written against the
implemented NODE-C code (`src/pipeline.py`, `src/remote_client.py`) and the two partner contracts
(`NODE_A_GAURAV_TASKS.md`, `NODE_B_SANKET_TASKS.md`). Companion to `INTEGRATION_PLAN.md`.

Last updated: 2026-07-09.

---

## 1. The three machines

| Node | Owner | Role | Address | Kind |
|---|---|---|---|---|
| **NODE-A** | Gaurav | Denoise (DeepFilterNet3) + speaker diarization (DiariZen) | `192.168.10.11:8801` | Stateless HTTP service |
| **NODE-B** | Sanket | Language ID (8 langs) + Mandarin dialect ID (MMS-LID-4017) | `192.168.10.12:8802` | Stateless HTTP service |
| **NODE-C** | this repo | VAD, ASR, translation, keywords, ISUM, storage, GUI, **orchestrator** | `192.168.10.13:8501` | Streamlit app |

- One unmanaged gigabit switch, isolated `/24` LAN, **no internet, no gateway, no DNS**.
- **A and B are stateless** — they know nothing about each other or about VANI. No shared filesystem,
  no broker, no queue.
- **NODE-C is the only orchestrator.** It calls A and B as plain request/response HTTP services and
  assembles the final result.

---

## 2. End-to-end flow for one clip

```
Analyst uploads clip.wav in the Streamlit GUI                          [NODE-C]
        │
 1. VAD — strip silence, keep speech regions (Silero)                  [NODE-C]
        │   → vad.wav
        │
 2. Coarse MMS-LID probe on vad.wav                                    [NODE-C]
        │   → coarse language hint (e.g. "zh")  →  Gaurav tag "mandarin"
        │
 3. POST /process?lang=mandarin&variant=robust  (raw WAV bytes)  ────► [NODE-A]
        │                                                          A denoises per
        │                                                          speaker + diarizes
        │   ◄─── zip: diarization.json, summary.json,             ◄───┘
        │             mixed_denoised.wav, Speaker_1_Denoised.wav ×N
        │
 4. Normalize A's output; use mixed_denoised.wav as the audio          [NODE-C]
        │   (skip C's local denoise/bandpass; keep normalize)
        │
 5. POST /api/analyze  (each Speaker_N_Denoised.wav, normalized) ────► [NODE-B]
        │                                                          B runs LID +
        │   ◄─── per speaker: {top1_language, confidence, dialect} ◄─┘  dialect
        │
 6. Aggregate B's per-speaker LID → dominant language by talk time     [NODE-C]
        │   → picks the language-specific Whisper model
        │
 7. Chunk mixed_denoised.wav (VAD-aware)                               [NODE-C]
        │
 8. ASR — Whisper large-v3 CT2 (or SeamlessM4T for pa/ne),             [NODE-C]
        │   language forced from B's answer  → transcript + segments
        │
 9. Label each ASR segment by time-overlap vs A's diarization.json     [NODE-C]
        │   → SPEAKER_A / SPEAKER_B / …
        │
10. LanguageRouter vote: B's LID = audio vote, FastText = text vote    [NODE-C]
        │   → final language + translation route
        │
11. Translation — NLLB-200 (IndicTrans2 for Dogri)  → English          [NODE-C]
        │
12. Keyword detection → threat level + categories                     [NODE-C]
        │
13. ISUM — Ollama gemma3:4b → structured intelligence summary (5W)     [NODE-C]
        │
14. Persist to SQLite, render in GUI                                   [NODE-C]
            (per-speaker cards, noisy-vs-denoised audio, map, timeline)
```

---

## 3. Why the ordering is what it is

There is a genuine circular dependency between the two partner modules, and NODE-C's existing local
probe resolves it for free:

- **NODE-A needs a language tag *before* it runs.** Its clustering has a per-language tuned operating
  point; leaving it `"default"` measurably hurts (one observed case found 2 speakers instead of 4).
  It does no language ID itself.
- **NODE-B wants to run *after* NODE-A.** Its model is trained on a post-denoiser, non-reverberant
  distribution, so it expects clean per-speaker tracks — i.e. NODE-A's output.

Resolution: NODE-C already has a local `mms-lid-256` probe. Running it early (step 2, on `vad.wav`)
gives NODE-A the coarse language tag it needs. NODE-B then runs on NODE-A's clean per-speaker tracks
and produces the authoritative answer. **Neither partner changes to accommodate the other.**

---

## 4. What each node does internally

### NODE-A (Gaurav) — "who spoke when, and clean it up"

- Loads DiariZen (WavLM + Conformer) and DeepFilterNet3 (`ep26`) **once at startup** and holds them
  resident — the single biggest performance win (a ~90 s cold call becomes ~5–10 s warm).
- Diarizes: how many speakers, which time segments belong to each.
- Cuts the input into per-speaker tracks and denoises each independently (the chosen DFN3 checkpoint
  *preserves* speaker identity: 15.62% EER, better than raw audio).
- Reconstructs **`mixed_denoised.wav`** — a single full-length denoised track on the original
  wall-clock timeline (so VANI's timestamp-based downstream needs no rework).
- Serializes GPU work behind a lock (one instance uses ~6 GB of 8 GB VRAM); returns `503` when busy.
- Checkpoint `variant`: `clean` vs `robust` — **not interchangeable**; picked per request by the
  clip's condition (real radio → `robust`, clean/synthetic → `clean`).

### NODE-B (Sanket) — "what language, and which Mandarin dialect"

- MMS-LID-4017 (≈1B-param Wav2Vec2 + LoRA); one forward pass → 8-way language distribution + 7-way
  Mandarin dialect distribution.
- Trained on a **post-denoiser distribution (no reverb)** — the reason it runs *after* NODE-A.
- Analyzes only the first **10 s** of a track (no chunking on its side); NODE-C handles longer tracks
  by sending up to 3 windows and averaging the probability vectors.
- RMS < 0.01 **silence gate** returns `{"no_speech_detected": true}`; NODE-C treats that as "drop
  this speaker from the vote," not an error.
- Frozen checkpoint pair: **v1 language (`stage2_v3`) + v1 dialect (`stage4_phaseB`)** — the only
  internally-consistent combination. Reported: language macro-F1 0.9764, dialect macro-F1 0.601.
- Always reached via its HTTP endpoint — never the raw `VANIModel` (whose `punjabi_ldcil` head is
  architecturally present but never trained, and indistinguishable by shape from the real head).

### NODE-C (VANI) — "everything else + orchestration"

- Owns VAD, the coarse MMS probe, chunking, ASR, diarization *labeling*, the language vote,
  translation, keywords, ISUM, SQLite storage, and the GUI.
- `src/remote_client.py`: `health()`, `denoise_diarize()`, `identify_language()`, the two mapping
  tables, plus `assign_speakers_from_diarization()` and `dominant_language()`.
- Normalizes NODE-A's audio before forwarding to NODE-B (RMS-normalize toward 0.08 with a 20× gain
  cap so quiet real speech clears B's gate while true silence still gates) — closes the
  gain-convention mismatch across the denoiser boundary.

---

## 5. The two handoffs, precisely

**NODE-C → NODE-A**
```
POST http://192.168.10.11:8801/process?lang=<mandarin|urdu|punjabi|pashto|default>
                                       &variant=<clean|robust>&mode=diarization-guided
     Content-Type: audio/wav          body = raw 16 kHz mono WAV bytes
 200 → application/zip { diarization.json, summary.json,
                         mixed_denoised.wav, Speaker_1_Denoised.wav … }
 503 → busy (GPU lock held)
```

**NODE-C → NODE-B**
```
POST http://192.168.10.12:8802/api/analyze          multipart file = one Speaker_N_Denoised.wav
 200 → { top1_language, top1_language_confidence,
         top1_dialect, top1_dialect_confidence, dialect_engaged, language_probs }
 200 → { no_speech_detected: true, rms }             (silence gate)
```

Both directions stay **16 kHz mono** end to end, so nothing is ever resampled twice.

---

## 6. Language-code mapping (get this right or the demo silently degrades)

**VANI → NODE-A** (Gaurav's per-language clustering knob):

| VANI | Gaurav |
|---|---|
| `zh` | `mandarin` |
| `ur` | `urdu` |
| `pa` | `punjabi` |
| `ps` | `pashto` |
| anything else | `default` (logged as a warning — no tuned operating point) |

**NODE-B → VANI** (Sanket's 8 classes → VANI codes):

| Sanket | VANI | Fine-tuned VANI ASR? |
|---|---|---|
| `urdu` | `ur` | yes |
| `pashto` | `ps` | yes |
| `kashmiri` | `ks` | yes |
| `dogri` | `doi` | no — default Whisper + IndicTrans2 |
| `punjabi` | `pa` | yes (routed to SeamlessM4T) |
| `mandarin` | `zh` | yes |
| `cantonese` | `zh` | no `yue` model → falls back to `zh` |
| `tibetan` | `bo` | no — default Whisper |

**Critical asymmetry:** `hi` (Hindi) and `ne` (Nepali) are **not** in NODE-B's 8-class set, but VANI
supports both and has fine-tuned models for them. So NODE-B's answer is accepted as the audio vote
**only when** it is confident *and* maps to something VANI can act on; otherwise NODE-C defers to its
local MMS-LID probe, which covers the wider label set. The local probe is never removed — it is the
safety net for Hindi, Nepali, and for the case where NODE-B is down.

---

## 7. The safety net — why this cannot regress the demo

Every remote hop is behind a config flag with `fallback_on_error: true`:

- **NODE-A unreachable** → NODE-C runs its own local denoise + MFCC diarization. Pipeline continues.
- **NODE-B unreachable / not confident** → NODE-C's local MMS-LID vote carries the language decision.
- **Both down** → NODE-C is exactly the original single-machine VANI, byte-for-byte.

### Network mode (one build, either environment)

NODE-C's GUI exposes an **Auto / Standalone / Networked** switch backed by a once-per-session health
probe (short timeout, cached):

| Mode | Behaviour |
|---|---|
| **Auto** (default) | Probe A and B once; use whichever are reachable; otherwise run fully local. Unreachable nodes are skipped so there is no per-file connect-timeout latency. |
| **Standalone** | Never touch the network — pure local pipeline. |
| **Networked** | Trust the configured LAN nodes; fall back per-node on failure. |

This is what lets the identical build run on the 3-node LAN or on a single laptop with no code change.

---

## 8. What NODE-C adds to the result

Beyond the existing VANI result fields, each run now carries:

- `remote_nodes` — which nodes actually served this run (e.g. `["A","B"]`).
- `diarizer_variant` — `clean` or `robust`, as reported by NODE-A.
- `der_source` — `remote:A` or `local`.
- `speakers[]` — per-speaker label, talk time, denoised track path, language, dialect, confidence.
- `denoised_audio` — path to NODE-A's `mixed_denoised.wav` for playback.

The GUI renders these as a per-result "REMOTE NODE ANALYSIS" panel (nodes served, variant, DER
source), an **Original (noisy) vs Denoised** audio comparison, and per-speaker cards each with a
player for that speaker's denoised track.

---

## 9. Roles at a glance

| Concern | NODE-A | NODE-B | NODE-C |
|---|---|---|---|
| Denoising | ✅ per-speaker | — | fallback (local) |
| Diarization | ✅ | — | fallback (MFCC) + labeling |
| Language ID | — (needs a tag) | ✅ authoritative | coarse probe + vote + fallback |
| Dialect (Mandarin) | — | ✅ | display |
| VAD / chunking | — | — | ✅ |
| ASR (transcription) | — | — | ✅ |
| Translation | — | — | ✅ |
| Keywords / threat | — | — | ✅ |
| ISUM (summary) | — | — | ✅ |
| Orchestration / GUI / storage | — | — | ✅ |

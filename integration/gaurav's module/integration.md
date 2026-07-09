# Project VANI — Integration Reference (DiariZen Diarization + DeepFilterNet3 Denoising Module)

_This document assumes no prior knowledge of Project VANI. It describes ONE module — speaker
diarization (via DiariZen) + speaker-wise speech denoising (via DeepFilterNet3) — as it currently
stands, so it can be integrated into a larger pipeline by someone who was not part of its
development. It is the integration contract for this module: what it needs, what it accepts, what
it produces, and how a downstream module should consume its output.

**Scope note:** Project VANI also has a separate, independently-developed diarization system
based on PyAnnote (used for an interim project demonstration). That system is explicitly OUT OF
SCOPE for this document — this handover covers only the DiariZen + DeepFilterNet3 module. Last
updated: 2026-07-09._

---

## 1. Module Overview

This module takes a single audio recording containing multiple speakers and produces clean,
separated, per-speaker audio tracks. Concretely, it:

- Accepts one audio file as input (a recording of a multilingual conversation — Mandarin, Urdu,
  Punjabi, or Pashto — possibly degraded by real radio-channel transmission or background noise).
- Performs **speaker diarization** (DiariZen): determines how many distinct speakers are present
  and which time segments belong to each speaker ("who spoke when").
- Performs **speaker-wise extraction**: cuts the input audio into one track per detected speaker.
- Performs **speech denoising** (DeepFilterNet3): cleans each speaker's track independently,
  removing channel distortion and background noise while preserving speaker identity.
- Produces one clean, denoised audio file per detected speaker, plus structured metadata
  describing what was found (speaker count, per-speaker timing, file mappings).
- Exposes these outputs (audio files + metadata) for any downstream module to consume — this
  module does not know or care what happens after it.

This module does **not** perform transcription, language/dialect identification, translation, or
any semantic analysis of the audio content. Those are explicitly out of scope and are expected to
be separate, independent downstream modules (see §7-8).

---

## 2. Current Version

| Component | Current implementation | Status |
|---|---|---|
| **Diarization engine** | **DiariZen** — a WavLM-large foundation model + Conformer head, fine-tuned in-house (the "unfreeze" recipe: the WavLM backbone itself was fine-tuned, not just a lightweight head on top of it) | **Research-grade, best-known accuracy. Not yet integration-hardened — see the license note and §9.** Results: **12.74% Diarization Error Rate (DER)**, single checkpoint, applied uniformly across all 4 languages (Mandarin 7.87%, Urdu 14.67%, Punjabi 1.38%, Pashto 27.05%). A specialized checkpoint variant exists for real-radio-channel audio specifically: **32.02% DER on real DMR (radio) recordings** — better on that condition, but measurably worse on clean/synthetic-noise audio, so it is a separate, condition-specific checkpoint, not a universal upgrade (see §9). |
| **Denoiser** | **DeepFilterNet3**, fine-tuned in-house via synthetic radio-channel-characterized data augmentation (checkpoint nicknamed `ep26`) | **Stable, the most mature component of this module.** Improves perceptual quality (DNSMOS +0.181), intelligibility (STOI +0.107), and — critically — speaker-verification accuracy on the cleaned audio (15.62% Equal Error Rate, beating even the raw unprocessed signal at 18.75%; a generic/stock denoiser instead makes this *worse*, 28.58% — i.e. an untrained denoiser actively damages speaker identity, this one doesn't). |
| **Speaker separation (alternative to diarization-guided extraction)** | MossFormer2_SS_16K (via the ClearVoice toolkit) | **Stable, but limited to exactly 2 output tracks** regardless of true speaker count. Kept as an alternate mode for true overlapping-speech un-mixing; not the default. |
| **Reference implementation / live demo** | A Tkinter desktop GUI (`E:\Gaurav\demo\diarizen\`) | **Frozen as of 2026-07-08** for an interim project presentation — treat as read-only reference code showing how to call the underlying stage scripts, not as the integration target itself (see §10). |

> **⚠ License — read before integrating anywhere commercial.** DiariZen's model weights are
> **CC BY-NC 4.0 — non-commercial use only** (the code/recipe itself is MIT, but that does not
> change the weights' license). Do not integrate this diarization engine into any commercial
> product or service without a separate legal review / a different license arrangement. This is
> the single most important constraint in this entire document for anyone planning to productize
> this module. DeepFilterNet3 does not carry this restriction (standard open license).

**Overall module version:** diarization-guided extraction (DiariZen) → speaker-wise extraction →
denoising (`ep26`) is the current, best-tested pipeline configuration. A "clean-conditions" and
a "real-radio-channel" DiariZen checkpoint both exist — pick per deployment condition (see §9),
they are not interchangeable drop-ins for each other.

---

## 3. Software Environment

Project VANI currently runs on a **single Windows workstation**, not a server/cloud environment.
There is no containerization (no Docker) as of this writing — the environment below is the actual
bare-metal/venv setup, documented so it can be reproduced or containerized by whoever integrates
this module.

| Component | Value |
|---|---|
| **Operating System** | Windows 10 Pro, version 10.0.19045 (Build 19045) |
| **GPU** | NVIDIA Quadro P4000, 8 GB VRAM |
| **NVIDIA driver version** | 531.14 |
| **CUDA (as built into PyTorch)** | 12.1 |
| **cuDNN** | 8.8.1 |
| **Python** | 3.11.9 (identical across both environments below) |

This module's code spans **two separate Python virtual environments**, not one — a real
integration effort needs to either replicate this separation or consolidate it deliberately (not
accidentally):

| Environment | Path | Used for | Key package versions |
|---|---|---|---|
| **DiariZen** | `sota_exploration/diarizer_bench/.venv_diarizen/` | DiariZen diarization | PyTorch 2.1.1+cu121, torchaudio 2.1.1+cu121, `pyannote.audio` 3.1.1 (DiariZen bundles its own vendored/modified copy of pyannote-audio internally — this is a dependency of DiariZen's own pipeline code, not a separate diarization system) |
| **Denoising** | `denoising_exploration/.venv_denoise/` | DeepFilterNet3 denoising, MossFormer2 separation | PyTorch 2.1.1+cu121, torchaudio 2.1.1+cu121, `DeepFilterNet` 0.5.6, `clearvoice` 0.1.2, `speechbrain` 1.1.0 |

Both environments happen to share the same PyTorch/torchaudio version (2.1.1+cu121) — they were
kept as separate venvs primarily to isolate DiariZen's and the denoiser's other, more divergent
dependencies (e.g. `clearvoice`/`speechbrain` vs. DiariZen's own NeMo/pyannote-audio stack) from
each other, not because of a PyTorch version conflict between these two specifically. Consolidating
them into one environment is plausible and worth attempting if it simplifies integration — unlike
the (out-of-scope) PyAnnote track, there is no known hard version conflict blocking this.

**Known environment quirks worth knowing about before you hit them:**
- `speechbrain` (1.1.0, in the denoising environment) expects a newer, device-agnostic
  `torch.amp.custom_fwd` API that does not exist in the pinned PyTorch (2.1.1). Any code that
  loads a `speechbrain` ECAPA embedding model in this environment needs a small compatibility
  shim (already implemented in `denoising_exploration/scripts/dfn_finetune.py`'s `make_ecapa()` —
  copy this pattern rather than re-deriving it; a second, related shim for a `LazyModule`
  attribute-lookup bug is also needed and is in the same function).
- Model weights are routed through a local, offline cache (`VANI_Models_Cache/`) with
  `HF_HUB_OFFLINE=1` set, so the pipeline does not require internet access at run time once set
  up. A fresh environment will need one online run first to populate this cache from Hugging Face.
- A single DiariZen pipeline instance steady-states at ~6 GB of the 8 GB available VRAM at
  97-99% compute utilization — see §9 for what this means for concurrency.

---

## 4. Pipeline Description

```
Input Audio (.wav)
        │
        ▼
┌───────────────────────┐
│  Stage 1: Speaker      │   Detects how many speakers are present and which time
│  Diarization           │   segments belong to each. Outputs a speaker-labelled
│  (DiariZen)            │   timeline (RTTM format) + JSON metadata.
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Stage 2a: Speaker-wise │   Cuts the ORIGINAL input audio into one track per
│  Extraction             │   detected speaker, using Stage 1's timeline. Adapts
│  (diarization-guided,   │   to however many speakers were actually found — no
│  the default mode)      │   fixed track count.
└───────────────────────┘
        │        (alternate: Stage 2b, blind 2-track separation via MossFormer2 —
        │         used instead of 2a when overlap un-mixing matters more than
        │         adapting to the true speaker count; fixed at exactly 2 tracks)
        ▼
┌───────────────────────┐
│  Stage 3: Denoising     │   Each speaker's track is denoised independently
│  (DeepFilterNet3,        │   (channel distortion + background noise removed),
│  fine-tuned "ep26")      │   with the fine-tuned model chosen specifically
│                          │   because it preserves speaker identity better than
│                          │   both the raw audio and a generic/stock denoiser.
└───────────────────────┘
        │
        ▼
Clean, Speaker-Separated Audio Outputs + Metadata
(one file per speaker, plus JSON describing what was found)
```

---

## 5. Input Specification

| Property | Requirement |
|---|---|
| **File format** | WAV (`.wav`). Other formats are not currently validated. |
| **Sample rate** | No fixed requirement enforced at the API level — the pipeline resamples internally as needed (DiariZen's segmentation runs at 16 kHz; the denoiser runs at 48 kHz internally and resamples back to the input's own rate on output). Source recordings used in development are 16 kHz mono. |
| **Channels** | Mono expected. If given stereo, only one channel is used (not averaged) — verify this is acceptable for your source audio, or downmix before input. |
| **Duration** | No hard limit enforced, but development/validation was done on clips in the 10-45 second range. Behavior on much longer recordings (multi-minute+) has not been characterized. |
| **Language** | Mandarin, Urdu, Punjabi, or Pashto. Accuracy depends on applying the correct **per-language operating point** to the clustering step — see the integration note in §7. There is no automatic language detection from audio content; the reference implementation only auto-fills this from an input file's folder path (a convenience of the curated demo set, not a real language-ID capability). |
| **Diarizer variant (condition)** | The DiariZen checkpoint comes in a "clean-conditions" and a "real-radio-channel" variant (§2, §9) — the caller must pick the right one for the expected input condition; there is no automatic condition detection. |
| **Directory structure** | No structure is required by the underlying processing scripts themselves — they take a single file path. The reference implementation's curated sample set uses `demo_inputs/<Language>/<DMR\|Synthetic>/<sample_name>/{clean,noisy}.wav`, but this is a convention of the reference GUI, not a requirement of the processing pipeline. |
| **Naming conventions** | None enforced by the processing pipeline. |
| **Assumptions the pipeline makes** | (1) The recording contains genuine speech from 1 or more distinct human speakers. (2) Overlapping speech (two people talking simultaneously) is not resolved by the default diarization-guided mode — an overlapping segment is assigned to whichever single speaker the segmentation model considers dominant; true overlap un-mixing requires the alternate blind-separation mode, which is itself capped at 2 output tracks. (3) The true number of speakers is NOT known in advance by the diarization-guided mode (it is discovered); the blind-separation mode assumes exactly 2. |

---

## 6. Output Specification

The reference implementation writes one output folder per processed input, structured as follows
(this exact layout is a convention of the reference scripts, not an inherent API — see §7 for
what a downstream integration should actually rely on):

```
<output_root>/<sample_id>/
    speakers.rttm                  Stage-1 output: standard RTTM format, one line per
                                    speaker turn (start time, duration, speaker label)
    diarization.json               Stage-1 metadata (see schema below)
    Speaker_1.wav                  Stage-2 output: speaker 1's raw extracted audio
    Speaker_1_Denoised.wav         Stage-3 output: speaker 1's audio after denoising
    Speaker_2.wav, Speaker_2_Denoised.wav, ...   (one pair per detected speaker, N adapts
                                                   to however many speakers were found)
    summary.json                   Stage-2/3 metadata (see schema below)
```

**`diarization.json` schema:**
```json
{
  "wav": "<path to the input file that was processed>",
  "duration": 32.1,
  "samplerate": 16000,
  "variant": "clean",
  "lang_knobs": "mandarin",
  "n_speakers": 4,
  "speakers": {
    "SPEAKER_00": {
      "talk_time": 12.35,
      "segments": [[20.332, 32.684]]
    },
    "SPEAKER_01": { "talk_time": 6.16, "segments": [[0.031, 0.048], [0.318, 0.858], ...] }
  }
}
```
`variant` is `"clean"` or `"robust"` (which DiariZen checkpoint was used — see §2, §9).
`lang_knobs` is the language tag that was applied, or `"default"` if none was supplied (see the
integration note in §7 — leaving this as `"default"` measurably hurts accuracy, it is not a safe
no-op). `segments` are `[start_seconds, end_seconds]` pairs. `n_speakers` is the count a
downstream module should treat as authoritative for "how many speakers were found."

**`summary.json` schema:**
```json
{
  "input": "<path to the input file>",
  "n_tracks": 4,
  "mode": "diarization-guided",
  "source": "stage-1 diarization segments (speakers.rttm / diarization.json)",
  "denoiser": "DFN3 fine-tuned ep26 (dfn_vani3b)",
  "files": [
    {"speaker": 1, "label": "SPEAKER_00", "raw": "Speaker_1.wav",
     "clean": "Speaker_1_Denoised.wav", "duration": 11.77, "n_segments": 1},
    ...
  ]
}
```
The `files` array is the authoritative mapping from speaker index → output filenames — a
downstream module should read this rather than assuming a naming pattern, in case the naming
convention changes in a future version.

**Logs:** the reference scripts print progress to stdout (prefixed `##PROG <percent> <message>`
in the GUI-oriented versions); there is no structured/persistent log file produced per run beyond
the JSON metadata above. A production integration should add its own logging around calls to
this module rather than relying on stdout parsing.

---

## 7. Interface for Future Integration

The long-term architecture of Project VANI is expected to be a chain of independently-developed
pipelines, of which this module is the first:

```
Audio Input
    │
    ▼
Pipeline 1 — THIS MODULE (Speaker Diarization + Speaker-wise Denoising)
    │
    ▼
Pipeline 2 — Dialect Identification
    │
    ▼
Pipeline 3 — Speech-to-Text
    │
    ▼
Pipeline 4 — Translation / Analysis / Downstream Processing
```

**What this module expects as input:** one audio file (§5), a diarizer-variant selection
(clean/robust, §2), and ideally a language tag supplied by the caller (see the integration note
below — this materially affects accuracy and should not be left unspecified).

**What this module produces as output, and what a downstream module should consume:**
- A set of per-speaker audio files (denoised) — this is the primary payload for any downstream
  module that itself processes audio (e.g. Speech-to-Text should run on `Speaker_N_Denoised.wav`
  per speaker, not on the original mixed input).
- `diarization.json` and `summary.json` — the metadata contract. A downstream module should treat
  `summary.json`'s `files` array as the authoritative index of "what speakers exist and where
  their audio lives," and `diarization.json`'s `segments` as the authoritative per-speaker timing
  if downstream processing needs to reconstruct a timeline (e.g. for aligning transcripts back to
  the original recording's timeline).
- **No speaker identity persists across separate input files.** `SPEAKER_00`/`SPEAKER_01`/... are
  labels local to a single diarization run — this module does not currently perform speaker
  recognition/re-identification across different recordings. If a downstream module needs "is
  this the same person as in a different file," that is new capability, not something this
  module's output already encodes.

**How data should be passed between pipelines:** this module does not currently expose an API,
message queue, or service endpoint — it is a set of command-line Python scripts writing to a
local filesystem (see §6, §8). A downstream module today would need to either (a) invoke these
scripts as subprocesses and read the resulting files/folder, or (b) wrap the underlying Python
functions directly if running in the same process/environment. Neither a REST API nor a queue-
based interface exists yet — this is the most significant integration gap for a true multi-service
architecture (see §8).

**Metadata that accompanies outputs:** language tag used, which diarizer variant was used,
per-speaker talk time and segment timing, denoiser identity, and the raw/denoised filename
mapping — all in `diarization.json` + `summary.json` as specified in §6. No confidence scores are
currently emitted (DiariZen does not expose per-segment confidence in the current integration;
this would need to be added if a downstream module wants to weight its own processing by
diarization confidence).

**Integration note on language:** accuracy depends materially on knowing the input's language
ahead of time (each language has its own tuned clustering operating point — see §2, §9). This
module does **not** perform language identification itself. A calling pipeline that doesn't
already know the language should either supply a best-guess tag and accept reduced accuracy, or
insert a language/dialect-ID step *before* this module rather than after it — worth deciding
explicitly with whoever owns the "Dialect Identification" pipeline stage in the diagram above,
since it may make more sense to run before, not after, diarization.

---

## 8. Future Integration Considerations

Development to date has taken place entirely on a single workstation (§3), with all "pipelines"
implemented as scripts invoked directly against a shared local filesystem. This was appropriate
for research iteration speed, but the long-term intent for Project VANI is different: **each
processing stage (diarization+denoising, dialect ID, speech-to-text, translation/analysis) is
expected to eventually become an independently deployable service**, communicating through
well-defined interfaces rather than shared local files.

Consequences worth designing for now, even though they aren't implemented yet:
- **Do not assume a shared filesystem** between this module and its consumers in the target
  architecture — today's file-path-based handoff (§6, §7) is a placeholder for what should
  eventually be an explicit data-transfer contract (e.g. object storage references, a message
  payload, or an API response body containing the same JSON schemas already defined in §6).
- **Do not assume synchronous, single-process execution.** This module currently takes on the
  order of tens of seconds to a few minutes per input (model loading dominates for short clips —
  see §9's timing table). A service architecture should plan for asynchronous invocation
  (submit → poll/callback), not a blocking call in a request path.
- **Do not assume the diarizer variant (clean/robust) or checkpoint choice is fixed.** The
  clean-conditions and real-radio-channel checkpoints (§2) have different accuracy profiles per
  condition — a service interface should treat "which variant" as a configuration parameter of
  the diarization step, not something hard-coded into a downstream consumer's expectations.
- **The JSON schemas in §6 are a reasonable starting point for a formal service contract** (e.g.
  an OpenAPI schema) but have not been designed as one — expect to formalize field types,
  optionality, and versioning if/when this becomes a real service boundary.
- **The CC BY-NC 4.0 license on DiariZen's weights (§2) is a hard blocker for commercial
  deployment as-is** — factor this into any service/productization roadmap now, not after
  integration work is done. This is the single highest-priority open item for whoever owns
  turning this into a real service.

---

## 9. Known Constraints

**GPU / resource requirements:**
- A single NVIDIA GPU with **at least 8 GB VRAM** is required for the current model set at their
  current batch sizes. The development GPU (Quadro P4000, 8 GB) runs a single DiariZen instance
  at ~6 GB / 97-99% compute utilization — **there is no safe headroom to run a second GPU job
  concurrently** on hardware this size. Plan for either a bigger GPU or strictly serialized
  processing if this module needs to handle concurrent requests.
- CPU-only execution is possible for both environments (confirmed working, used for development
  smoke tests) but is dramatically slower — not recommended for any real-time or near-real-time
  use case.

**Performance / expected execution time** (measured on the Quadro P4000, single-clip runs,
10-45 second input clips):
| Stage | Cold start (first call — includes model load) | Warm (model already loaded) |
|---|---|---|
| Diarization (DiariZen) | ~30-60 seconds (dominated by model load) | a few seconds |
| Denoising + extraction | ~20-40 seconds (dominated by model load) | a few seconds per speaker |

The reference implementation reloads models from scratch on every single invocation (no
persistent model-serving process) — this is the dominant cost for short clips and is the first
thing worth fixing in a service architecture (keep models loaded, serve many requests per load).

**Accuracy / quality limitations, honestly stated:**
- **Real-radio-channel audio is substantially harder than clean audio.** The clean-conditions
  checkpoint gets 12.74% DER on clean audio; the specialized real-radio checkpoint gets 32.02% DER
  on real radio-channel recordings — a large gap that reflects genuine domain difficulty, not a
  tuning oversight. **These two checkpoints are not interchangeable**: the real-radio checkpoint
  is measurably *worse* than the clean checkpoint on clean/synthetic-noise audio (confirmed via
  direct A/B testing) — pick the one matching your actual deployment condition, don't assume one
  dominates the other.
- **One language (Pashto) has a persistent, only-partially-solved speaker-confusion and
  over-counting problem.** Pashto's error rate is markedly higher than the other three languages
  regardless of configuration; a systematic investigation (raising the clustering merge threshold,
  capping the maximum detected speaker count) found no clustering-parameter fix that improves
  this without making something else worse — root-cause analysis indicates this needs more real
  training data for this specific language, not further algorithmic tuning.
- **The blind-separation alternate mode is hard-capped at exactly 2 output tracks** regardless of
  the true number of speakers — only use it when you specifically need overlap un-mixing and know
  the input has (at most) 2 speakers; otherwise use the default diarization-guided mode.
- **No speaker re-identification across files** (§7) — every run's speaker labels are local to
  that run only.
- **License: CC BY-NC 4.0 on the diarization model's weights** (§2) — non-commercial use only.
  This is a hard constraint, not a soft recommendation; treat any commercial integration plan as
  blocked on resolving this first.

---

## 10. Notes for Future Developers

- **Read this document, not the full research history, to get started.** The project has an
  extensive day-by-day research log (`GPU_online/GPU_online/docs/daily/`) and per-track status
  files (`sota_exploration/STATUS.md` for DiariZen, `denoising_exploration/STATUS.md` for the
  denoiser) documenting how the current numbers were reached — useful for understanding *why* a
  design choice was made, but not necessary to integrate the module as it stands today.
- **The reference GUI is a demonstration tool, not the integration target.** It is currently
  frozen (read-only) pending an internal presentation and should be treated as a working example
  of how to call the underlying stage scripts, not as something to build a service around
  directly. The underlying stage scripts it calls (`stage_diarize.py`, `stage_extract_denoise.py`,
  `stage_separate_denoise.py`) are the actual reusable units of logic.
- **Always supply the language explicitly if it's known.** One recurring, easy-to-miss mistake
  during development was leaving the diarizer's language parameter unset/"auto," which silently
  skips a per-language accuracy tuning step and produces measurably worse results (in one observed
  case on the sibling PyAnnote-based system, finding 2 speakers instead of the correct 4 — the
  same failure mode applies here). If the calling pipeline knows the language, pass it; if it
  doesn't, get it from somewhere before this stage, not after.
- **The two virtual environments (§3) are believed compatible enough to merge** (both already run
  the same PyTorch/torchaudio version) — worth attempting consolidation into one environment
  rather than assuming the current split is load-bearing.
- **All checkpoints referenced in this document are file-based** (PyTorch `.pt`/`.bin` files on
  local disk, not a model registry or artifact store) — an integration effort should decide where
  these live long-term (object storage, a model registry, etc.) rather than assuming the current
  local paths are stable.
- **Nothing in this module currently exposes a network interface.** Every invocation described in
  this document is a local Python process reading/writing local files. Building an actual service
  boundary (REST/gRPC/queue) around this module is real, not-yet-done work, not a small wrapper.
- **Resolve the CC BY-NC 4.0 licensing question early**, not late — it affects whether this module
  can be part of anything beyond an internal/research deployment, and that answer should shape
  integration decisions from the start, not be discovered after the fact.

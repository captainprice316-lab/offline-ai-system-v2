# VANI LID Module — Integration Context

**Purpose of this document:** brief a system integrator planning to wire VANI into a
pipeline of the form `[Denoiser] -> [VANI LID] -> [Downstream Module]`. Every claim
below is sourced from reading the actual source in this repository (`src/`, `demo/`,
`scripts/evaluate.py`, `configs/config.yaml`) as of 2026-07-09, not from filenames or
prior documentation alone. Where the code does not answer a question, that is stated
explicitly in Section 8 rather than assumed.

---

## 1. System Overview

VANI (Versatile Acoustic Neural Identifier) is an offline, single-model language and
dialect identification (LID) system. It takes a single audio clip as input and
produces, in one forward pass, (a) a probability distribution over 8 languages (Urdu,
Pashto, Kashmiri, Dogri, Punjabi, Mandarin, Cantonese, Tibetan) and (b) a probability
distribution over 7 Mandarin regional dialect groups (from KeSpeech), the latter
intended to be consulted only when the language head's top choice is Mandarin. In the
target pipeline, VANI is meant to sit immediately after a denoiser and immediately
before whatever downstream module consumes a language/dialect decision (e.g. routing,
transcription-model selection, or a reporting/triage layer) — but as detailed in
Section 3 and Section 7, no code in this repository currently accepts or has been
validated against actual denoiser output; every existing entry point reads audio
directly from disk, an in-memory array, or an HTTP upload, and every reported accuracy
number was measured on source-corpus audio passed through VANI's own synthetic
augmentation pipeline, not real denoised audio.

---

## 2. Architecture

### 2.1 Model architecture

- **Backbone:** `facebook/mms-lid-4017` (Meta AI "Massively Multilingual Speech"),
  loaded via `transformers.Wav2Vec2Model.from_pretrained(...)` in `src/model.py`.
  Wav2Vec2 architecture, ~1 billion parameters, hidden size 1280, 48 transformer
  layers on top of a convolutional feature extractor, pretrained for LID across 4017
  languages. `output_hidden_states=True` is set, so the model exposes 49 hidden
  states per forward pass (CNN output + 48 transformer layers).
- **Adaptation (LoRA):** `peft.LoraConfig` applied via `VANIModel.apply_lora()`
  — rank 16, `lora_alpha = rank * 2 = 32` (hardcoded in `src/model.py`, **not** in
  `configs/config.yaml`), target modules `q_proj`/`v_proj`, applied only to
  transformer layers 24-47 (the top 24 of 48; `layers_to_transform=list(range(24,
  48))`). The rest of the backbone, including all layers below 24, is frozen
  (`p.requires_grad_(False)` on every backbone parameter before LoRA is applied).
- **Per-head layer weighting (`PerHeadLayerWeighting`):** each task head learns its
  own softmax distribution (49 learnable scalars) over the 49 hidden states, and
  computes a weighted sum — i.e. the language head and each dialect head can each
  learn to emphasize a different depth of the backbone. An ablation variant
  (`FixedUniformLayerWeighting`, α = 1/49 fixed, no learnable parameters) exists in
  the same file for comparison but is not used in any deployed checkpoint.
- **Pooling (`AttentiveStatisticsPooling`):** a learned linear attention layer scores
  each time step, softmaxes across time, and produces a `[batch, 2*hidden]` vector by
  concatenating the attention-weighted mean and standard deviation. A `MeanPooling`
  ablation variant (simple temporal mean, mean duplicated to match the `2*hidden`
  output shape) also exists but again is not used in any deployed checkpoint.
- **Classifier head (`HeadMLP`):** `Linear(2*hidden -> 512) -> ReLU -> Dropout(0.1)
  -> Linear(512 -> num_classes)`, one instance per task (language head: 8-way; each
  dialect head: scheme-specific class count).
- **Multi-head structure:** `VANIModel` builds one language head plus one head per
  entry in `configs/config.yaml`'s `dialect_heads:` block. As configured today that is
  **two** dialect heads — `mandarin_kespeech` (7 classes) and `punjabi_ldcil` (3
  classes) — both of which are always constructed and always contribute to
  `forward()`'s output dict regardless of whether either has ever been trained (see
  Section 4 for why `punjabi_ldcil` matters here).
- **Loss:** `FocalLoss` (`src/focal_loss.py`), γ=2, with `ignore_index=-1` used to mask
  out loss contribution from rows that don't apply to a given head (e.g. a Mandarin
  KeSpeech dialect row contributes to the dialect head's loss but is masked out
  — `ignore_index=-1` — for `punjabi_ldcil`).

### 2.2 Data flow, raw input to output

This is the path implemented in `demo/inference.py` (`VaniDemoModel.infer_file`),
which is the most complete single "reference" inference path in the repo:

1. Read audio file via `soundfile.read(..., dtype="float32", always_2d=False)`.
2. If the array is 2-D (multi-channel), downmix by averaging channels
   (`wav.mean(axis=1)`) — a fixed, non-configurable downmix, no channel selection.
3. If the source sample rate isn't 16000 Hz, resample via `librosa.resample`.
4. Truncate to the first `MAX_DUR_SEC` (10.0 seconds in the demo path) —
   **anything beyond the first 10 seconds is discarded, with no chunking,
   sliding window, or aggregation of any kind.**
5. Build a `[1, T]` float tensor and an all-ones `[1, T]` attention mask (single-clip
   inference; no dynamic padding is needed with batch size 1 — the padding-to-a-
   multiple-of-160 logic in `VANICollator` only matters for the batched
   training/evaluation path).
6. Forward pass through the backbone (bf16 autocast on CUDA if
   `torch.cuda.is_bf16_supported()`, else fp16; CPU fallback with no autocast) to get
   49 hidden states.
7. For the language head and each dialect head independently: layer-weighted sum of
   the 49 hidden states -> attentive statistics pooling -> MLP -> raw logits.
8. Softmax each head's logits; take argmax for top-1 class and its probability per
   head.

### 2.3 Pipeline diagram

```
                         ┌───────────────────────────────────────────┐
                         │              VANI LID module                │
[Denoiser] ──(audio)──▶  │                                              │ ──(JSON)──▶ [Downstream Module]
                         │  1. soundfile.read -> float32 mono           │
                         │  2. downmix if stereo (mean of channels)     │
                         │  3. resample to 16 kHz if needed             │
                         │  4. truncate to first 10 s (NO chunking)     │
                         │  5. Wav2Vec2 backbone (MMS-LID-4017 + LoRA)  │
                         │        │                                     │
                         │        ▼ 49 hidden states (CNN + 48 layers)  │
                         │   ┌────────────┐        ┌──────────────┐    │
                         │   │ Language   │        │ Dialect head(s)│  │
                         │   │ layer-wt   │        │ layer-wt        │  │
                         │   │ + ASP pool │        │ + ASP pool      │  │
                         │   │ + MLP (8)  │        │ + MLP (7 / 3)   │  │
                         │   └────────────┘        └──────────────┘    │
                         │        │                       │            │
                         │        ▼                       ▼            │
                         │  softmax -> top-1        softmax -> top-1    │
                         └───────────────────────────────────────────┘
```

Both heads run on every call — there is no code path that skips the dialect head when
the language head's top-1 isn't Mandarin. The demo UI only *displays* the dialect
panel conditionally (`"dialect_engaged": top1_lang == "mandarin"` in
`demo/inference.py`); the dialect logits are computed and returned regardless.

---

## 3. Interfaces

This is the section most load-bearing for integration, and also the section with the
most gaps — there is **no single, unified "module" entry point** in this codebase.
Three different, mutually-inconsistent entry points exist, each described below.

### 3.1 Input contract

| Property | Finding |
|---|---|
| Sample rate | Hardcoded to 16000 Hz (`SAMPLE_RATE = 16000` in `demo/inference.py`; same constant implicitly in `src/dataset.py`). Anything else is auto-resampled via `librosa.resample`, so a denoiser does not strictly have to emit 16 kHz, but every reported eval number was measured on audio that was already 16 kHz or resampled once at load time — resampling *twice* (denoiser's internal rate -> its output rate -> VANI's 16 kHz) is unvalidated. |
| Bit depth / container | Read via `soundfile.read(dtype="float32")` (libsndfile-backed) — supports WAV/FLAC/OGG and, depending on the installed libsndfile build, some MP3. `demo/server.py` additionally has a **PyAV fallback** (`_decode_with_pyav`) specifically for WebM/Opus, because that's what a browser's `MediaRecorder` produces for live mic capture; this fallback only exists in `server.py`, not in `demo/inference.py`'s file-path loader or in `scripts/evaluate.py`. |
| Channels | Mono expected. Stereo/multi-channel input is silently downmixed by averaging channels (`wav.mean(axis=1)`) wherever audio is loaded (`demo/inference.py`, `demo/server.py`, `src/dataset.py` all repeat this same three-line pattern independently — it is not a shared utility function). No channel-selection option exists. |
| Amplitude / normalization | **No normalization step exists anywhere in the reviewed code.** Raw float32 samples (as returned by `soundfile`, natively in roughly [-1, 1] for PCM sources) are passed directly into `Wav2Vec2Model` — no `Wav2Vec2FeatureExtractor`, no zero-mean/unit-variance step, no loudness normalization. Whatever amplitude convention the training corpora happened to have is implicitly "the" expected convention; this was not verified against denoiser output. |
| Duration / chunking | **No chunking or sliding-window inference exists anywhere in the inference code.** `demo/inference.py` and `demo/server.py` both truncate to the first `MAX_DUR_SEC` (10.0 s) and discard the rest. `scripts/evaluate.py` truncates via `VANIDataset`'s `max_dur_sec` argument (CLI default 10.0 via `--max-dur`, but the dataset class itself defaults to 30.0 if not overridden — the two default values are not the same, which is worth being deliberate about downstream). No minimum-length floor is enforced in code, though the batch collator (`VANICollator`) pads short clips to a multiple of 160 samples purely for tensor-shape reasons. |
| Silence / gating | `demo/server.py`'s `/api/analyze` computes RMS on the truncated waveform and, if RMS < 0.01, returns `{"no_speech_detected": true, ...}` **without running the model**. This heuristic threshold is tuned against this project's own corpora ("real speech clips ... run RMS ~0.07-0.11", per an inline comment) and exists **only** in `server.py`'s upload/mic path — it is absent from `demo/inference.py`'s file-based path and from `scripts/evaluate.py`'s batch path, both of which will run a full inference pass (and return a confident-looking but meaningless top-1 class) on silent or near-silent input. |

**How input is currently received (three inconsistent paths, no unified API):**

1. **Batch/offline** — `scripts/evaluate.py` / `src/train.py`: a CSV manifest of file
   paths (columns include `audio_path`, `label_language`, `dialect_scheme`,
   `label_dialect`, `split`), loaded via `VANIDataset` + a `torch.utils.data.DataLoader`
   with a configurable `--batch-size` (default 4).
2. **Direct Python call** — `demo/inference.py`'s `VaniDemoModel.infer_file(path)` or
   `.infer_waveform(wav_array)` (a 1-D float32 numpy array, already assumed truncated
   by the caller).
3. **HTTP** — `demo/server.py`, a FastAPI app:
   - `POST /api/analyze` — either `clip_id` (a string referencing one of a fixed,
     pre-staged set of demo files listed in `demo/data/demo_manifest.csv`) or a
     multipart `file` upload.
   - `GET /api/analyze_dir?folder=...` — a **server-side** folder path (must already
     exist under `demo/data/`), streamed back one file at a time via Server-Sent
     Events (SSE); this is not a client upload of a folder, it's a pre-staged-data
     demo feature.

   There is no gRPC interface, no websocket/streaming-audio interface, and no formal
   OpenAPI/pydantic request or response schema — FastAPI infers the wire format from
   plain Python dicts returned by each handler.

### 3.2 Output contract

The core model (`VANIModel.forward`) returns a plain `dict[str, Tensor]`:
`{"language": logits [B,8], "mandarin_kespeech": logits [B,7], "punjabi_ldcil": logits
[B,3]}` — **raw, un-softmaxed logits**; the caller is responsible for `softmax`/argmax.
Nothing in this dict distinguishes a trained head's logits from an untrained head's
logits (see Section 4 — `punjabi_ldcil` is architecturally present but never trained).

`demo/inference.py` wraps this into a JSON-serializable dict, which is the closest
thing to a documented output schema in the repo:

```json
{
  "language_probs": {"urdu": 0.001, "pashto": 0.0, "...": "... (all 8 classes)"},
  "top1_language": "mandarin",
  "top1_language_display": "Mandarin",
  "top1_language_confidence": 0.97,
  "dialect_probs": {"Standard": 0.61, "Ji-Lu": 0.05, "...": "... (all 7 KeSpeech classes)"},
  "top1_dialect": "southwestern",
  "top1_dialect_display": "Southwestern",
  "top1_dialect_confidence": 0.72,
  "dialect_engaged": true,
  "latency_ms": 184.2,
  "duration_sec": 14.2,
  "analyzed_sec": 10.0
}
```

Notes on this schema:
- `dialect_probs` only ever surfaces the 7 `mandarin_kespeech` classes;
  `punjabi_ldcil` logits are computed by the underlying model but never appear here —
  this wrapper effectively hides the untrained head, but only because it hardcodes
  `self.dialect_scheme = "mandarin_kespeech"`, not because of any architectural
  safeguard.
- `dialect_engaged` is a UI hint (`top1_language == "mandarin"`), not an indication
  that the dialect computation was skipped — it always runs.
- `duration_sec` is the *original* file duration; `analyzed_sec` is what was actually
  fed to the model (capped at `MAX_DUR_SEC`). This pairing exists only in the demo
  layer — `scripts/evaluate.py`'s batch path does not report per-clip truncation.
- `demo/server.py`'s `/api/analyze` adds `file`/`clip_id` fields, and, on the
  RMS-gated silence path, returns `{"no_speech_detected": true, "rms": ..., ...}`
  instead of the above.
- **No embeddings are exposed at any current entry point.** The pooled `[B, 2560]`
  vector (`lang_embed` in `VANIModel.forward`, called `lang_repr`/`lang_embed`
  internally) exists as an intermediate value but is not returned by any reviewed
  code path. If a downstream module needs the embedding rather than a class decision,
  that requires a code change (flagged in Section 8).
- No formal JSON Schema / OpenAPI document exists anywhere in the repo; the shape
  above was reconstructed by reading `demo/inference.py` and `demo/server.py`
  directly.

### 3.3 Assumptions about upstream preprocessing (relevant to the denoiser boundary)

- **Trained on a "post-denoiser distribution," explicitly without reverb.**
  `src/dataset.py`'s augmentation function carries the comment *"Augmentation
  (post-denoiser distribution — NO RIR/reverb)"* and implements: random time-masking,
  time-stretch (0.9-1.1x), polarity inversion, additive Gaussian noise (SNR sampled
  15-30 dB), and a 300-3400 Hz "telephone band" bandpass filter — but **no
  reverberant/RIR augmentation of any kind**. This is the clearest evidence in the
  codebase of what upstream signal characteristics the model expects: a denoised,
  non-reverberant signal at roughly telephone-to-clean quality, not a raw far-field or
  heavily reverberant recording.
- **No explicit SNR floor is enforced or documented anywhere in code.** The RMS ≥ 0.01
  gate in `demo/server.py` is an amplitude/silence check, not a noise-floor or SNR
  check — it will not catch loud-but-noisy input.
- **No voice-activity detection (VAD) or silence-trimming happens inside VANI.**
  Leading/trailing silence is not trimmed anywhere in the reviewed code; if the
  denoiser leaves silence at the start of a clip, that silence occupies part of the
  10-second analysis window, at the potential cost of speech content being pushed
  past the truncation point.
- No documentation or code comment anywhere states an assumed minimum SNR, a maximum
  reverberation time, or an expected loudness/LUFS target for denoiser output — these
  are open questions (Section 8), not something the code silently handles gracefully.

---

## 4. Models

| Item | Detail |
|---|---|
| Backbone | `facebook/mms-lid-4017` (Meta AI MMS project), Wav2Vec2 architecture, ~1B parameters, hidden size 1280, 48 transformer layers, pretrained for LID across 4017 languages. Loaded from the local HuggingFace cache (`~/.cache/huggingface/hub/models--facebook--mms-lid-4017`, 7.3 GB on disk in this environment). |
| **License** | **CC-BY-NC-4.0 (non-commercial)** — confirmed directly from the cached model card (`license: cc-by-nc-4.0`). **This is a first-order integration/legal question**: if the target deployment is anything beyond research/demonstration use, this license needs explicit legal sign-off before the current backbone can be used. |
| Adaptation | LoRA (rank 16, α=32) on `q_proj`/`v_proj` of transformer layers 24-47 only, via HuggingFace `peft`. Rest of backbone frozen. |
| Trained checkpoint: language head | `checkpoints/stage2_v3/best.pt` — 71 MB (LoRA deltas + language layer-weighting + pooling + MLP). Reported Language Macro-F1 = 0.9764 under a clip-level (non-speaker-disjoint) protocol, per `reports/eval_full_v3.txt`. A separately-trained sibling, `checkpoints/v2/stage2_speakerfix/best.pt` (also 71 MB), reports 0.970 under a speaker-disjoint protocol (`reports/v2/eval_speakerfix_fleurstest.txt`) — **these are two different checkpoints, not two views of the same weights.** |
| Trained checkpoint: dialect head | `checkpoints/stage4_phaseB/best.pt` — 31.6 MB (dialect layer-weighting + pooling + MLP for the `mandarin_kespeech` scheme only). Reported Dialect Macro-F1 = 0.601. Trained with the backbone and language head fully frozen ("gradient isolation") to avoid a newly-initialized head's noisy gradients degrading the already-trained language head. |
| **Untrained head present in the architecture** | `configs/config.yaml` also declares a `punjabi_ldcil` dialect scheme (3 classes: malwa, doab, puadh), `status: pending_ldcil_acquisition`. `VANIModel.__init__` unconditionally constructs a full head (layer-weighting + pooling + MLP) for it, and I confirmed by inspecting both checkpoints directly that `dialect_heads.punjabi_ldcil.*` weight tensors exist in both — but no LDCIL data was ever acquired, so every training row masks this head's loss (`ignore_index=-1` in `src/focal_loss.py`); its weights are effectively random and were never updated by gradient descent. **Any code that calls the raw model and reads `logits["punjabi_ldcil"]` will get meaningless output, with nothing in the tensor shapes or checkpoint file distinguishing it from the real, trained `mandarin_kespeech` head.** `demo/inference.py` avoids this only by convention — it hardcodes `self.dialect_scheme = "mandarin_kespeech"` and never reads the other head — not by any architectural guard. |
| Which checkpoint pair is "production" | Ambiguous as of this writing. The demo (`demo/inference.py`, `demo/server.py`) deliberately loads **v1 language (`stage2_v3`) + v1 dialect (`stage4_phaseB`)** together, because pairing the newer, more rigorously speaker-disjoint-evaluated v2 language checkpoint with the v1 dialect head is documented (in the same file's docstring, and in `vani_gui_context.md`) to "collapse dialect F1 to near-zero." The demo UI displays the *v2* validated metrics (0.970 / 0.601 / 0.948) as "the numbers" while actually running *v1* weights for the language head — reconciled only because the v1/v2 language gap happens to be small (0.9764 vs 0.970), not because they are the same weights. An integrator needs one explicit, final answer on which checkpoint pair ships. |
| Hardware used for development/eval | Single NVIDIA RTX 4050 Laptop GPU, 6141 MiB (~6 GB) VRAM, driver 581.42, CUDA 12.1 (via `torch==2.5.1+cu121`). Runs bf16 autocast if `torch.cuda.is_bf16_supported()`, else fp16; falls back to CPU if no CUDA device is found (`torch.cuda.is_available()`), with no CPU-specific optimization or benchmark in the reviewed code. |
| Latency | `demo/server.py` reports a live warm-up latency at startup (read from the model object, not a fixed constant) and a per-request `latency_ms` field, but **I did not find any logged, reproducible steady-state latency or throughput benchmark anywhere in `reports/`** — this is an open question (Section 8), not a documented number. |

---

## 5. Dependencies & Environment

**No `requirements.txt`, `pyproject.toml`, `environment.yml`, or `Pipfile` exists
anywhere in this repository** (confirmed by an exhaustive search) — this is a real gap,
not an oversight in this document. The versions below are what is actually installed
in this development environment, captured directly (`pip`/`python -c "import x;
print(x.__version__)"`), and should be treated as "known-working in this one
environment," not as a pinned, portable dependency manifest.

| Package | Version (this dev environment) | Used for |
|---|---|---|
| Python | 3.12.7 | — |
| torch | 2.5.1+cu121 | model, training, inference |
| transformers | 5.9.0 | `Wav2Vec2Model` backbone |
| peft | 0.19.1 | LoRA |
| scikit-learn | 1.8.0 | eval metrics (`classification_report`, `f1_score`, `confusion_matrix`) |
| numpy | 2.4.6 | array handling |
| soundfile | 0.13.1 | primary audio decode (libsndfile) |
| librosa | 0.11.0 | resampling; time-stretch augmentation |
| scipy | (present, version not separately pinned) | bandpass-filter augmentation (`butter`/`sosfilt`) |
| PyYAML | 6.0.3 | `configs/config.yaml` parsing |
| fastapi | 0.136.3 | demo server only |
| uvicorn | 0.49.0 | demo server ASGI runner only |
| PyAV (`av`) | 18.0.0 | demo server only — WebM/Opus mic-recording decode fallback |

**Non-Python:**
- CUDA 12.1 (bundled via `torch`'s wheel; NVIDIA driver 581.42 present on the dev
  machine). No evidence a separately-installed system CUDA toolkit is required beyond
  the driver.
- `ffmpeg` CLI was **not found on PATH** in this environment. PyAV statically bundles
  its own compiled libav libraries, so the WebM/Opus decode fallback in
  `demo/server.py` likely does not need a system `ffmpeg` binary — but this was not
  verified end-to-end, and other audio formats decoded via `soundfile` depend on
  whatever codecs the installed `libsndfile` build supports (MP3 support in particular
  varies by `libsndfile` version). Flagged as an open question for a different target
  environment.
- No `Dockerfile` or container definition exists anywhere in the repo.

---

## 6. Configuration

The single configuration file is `configs/config.yaml`:

- `language_classes:` — fixed, ordered list of 8 language names; **list order defines
  the classifier's output index order** (the file's own comment: *"edit only via an
  approval gate"*).
- `num_language_classes: 8` — must stay in sync with the list above; a second boolean
  flag (`wenetspeech_wu_acquired: false`) gates a documented future 9th class ("wu"),
  meaning the class count is not permanently fixed at 8 from the integrator's
  perspective, only fixed *for the currently-shipped checkpoints*.
- `dialect_heads:` — two schemes: `mandarin_kespeech` (7 classes, active/trained) and
  `punjabi_ldcil` (3 classes, declared but never trained — see Section 4).
- `backbone_hf_id`, `backbone_hidden_size` (1280), `backbone_num_hidden_layers` (48),
  `backbone_num_hidden_states` (49) — architecture constants, must match the loaded
  backbone.
- `lora_rank: 16`, `lora_target_modules: [q_proj, v_proj]` — LoRA config.
  `lora_alpha` is **not** in this file; it's hardcoded as `lora_rank * 2` inside
  `VANIModel.apply_lora()` in `src/model.py`.
- Training-only fields not relevant to inference: `lr`, `batch_size` (6),
  `num_epochs` (15), `val_frac`, `test_frac`.

**Not present anywhere in configuration:**
- No confidence threshold / abstention setting. `VANIModel` always emits a top-1
  class for every head on every call; there is no "reject below confidence X"
  behavior built into the model itself. The only rejection logic anywhere is
  `demo/server.py`'s pre-inference RMS-silence gate (Section 3.1), which is an
  amplitude check, not a post-inference confidence check.
- No environment variables control model behavior, except `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1`, set unconditionally at the top of `demo/inference.py` to
  force offline weight loading (prevents `transformers` from phoning home to check
  for model updates). No proxy or network configuration is surfaced anywhere.
- `MAX_DUR_SEC` (10.0) and `SAMPLE_RATE` (16000) are hardcoded Python constants in
  `demo/inference.py`, not read from `configs/config.yaml` and not overridable via
  environment variable.

---

## 7. Known Constraints & Integration Risks

1. **License.** MMS-LID-4017 is CC-BY-NC-4.0 — non-commercial only. Any deployment
   beyond research/demonstration needs explicit legal clearance before using this
   backbone as-is.
2. **No streaming or chunking.** Only the first 10 seconds of any clip are ever
   analyzed (demo path); anything longer is silently dropped with no aggregation, no
   sliding window, and (outside the demo layer) no signal to the caller that
   truncation even happened.
3. **No batched throughput path in serving.** `demo/server.py`'s folder/batch mode
   (`/api/analyze_dir`) processes one file at a time sequentially via SSE; only the
   offline `scripts/evaluate.py` path uses real batched tensors (`DataLoader`,
   `--batch-size`, default 4). Expect serving throughput to scale roughly linearly
   with file count, not to benefit from GPU batching, unless re-architected.
4. **No input normalization.** Raw float32 PCM goes straight into the backbone with
   no zero-mean/unit-variance or loudness normalization step anywhere in the code. Any
   gain/loudness mismatch between a denoiser's output convention and the training
   corpora's implicit convention is currently unvalidated.
5. **Inconsistent silence/rejection handling across entry points.** The RMS < 0.01
   near-silence gate exists only in `demo/server.py`'s upload/mic path.
   `demo/inference.py`'s file path and `scripts/evaluate.py`'s batch path have no
   equivalent — heavily attenuated denoiser output could produce a full, confident-
   looking, but meaningless prediction through those paths.
6. **Untrained `punjabi_ldcil` head is live in the architecture** (Section 4) — a real
   trap for any code that talks to `VANIModel` directly instead of going through
   `demo/inference.py`'s hardcoded head selection.
7. **Ambiguous "production" checkpoint pairing** (Section 4) — three combinations
   exist in the repo; only v1 language + v1 dialect is validated as internally
   consistent, and that's the one the demo ships, not necessarily the one with the
   best-audited language numbers (v2).
8. **No documented steady-state latency/throughput benchmark** exists in `reports/` —
   only a live warm-up-time readout at server start and a per-request timer. Expect
   to need to benchmark this yourselves against the target hardware.
9. **Known accuracy weak points**, per this project's own eval reports (measured on
   source-corpus audio through VANI's own augmentation pipeline, not on real denoiser
   output): Kashmiri/Dogri/Punjabi/Urdu confuse with each other (same Indo-
   Aryan/Indo-Iranian language cluster); the Jiao-Liao Mandarin dialect class scores
   F1 = 0.311 on only 43 test clips (small-sample, not fully trustworthy in
   isolation); Tibetan shows an unexplained cross-corpus confusion pattern with
   Cantonese in 15 of 31 errors in one out-of-corpus evaluation.
10. **No evaluation exists yet against real denoiser output.** Every accuracy number
    in this repository's `reports/` and `reports/v2/` directories was measured on
    source-corpus audio (FLEURS, IndicVoices-R, KeSpeech, OpenSLR corpora, etc.),
    optionally passed through VANI's own synthetic augmentation (noise, telephone-
    band filtering, time-masking — explicitly *not* reverb). The
    `[Denoiser] -> [VANI LID]` boundary this document is meant to support has no
    empirical validation anywhere in the codebase.

---

## 8. Open Questions for the Integrator

- What sample rate, bit depth, and channel layout does the denoiser actually emit?
  VANI will auto-resample/downmix if given something other than 16 kHz mono, but this
  has never been exercised against real denoiser output in this repo.
- What amplitude/gain convention does the denoiser use? VANI performs no
  normalization of its own; a mismatch with whatever convention the training corpora
  happened to use is unverified.
- Does the denoiser's output ever contain long silences, or has silence-
  trimming/VAD already happened upstream? VANI does not trim silence itself, and only
  one of its three entry points even checks gross RMS.
- Is the denoiser expected to dereverberate as well as denoise? Training augmentation
  explicitly excludes reverberant/RIR conditions, so reverberant output is out of
  the model's training distribution.
- What audio duration will the downstream module typically hand to VANI? If routinely
  longer than 10 seconds, a chunking/aggregation strategy must be designed —
  none currently exists anywhere in the code.
- Which of the three existing input paths (batch CSV manifest, direct Python call, or
  the demo HTTP API) is meant to become the real integration point? None was built
  with the others' constraints in mind — for instance, the silence gate exists in
  only one of the three.
- Which checkpoint pairing is meant to ship to production: v1 language + v1 dialect
  (the only validated-consistent pair, and the demo default), v2 language + v1
  dialect (explicitly warned against in the code), or a future v2 language + v2
  dialect pair that does not yet exist?
- Should the untrained `punjabi_ldcil` dialect head be trained, or removed from
  `configs/config.yaml`/the architecture before integration, to eliminate the
  untrained-head trap described in Section 4?
- Does the downstream module need class probabilities only, or does it need the
  pooled embedding vector directly? No current code path returns embeddings — this
  would require a code change.
- Is the CC-BY-NC-4.0 license on the MMS-LID-4017 backbone compatible with the
  target deployment? This needs an explicit legal answer before any non-research use.
- What error-handling/retry contract should malformed or undecodable audio follow?
  Current behavior is inconsistent: `src/dataset.py`'s `VANIDataset.__getitem__`
  silently substitutes a 1-second zero (silent) waveform on any read/decode
  exception, which will then produce a real-looking (but meaningless) prediction
  rather than an explicit error; `demo/server.py` raises an HTTP 404 only for unknown
  `clip_id`s, and falls back from `soundfile` to PyAV for undecodable upload bytes,
  but has no further fallback if both decoders fail.
- Is there a target latency or throughput SLA from the downstream module? No
  benchmark currently exists in this repository to compare against.
- Should VANI expose a versioned, formal API contract (OpenAPI/pydantic schema)? The
  only HTTP surface today (`demo/server.py`) is explicitly a demo server, not a
  production service.

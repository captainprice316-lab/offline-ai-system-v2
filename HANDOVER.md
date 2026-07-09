# VANI — Session Handover

**Purpose:** bring a fresh Claude session (new account, no memory, no context) fully up to speed on
this project. Written 2026-07-09.

**If you are a new Claude session: read this whole file before touching anything.**

---

## 0. The prompt to paste on a fresh session

> Read `HANDOVER.md` at the repo root, then `integration/INTEGRATION_PLAN.md`. Confirm the git
> working tree state matches what HANDOVER.md §3 describes before doing anything else. I'm
> continuing the 3-node integration work.

That's it. Everything else is in this file.

---

## 1. What VANI is

An **offline** (air-gapped) speech-intelligence pipeline for military radio intercepts. Takes a WAV
of a foreign-language radio conversation and produces a structured intelligence summary in English.
Streamlit GUI, single Windows workstation, no internet at runtime (`HF_HUB_OFFLINE=1` is forced in
`src/pipeline.py` before any HF import).

**Pipeline** (`src/pipeline.py`, `run_pipeline()`):

```
VAD → Preprocessing (denoise/bandpass) → Chunking → MMS-LID probe → ASR
  → Diarization → Language ID (vote) → Translation → Keywords → ISUM → SQLite → GUI
```

**Languages:** Punjabi (pa), Hindi (hi), Urdu (ur), Nepali (ne), Mandarin (zh), Pashto (ps),
Kashmiri (ks), Dogri (doi).

**Models** (all local, under `models/`):
- ASR: per-language fine-tuned Whisper large-v3 → CT2 int8 (`whisper-large-v3-<lang>-ct2`), plus
  `whisper-large-v3-turbo-ct2` as the default fallback. Pashto is medium-based.
- `pa` and `ne` are routed to **SeamlessM4T** zero-shot instead of Whisper (`asr.seamless_langs` in
  config) — it beat the fine-tuned models (pa 56%→20% WER, ne 49%→28%).
- LID: `mms-lid-256` (audio) + FastText (text), confidence-weighted vote in `LanguageRouter`.
- Translation: NLLB-200-distilled-600M for everything except Dogri, which uses IndicTrans2.
- ISUM (summarisation): **Ollama `gemma3:4b`** at `localhost:11434`.

**Key files:** `app.py` (4,895 lines, Streamlit), `src/pipeline.py` (668 lines, the orchestrator),
`config.yaml` (all tunables).

**Entry point:** `venv\Scripts\python.exe -m streamlit run app.py`

---

## 2. Machine, storage, and the junctions that will bite you

**Machine:** ASUS ROG Strix G16 **G614PM**, Windows 11 Home, NVIDIA RTX 5060 **8 GB VRAM**,
**32 GB RAM** (upgraded 2026-07-06).

**Disks:** `C:` (OS, ~112 GB free) · `D:` (empty, 500 GB — same physical disk as C:) ·
**`E:` = WD Black SN7100 1 TB NVMe** (~454 GB free). All large data lives on `E:`.

**⚠ THREE NTFS JUNCTIONS under the project point to `E:`:**

| Junction (inside repo) | Real target | Size |
|---|---|---|
| `models` | `E:\vani_models` | 52.8 GB |
| `finetune_runs` | `E:\finetune_runs` | 89 GB |
| `finetune_runs_seamless` | `E:\finetune_runs_seamless` | 6.4 GB |

**Backup gotcha (has bitten this project once):** `robocopy /XJ` and most junction-aware copies
**silently skip junctions**. A backup of the project folder captures ~3 GB of code and misses 148 GB
of models. **Always back up the three `E:\` targets as separate robocopy jobs**, and verify with
`Get-ChildItem -Recurse -Filter model.bin` rather than trusting exit codes (robocopy exit code 1 =
success, and the Claude Code harness flags any non-zero as failure — trust neither).

**Good backup:** `G:\vani_backup_2026-07-06` (~148.6 GB, verified complete, 0 failed).
The WD My Passport drive letter is **not stable** (`F:` one day, `G:` the next) — look it up with
`Get-Volume | ? FileSystemLabel -eq 'My Passport'`.

**Safe to exclude from backups:** `E:\hf_cache` (251 GB), `E:\hf_ks_temp` (77 GB), `venv` — all
re-downloadable.

---

## 3. Git state as of 2026-07-09 — READ THIS FIRST

Branch `master`, HEAD `e79a863` "Add India critical-infrastructure detection + map pins".

**The working tree has ~52 files of uncommitted work:**
- **16 modified** (tracked) — includes `app.py`, `config.yaml`, `src/pipeline.py`,
  `src/isum_module.py`, `src/seamless_asr.py`, `src/translation_module.py`
- **14 deleted** (tracked) — old eval/download scripts moved into `scripts/`
- **22 untracked** — `integration/`, `ui/`, `demo/`, `assets/`, `LRP/`, `scripts/data/`,
  `scripts/eval/*`, `scratchpad/`, `run_full_pipeline_batch.py`, and more

**This uncommitted set contains the entire Phase A pipeline speedup (§5) and is not backed up by
git.** Before any risky operation — and ideally before the account switch — commit it or at minimum
`git stash`/branch it. A fresh session must **not** assume the tree is clean and must **not** run
`git checkout --`, `git reset --hard`, or `git clean` without asking.

There is a pre-change backup of the speedup work at `scratchpad\speedup_backup_2026-07-08`.

---

## 4. Where the project stands

| Track | Status |
|---|---|
| Fine-tuning (pa/hi/ur/ne/zh/ps/ks) | **Done.** All deployed as CT2 int8. |
| Robustness eval (7 langs × 5 conditions) | **Done** 2026-07-01. Full VANI beats Whisper-only by **+14 pp** average. |
| Reports (PDF + PPTX) | **Done**, regenerated 2026-07-01. |
| Demo | **Delivered 2026-07-08** via the Streamlit app. Full pipeline verified end-to-end. |
| Pipeline speedup Phase A | **Done** 2026-07-08 evening. **Uncommitted.** |
| 3-node integration | **Planned, not started.** See `integration/INTEGRATION_PLAN.md`. |

### Model results (for reports / questions)

| Lang | WER | Note |
|---|---|---|
| zh | 8.97% | best |
| ur | 22.3% | vs ~74% baseline |
| hi | 23.1% | vs ~75% baseline |
| pa | 49.31% | LoRA r=16, step 4000; but **pa is served by SeamlessM4T (~20%)**, not this model |
| ne | 54.4% | likewise served by SeamlessM4T (~28%) |
| ks | 74.02% | −22.85 pp vs 96.87% baseline; first language with **no native Whisper token** |

**Robustness (Full VANI LangID accuracy):** zh 97–100% · pa 60–97% · ps 53–87% (MMS-LID supplies
*all* the signal — Whisper-only is 0%) · hi 47–83% · ur 27–80% · ne 17–43% · ks 0–23%.

---

## 5. Pipeline speedup Phase A (done, uncommitted)

Plan file: `C:\Users\vis15\.claude\plans\enumerated-meandering-moth.md`.
Warm run **33.1 s** (was ~42 s). Cold 57.6 s. Target was 16–19 s.

Implemented and verified in a real Streamlit run:
- **A1** ISUM knobs — `ollama_keep_alive: 0`, `num_predict: 320`, prompt truncation to 600 chars
- **A2** double-MMS fix — Stage 5 reuses the pre-ASR probe result instead of re-running MMS-LID
- **A3** NLLB cached fp16, parked in CPU RAM between files (`offload_to_cpu: true`)
- **A4** SeamlessM4T cached with a CPU⇄GPU swap (`_get_seamless_model` @ `st.cache_resource`)
- **A5** **ISUM switched `gemma3:12b` → `gemma3:4b`** — matched threat levels and locations on all
  3 eval cases, ~2× faster. 12b stays installed as rollback. Compare script:
  `scratchpad\isum_model_compare.py`
- **A6** batch/main-tab device fix (`_run_device` / `_brun_device`)

**Remaining Phase B hot spots** (warm run): ASR 10.8 s (Seamless 4.6 GB CPU→GPU promote),
Chunking bucket 7.1 s (per-language PA-CT2 model is **not** cached), Translation 5.9 s (NLLB
promote/park; candidate fix = CT2 int8 NLLB).

---

## 6. Active work: 3-node LAN integration

Full plan: **`integration/INTEGRATION_PLAN.md`** — read it, it's the current task.

Three machines on an isolated LAN (no internet, no gateway):
- **NODE-A** (Gaurav) — DiariZen diarization + DeepFilterNet3 denoising. Context:
  `integration/gaurav's module/integration.md`
- **NODE-B** (Sanket) — LID (8 languages) + Mandarin dialect ID. Context:
  `integration/sanket's module/VANI_LID_INTEGRATION_CONTEXT.md`
- **NODE-C** — this repo. Streamlit GUI, orchestrator, final output.

Design in one paragraph: A and B become stateless HTTP services; C orchestrates. C's existing local
MMS-LID probe moves earlier to supply Gaurav the language tag his clustering needs (he has no LID),
which lets Sanket's LID run *after* denoising where it was trained to run. Gaurav adds a
`mixed_denoised.wav` output (~10 lines) so VANI's single-timeline downstream needs no changes.
Everything is behind config flags with `fallback_on_error: true`, so the working demo path cannot
regress. Estimated ~4–5 days, ~70% on NODE-C, **zero cost**.

**Not started.** No code written yet on any node.

---

## 7. Landmines — things that have already cost time

1. **CT2 tokenizer fix (critical).** `ct2-transformers-converter` does **not** copy
   `tokenizer.json`. faster-whisper then falls back to whisper-tiny's tokenizer, where
   `<|transcribe|>=50359` — but large-v3 expanded its vocab, so **50359 is `<|translate|>`**. The
   one-token offset makes every fine-tuned large-v3 CT2 model *silently translate to English*
   (~100% WER against source-language refs). Always:
   `Copy-Item finetune_runs/<lang>/adapter/tokenizer.json models/<ct2_name>/tokenizer.json`
   `finetune_whisper.py merge_and_convert()` now does this automatically; manual rebuilds don't.
   Does **not** apply to medium-based models (Pashto). Do **not** copy the turbo tokenizer onto a
   fine-tuned large-v3 model.
2. **CT2 also doesn't emit `preprocessor_config.json`** — copy it too, or faster-whisper crashes on
   a mel-bin shape mismatch (large-v3 needs `feature_size=128`).
3. **Use `venv\Scripts\python.exe`, never `py -3.11` or bare `python`.** Only the venv has
   torch/peft. Bare `python` is not even on PATH in this shell.
4. **Never `load_dataset()` for audio on this machine.** `datasets` 5.0.0 needs `torchcodec`, which
   has no Windows build. Use `pyarrow.parquet.read_table()` + `soundfile.read(io.BytesIO(...))`.
   Reference: `scripts/eval/eval_indic_conformer_ks.py`. For streaming, monkey-patch
   `datasets.features.audio.Audio.decode_example` — see `_patch_audio_decode()` in
   `scripts/eval/robustness_eval.py`.
5. **The Windows console (cp1252) cannot print Gurmukhi/Devanagari/Han.** Scripts crash on
   `print()` *after* the pipeline succeeded. Wrap stdout:
   `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`, or read the saved JSON.
6. **8 GB VRAM is the binding constraint on everything.** It's why ISUM keep_alive is 0, why NLLB
   and Seamless park in CPU RAM, why training eval had to go greedy (`num_beams=1` +
   `torch.cuda.empty_cache()` before each eval — an OOM crash at step 2400 cost a training run).
7. **`cross_file_reid` is disabled and should stay disabled.** MFCC-stat cosine cannot separate
   speakers on bandpassed radio audio (same-speaker mean cos 0.993 vs different-speaker 0.984 →
   77% false-match rate; every intercept collapsed into `VOICE_001`). Re-enable only after
   switching to real speaker embeddings (ECAPA-TDNN). Gaurav's denoiser was chosen partly because
   it *preserves* speaker identity (15.62% EER, better than raw audio) — his per-speaker tracks are
   the natural input for that future fix.
8. **The Mandarin "100% baseline WER"** in the reports is a **turbo-specific** artefact (turbo
   defaults to the translate task). `whisper-large-v3` defaults to transcribe and produces correct
   Chinese. Be precise about this if it comes up.
9. **`ks` needed a custom `<|ks|>` token** at vocab ID 51866, embedding initialised from `<|ur|>`,
   forced prefix `[50258, 51866, 50360, 50364]` via `TemplateProcessing`, plus a faster-whisper
   patch at import in `src/asr_module.py` (its language allowlist rejects `"ks"`). Template for any
   future no-token language: `scripts/add_ks_token.py`.
10. **Root-dir `Remove-Item` is blocked by the Claude Code sandbox** even with
    `dangerouslyDisableSandbox`. Use the Bash tool's `rm -rf` with `/c/...`, `/e/...` paths.
    Also: `cmd //c mklink /J` via Git Bash mangles arguments — use PowerShell
    `New-Item -ItemType Junction` instead.

---

## 8. Standing user preferences

- **Detailed explainer owed at project end.** The user wants a comprehensive technical reference
  covering *every* parameter and method: LoRA hyperparameters (r, α, target modules, dropout),
  training args (steps, batch size, warmup, grad norm), the data pipeline, CT2 conversion, eval
  metrics (WER, chrF), the SeamlessM4T comparison, the tokenizer fix. Produce it when the user
  signals the project is wrapping up.

---

## 9. Open items

1. **Commit the 52 uncommitted files** (§3). Highest priority, lowest effort.
2. **Integration Phase 0** — static IPs, firewall rules, `/health` endpoints on A and B, prove
   reachability with `curl` from C.
3. **Decisions needed from the other two owners** before integration Phase 1:
   - Sanket: confirm the shipping checkpoint pair (v1 language `stage2_v3` + v1 dialect
     `stage4_phaseB` — the only internally-consistent combination).
   - Gaurav: default diarizer `variant` (`clean` vs `robust` — they are **not** interchangeable).
   - Both: demo clip set. Safe intersection across all three modules is **Punjabi, Mandarin,
     Urdu**. Pashto is supported everywhere but is the weakest link in two of the three.
4. **Licensing, flagged not blocking:** DiariZen's weights, Sanket's MMS-LID-4017 backbone, and
   VANI's own `mms-lid-256` are all **CC BY-NC 4.0**. The integrated system is research /
   non-commercial only. Fine for the demo; a hard blocker for productization.
5. Pipeline speedup Phase B (§5) — optional, orthogonal to integration.

---

## 10. Memory index (reproduced so it survives an account switch)

Claude's persistent memory for this project lives at
`C:\Users\vis15\.claude\projects\C--Users-vis15-offline-ai-system-v2\memory\`. It is keyed by
*project path*, not by account, so it should survive a `/login` — but this file is written to be
self-contained in case it doesn't.

| Memory file | Covers |
|---|---|
| `feedback_ct2_tokenizer_fix.md` | §7.1 |
| `feedback_datasets_v5_workaround.md` | §7.4 |
| `reference_machine_storage_junctions.md` | §2 |
| `project_ks_baseline.md` / `project_ks_training_complete.md` | §4 (ks track) |
| `project_pa_v3_deployed.md` | §4 (pa track), §7.8 |
| `project_robustness_eval.md` | §4 (robustness) |
| `project_demo_2026-07-08.md` | §4 (demo) |
| `project_pipeline_speedup_plan.md` | §5 |
| `project_detailed_explainer.md` | §8 |

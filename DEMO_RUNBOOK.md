# VANI — Demo Runbook (08 Jul 2026)

## 0. Pre-demo checklist (do ~15 min before)
1. **Plug in power** (GPU throttles on battery) and connect to a screen.
2. **Start Ollama** (for ISUM) — it should auto-run; verify: `ollama ps`.
3. **Launch VANI**: from the project folder, `venv\Scripts\streamlit run app.py`
   → opens `http://localhost:8501`.
4. **Warm-up run** (critical — first run loads models, ~40 s): process
   `demo_audio/08_hi_clear.wav` once. After this, runs are fast (~10–15 s).
5. **Click through each tab once** so they're rendered and cached.
6. **Check the sidebar GPU VRAM bar** — if it's already amber/red before you
   start, Gemma is still resident from warm-up; wait ~1 min or run
   `ollama stop gemma3:12b`.
7. **Mic**: click ● Start recording once and Allow the browser mic permission
   prompt now, so it doesn't pop up mid-demo.

## 1. The 13 demo clips  (`demo_audio/`)
Filenames encode `NN_lang_threat_location`. Manifest: `demo_audio/manifest.json`.

| Clip | Lang | Threat | Map pin | Shows |
|------|------|--------|---------|-------|
| 01 srinagar | HI | 🔴 CRITICAL | Srinagar | enemy + attack + bomb |
| 02 anantnag **coded** | HI | 🟠 HIGH | Anantnag | **aloo→grenades, mirchi→bullets, mehmaan→infiltrators** |
| 03 pulwama | HI | 🟠 HIGH | Pulwama | weapons + direction |
| 04 anantnag | HI | 🟠 HIGH | Anantnag | retreat + casualties |
| 05 kupwara | HI | 🟡 MEDIUM | Kupwara | surveillance |
| 06 jammu | HI | 🟡 MEDIUM | Jammu | logistics |
| 07 comms | HI | 🔵 LOW | – | radio/callsign |
| 08 clear | HI | 🟢 CLEAR | – | benign (use as warm-up) |
| 09 muzaffarabad | UR | 🔴 CRITICAL | Muzaffarabad | enemy + explosives |
| 10 baramulla **coded** | UR | 🟠 HIGH | Baramulla | **doctor→IED maker, saman→weapons** |
| 11 baramulla | UR | 🟠 HIGH | Baramulla | pre-attack + direction |
| 12 command | UR | 🟡 MEDIUM | – | await orders |
| 13 clear | UR | 🟢 CLEAR | – | benign |
| 14 srinagar | NE | 🔴 CRITICAL | Srinagar | enemy + bomb (**SeamlessM4T ASR**) |
| 15 pulwama | NE | 🟠 HIGH | Pulwama | weapons + location |
| 16 jammu | NE | 🟡 MEDIUM | Jammu | logistics |
| 17 pa showcase | PA | – | (Berlin etc.) | **real speech — SeamlessM4T transcription quality** |
| 18 pa showcase | PA | 🟠 HIGH | – | real speech — SeamlessM4T quality |

**Punjabi/Nepali note:** clips 14–18 route ASR through **zero-shot SeamlessM4T**
(pa 56%→20%, ne 49%→28% WER vs fine-tuned Whisper). Clip 17 is the clean
showcase — Whisper mistranscribed it into *"a million logs had been altered"*;
SeamlessM4T gets *"a million people were displaced."* Punjabi has no neural TTS,
so 17/18 are real FLEURS speech (which shows the ASR win better anyway).

## 2. Suggested flow (~8 min)
1. **Open on the Map / Dashboard** — pre-populated with all 13 intercepts:
   threat spread + J&K/Pakistan pins. "This is an evening's worth of intercepts."
2. **Live single run** — Process tab, upload `01_hi_critical_srinagar.wav`.
   Narrate the 10 stages; land on the **CRITICAL** badge + Srinagar pin +
   Hindi→English translation + ISUM summary.
3. **The codeword reveal** — run `02_hi_coded_anantnag.wav`. Point at the amber
   **"POSSIBLE CODED TERMINOLOGY DETECTED"** panel: potato→grenades,
   chilli→bullets, guest→infiltrator. This is the differentiator.
4. **Live mic moment** — click ● Start recording, speak a line
   (e.g. *"Muzaffarabad mein dushman par hamla, bomb tayyar hai"*), stop, RUN.
   Unscripted, real-time, fully offline.
5. **Close on scope** — 7 fine-tuned languages, 256-language ID, 100% offline.

## 3. Talking points
- **Fully offline** — no internet after setup (say it while showing the ● OFFLINE badge).
- **Language-specific fine-tuned Whisper** per language (LoRA), auto-selected by MMS-LID.
- **Coded terminology** decoded from an open-source militant-tradecraft lexicon
  (be honest: illustrative, analyst-lead not ground truth).
- **GPU-accelerated** — point at the VRAM bar climbing during ASR.

## 4. Troubleshooting
- **First run slow (~40 s)** → expected (model load). Warm up beforehand (step 0.4).
- **CUDA out-of-memory on 2nd/3rd file** → Gemma held VRAM; we set keep_alive=0
  so it unloads after each ISUM. If it still bites: `ollama stop gemma3:12b`.
- **Urdu clip routes to turbo model (probe reads `hn`)** → output is still correct
  via the Arabic-script cascade; don't flinch.
- **Mic does nothing** → browser mic permission denied; re-allow in the address-bar
  lock icon and refresh.
- **A tab errors** → refresh the browser tab; Streamlit recovers.

## 5. Reset DB to demo-only (optional, for a clean aggregate)
The DB also holds earlier test runs. For a pristine Map/Dashboard showing only
the 13 demo intercepts: use the **CLEAR tab** in the app to wipe history, then
re-run `python populate_demo_db.py`.

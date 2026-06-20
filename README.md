# VANI – Voice Analysis & Neural Intelligence

**Offline AI-based Radio Intercept Analysis System**  
M.Tech Research Project · IIT Indore

---

## Overview

VANI processes radio intercepts in multiple Indic and foreign languages, producing:
- Real-time transcription via Whisper ASR
- Automatic language identification (Hindi, Punjabi, Dogri, Urdu, Pashto, Mandarin, Nepali, Burmese)
- English translation via IndicTrans2 and NLLB
- Keyword-based threat detection
- Structured Intelligence Summary (ISUM) reports
- Searchable transcript database

All processing is **fully offline** — no internet connection required after setup.

---

## Project Structure

```
offline_ai_system/
├── app.py                          # Streamlit UI entry point
├── config.yaml                     # All configuration
├── requirements.txt
├── models/
│   ├── whisper_medium/             # faster-whisper CTranslate2 format
│   ├── indictrans2-indic-en-1B/    # IndicTrans2 translation model
│   ├── nllb-200-distilled-600M/    # NLLB translation model
│   └── langid/lid.176.bin          # FastText language ID model
├── src/
│   ├── pipeline.py                 # Main orchestration pipeline
│   ├── vad_module.py               # Silero VAD
│   ├── preprocessing.py            # Audio preprocessing
│   ├── chunker.py                  # VAD-aware audio chunking
│   ├── asr_module.py               # Whisper ASR
│   ├── language_module.py          # LangID + routing
│   ├── translation_module.py       # IndicTrans2 + NLLB
│   ├── keyword_module.py           # Keyword detection
│   ├── isum_module.py              # ISUM generation
│   ├── database.py                 # SQLite storage
│   ├── search.py                   # Transcript search
│   └── utils.py                    # Shared utilities
├── ui/
│   └── streamlit_helpers.py        # Reusable UI components
├── alerts/
│   └── keyword_dictionary.json     # Multilingual keyword dictionary
├── database/
│   └── transcripts.db              # SQLite database
├── input_audio/                    # Place input WAV files here
├── output/                         # Pipeline JSON outputs
└── logs/                           # System logs
```

---

## Setup

### 1. Install dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install PyTorch (CPU)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

### 2. Download models (one time, requires internet)

```bash
# Whisper medium (CTranslate2 format for faster-whisper)
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8', download_root='models/whisper_medium')"

# FastText language ID
# Download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html
# Place at: models/langid/lid.176.bin

# IndicTrans2 (from HuggingFace)
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
AutoTokenizer.from_pretrained('ai4bharat/indictrans2-indic-en-1B', cache_dir='models/indictrans2-indic-en-1B')
AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-indic-en-1B', cache_dir='models/indictrans2-indic-en-1B')
"

# NLLB
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='models/nllb-200-distilled-600M')
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='models/nllb-200-distilled-600M')
"
```

### 3. Run

```bash
streamlit run app.py
```

---

## Usage

1. Open browser at `http://localhost:8501`
2. Upload a `.wav` audio file in the **PROCESS** tab
3. Click **RUN PIPELINE**
4. View results across all tabs:
   - **PROCESS** — transcript, translation, keyword highlights
   - **ISUM REPORT** — structured 5W intelligence summary
   - **SEARCH** — fuzzy keyword search across all intercepts
   - **DASHBOARD** — threat statistics and language distribution
   - **HISTORY** — all processed intercepts

---

## Supported Languages

| Language | ISO Code | ASR | Translation |
|----------|----------|-----|-------------|
| Hindi | hi | ✓ | IndicTrans2 |
| Punjabi | pa | ✓ | IndicTrans2 |
| Dogri | doi | ✓ | IndicTrans2 |
| Urdu | ur | ✓ | IndicTrans2 |
| Nepali | ne | ✓ | IndicTrans2 |
| Kashmiri | ks | ✓ | IndicTrans2 |
| Pashto | ps | ✓ | NLLB |
| Mandarin | zh | ✓ | NLLB |
| Burmese | my | ✓ | NLLB |

---

## Lab Server Upgrade

Change only two lines in `config.yaml`:
```yaml
device: cuda                         # was: cpu
translation:
  unload_after_use: false            # was: true (keep models in GPU VRAM)
```

---

## Architecture

```
Audio → VAD → Preprocessing → Chunking → Whisper ASR
                                              ↓
                               LangID (Whisper + FastText voting)
                                              ↓
                                       Translation
                                              ↓
                                    Keyword Detection
                                              ↓
                                   ISUM Generation
                                              ↓
                               SQLite DB + JSON Output
                                              ↓
                                       VANI UI
```

---

*Developed as M.Tech Research Project, IIT Indore*

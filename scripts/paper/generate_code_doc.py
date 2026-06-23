"""
generate_code_doc.py
Generates a detailed code-explanation PDF for the VANI project.
Run: python generate_code_doc.py
Output: VANI_Code_Documentation.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus.tableofcontents import TableOfContents

PAGE_W, PAGE_H = A4

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1a1a2e")
C_BLUE    = colors.HexColor("#16213e")
C_ACCENT  = colors.HexColor("#0f3460")
C_TEAL    = colors.HexColor("#1976d2")
C_ORANGE  = colors.HexColor("#e65100")
C_GREEN   = colors.HexColor("#2e7d32")
C_GREY    = colors.HexColor("#546e7a")
C_LIGHT   = colors.HexColor("#eceff1")
C_CODE_BG = colors.HexColor("#f5f5f5")
C_CODE_FG = colors.HexColor("#212121")
C_HEAD_BG = colors.HexColor("#0d47a1")
C_HEAD_FG = colors.white


def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"],
            fontSize=28, textColor=C_DARK, leading=36,
            spaceAfter=12, alignment=TA_CENTER, fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle", parent=base["Normal"],
            fontSize=14, textColor=C_GREY, leading=20,
            spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica",
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=18, textColor=C_ACCENT, leading=24,
            spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold",
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=14, textColor=C_TEAL, leading=20,
            spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"],
            fontSize=12, textColor=C_ORANGE, leading=16,
            spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, textColor=colors.black, leading=15,
            spaceAfter=6, alignment=TA_JUSTIFY, fontName="Helvetica",
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=10, textColor=colors.black, leading=14,
            spaceAfter=3, leftIndent=18, bulletIndent=6,
            fontName="Helvetica",
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"],
            fontSize=8.5, textColor=C_CODE_FG, leading=13,
            fontName="Courier", backColor=C_CODE_BG,
            leftIndent=10, rightIndent=10,
            spaceBefore=4, spaceAfter=4,
            borderPad=6,
        ),
        "code_comment": ParagraphStyle(
            "CodeComment", parent=base["Code"],
            fontSize=8.5, textColor=C_GREEN, leading=13,
            fontName="Courier", backColor=C_CODE_BG,
            leftIndent=10, rightIndent=10,
        ),
        "label": ParagraphStyle(
            "Label", parent=base["Normal"],
            fontSize=9, textColor=C_GREY, leading=12,
            fontName="Helvetica-Oblique", spaceAfter=2,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["Normal"],
            fontSize=10, textColor=C_DARK, leading=15,
            fontName="Helvetica", leftIndent=12, rightIndent=12,
            backColor=colors.HexColor("#e3f2fd"),
            borderPad=8, spaceBefore=6, spaceAfter=6,
        ),
        "warning": ParagraphStyle(
            "Warning", parent=base["Normal"],
            fontSize=10, textColor=colors.HexColor("#b71c1c"), leading=15,
            fontName="Helvetica-Bold", leftIndent=12,
            backColor=colors.HexColor("#ffebee"),
            borderPad=6, spaceBefore=4, spaceAfter=4,
        ),
    }
    return styles


def code_block(lines, styles, label=None):
    """Return a list of Paragraph flowables for a code block."""
    items = []
    if label:
        items.append(Paragraph(label, styles["label"]))
    for line in lines:
        safe = (line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        items.append(Paragraph(safe, styles["code"]))
    return items


def section_rule():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#bbdefb"), spaceAfter=4, spaceBefore=4)


def module_header_table(name, file_path, purpose, styles=None):
    data = [
        [Paragraph(f"<b>{name}</b>", ParagraphStyle(
            "MH", fontSize=13, textColor=C_HEAD_FG, fontName="Helvetica-Bold",
            leading=17,
        ))],
        [Paragraph(f"<font color='#90caf9'>{file_path}</font>", ParagraphStyle(
            "MF", fontSize=9, textColor=C_HEAD_FG, fontName="Courier", leading=13,
        ))],
        [Paragraph(purpose, ParagraphStyle(
            "MP", fontSize=10, textColor=colors.white, fontName="Helvetica",
            leading=14, alignment=TA_JUSTIFY,
        ))],
    ]
    t = Table(data, colWidths=[PAGE_W - 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_ACCENT]),
    ]))
    return t


def kv_table(rows, styles, col1=6*cm, col2=None):
    col2 = col2 or (PAGE_W - 4*cm - col1)
    data = []
    for k, v in rows:
        data.append([
            Paragraph(f"<b>{k}</b>", ParagraphStyle(
                "KV_K", fontSize=9.5, textColor=C_DARK, fontName="Helvetica-Bold",
                leading=13,
            )),
            Paragraph(v, ParagraphStyle(
                "KV_V", fontSize=9.5, textColor=colors.black, fontName="Helvetica",
                leading=13,
            )),
        ])
    t = Table(data, colWidths=[col1, col2])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LIGHT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
    ]))
    return t


def bullet(text, styles):
    return Paragraph(f"• &nbsp;{text}", styles["bullet"])


def callout(text, styles):
    return Paragraph(text, styles["callout"])


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_cover(s):
    return [
        Spacer(1, 3*cm),
        Paragraph("Offline Multilingual Spoken Language<br/>Processing Pipeline", s["title"]),
        Spacer(1, 0.4*cm),
        Paragraph("Complete Code Documentation &amp; Technical Reference", s["subtitle"]),
        Spacer(1, 0.2*cm),
        HRFlowable(width="60%", thickness=2, color=C_TEAL, hAlign="CENTER"),
        Spacer(1, 0.4*cm),
        Paragraph("For Professor Review — M.Tech Project", s["subtitle"]),
        Spacer(1, 4*cm),
        kv_table([
            ("Project",       "Multilingual Offline Audio Analysis Pipeline"),
            ("Entry Point",   "streamlit run app.py"),
            ("Hardware Target", "8 GB RAM · CPU-only · Fully Offline"),
            ("Languages Supported", "Hindi · Punjabi · Urdu · Nepali · Dogri · Kashmiri · Pashto · Sindhi · Mandarin · Tibetan · English + more"),
            ("Core Models",   "Whisper large-v3-turbo (ASR) · NLLB-200-600M (Translation) · MMS-LID-256 (LangID) · FastText lid.176 (LangID) · Qwen2.5-1.5B (Summarisation)"),
            ("Source Files",  "18 Python modules across src/ + app.py"),
            ("Database",      "SQLite with FTS5 full-text search"),
            ("Pipeline Stages", "10 sequential stages from raw audio to structured summary"),
        ], s),
        PageBreak(),
    ]


def build_toc_page(s):
    return [
        Paragraph("Table of Contents", s["h1"]),
        section_rule(),
        Spacer(1, 0.3*cm),
        Paragraph("1. Project Overview &amp; Architecture", s["body"]),
        Paragraph("2. Configuration — config.yaml", s["body"]),
        Paragraph("3. Module 1: Voice Activity Detection — vad_module.py", s["body"]),
        Paragraph("4. Module 2: Audio Preprocessing — preprocessing.py", s["body"]),
        Paragraph("5. Module 3: Audio Chunking — chunker.py", s["body"]),
        Paragraph("6. Module 4: ASR — asr_module.py", s["body"]),
        Paragraph("7. Module 5: Speaker Diarisation — diarize_module.py", s["body"]),
        Paragraph("8. Module 6: Language Identification — language_module.py", s["body"]),
        Paragraph("9. Module 7: Translation — translation_module.py", s["body"]),
        Paragraph("10. Module 8: Keyword Detection — keyword_module.py", s["body"]),
        Paragraph("11. Module 9: Structured Summarisation — isum_module.py", s["body"]),
        Paragraph("12. Module 10: Database — database.py", s["body"]),
        Paragraph("13. Pipeline Orchestrator — pipeline.py", s["body"]),
        Paragraph("14. Frontend — app.py", s["body"]),
        Paragraph("15. Key Design Decisions &amp; Technical Challenges", s["body"]),
        Paragraph("16. Data Flow: Audio File to Final Output", s["body"]),
        PageBreak(),
    ]


def build_overview(s):
    story = []
    story.append(Paragraph("1. Project Overview &amp; Architecture", s["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "This system is a fully offline, CPU-only pipeline for processing multilingual audio recordings "
        "and producing structured English-language summaries. It was designed around a specific hardware "
        "constraint: 8 GB RAM with no GPU and no internet access. Every design decision flows from that "
        "constraint — sequential model loading, explicit memory release, int8 quantisation, and beam_size=2 "
        "for ASR. The system integrates ten discrete processing stages, each implemented as an independent "
        "Python module, orchestrated by a central pipeline runner (pipeline.py).",
        s["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Pipeline Overview", s["h2"]))
    stage_data = [
        ["Stage", "Name", "Module", "Key Technology"],
        ["1",   "Voice Activity Detection",   "vad_module.py",      "Silero-VAD (PyTorch)"],
        ["2",   "Audio Preprocessing",         "preprocessing.py",   "noisereduce + scipy Butterworth"],
        ["3",   "VAD-aware Chunking",           "chunker.py",         "soundfile + librosa"],
        ["3.5", "Pre-ASR Language Probe",       "pipeline.py",        "MMS-LID-256 (wav2vec)"],
        ["4",   "Automatic Speech Recognition", "asr_module.py",      "Whisper large-v3-turbo (CTranslate2 int8)"],
        ["4.5", "Speaker Diarisation",          "diarize_module.py",  "MFCC + Agglomerative Clustering (sklearn)"],
        ["5",   "Language Identification",      "language_module.py", "FastText + MMS-LID + Whisper vote"],
        ["6",   "Translation",                  "translation_module.py","NLLB-200-600M / IndicTrans2"],
        ["7",   "Keyword Detection",            "keyword_module.py",  "Regex patterns + severity scoring"],
        ["8",   "Structured Summarisation",     "isum_module.py",     "Rule-based NLP + spaCy NER + Qwen2.5 LLM"],
    ]
    t = Table(stage_data, colWidths=[1.2*cm, 4.8*cm, 4.2*cm, 5.3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEAD_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEAD_FG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("BACKGROUND",    (0, 4), (0, 4), colors.HexColor("#fff3e0")),  # Stage 3.5
        ("BACKGROUND",    (0, 6), (0, 6), colors.HexColor("#fff3e0")),  # Stage 4.5
    ]))
    story.append(t)
    story.append(Paragraph(
        "Stages 3.5 and 4.5 (orange) are novel insertion points that constitute "
        "the primary architectural contributions of this project.",
        s["label"],
    ))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Repository Structure", s["h2"]))
    story.append(Paragraph(
        "The project root contains app.py (Streamlit frontend), config.yaml, and all "
        "evaluation/utility scripts. All processing logic lives in src/. The database is "
        "stored in database/transcripts.db. Models are stored in models/ (downloaded separately "
        "since they are too large for the repository).",
        s["body"],
    ))
    story += code_block([
        "offline_ai_system_v2/",
        "├── app.py                    # Streamlit frontend — 10-tab UI",
        "├── config.yaml               # All runtime parameters",
        "├── src/",
        "│   ├── pipeline.py           # Orchestrator: calls all 10 stages",
        "│   ├── vad_module.py         # Stage 1: silence removal",
        "│   ├── preprocessing.py      # Stage 2: audio cleaning",
        "│   ├── chunker.py            # Stage 3: VAD-aware splitting",
        "│   ├── asr_module.py         # Stage 4: Whisper transcription",
        "│   ├── diarize_module.py     # Stage 4.5: speaker labelling",
        "│   ├── language_module.py    # Stage 5: 3-source LangID vote",
        "│   ├── translation_module.py # Stage 6: NLLB / IndicTrans2",
        "│   ├── keyword_module.py     # Stage 7: keyword detection",
        "│   ├── isum_module.py        # Stage 8: structured summary",
        "│   ├── database.py           # SQLite storage + FTS5 search",
        "│   ├── mms_module.py         # MMS-LID audio language ID",
        "│   ├── geo_module.py         # Geographic entity extraction",
        "│   ├── metrics_module.py     # Auto-metrics computation",
        "│   └── report_exporter.py    # PDF/DOCX/SRT export",
        "├── models/                   # Downloaded AI models (git-ignored)",
        "├── database/transcripts.db   # SQLite database",
        "└── alerts/keyword_dictionary.json",
    ], s, "Directory Layout")
    story.append(PageBreak())
    return story


def build_config(s):
    story = []
    story.append(Paragraph("2. Configuration — config.yaml", s["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "All runtime parameters are centralised in config.yaml. The pipeline reads this file "
        "at startup and passes the relevant sub-dictionary to each module. This means you can "
        "change beam sizes, memory thresholds, or model paths without touching any Python code.",
        s["body"],
    ))
    story.append(Paragraph("Key Sections", s["h2"]))
    story.append(kv_table([
        ("device",           "cpu / mps / cuda:0 — controls which hardware PyTorch uses. CTranslate2 "
                             "always falls back to CPU since faster-whisper doesn't support MPS."),
        ("paths.*",          "All model paths relative to project root. Adding whisper_model_zh or "
                             "whisper_model_ps enables language-specific ASR model routing."),
        ("asr.*",            "beam_size=2 and temperature=0.0 give deterministic, fast output. "
                             "condition_on_previous_text=False prevents hallucination loops on noise. "
                             "initial_prompt primes Whisper with domain vocabulary."),
        ("vad.*",            "threshold=0.45 is the Silero-VAD speech probability cutoff. "
                             "speech_pad_ms=100 adds 100 ms padding around each detected segment."),
        ("preprocessing.*",  "bandpass_filter=true + bandpass_low_hz=300 + bandpass_high_hz=3400 "
                             "restricts audio to the radio telephony band. prop_decrease=0.75 controls "
                             "noise reduction aggressiveness."),
        ("language.*",       "confidence_threshold=0.60: ensemble scores below this set uncertain=True "
                             "which propagates as LOW_LANG_CONFIDENCE flag to the output."),
        ("memory.use_mms_lid","true: enables the 3rd vote source. Set false on very low RAM systems "
                             "to skip loading the 150 MB MMS-LID model."),
        ("isum.model",       "Priority chain: ollama (gemma3:4b via REST) → qwen (local Qwen2.5-1.5B) "
                             "→ rule_based (always available fallback)."),
    ], s))
    story.append(Spacer(1, 0.4*cm))
    story += code_block([
        "# How the pipeline reads config",
        "from utils import load_config",
        "config = load_config()           # reads config.yaml from project root",
        "",
        "# Each module receives only its sub-section:",
        "vad     = VADModule(cfg=config.get('vad', {}))",
        "pre     = AudioPreprocessor(cfg=config.get('preprocessing', {}))",
        "chunker = AudioChunker(cfg=config.get('chunking', {}))",
        "asr     = ASRModule(model_path=..., cfg=config.get('asr', {}))",
    ], s, "Config usage pattern")
    story.append(PageBreak())
    return story


def build_vad(s):
    story = []
    story.append(module_header_table(
        "Module 1: Voice Activity Detection",
        "src/vad_module.py — class VADModule",
        "Strips silence from the input audio before any model processing. This serves two purposes: "
        "it reduces ASR processing time by removing non-speech segments, and it prevents Whisper "
        "from hallucinating text on quiet/noise-only segments. The module uses Silero-VAD, a "
        "lightweight LSTM-based model that achieves near-perfect speech detection in under 50 MB.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Design Decisions", s["h2"]))
    story.append(kv_table([
        ("Model Choice",      "Silero-VAD was chosen over WebRTC-VAD because it is neural-based and "
                              "handles radio noise, accents, and low-volume speech far better. It uses "
                              "a small LSTM model (~2 MB)."),
        ("speech_pad_ms=100", "100 ms of padding is added before and after each detected speech segment. "
                              "Without this, the first/last phoneme of a word is often clipped, which "
                              "raises the WER significantly."),
        ("Segment Merging",   "After padding, overlapping segments are merged to avoid duplicate output. "
                              "The merge loop walks the padded list and extends the current segment if "
                              "the next one starts before the current one ends."),
        ("Return Value",      "Returns segments_seconds (list of {start_sec, end_sec} dicts). These "
                              "timestamps are passed directly to the AudioChunker in Stage 3 to ensure "
                              "chunk boundaries align with real speech boundaries."),
    ], s))

    story.append(Paragraph("Key Method: remove_silence()", s["h2"]))
    story += code_block([
        "def remove_silence(self, audio_path: str, output_path: str) -> dict:",
        "    wav = read_audio(audio_path, sampling_rate=16000)",
        "",
        "    # 1. Detect speech timestamps with Silero-VAD",
        "    raw_segments = get_speech_timestamps(",
        "        wav, self.model,",
        "        sampling_rate=16000,",
        "        threshold=0.45,           # probability cutoff",
        "        min_speech_duration_ms=250,",
        "        min_silence_duration_ms=600,",
        "    )",
        "",
        "    # 2. Add 100ms padding around each segment",
        "    pad = int(100 * 16000 / 1000)   # = 1600 samples",
        "    padded = [{'start': max(0, s['start']-pad),",
        "               'end': min(total-1, s['end']+pad)} for s in raw_segments]",
        "",
        "    # 3. Merge overlapping segments after padding",
        "    merged = [padded[0].copy()]",
        "    for seg in padded[1:]:",
        "        if seg['start'] <= merged[-1]['end']:",
        "            merged[-1]['end'] = max(merged[-1]['end'], seg['end'])",
        "        else:",
        "            merged.append(seg.copy())",
        "",
        "    # 4. Concatenate speech chunks and write to disk",
        "    speech_audio = torch.cat([wav[s['start']:s['end']] for s in merged])",
        "    sf.write(output_path, speech_audio.numpy(), 16000)",
        "",
        "    # 5. Return second-based timestamps for downstream use",
        "    return {'segments_seconds': [...], 'total_speech_sec': float}",
    ], s, "vad_module.py — remove_silence()")

    story.append(PageBreak())
    return story


def build_preprocessing(s):
    story = []
    story.append(module_header_table(
        "Module 2: Audio Preprocessing",
        "src/preprocessing.py — class AudioPreprocessor",
        "Cleans the VAD-stripped audio through a 5-step signal processing chain: pre-emphasis, "
        "bandpass filtering, stationary noise reduction, RMS normalisation, and silence trimming. "
        "The chain is specifically tuned for radio telephony audio, not studio speech.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Processing Chain", s["h2"]))
    story.append(kv_table([
        ("Step 1: Pre-emphasis",
         "Applies a first-order high-pass filter: audio[n] = audio[n] - 0.97×audio[n-1]. "
         "Radio transmission compresses high frequencies; pre-emphasis partially reverses this, "
         "improving ASR accuracy on fricatives (s, sh, f) and aspirated consonants common in Indic languages."),
        ("Step 2: Bandpass Filter",
         "A 4th-order Butterworth filter restricts audio to 300–3400 Hz — the standard "
         "telephone/radio voice band defined in ITU-T G.711. This removes sub-bass rumble "
         "(equipment vibration, wind) and high-frequency hiss above the voice range."),
        ("Step 3: Noise Reduction",
         "Uses the noisereduce library with stationary=True. In stationary mode, the algorithm "
         "estimates a noise profile from the entire clip (assuming the carrier/static noise is "
         "constant). prop_decrease=0.75 removes 75% of the estimated noise energy. "
         "Non-stationary mode (per-frame adaptation) can over-suppress quiet speech."),
        ("Step 4: RMS Normalisation",
         "Scales amplitude to a target RMS of 0.1 (~-20 dBFS). Peak normalisation is avoided "
         "because transient peaks (clicks, PTT pops) would cause the speech to become very quiet."),
        ("Step 5: Trim",
         "librosa.effects.trim() removes leading/trailing silence below 20 dB."),
        ("SNR Measurement",
         "Before and after SNR is estimated by computing per-frame energy, then comparing the "
         "80th percentile (signal) to the 20th percentile (noise). This gives a quick quality "
         "indicator without needing a reference clean signal."),
    ], s))
    story += code_block([
        "# Step 2 — Butterworth bandpass filter (scipy)",
        "def _bandpass_filter(audio, sr, low_hz=300, high_hz=3400, order=4):",
        "    nyq  = sr / 2.0                       # Nyquist = 8000 Hz at 16kHz",
        "    low  = 300  / 8000                    # = 0.0375 (normalised frequency)",
        "    high = 3400 / 8000                    # = 0.425",
        "    sos  = butter(order, [low, high], btype='band', output='sos')",
        "    return sosfilt(sos, audio).astype(np.float32)",
        "",
        "# Step 4 — RMS normalisation",
        "rms = np.sqrt(np.mean(audio ** 2))        # root mean square energy",
        "if rms > 1e-6:                            # guard against silent audio",
        "    target_rms = 0.1                      # -20 dBFS target level",
        "    audio = audio * (target_rms / rms)",
        "audio = np.clip(audio, -1.0, 1.0)         # hard clip to prevent overflow",
    ], s, "preprocessing.py — key implementation snippets")

    story.append(PageBreak())
    return story


def build_chunker(s):
    story = []
    story.append(module_header_table(
        "Module 3: VAD-Aware Audio Chunking",
        "src/chunker.py — class AudioChunker",
        "Splits the preprocessed audio into segments for Whisper. Whisper was trained on 30-second "
        "windows; feeding audio longer than 30 seconds causes the model to truncate or hallucinate. "
        "The chunker respects Whisper's window by using the VAD segment timestamps from Stage 1 to "
        "split at natural speech boundaries rather than arbitrary time points.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Chunking Strategy", s["h2"]))
    story.append(kv_table([
        ("max_chunk_duration=29s",
         "Keeps each chunk safely under Whisper's 30 s training window. Using 29 s (not 30) "
         "adds a 1-second safety margin for floating-point rounding."),
        ("VAD-aware grouping",
         "Instead of blindly splitting at 29 s, the chunker greedily packs adjacent VAD "
         "segments into groups. A group is closed when adding the next segment would push the "
         "total past 29 s. This ensures cuts never fall mid-word."),
        ("Fixed fallback",
         "If no VAD segments are provided (e.g., VAD was disabled), the chunker falls back "
         "to fixed-size 29 s windows."),
        ("Timestamp preservation",
         "Each chunk dict stores start_sec and end_sec in the original audio timeline. "
         "The ASR module adds these offsets to all segment timestamps so the final output "
         "always references absolute time in the original file."),
    ], s))
    story += code_block([
        "def _group_vad_segments(self, vad_segs, total_dur):",
        "    # Greedy packing: extend current group until max_duration exceeded",
        "    groups = []",
        "    group_start = group_end = None",
        "",
        "    for seg in vad_segs:",
        "        s, e = seg['start_sec'], seg['end_sec']",
        "        if group_start is None:",
        "            group_start, group_end = s, e",
        "            continue",
        "        if (e - group_start) > self.max_duration:  # would exceed 29s",
        "            groups.append((group_start, group_end))",
        "            group_start, group_end = s, e          # start new group",
        "        else:",
        "            group_end = e                          # extend current group",
        "",
        "    if group_start is not None:",
        "        groups.append((group_start, group_end))",
        "    return groups",
    ], s, "chunker.py — VAD-aware grouping algorithm")

    story.append(PageBreak())
    return story


def build_asr(s):
    story = []
    story.append(module_header_table(
        "Module 4: Automatic Speech Recognition",
        "src/asr_module.py — class ASRModule",
        "Wraps faster-whisper (CTranslate2-based Whisper) with configuration specifically tuned "
        "for multilingual, noisy audio. The key contribution here is not the ASR model itself "
        "(Whisper large-v3-turbo is an off-the-shelf model) but the parameter choices that prevent "
        "hallucination, improve speed, and produce per-word timestamps.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Why faster-whisper (CTranslate2)?", s["h2"]))
    story.append(Paragraph(
        "The original whisper Python package loads the model in float32, requiring ~3.5 GB RAM for "
        "large-v3-turbo. CTranslate2 int8 quantisation reduces this to ~1.9 GB — a difference that "
        "determines whether the full pipeline fits in 8 GB. The speed improvement is also significant: "
        "int8 on CPU is 2–3× faster than float32 for the same model.",
        s["body"],
    ))

    story.append(Paragraph("Critical Parameter Choices", s["h2"]))
    story.append(kv_table([
        ("beam_size=2, best_of=1",
         "Beam search width of 2 reduces memory from ~4× (beam_size=5 default) to ~1.2× base. "
         "Accuracy loss is minimal for clean speech; the main impact is on very noisy inputs."),
        ("temperature=0.0",
         "Forces greedy/deterministic decoding. Non-zero temperature adds stochastic sampling, "
         "which occasionally produces better outputs on ambiguous audio but is inconsistent."),
        ("condition_on_previous_text=False",
         "This is the most important setting for radio audio. With conditioning enabled, "
         "Whisper feeds its own previous output as a prompt for the next segment. On radio "
         "static or silence, this creates infinite repetition loops (the model keeps predicting "
         "the same phrase). Disabling it makes each segment independent."),
        ("no_speech_threshold=0.70",
         "Whisper assigns a no_speech_prob to each segment. Above 0.70, the segment is "
         "assumed to be noise/silence and discarded. This was raised from 0.60 to 0.70 "
         "because quiet Punjabi and Indic speech was occasionally being filtered out."),
        ("word_timestamps=True",
         "Forces Whisper to output per-word timestamps from cross-attention alignment. "
         "Required for the keyword detection stage which needs to know when a keyword was "
         "spoken, and for the ASR confidence heatmap in the UI."),
        ("Language hint caching",
         "The language is detected on the first chunk and then passed as language_hint to "
         "all subsequent chunks. This avoids the 0.5 s per-chunk language detection overhead "
         "and ensures consistent language assignment across a multi-chunk recording."),
    ], s))

    story.append(Paragraph("Hallucination Filter", s["h2"]))
    story.append(Paragraph(
        "Whisper is known to hallucinate common English phrases on silence or background music. "
        "A hardcoded set of known hallucination patterns is checked for each segment:",
        s["body"],
    ))
    story += code_block([
        "_HALLUCINATION_PHRASES = frozenset([",
        "    'thank you for watching', 'please subscribe',",
        "    '[music]', '[ music ]', '[applause]',",
        "    'subtitles by', 'transcribed by',",
        "    'www.', '.com', '.org', ...",
        "])",
        "",
        "# Also detects repetition loops: same word ≥5 times in a row",
        "words = text.lower().split()",
        "if len(words) >= 5:",
        "    for i in range(len(words) - 4):",
        "        if len(set(words[i:i+5])) == 1:   # all 5 identical",
        "            return True   # is_hallucination",
    ], s, "asr_module.py — hallucination detection")

    story.append(Paragraph("Confidence Score Formula", s["h2"]))
    story.append(Paragraph(
        "Whisper outputs avg_logprob (average log probability) per segment, which ranges from "
        "roughly -5 to 0. This is converted to a 0–1 confidence score:",
        s["body"],
    ))
    story += code_block([
        "# avg_logprob is negative; -4.0 → conf=0.0, 0.0 → conf=1.0",
        "confidence = min(1.0, max(0.0, 1.0 + seg.avg_logprob / 4.0))",
        "",
        "# Example:",
        "#   avg_logprob = -0.5  →  confidence = 1.0 + (-0.5/4.0) = 0.875",
        "#   avg_logprob = -2.0  →  confidence = 1.0 + (-2.0/4.0) = 0.500",
        "#   avg_logprob = -4.0  →  confidence = 1.0 + (-4.0/4.0) = 0.000",
    ], s, "asr_module.py — confidence formula")

    story.append(PageBreak())
    return story


def build_diarize(s):
    story = []
    story.append(module_header_table(
        "Module 5: Speaker Diarisation",
        "src/diarize_module.py — function diarize()",
        "Assigns speaker labels (SPEAKER_A, SPEAKER_B, ...) to Whisper segments before language "
        "routing. This enables per-speaker transcript export and multi-party analysis without "
        "requiring any pre-trained speaker embedding model. The entire module uses only librosa "
        "and scikit-learn — libraries already in the project requirements.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Why Avoid Pre-trained Speaker Embeddings?", s["h2"]))
    story.append(Paragraph(
        "State-of-the-art speaker diarisation systems (pyannote, NeMo) use x-vector or ECAPA "
        "speaker embeddings that require 100–300 MB model files, GPU for real-time use, and "
        "sometimes online model downloads. This conflicts with the 8 GB RAM / offline constraint. "
        "MFCC-based clustering is the offline-friendly alternative: no pre-trained encoder, "
        "no GPU, under 10 MB memory footprint.",
        s["body"],
    ))

    story.append(Paragraph("Algorithm Step-by-Step", s["h2"]))
    story.append(kv_table([
        ("Step 1: MFCC Embedding",
         "For each Whisper segment, 40 MFCC coefficients + delta + delta-delta are computed "
         "using librosa. These 3 × 40 = 120 coefficient streams are flattened by mean and std "
         "to produce a 240-dimensional speaker embedding vector."),
        ("Step 2: Normalisation",
         "All embedding vectors are L2-normalised (sklearn.preprocessing.normalize). "
         "This ensures cosine distance = 1 - dot product, making the clustering metric consistent."),
        ("Step 3: Speaker Count Detection",
         "_pick_n_speakers() runs Agglomerative Clustering for k=2, 3, ..., max_speakers "
         "and selects k by maximising the silhouette score (a measure of cluster separation). "
         "Silhouette score > 0.5 indicates well-separated clusters."),
        ("Step 4: Cluster Assignment",
         "AgglomerativeClustering with cosine distance assigns each segment to a speaker cluster. "
         "Cluster indices are mapped to SPEAKER_A, SPEAKER_B, etc."),
        ("Step 5: Short Segment Fallback",
         "Segments shorter than 0.5 s produce unreliable MFCC embeddings. These inherit "
         "the speaker label of the nearest longer segment (left-first search)."),
    ], s))

    story += code_block([
        "def _mfcc_embedding(audio, sr):",
        "    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)",
        "    delta  = librosa.feature.delta(mfcc)       # velocity",
        "    delta2 = librosa.feature.delta(mfcc, order=2)  # acceleration",
        "    feats  = np.vstack([mfcc, delta, delta2])  # shape: (120, T)",
        "    # Summarise over time by mean + std → fixed-size vector",
        "    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)])  # (240,)",
        "",
        "def _pick_n_speakers(X, max_n):",
        "    best_score, best_n = -1.0, 2",
        "    for n in range(2, min(max_n+1, len(X))):",
        "        clust  = AgglomerativeClustering(n_clusters=n,",
        "                     metric='cosine', linkage='average')",
        "        labels = clust.fit_predict(X)",
        "        score  = silhouette_score(X, labels, metric='cosine')",
        "        if score > best_score:",
        "            best_score, best_n = score, n",
        "    return best_n",
    ], s, "diarize_module.py — embedding and speaker count selection")

    story.append(PageBreak())
    return story


def build_langid(s):
    story = []
    story.append(module_header_table(
        "Module 6: Language Identification",
        "src/language_module.py — FastTextLangDetector, DialectDetector, LanguageRouter",
        "The most technically novel module. Rather than trusting a single ASR language prediction, "
        "it fuses three independent sources in a confidence-weighted vote, with hard Unicode script "
        "overrides (the Script-Cascade Algorithm) that correct systematic Whisper failures on "
        "Punjabi and Urdu.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Three Source Architecture", s["h2"]))
    story.append(kv_table([
        ("Source 1: Whisper language head",
         "Whisper's encoder outputs a language probability distribution as part of its "
         "transcription. This is read from info.language and info.language_probability. "
         "Weakness: biased towards languages with more training data (Hindi >> Punjabi)."),
        ("Source 2: FastText lid.176.bin",
         "Text-based language ID using character n-gram features. Operates on the Whisper "
         "transcript. Achieves >97% on balanced benchmarks but is unreliable when Whisper "
         "transcribes the wrong script (e.g., Devanagari output for Punjabi audio)."),
        ("Source 3: MMS-LID-256",
         "Audio-based language ID from Meta's Massively Multilingual Speech project. "
         "Processes raw waveforms via wav2vec features, completely independent of the transcript. "
         "This is the only source that is immune to transcript script errors, making it the "
         "critical differentiator for Punjabi/Hindi confusion."),
    ], s))

    story.append(Paragraph("Script-Cascade Algorithm (Novel Contribution)", s["h2"]))
    story.append(Paragraph(
        "The DialectDetector class computes Unicode block ratios for the ASR transcript. "
        "The LanguageRouter then applies a two-tier cascade based on these ratios:",
        s["body"],
    ))
    story.append(kv_table([
        ("Tier 1: Gurmukhi (Force Override)",
         "If Gurmukhi character ratio > 20%, language is forced to Punjabi (pa) regardless "
         "of all probabilistic sources. Gurmukhi is exclusive to Punjabi in South Asian usage, "
         "making this a safe hard rule. Confidence floor = 0.75."),
        ("Tier 2: Arabic/Nastaliq (Filter-then-Vote)",
         "If Arabic character ratio > 20%, the candidate pool is restricted to "
         "{ur, ks, sd, ps, fa, ar} before voting. This prevents Whisper's 'hi' prediction "
         "from winning the vote for Urdu audio. If no source votes within this set "
         "(FastText returns 'unknown' for Nastaliq ~90% of the time), the system defaults to 'ur'."),
    ], s))

    story += code_block([
        "# DialectDetector: count Unicode code points per script",
        "for ch in text:",
        "    cp = ord(ch)",
        "    if 0x0900 <= cp <= 0x097F: devanagari += 1   # Hindi/Nepali",
        "    elif 0x0A00 <= cp <= 0x0A7F: gurmukhi += 1   # Punjabi",
        "    elif 0x0600 <= cp <= 0x06FF: arabic   += 1   # Urdu/Pashto/Kashmiri",
        "    elif 0x4E00 <= cp <= 0x9FFF: chinese  += 1",
        "",
        "# LanguageRouter: Script-Cascade (Tier 1 — Gurmukhi)",
        "if d == 'gurmukhi_indic':    # dialect from DialectDetector",
        "    return self._make('pa', 'nllb', max(pa_conf, 0.75),",
        "                      'indic', False, 'gurmukhi-script override')",
        "",
        "# Script-Cascade (Tier 2 — Arabic filter-then-vote)",
        "if d == 'arabic_script':",
        "    ar_cands = [(lang, conf, src) for lang, conf, src in candidates",
        "                if lang in {'ur','ks','sd','ps','fa','ar'}]",
        "    if ar_cands:",
        "        best_al = Counter(lang for lang,_,_ in ar_cands).most_common(1)[0][0]",
        "        return self._make(best_al, 'nllb', max(conf_al, 0.70), ...)",
        "    return self._make('ur', 'nllb', 0.65, 'arabic', True,",
        "                      'arabic-script default->ur')",
    ], s, "language_module.py — Script-Cascade Algorithm")

    story.append(Paragraph("Voting Logic", s["h2"]))
    story += code_block([
        "# Build candidate list from available sources",
        "candidates = []",
        "if wl not in ('', 'unknown'): candidates.append((wl, whisper_prob, 'whisper'))",
        "if fl not in ('', 'unknown'): candidates.append((fl, ft_conf, 'fasttext'))",
        "if ml not in ('', 'unknown'): candidates.append((ml, mms_conf, 'mms'))",
        "",
        "vote_counts = Counter(lang for lang, _, _ in candidates)",
        "majority_lang, majority_votes = vote_counts.most_common(1)[0]",
        "",
        "if majority_votes == len(candidates):     # Unanimous (3/3 or 2/2)",
        "    conf = min(0.99, avg_conf * 1.10)     # +10% bonus for agreement",
        "elif majority_votes >= 2:                 # Majority (2/3)",
        "    conf = avg of agreeing sources",
        "else:                                     # All disagree",
        "    best = max(candidates, key=lambda x: x[1])  # highest single confidence",
        "    conf = best_conf * 0.85               # 15% penalty for no consensus",
    ], s, "language_module.py — voting logic summary")

    story.append(Paragraph("Language Routing Sets", s["h2"]))
    story += code_block([
        "INDIC_LANGS  = {'doi'}   # Dogri only — not in NLLB-200 vocab → IndicTrans2",
        "NLLB_LANGS   = {'hi','pa','ur','ne','bn','mai','ks','sd','si',",
        "                'ps','zh','my','bo','fa','ar','tg','uz','kk'}",
        "ENGLISH_LIKE = {'en'}    # No translation needed",
        "",
        "# Result structure from detect_family():",
        "# {'final_language': 'pa', 'route': 'nllb', 'confidence': 0.993,",
        "#  'script_hint': 'indic', 'uncertain': False,",
        "#  'vote_note': 'gurmukhi-script override'}",
    ], s, "language_module.py — routing constants")

    story.append(PageBreak())
    return story


def build_translation(s):
    story = []
    story.append(module_header_table(
        "Module 7: Translation",
        "src/translation_module.py — class TranslationModule",
        "Routes the transcript to one of two translation models depending on the detected language: "
        "NLLB-200-distilled-600M for all supported languages, or IndicTrans2 for Dogri (the one "
        "Indic language absent from NLLB-200's distilled vocabulary). Models are loaded on demand "
        "and explicitly unloaded after each translation to keep memory usage below 8 GB.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Model Selection Rationale", s["h2"]))
    story.append(kv_table([
        ("NLLB-200-distilled-600M",
         "Primary translation engine. Covers 200 languages including all target languages except "
         "Dogri. The distilled 600M variant fits in ~2.4 GB RAM and achieves +44% BLEU over "
         "previous low-resource translation systems (Meta 2022). Target is always 'eng_Latn'."),
        ("IndicTrans2",
         "Used exclusively for Dogri (ISO code: doi). IndicTrans2 covers all 22 constitutionally "
         "scheduled Indian languages including Dogri. However, it has a transformers>=5.x "
         "DynamicCache incompatibility, so all other Indic languages that work fine in NLLB "
         "are routed through NLLB to avoid the bug."),
        ("Memory management",
         "unload_after_use=True (default) deletes the model and calls gc.collect() + "
         "torch.cuda.empty_cache() / torch.mps.empty_cache() after every translation. "
         "This prevents the 2.4 GB NLLB model from coexisting in memory with the next stage's model."),
    ], s))

    story.append(Paragraph("Long Text Chunking", s["h2"]))
    story.append(Paragraph(
        "NLLB-200 has a 256-token input limit. Transcripts longer than this are split into "
        "sentence-level chunks that each fit within the limit, translated individually, and "
        "then joined. The split is at sentence boundaries (. ! ? ।) to avoid cutting mid-clause.",
        s["body"],
    ))
    story += code_block([
        "token_ids = tokenizer.encode(text, add_special_tokens=True)",
        "if len(token_ids) <= 256:                   # fits in one call",
        "    return self._nllb_translate_chunk(text, src_code)",
        "",
        "# Long text: split into sentence chunks",
        "sentences = re.split(r'(?<=[.!?।])\\s+|\\n+', text)  # split on Devanagari danda too",
        "chunks = []",
        "current, current_len = [], 0",
        "for sent in sentences:",
        "    sent_len = len(tokenizer.encode(sent))",
        "    if current_len + sent_len > 246 and current:  # 10 token margin",
        "        chunks.append(' '.join(current))",
        "        current, current_len = [sent], sent_len",
        "    else:",
        "        current.append(sent)",
        "        current_len += sent_len",
        "return ' '.join(translate_chunk(c) for c in chunks)",
    ], s, "translation_module.py — sentence chunking for long inputs")

    story.append(Paragraph("IndicTrans2 Compatibility Fixes", s["h2"]))
    story.append(callout(
        "<b>Technical note for professor:</b> IndicTrans2 uses a custom tokeniser that requires "
        "explicit mode switching between encoding (input) and decoding (output). The code calls "
        "tokenizer._switch_to_input_mode() before encoding and tokenizer._switch_to_target_mode() "
        "before decoding. Without this, the wrong SentencePiece model is used and the output is "
        "garbage. Additionally, use_cache=False and no_repeat_ngram_size=3 are required to work "
        "around a DynamicCache incompatibility in transformers>=5.",
        s,
    ))

    story.append(PageBreak())
    return story


def build_keywords(s):
    story = []
    story.append(module_header_table(
        "Module 8: Keyword Detection",
        "src/keyword_module.py — class KeywordDetector",
        "Scans both the original-language transcript and the English translation for domain-specific "
        "keywords organised into 8 categories with severity weights. Every match is linked to its "
        "source segment with timestamps, enabling audio playback to the exact moment a keyword "
        "was spoken.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Keyword Categories and Severity", s["h2"]))
    cat_data = [
        ["Category",        "Severity",  "Example Keywords"],
        ["enemy_activity",  "CRITICAL",  "enemy, hostile, dushman, atankwadi, insurgent"],
        ["attack",          "CRITICAL",  "attack, firing, ambush, mortar, IED, hamla, goli"],
        ["weapons",         "HIGH",      "rifle, RPG, explosive, bandook, hathiyar"],
        ["movement",        "HIGH",      "advance, retreat, crossing, flanking, aage badho"],
        ["location",        "HIGH",      "sector, grid, border, LOC, LAC, ridge, seema"],
        ["support_request", "HIGH",      "casualty, medevac, extraction, madad, zakhmi"],
        ["command",         "MEDIUM",    "orders, roger, wilco, abort, stand by, ruko"],
        ["comms",           "LOW",       "callsign, frequency, channel, say again"],
    ]
    t = Table(cat_data, colWidths=[3.8*cm, 2.2*cm, 9.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEAD_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEAD_FG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("BACKGROUND", (1, 1), (1, 2), colors.HexColor("#ffebee")),  # CRITICAL rows
        ("BACKGROUND", (1, 3), (1, 6), colors.HexColor("#fff8e1")),  # HIGH rows
        ("BACKGROUND", (1, 7), (1, 7), colors.HexColor("#f3e5f5")),  # MEDIUM
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Confidence-Adjusted Severity", s["h2"]))
    story.append(Paragraph(
        "Each keyword alert inherits the ASR segment confidence of the segment it was found in. "
        "If segment confidence < 0.40, the severity is downgraded one level (critical→high, "
        "high→medium, etc.). This prevents low-confidence ASR output from triggering CRITICAL "
        "alerts on words that may have been hallucinated.",
        s["body"],
    ))
    story += code_block([
        "def _downgrade_severity(severity: str, confidence: float) -> str:",
        "    if confidence >= 0.40:          # high confidence → keep as-is",
        "        return severity",
        "    _order = ['low', 'medium', 'high', 'critical']",
        "    idx = _order.index(severity)",
        "    return _order[idx - 1] if idx > 0 else severity  # drop one level",
        "",
        "# Threat level = highest effective_severity across all alerts",
        "def _compute_threat_level(alerts):",
        "    if not alerts: return 'CLEAR'",
        "    SEVERITY_ORDER = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}",
        "    max_sev = max(SEVERITY_ORDER.get(a.effective_severity, 0) for a in alerts)",
        "    return {4:'CRITICAL', 3:'HIGH', 2:'MEDIUM', 1:'LOW'}.get(max_sev, 'CLEAR')",
    ], s, "keyword_module.py — confidence-adjusted severity")

    story.append(Paragraph("Dual-Language Scanning", s["h2"]))
    story.append(Paragraph(
        "Keyword matching is run on both the original-language transcript AND the English translation. "
        "This catches cases where a domain term appears in the source language (e.g., 'dushman' in "
        "Hindi for 'enemy') but was not caught by English-only matching. The alert records "
        "which source (transcript vs translation) triggered it.",
        s["body"],
    ))
    story.append(PageBreak())
    return story


def build_isum(s):
    story = []
    story.append(module_header_table(
        "Module 9: Structured Summarisation (SSUM)",
        "src/isum_module.py — class ISUMGenerator",
        "Converts the pipeline output into a structured 5W report (Who, What, Where, When, "
        "Assessment) with severity classification and quality flags. Operates in three modes: "
        "Ollama (gemma3:4b via REST), Qwen2.5-1.5B-Instruct (local LLM), or rule-based "
        "(always available). The rule-based layer always runs; LLMs override specific fields.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Three-Tier Architecture", s["h2"]))
    story.append(kv_table([
        ("Tier 1: Rule-Based (Always Present)",
         "Uses regular expressions to extract named entities, callsigns, unit designators, "
         "grid references, time expressions, and direction words. Also uses spaCy NER "
         "(en_core_web_sm) for PERSON, ORG, GPE, and LOC entities from the English translation. "
         "Produces a guaranteed structured output even on offline CPU-only hardware."),
        ("Tier 2: Qwen2.5-1.5B (Optional Local LLM)",
         "If the model is present, the same 5W extraction prompt is sent to the quantised "
         "(int4) Qwen2.5-1.5B-Instruct model. The LLM output overrides the rule-based "
         "fields where it provides non-empty values. The model is loaded, used, then deleted "
         "with explicit GPU memory clearing to prevent OOM."),
        ("Tier 3: Ollama (Optional REST API)",
         "If ollama is running locally (gemma3:4b or any configured model), the system sends "
         "a structured JSON extraction prompt via HTTP POST to http://localhost:11434. This "
         "provides the best quality output and uses minimal Python-side memory."),
    ], s))

    story.append(Paragraph("5W Extraction — Who Field", s["h2"]))
    story.append(Paragraph(
        "The _extract_who() method illustrates the depth of the rule-based system. It searches "
        "for seven distinct actor types:",
        s["body"],
    ))
    story += code_block([
        "# 1. NATO phonetic callsigns: Alpha-3, Bravo-2, ...",
        "callsigns = re.findall(",
        "    r'\\b(?:alpha|bravo|charlie|delta|...)\\s*\\d*\\b', text, re.IGNORECASE)",
        "",
        "# 2. Alphanumeric unit designators: TF-7, SF-3, OP-4",
        "num_callsigns = re.findall(r'\\b[A-Z]{1,3}[-\\s]\\d{1,3}\\b', text)",
        "",
        "# 3. Formation names: platoon, company, battalion, task force, QRF",
        "units = re.findall(",
        "    r'\\b(?:unit|team|squad|platoon|company|battalion|task force|QRF)\\s+[\\w\\d-]+',",
        "    text, re.IGNORECASE)",
        "",
        "# 4. Military ranks (English + South Asian transliterations)",
        "ranks = re.findall(",
        "    r'\\b(?:captain|major|colonel|havildar|subedar|naik|sepoy|jawan)\\b',",
        "    text, re.IGNORECASE)",
        "",
        "# 5. PLA (Chinese military) unit codes: '第72集团军', 'Unit 61398'",
        "pla_codes = re.findall(r'\\bUnit\\s+\\d{5}\\b', text, re.IGNORECASE)",
        "zh_units  = re.findall(r'第\\d+\\s*(?:集团军|战区|军区|师|旅|团)', zh_text)",
        "",
        "# 6. spaCy NER: PERSON and ORG entities from English translation",
        "doc = nlp(translation[:1000])",
        "persons = [e.text for e in doc.ents if e.label_ == 'PERSON']",
        "orgs    = [e.text for e in doc.ents if e.label_ == 'ORG']",
        "",
        "# 7. Friendly/hostile force indicators from pronouns",
        "if re.search(r'\\b(?:we are|our forces|our position)\\b', text, re.I):",
        "    actors.append('Friendly forces (self-referenced)')",
    ], s, "isum_module.py — Who field extraction rules")

    story.append(Paragraph("Quality Flags System", s["h2"]))
    story += code_block([
        "def _quality_flags(self, r, uncertain, lang_conf):",
        "    flags = []",
        "    if uncertain:                          # LangID confidence < 0.60",
        "        flags.append('LOW_LANG_CONFIDENCE')",
        "    if lang_conf < 0.50:                   # Translation reliability below threshold",
        "        flags.append('TRANSLATION_UNRELIABLE')",
        "    if r.get('whisper_language_probability', 1.0) < 0.60:",
        "        flags.append('ASR_LOW_CONFIDENCE')",
        "    avg_conf = mean(seg['confidence'] for seg in r['segments'])",
        "    if avg_conf < 0.55:                    # Overall transcript quality low",
        "        flags.append('TRANSCRIPTION_LOW_CONFIDENCE')",
        "    if not r.get('translation', {}).get('success', True):",
        "        flags.append('TRANSLATION_FAILED')",
        "    return flags",
    ], s, "isum_module.py — quality flags")

    story.append(PageBreak())
    return story


def build_database(s):
    story = []
    story.append(module_header_table(
        "Module 10: Database Layer",
        "src/database.py — class TranscriptDB",
        "Stores all pipeline results in a SQLite database with 6 tables, full-text search via "
        "FTS5, actor profile tracking, cross-intercept correlation, and annotation export. "
        "SQLite was chosen over PostgreSQL/MySQL because it requires zero server setup, stores "
        "everything in a single file, and handles the expected record volumes (thousands, not millions) "
        "with sub-millisecond query times.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Database Schema", s["h2"]))
    schema_data = [
        ["Table",           "Primary Key",   "Purpose"],
        ["intercepts",      "id (AUTOINCREMENT)", "One row per audio file. Stores all LangID, ASR, translation metadata."],
        ["segments",        "id",            "Per-segment ASR output with timestamps and confidence scores."],
        ["keyword_alerts",  "id",            "All keyword matches with category, severity, timestamps."],
        ["isums",           "id",            "5W structured summary fields linked to intercept."],
        ["annotations",     "id",            "Human corrections to ASR/translation/ISUM for training data export."],
        ["metrics",         "id",            "Auto-computed quality metrics (RTF, confidence, ensemble score) per run."],
        ["intercepts_fts",  "rowid (virtual)", "FTS5 virtual table indexing transcript + translation + ISUM text."],
    ]
    t = Table(schema_data, colWidths=[3.5*cm, 4.0*cm, 8.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEAD_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEAD_FG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("FTS5 Full-Text Search", s["h2"]))
    story.append(Paragraph(
        "SQLite FTS5 is used for full-text search across transcript + translation + all ISUM fields. "
        "The FTS5 virtual table is kept in sync with the main tables via INSERT/DELETE triggers in "
        "the save_result() method. FTS5 query syntax allows phrase matching (\"alpha bravo\") and "
        "is 10–100× faster than LIKE-based search on large datasets.",
        s["body"],
    ))
    story += code_block([
        "# FTS5 virtual table definition",
        "CREATE VIRTUAL TABLE intercepts_fts USING fts5(",
        "    report_id   UNINDEXED,    -- not searchable, just an identifier",
        "    transcript,               -- original language text",
        "    translation,              -- English translation",
        "    isum_text,                -- concatenated 5W fields",
        "    tokenize = 'unicode61 remove_diacritics 2'",
        "                              -- removes accent marks for transliteration matching",
        ");",
        "",
        "# Search query: wrap each word in quotes for exact term matching",
        "fts_query = '\"enemy\" \"north\"'   # FTS5 implicit AND",
        "cursor.execute(",
        "    'SELECT * FROM intercepts_fts JOIN intercepts i ON i.id=f.rowid",
        "     WHERE intercepts_fts MATCH ?', [fts_query])",
    ], s, "database.py — FTS5 setup and query")

    story.append(Paragraph("Cross-Intercept Correlation", s["h2"]))
    story.append(Paragraph(
        "get_related_intercepts() scores all other database records against a given report "
        "across four dimensions. This enables the ISUM tab to show related recordings automatically:",
        s["body"],
    ))
    story.append(kv_table([
        ("Shared keywords",   "Each shared high/critical keyword = +3 points, capped at +12"),
        ("Shared actors",     "Each shared actor token from who_field = +4 points, capped at +16"),
        ("Language + threat", "Same language = +1; same non-trivial threat level = +2"),
        ("Time proximity",    "Within 24h = +3; within 3 days = +2; within 7 days = +1"),
    ], s))

    story.append(Paragraph("Actor Profile Tracking", s["h2"]))
    story.append(Paragraph(
        "get_actor_profiles() parses who_field strings from all SSUM records and builds a "
        "per-entity profile: appearance count, first/last seen timestamps, languages, and "
        "highest threat level associated with that entity. Results are cached for 5 minutes "
        "to avoid re-parsing on every UI refresh.",
        s["body"],
    ))
    story.append(PageBreak())
    return story


def build_pipeline(s):
    story = []
    story.append(module_header_table(
        "Pipeline Orchestrator",
        "src/pipeline.py — function run_pipeline()",
        "The central function that calls all 10 stages in order, manages stage timing, "
        "tracks peak memory, caches MMS-LID results between Stage 3.5 and Stage 5, "
        "and assembles the final result dict. It exposes a progress_cb parameter so "
        "the Streamlit UI can display live stage progress updates.",
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Memory Management Strategy", s["h2"]))
    story.append(Paragraph(
        "The 8 GB ceiling is managed through sequential model loading: only one model is in "
        "memory at a time. After each stage, the module object is deleted and free_memory() "
        "is called (gc.collect() + torch cache clear). The pipeline tracks peak RAM via psutil.",
        s["body"],
    ))
    story += code_block([
        "# Pattern used after every stage that loads a model",
        "del vad;  free_memory(logger)    # Stage 1: ~0.2 GB freed",
        "del pre;  free_memory(logger)    # Stage 2: ~0.1 GB freed",
        "del asr;  free_memory(logger)    # Stage 4: ~1.9 GB freed",
        "del translator; free_memory(logger)  # Stage 6: ~2.4 GB freed",
        "",
        "def free_memory(logger=None):",
        "    import gc, torch",
        "    gc.collect()",
        "    try: torch.cuda.empty_cache()",
        "    except: pass",
        "    try: torch.mps.empty_cache()   # Apple Silicon",
        "    except: pass",
    ], s, "pipeline.py — sequential memory management")

    story.append(Paragraph("MMS-LID Result Caching (Stage 3.5 → Stage 5)", s["h2"]))
    story.append(Paragraph(
        "MMS-LID (150 MB model) is needed in two places: Stage 3.5 for ASR model selection, "
        "and Stage 5 for the LangID vote. Loading it twice would waste ~1.2 GB of RAM and "
        "~5 seconds. The pipeline solves this by caching the Stage 3.5 result in "
        "_pre_asr_mms_result and passing it directly to the LangID vote in Stage 5.",
        s["body"],
    ))
    story += code_block([
        "# Stage 3.5: run MMS-LID probe, cache result",
        "_pre_asr_mms_result = None",
        "if not config.get('language_override'):",
        "    whisper_path, _pre_asr_mms_result = _probe_and_select_asr_model(...)",
        "",
        "# Stage 5: reuse cached result instead of loading MMS-LID again",
        "elif _pre_asr_mms_result is not None:            # cache hit",
        "    mms_result = _pre_asr_mms_result",
        "    logger.info('MMS-LID (cached from pre-ASR probe)')",
        "elif _mms_lid_available(paths):                  # cache miss: load fresh",
        "    mms_det    = MMSLangDetector(...)",
        "    mms_result = mms_det.detect(str(pre_out))",
        "    mms_det.unload()",
    ], s, "pipeline.py — MMS-LID result caching")

    story.append(Paragraph("Result Dict Structure", s["h2"]))
    story.append(Paragraph(
        "The pipeline assembles a comprehensive result dict that is saved to "
        "output/{stem}_result.json and also passed to the database layer. Key fields:",
        s["body"],
    ))
    story.append(kv_table([
        ("report_id",          "ISUM-YYYYMMDDHHMMSS — unique identifier"),
        ("transcript",         "Full concatenated transcript from all chunks"),
        ("translation{}",      "{translated_text, route_used, success, error}"),
        ("segments[]",         "Per-segment: {start, end, text, confidence, no_speech_prob, speaker}"),
        ("final_language",     "ISO-639-1 code from LangID vote"),
        ("route_confidence",   "0–1 ensemble confidence"),
        ("language_uncertain", "True if confidence < 0.60"),
        ("vote_note",          "Human-readable vote explanation, e.g. 'gurmukhi-script override'"),
        ("keyword_alerts{}",   "{alerts[], summary_counts{}, threat_level, top_categories[]}"),
        ("isum{}",             "Full ISUMReport: all 5W fields, flags, timestamps"),
        ("stage_timings{}",    "{VAD: 1.2, ASR: 45.3, ...} in seconds"),
        ("mem_peak_mb",        "Peak RAM usage during pipeline run"),
        ("backtrans_chrf",     "chrF score against back-translation (if enabled)"),
    ], s))

    story.append(PageBreak())
    return story


def build_frontend(s):
    story = []
    story.append(Paragraph("14. Frontend — app.py", s["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "app.py is the Streamlit frontend. It provides a 10-tab web UI that wraps the pipeline "
        "and database without containing any processing logic itself. All tabs read from "
        "st.session_state['last_result'] (the most recent pipeline output) or query the "
        "database through the TranscriptDB API.",
        s["body"],
    ))
    story.append(Spacer(1, 0.2*cm))

    tab_data = [
        ["Tab",        "Name",        "Primary Functionality"],
        ["1",  "PROCESS",   "Upload audio, select device/language, run pipeline with live stage progress. "
                            "Shows ASR confidence heatmap (word-level colours), VAD timeline, per-chunk re-transcription."],
        ["2",  "ISUM",      "Displays the 5W structured summary, severity badge, quality flags, "
                            "confidence gauge, and related intercepts from cross-correlation."],
        ["3",  "SEARCH",    "FTS5 full-text search across all records with language/threat filters, "
                            "date range, and matched segment timestamps."],
        ["4",  "DASHBOARD", "Aggregate statistics: total records, threat level distribution, "
                            "language breakdown, most active callsigns."],
        ["5",  "MAP",       "Geographic entity extraction results plotted on a Plotly dark-theme "
                            "globe using India SoI boundary GeoJSON overlay."],
        ["6",  "HISTORY",   "All historical intercepts in a sortable table. Actor profiles showing "
                            "per-entity appearance counts and threat history."],
        ["7",  "EXPORT",    "Download current result as PDF report, DOCX document, or SRT subtitle "
                            "file. Export all annotations as training data JSON."],
        ["8",  "METRICS",   "Per-run quality metrics (Tier 1 auto-metrics) and manual WER/BLEU "
                            "computation (Tier 2) with trend plots across runs."],
        ["9",  "ANNOTATE",  "Human correction interface: edit transcript, translation, language, "
                            "ISUM fields, mark false-positive keywords, add notes."],
        ["10", "CLEAR",     "Delete all database records and reset session state."],
    ]
    t = Table(tab_data, colWidths=[1.0*cm, 2.6*cm, 11.9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEAD_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEAD_FG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("ALIGN",         (0, 0), (1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(PageBreak())
    return story


def build_design_decisions(s):
    story = []
    story.append(Paragraph("15. Key Design Decisions &amp; Technical Challenges", s["h1"]))
    story.append(section_rule())

    story.append(Paragraph("Why 3-Source LangID Instead of Whisper Alone?", s["h2"]))
    story.append(Paragraph(
        "On sent_1.wav (7s Punjabi broadcast clip), Whisper returned hi with confidence 0.707 "
        "and FastText also returned hi at 0.986. Both systems were reading the same signal — "
        "romanised Punjabi text that looks identical to Hindi text to a character n-gram model. "
        "MMS-LID, operating on raw waveforms via wav2vec features, returned pa at 0.9992. "
        "Without an audio-only LID signal in the pipeline, this entire failure class "
        "(unanimous, high-confidence, wrong) has no recovery path.",
        s["body"],
    ))

    story.append(Paragraph("Why Not Use a Single Large Model?", s["h2"]))
    story.append(Paragraph(
        "The 8 GB RAM constraint forces modular sequential loading. A hypothetical single model "
        "that does ASR + LangID + Translation + Summarisation (e.g., Gemini Nano, Qwen-72B) "
        "would not fit in RAM and would require internet connectivity. The modular approach "
        "trades some efficiency (reload time between stages) for deployability on consumer hardware.",
        s["body"],
    ))

    story.append(Paragraph("Why SQLite Instead of a Vector Database?", s["h2"]))
    story.append(Paragraph(
        "SQLite FTS5 with the unicode61 tokenizer (which removes diacritics) handles "
        "transliterated Indic text search adequately for the expected dataset size. "
        "A vector database (Chroma, FAISS) would enable semantic search but requires "
        "an embedding model loaded at query time — adding ~500 MB RAM overhead for queries. "
        "For the current use case (keyword/callsign search), FTS5 is faster and simpler.",
        s["body"],
    ))

    story.append(Paragraph("Whisper condition_on_previous_text — The Critical Bug Fix", s["h2"]))
    story.append(callout(
        "During development, enabling condition_on_previous_text=True caused the pipeline to produce "
        "infinite repetition loops on silence and radio static — outputs like 'you you you you you...'. "
        "This is a known Whisper behaviour when it encounters low-SNR audio: the conditioning causes "
        "it to repeat its previous prediction with increasing confidence. "
        "Setting condition_on_previous_text=False treats every chunk independently and completely "
        "eliminates this class of failure.",
        s,
    ))

    story.append(Paragraph("IndicTrans2 DynamicCache Incompatibility", s["h2"]))
    story.append(callout(
        "IndicTrans2's custom attention implementation is incompatible with the DynamicCache class "
        "introduced in transformers>=5.0. The fix is two-part: (1) pass attn_implementation='eager' "
        "to prevent SDPA dispatch, and (2) pass use_cache=False to generate() to avoid the "
        "EncoderDecoderCache path entirely. Without these, IndicTrans2 raises a cryptic "
        "AttributeError: 'EncoderDecoderCache' object has no attribute 'get_seq_length'. "
        "Additionally, a stub module for the removed transformers.onnx package must be injected "
        "before IndicTrans2 is imported, as it still imports from that removed namespace.",
        s,
    ))

    story.append(Paragraph("Arabic-Script Disambiguation — Why the Ablation Shows No Gain", s["h2"]))
    story.append(Paragraph(
        "The ablation study (language_hint=None) shows Urdu at 61% for all configurations "
        "including Full System. This is not a bug in the Arabic-script rule — it is a demonstration "
        "of a prerequisite failure. When language_hint=None, Whisper misidentifies Urdu audio as "
        "Hindi and outputs Devanagari text. The Arabic character ratio in Devanagari text is 0.00, "
        "so the script filter never fires. In the integrated pipeline (Technique 7 active), "
        "MMS-LID in Stage 3.5 supplies language_hint='ur', Whisper outputs Nastaliq, and the "
        "Arabic filter activates — producing 70% Urdu accuracy. The ablation intentionally breaks "
        "this feedback loop to isolate components.",
        s["body"],
    ))

    story.append(PageBreak())
    return story


def build_dataflow(s):
    story = []
    story.append(Paragraph("16. Data Flow: Audio File to Final Output", s["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "The following traces a single audio file through all 10 stages, showing what each stage "
        "reads, what it produces, and what data structure it passes to the next stage.",
        s["body"],
    ))
    story.append(Spacer(1, 0.2*cm))

    flow_data = [
        ["Stage", "Input",                       "Processing",                             "Output / Passes to Next"],
        ["1 VAD",
         "Raw audio file (any format)",
         "Silero-VAD detects speech timestamps. Strips silence. Pads 100ms around each segment. Merges overlapping.",
         "→ output/*_vad.wav\n→ segments_seconds[] (timestamps) to Stage 3"],
        ["2 Preprocess",
         "_vad.wav",
         "Pre-emphasis, bandpass 300–3400Hz, stationary noise reduction, RMS normalisation, silence trim.",
         "→ output/*_preprocessed.wav\n→ SNR before/after to result dict"],
        ["3 Chunk",
         "_preprocessed.wav\n+ segments_seconds[]",
         "Greedy VAD-aware splitting into ≤29s chunks aligned with speech boundaries.",
         "→ output/*_chunks/chunk_XXXX.wav\n→ chunks[] with start/end timestamps"],
        ["3.5 Probe",
         "_preprocessed.wav",
         "MMS-LID-256 on full preprocessed audio. If lang matches whisper_model_<lang> config key and conf≥0.65, switch ASR model.",
         "→ whisper_path (possibly specialised model)\n→ mms_result dict (cached for Stage 5)"],
        ["4 ASR",
         "chunk_XXXX.wav files",
         "Whisper large-v3-turbo CTranslate2 int8. Language hint from chunk 1 reused for all. Hallucination filter.",
         "→ transcript (full text)\n→ all_segments[] with {start,end,text,conf,no_speech_prob,words[]}"],
        ["4.5 Diarise",
         "_preprocessed.wav\n+ all_segments[]",
         "40 MFCC + delta + delta2 embeddings per segment. Agglomerative clustering. Silhouette score selects k.",
         "→ all_segments[] with speaker field added\n(in-place modification)"],
        ["5 LangID",
         "transcript text\n+ mms_result (cached)\n+ whisper language+prob",
         "FastText on transcript. DialectDetector Unicode ratios. LangID vote with Script-Cascade overrides.",
         "→ routing dict: {final_language, route, confidence, uncertain, vote_note}"],
        ["6 Translate",
         "transcript\n+ routing.route\n+ routing.final_language",
         "NLLB-200 or IndicTrans2 based on route. Long-text sentence chunking. Explicit model unload after use.",
         "→ translation dict: {translated_text, success, route_used}\n→ back_translation (if enabled)"],
        ["7 Keywords",
         "transcript\n+ translated_text\n+ all_segments[]",
         "Regex word-boundary scan on both. Confidence-adjusted severity. Threat level from max effective severity.",
         "→ keyword_alerts dict: {alerts[], summary_counts{}, threat_level, top_categories[]}"],
        ["8 SSUM",
         "All previous outputs\n(intermediate dict)",
         "Rule-based 5W extraction (regex+spaCy). Optional Qwen/Ollama LLM override. Quality flags.",
         "→ isum dict: all ISUMReport fields\n→ Final result dict assembled\n→ Saved to JSON + SQLite"],
    ]
    t = Table(flow_data, colWidths=[1.8*cm, 3.5*cm, 6.2*cm, 4.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEAD_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEAD_FG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("LEADING",       (0, 0), (-1, -1), 11),
        # Highlight novel stages
        ("BACKGROUND",    (0, 4), (-1, 4), colors.HexColor("#fff3e0")),  # Stage 3.5
        ("BACKGROUND",    (0, 6), (-1, 6), colors.HexColor("#fff3e0")),  # Stage 4.5
    ]))
    story.append(t)
    story.append(Paragraph(
        "Rows highlighted in orange represent the novel Stage 3.5 (Pre-ASR Language Probe) and "
        "Stage 4.5 (Speaker Diarisation) insertion points.",
        s["label"],
    ))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Model Memory Budget Across Stages", s["h2"]))
    mem_data = [
        ["Stage", "Model Loaded",              "Peak RAM",    "Notes"],
        ["1",   "Silero-VAD",                  "< 0.2 GB",   "LSTM model, loaded once"],
        ["2",   "noisereduce (no model)",       "< 0.1 GB",   "NumPy only"],
        ["3",   "None",                         "< 0.1 GB",   "soundfile + librosa"],
        ["3.5", "MMS-LID-256",                  "~ 0.6 GB",   "Loaded + unloaded; result cached"],
        ["4",   "Whisper large-v3-turbo CT2",   "~ 1.9 GB",   "int8 quantised; largest single model"],
        ["4.5", "None (librosa + sklearn)",     "< 0.1 GB",   "No pre-trained model"],
        ["5",   "FastText lid.176.bin",         "~ 0.9 GB",   "+ MMS-LID cached (0 extra cost)"],
        ["6",   "NLLB-200-dist-600M",           "~ 2.4 GB",   "Largest memory stage; unloaded immediately"],
        ["7",   "None (regex)",                 "< 0.05 GB",  ""],
        ["8",   "spaCy en_core_web_sm",         "~ 0.1 GB",   "Lazy-loaded, shared with geo_module"],
        ["8",   "Qwen2.5-1.5B int4 (optional)", "~ 1.1 GB",   "Only loaded if model file present"],
        ["Total", "Sequential loading",         "< 7.8 GB",   "No two large models in RAM simultaneously"],
    ]
    t2 = Table(mem_data, colWidths=[1.4*cm, 5.5*cm, 2.4*cm, 6.2*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEAD_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEAD_FG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#e8f5e9")),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2), [C_LIGHT, colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("ALIGN",         (2, 0), (2, -1), "CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))
    return story


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_path = "/Users/vik/offline_ai_system_v2/VANI_Code_Documentation.pdf"
    s = build_styles()

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title="Multilingual Speech Processing Pipeline — Code Documentation",
        author="VANI Project",
    )

    story = []
    story += build_cover(s)
    story += build_toc_page(s)
    story += build_overview(s)
    story += build_config(s)
    story += build_vad(s)
    story += build_preprocessing(s)
    story += build_chunker(s)
    story += build_asr(s)
    story += build_diarize(s)
    story += build_langid(s)
    story += build_translation(s)
    story += build_keywords(s)
    story += build_isum(s)
    story += build_database(s)
    story += build_pipeline(s)
    story += build_frontend(s)
    story += build_design_decisions(s)
    story += build_dataflow(s)

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_GREY)
        page_num = canvas.getPageNumber()
        canvas.drawRightString(PAGE_W - 2*cm, 1.2*cm, f"Page {page_num}")
        canvas.drawString(2*cm, 1.2*cm, "Offline Multilingual Speech Processing Pipeline — Code Documentation")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF generated: {out_path}")


if __name__ == "__main__":
    main()

"""
Render integration/PROCESS_OVERVIEW.md content into a clean PDF via reportlab.

Content is authored inline (not parsed from the .md) so tables and the ASCII flow
diagram render reliably. Keep in sync with PROCESS_OVERVIEW.md.

Run:  venv\Scripts\python.exe scripts\paper\generate_process_overview_pdf.py
Out:  integration/PROCESS_OVERVIEW.pdf
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted,
)

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "integration" / "PROCESS_OVERVIEW.pdf"

INK   = colors.HexColor("#1b2733")
ACC   = colors.HexColor("#0f6fb5")
HEAD  = colors.HexColor("#12374f")
ROW   = colors.HexColor("#eef3f7")
LINE  = colors.HexColor("#c7d3dd")
MONOBG = colors.HexColor("#f4f6f8")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=HEAD, fontSize=20,
                    spaceAfter=4, leading=24)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=colors.HexColor("#5a6b7a"),
                     fontSize=9, spaceAfter=12, leading=13)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACC, fontSize=13,
                    spaceBefore=14, spaceAfter=5, leading=16)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], textColor=INK, fontSize=9.3,
                      leading=13.5, alignment=TA_LEFT, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=12, bulletIndent=2,
                        spaceAfter=2)
MONO = ParagraphStyle("MONO", parent=ss["Code"], fontName="Courier", fontSize=6.9,
                      leading=8.4, textColor=INK, backColor=MONOBG,
                      borderPadding=6, spaceAfter=8)


def P(t, style=BODY):  return Paragraph(t, style)
def B(t):              return Paragraph("• " + t, BULLET)


def table(data, col_widths=None, header=True, font=8.2):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), HEAD),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


FLOW = """Analyst uploads clip.wav in the Streamlit GUI                          [NODE-C]
        |
 1. VAD - strip silence, keep speech regions (Silero)                  [NODE-C]
        |   -> vad.wav
 2. Coarse MMS-LID probe on vad.wav                                    [NODE-C]
        |   -> coarse language hint (e.g. "zh")  ->  Gaurav tag "mandarin"
 3. POST /process?lang=mandarin&variant=robust  (raw WAV) ----------> [NODE-A]
        |                                                    A denoises per
        |   <--- zip: diarization.json, summary.json,        speaker + diarizes
        |            mixed_denoised.wav, Speaker_N_Denoised.wav
 4. Normalize A's output; use mixed_denoised.wav as the audio          [NODE-C]
        |   (skip C's local denoise/bandpass; keep normalize)
 5. POST /api/analyze (each Speaker_N_Denoised.wav) ---------------> [NODE-B]
        |   <--- per speaker: {top1_language, confidence, dialect}    B: LID+dialect
 6. Aggregate B's per-speaker LID -> dominant language by talk time    [NODE-C]
        |   -> picks the language-specific Whisper model
 7. Chunk mixed_denoised.wav (VAD-aware)                               [NODE-C]
 8. ASR - Whisper large-v3 CT2 (or SeamlessM4T for pa/ne),             [NODE-C]
        |   language forced from B's answer -> transcript + segments
 9. Label each ASR segment by time-overlap vs A's diarization.json     [NODE-C]
        |   -> SPEAKER_A / SPEAKER_B / ...
10. LanguageRouter vote: B's LID = audio vote, FastText = text vote    [NODE-C]
11. Translation - NLLB-200 (IndicTrans2 for Dogri) -> English          [NODE-C]
12. Keyword detection -> threat level + categories                    [NODE-C]
13. ISUM - Ollama gemma3:4b -> structured intelligence summary (5W)    [NODE-C]
14. Persist to SQLite, render in GUI                                   [NODE-C]"""

HANDOFF = """NODE-C -> NODE-A
  POST http://192.168.10.11:8801/process?lang=<mandarin|urdu|punjabi|pashto|default>
                                         &variant=<clean|robust>&mode=diarization-guided
       Content-Type: audio/wav      body = raw 16 kHz mono WAV bytes
   200 -> application/zip { diarization.json, summary.json,
                           mixed_denoised.wav, Speaker_1_Denoised.wav ... }
   503 -> busy (GPU lock held)

NODE-C -> NODE-B
  POST http://192.168.10.12:8802/api/analyze   multipart file = one Speaker_N_Denoised.wav
   200 -> { top1_language, top1_language_confidence,
           top1_dialect, top1_dialect_confidence, dialect_engaged, language_probs }
   200 -> { no_speech_detected: true, rms }     (silence gate)"""


def build():
    story = []
    story.append(P("VANI 3-Node Integration &mdash; Process Overview", H1))
    story.append(P("How one radio intercept flows through NODE-A (denoise + diarization), "
                   "NODE-B (language + dialect ID), and NODE-C (VANI orchestrator). "
                   "Generated 2026-07-09.", SUB))

    story.append(P("1. The three machines", H2))
    story.append(table([
        ["Node", "Owner", "Role", "Address"],
        ["NODE-A", "Gaurav", "Denoise (DeepFilterNet3) + diarization (DiariZen)", "192.168.10.11:8801"],
        ["NODE-B", "Sanket", "Language ID (8) + Mandarin dialect (MMS-LID-4017)", "192.168.10.12:8802"],
        ["NODE-C", "this repo", "VAD, ASR, translation, ISUM, GUI, orchestrator", "192.168.10.13:8501"],
    ], col_widths=[1.6*cm, 2.0*cm, 8.4*cm, 3.6*cm]))
    story.append(Spacer(1, 4))
    story.append(B("Isolated /24 LAN, one switch, no internet / gateway / DNS."))
    story.append(B("A and B are <b>stateless</b> HTTP services; they know nothing of each other or of VANI."))
    story.append(B("<b>NODE-C is the only orchestrator</b> &mdash; it calls A and B and assembles the result."))

    story.append(P("2. End-to-end flow for one clip", H2))
    story.append(Preformatted(FLOW, MONO))

    story.append(P("3. Why the ordering is what it is", H2))
    story.append(P("A circular dependency between the partners, resolved for free by NODE-C's existing probe:"))
    story.append(B("<b>NODE-A needs a language tag before it runs</b> (per-language clustering; "
                   "\"default\" measurably hurts). It does no language ID itself."))
    story.append(B("<b>NODE-B wants to run after NODE-A</b> (trained on a post-denoiser, non-reverberant "
                   "distribution &mdash; it expects clean per-speaker tracks)."))
    story.append(P("Resolution: NODE-C runs its local MMS-LID probe early (on vad.wav) to give NODE-A the "
                   "coarse tag; NODE-B then runs on NODE-A's clean per-speaker tracks for the authoritative "
                   "answer. Neither partner changes to accommodate the other."))

    story.append(P("4. What each node does internally", H2))
    story.append(P("<b>NODE-A &mdash; who spoke when, and clean it up.</b> Loads DiariZen + DeepFilterNet3 "
                   "once at startup (a ~90 s cold call becomes ~5&ndash;10 s warm); diarizes; denoises each "
                   "speaker track independently (preserves speaker identity, 15.62% EER); reconstructs "
                   "<b>mixed_denoised.wav</b> on the original wall-clock timeline; serializes GPU work behind "
                   "a lock (returns 503 when busy). The clean/robust variant is not interchangeable."))
    story.append(P("<b>NODE-B &mdash; what language, which Mandarin dialect.</b> One MMS-LID-4017 forward pass "
                   "&rarr; 8-way language + 7-way dialect distribution. Trained without reverb (runs after A). "
                   "Analyzes the first 10 s only; NODE-C windows longer tracks. RMS&lt;0.01 silence gate. "
                   "Frozen v1 language + v1 dialect pair (macro-F1 0.9764 / 0.601)."))
    story.append(P("<b>NODE-C &mdash; everything else + orchestration.</b> VAD, coarse probe, chunking, ASR, "
                   "diarization labeling, language vote, translation, keywords, ISUM, storage, GUI. Normalizes "
                   "A's audio before forwarding to B (closes the gain-convention mismatch)."))

    story.append(P("5. The two handoffs", H2))
    story.append(Preformatted(HANDOFF, MONO))
    story.append(P("Both directions stay 16 kHz mono end to end, so nothing is ever resampled twice."))

    story.append(P("6. Language-code mapping", H2))
    story.append(P("<b>VANI &rarr; NODE-A</b> (Gaurav's clustering knob):"))
    story.append(table([
        ["VANI", "Gaurav tag"],
        ["zh", "mandarin"], ["ur", "urdu"], ["pa", "punjabi"], ["ps", "pashto"],
        ["anything else", "default (logged &mdash; no tuned operating point)"],
    ], col_widths=[3.0*cm, 9.0*cm]))
    story.append(Spacer(1, 5))
    story.append(P("<b>NODE-B &rarr; VANI</b> (Sanket's 8 classes):"))
    story.append(table([
        ["Sanket", "VANI", "Fine-tuned VANI ASR?"],
        ["urdu", "ur", "yes"], ["pashto", "ps", "yes"], ["kashmiri", "ks", "yes"],
        ["dogri", "doi", "no &mdash; default Whisper + IndicTrans2"],
        ["punjabi", "pa", "yes (routed to SeamlessM4T)"],
        ["mandarin", "zh", "yes"], ["cantonese", "zh", "no yue model -> zh"],
        ["tibetan", "bo", "no &mdash; default Whisper"],
    ], col_widths=[3.0*cm, 2.0*cm, 7.0*cm]))
    story.append(Spacer(1, 5))
    story.append(P("<b>Critical asymmetry:</b> hi (Hindi) and ne (Nepali) are not in NODE-B's 8-class set, "
                   "but VANI supports both. NODE-B's answer is accepted only when confident AND mappable; "
                   "otherwise NODE-C defers to its local MMS-LID probe. The local probe is never removed "
                   "&mdash; it is the safety net for Hindi, Nepali, and for NODE-B being down."))

    story.append(P("7. The safety net &mdash; why this cannot regress the demo", H2))
    story.append(B("<b>NODE-A unreachable</b> &rarr; NODE-C runs local denoise + MFCC diarization."))
    story.append(B("<b>NODE-B unreachable / unsure</b> &rarr; local MMS-LID vote carries the decision."))
    story.append(B("<b>Both down / Standalone</b> &rarr; NODE-C is the original single-machine VANI, byte-for-byte."))
    story.append(Spacer(1, 4))
    story.append(P("<b>Network mode (one build, either environment)</b> &mdash; an Auto / Standalone / "
                   "Networked switch backed by a once-per-session health probe:"))
    story.append(table([
        ["Mode", "Behaviour"],
        ["Auto (default)", "Probe A and B once; use whichever are reachable; else run fully local. "
                           "Unreachable nodes skipped (no per-file latency)."],
        ["Standalone", "Never touch the network &mdash; pure local pipeline."],
        ["Networked", "Trust the configured LAN nodes; fall back per-node on failure."],
    ], col_widths=[3.0*cm, 9.0*cm]))

    story.append(P("8. Roles at a glance", H2))
    story.append(table([
        ["Concern", "NODE-A", "NODE-B", "NODE-C"],
        ["Denoising", "per-speaker", "-", "fallback (local)"],
        ["Diarization", "yes", "-", "fallback + labeling"],
        ["Language ID", "needs a tag", "authoritative", "coarse probe + vote + fallback"],
        ["Dialect (Mandarin)", "-", "yes", "display"],
        ["VAD / chunking", "-", "-", "yes"],
        ["ASR / translation", "-", "-", "yes"],
        ["Keywords / ISUM", "-", "-", "yes"],
        ["Orchestration / GUI / storage", "-", "-", "yes"],
    ], col_widths=[5.0*cm, 2.4*cm, 2.4*cm, 4.2*cm], font=8.0))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm,
                            title="VANI 3-Node Integration - Process Overview")
    doc.build(story)
    print(f"  wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()

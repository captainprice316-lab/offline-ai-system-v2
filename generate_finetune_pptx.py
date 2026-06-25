"""
generate_finetune_pptx.py  --  VANI Fine-Tuning 10-slide PPTX
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colour palette — Modern Minimal ───────────────────────────────────────────
C_DARK   = RGBColor(0xF5, 0xF5, 0xF5)   # light grey background
C_ACCENT = RGBColor(0x00, 0x69, 0x5C)   # teal
C_GOLD   = RGBColor(0x6A, 0x1B, 0x9A)   # purple
C_WHITE  = RGBColor(0x21, 0x21, 0x21)   # dark text
C_LGRAY  = RGBColor(0x55, 0x55, 0x55)   # medium grey secondary text
C_GREEN  = RGBColor(0x00, 0x89, 0x7B)   # teal-green for good WER
C_RED    = RGBColor(0xC6, 0x28, 0x28)   # red for poor WER
C_PANEL  = RGBColor(0xFF, 0xFF, 0xFF)   # white card background
C_BORDER = RGBColor(0xDD, 0xDD, 0xDD)   # light border
C_TBGHDR = RGBColor(0x00, 0x69, 0x5C)   # teal table header bg
C_TBGALT = RGBColor(0xE8, 0xF5, 0xE9)   # very light teal alternate row

OUT_PATH = Path(__file__).parent / "docs" / "VANI_Finetune_Presentation.pptx"

# ── Helpers ────────────────────────────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    return prs

def blank_slide(prs):
    blank_layout = prs.slide_layouts[6]  # completely blank
    return prs.slides.add_slide(blank_layout)

C_TITLE_TEXT = RGBColor(0x00, 0x47, 0x40)   # darker teal for slide titles

def bg(slide, color=C_DARK):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def box(slide, left, top, width, height, fill_color=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.line.width = line_width
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width if line_width else Pt(1)
    else:
        shape.line.fill.background()
    return shape

def txbox(slide, text, left, top, width, height,
          font_size=18, bold=False, color=C_WHITE,
          align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tb.word_wrap = wrap
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return tb

def accent_bar(slide, top=0.92, height=0.05):
    box(slide, 0, top, 13.33, height, fill_color=C_ACCENT)

def slide_number(slide, n):
    txbox(slide, str(n), 12.7, 7.15, 0.5, 0.3,
          font_size=11, color=C_LGRAY, align=PP_ALIGN.RIGHT)

def section_tag(slide, text, left=0.4, top=0.22):
    box(slide, left, top, 2.4, 0.30, fill_color=C_ACCENT)
    txbox(slide, text.upper(), left+0.08, top+0.03, 2.3, 0.26,
          font_size=9, bold=True, color=RGBColor(0xFF,0xFF,0xFF))

def hline(slide, top, color=C_ACCENT, width_in=12.5, left=0.4):
    box(slide, left, top, width_in, 0.025, fill_color=color)

def bullet_block(slide, items, left, top, width, height,
                 font_size=15, bullet="▸", color=C_WHITE, spacing=0.38):
    y = top
    for item in items:
        txbox(slide, f"{bullet}  {item}", left, y, width, 0.38,
              font_size=font_size, color=color)
        y += spacing

def wer_bar(slide, left, top, lang, wer, best=False):
    bar_max = 3.5
    pct = min(wer / 110, 1.0)
    bar_w = bar_max * pct
    color = C_GREEN if wer < 30 else (C_GOLD if wer < 60 else C_RED)

    box(slide, left, top, bar_max, 0.30, fill_color=RGBColor(0xE0, 0xF2, 0xF1))
    if bar_w > 0.02:
        box(slide, left, top, bar_w, 0.30, fill_color=color)
    txbox(slide, lang, left - 1.05, top + 0.03, 1.0, 0.28,
          font_size=13, bold=best, color=C_GOLD if best else C_WHITE, align=PP_ALIGN.RIGHT)
    txbox(slide, f"{wer}%", left + bar_max + 0.08, top + 0.03, 0.7, 0.28,
          font_size=13, bold=best, color=color)


# ── Slide builders ─────────────────────────────────────────────────────────────

def slide_01_title(prs):
    s = blank_slide(prs)
    bg(s)
    # Top teal band
    box(s, 0, 0, 13.33, 2.8, fill_color=C_ACCENT)
    # Bottom thin accent line
    box(s, 0, 7.45, 13.33, 0.05, fill_color=C_GOLD)

    txbox(s, "VANI", 1.0, 0.25, 11.0, 1.5,
          font_size=80, bold=True, color=RGBColor(0xFF,0xFF,0xFF), align=PP_ALIGN.CENTER)
    txbox(s, "Voice Analysis & Neural Intelligence", 1.0, 1.75, 11.0, 0.55,
          font_size=20, bold=False, color=RGBColor(0xCC,0xE8,0xE5), align=PP_ALIGN.CENTER)

    txbox(s, "Whisper Fine-Tuning Report", 1.0, 3.2, 11.0, 0.6,
          font_size=26, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)
    hline(s, 3.92, width_in=5.0, left=4.17, color=C_GOLD)
    txbox(s, "LoRA Adaptation for 6 Border-Region Languages", 1.0, 4.05, 11.0, 0.45,
          font_size=16, color=C_LGRAY, align=PP_ALIGN.CENTER)
    txbox(s, "M.Tech Research Project  ·  IIT Indore  ·  June 2026", 1.0, 5.0, 11.0, 0.38,
          font_size=13, italic=True, color=C_LGRAY, align=PP_ALIGN.CENTER)
    txbox(s, "Hardware: NVIDIA RTX 5060 8 GB  ·  Windows 11  ·  CUDA", 1.0, 5.45, 11.0, 0.3,
          font_size=11, color=RGBColor(0x99,0x99,0x99), align=PP_ALIGN.CENTER)
    slide_number(s, 1)

def slide_02_motivation(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Background")
    slide_number(s, 2)

    txbox(s, "Why Fine-Tune?", 0.4, 0.55, 12.0, 0.6,
          font_size=28, bold=True, color=C_ACCENT)
    hline(s, 1.25)

    # Left — problem
    box(s, 0.4, 1.4, 5.8, 4.9, fill_color=C_PANEL,
        line_color=C_RED, line_width=Pt(1.5))
    box(s, 0.4, 1.4, 5.8, 0.38, fill_color=C_RED)
    txbox(s, "The Problem", 0.55, 1.44, 5.4, 0.33,
          font_size=14, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
    issues = [
        "Baseline Whisper large-v3 WER\non Punjabi: ~75%",
        "Urdu baseline WER: ~74%",
        "Hindi baseline WER: ~75%",
        "Radio intercept audio is noisy\n& domain-specific",
        "Generic model not trained on\nborder-language acoustics",
    ]
    bullet_block(s, issues, 0.6, 1.95, 5.4, 3.5, font_size=13, color=C_WHITE, spacing=0.72)

    # Right — solution
    box(s, 6.7, 1.4, 5.9, 4.9, fill_color=C_PANEL,
        line_color=C_ACCENT, line_width=Pt(1.5))
    box(s, 6.7, 1.4, 5.9, 0.38, fill_color=C_ACCENT)
    txbox(s, "The Solution — LoRA Fine-Tuning", 6.85, 1.44, 5.6, 0.33,
          font_size=14, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
    solutions = [
        "Adapt only 0.25% of parameters\n(LoRA r=8, alpha=16)",
        "Train on FLEURS speech corpus\n(no military data needed)",
        "Keep base model frozen —\nno catastrophic forgetting",
        "Export to int8 CT2 format\nfor fast offline inference",
        "Plug directly into VANI\n10-stage pipeline",
    ]
    bullet_block(s, solutions, 6.9, 1.95, 5.5, 3.5, font_size=13, color=C_WHITE, spacing=0.72)

def slide_03_methodology(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Methodology")
    slide_number(s, 3)

    txbox(s, "LoRA Fine-Tuning Pipeline", 0.4, 0.65, 12.0, 0.6,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.3)

    # Pipeline flow boxes
    steps = [
        ("1", "FLEURS\nDataset", C_ACCENT),
        ("2", "WhisperProcessor\nFeature Extract", C_ACCENT),
        ("3", "LoRA Adapter\nr=8, q+v proj", C_GOLD),
        ("4", "Seq2Seq\nTrainer", C_GOLD),
        ("5", "Merge &\nConvert CT2", C_GREEN),
        ("6", "VANI\nDeployment", C_GREEN),
    ]
    x = 0.3
    for i, (num, label, col) in enumerate(steps):
        box(s, x, 1.7, 1.9, 1.3, fill_color=C_PANEL, line_color=col, line_width=Pt(1.5))
        txbox(s, num, x, 1.72, 1.9, 0.45, font_size=22, bold=True, color=col, align=PP_ALIGN.CENTER)
        txbox(s, label, x, 2.2, 1.9, 0.75, font_size=11, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            txbox(s, "→", x + 1.9, 2.1, 0.35, 0.4, font_size=22, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)
        x += 2.22

    # Config table
    txbox(s, "LoRA Configuration", 0.4, 3.3, 6.0, 0.4,
          font_size=16, bold=True, color=C_GOLD)
    params = [
        ("LoRA Rank (r)", "8"),
        ("LoRA Alpha", "16"),
        ("Target Modules", "q_proj, v_proj"),
        ("Trainable Params", "~3.9M / 1.55B (0.25%)"),
        ("Learning Rate", "5×10⁻⁵ with warmup"),
        ("Quantization", "int8 via CTranslate2"),
    ]
    y = 3.78
    for k, v in params:
        box(s, 0.4, y, 5.8, 0.38, fill_color=C_PANEL)
        txbox(s, k, 0.5, y+0.05, 2.8, 0.3, font_size=12, color=C_LGRAY)
        txbox(s, v, 3.3, y+0.05, 2.8, 0.3, font_size=12, bold=True, color=C_WHITE)
        y += 0.4

    # Right side notes
    txbox(s, "Training Notes", 6.7, 3.3, 5.8, 0.4,
          font_size=16, bold=True, color=C_GOLD)
    notes = [
        "Batch size 2, no gradient accumulation",
        "Early stopping: patience = 3 evals",
        "Eval metric: Word Error Rate (WER)",
        "Best checkpoint auto-selected",
        "Windows 11 CUDA — no torchcodec",
        "fp16 training, float32 eval",
    ]
    bullet_block(s, notes, 6.7, 3.75, 6.0, 3.0, font_size=12, spacing=0.37)

def slide_04_results_overview(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Results")
    slide_number(s, 4)

    txbox(s, "Fine-Tuning Results — All 7 Languages", 0.4, 0.65, 12.0, 0.6,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.3)

    # Header
    cols = [1.15, 3.0, 4.5, 6.2, 8.3, 10.0, 11.3]
    headers = ["Language", "Script", "Base Model", "Dataset", "Steps", "Best WER", "Size"]
    y = 1.45
    box(s, 0.3, y, 12.7, 0.38, fill_color=C_ACCENT)
    for i, h in enumerate(headers):
        txbox(s, h, cols[i], y+0.05, 1.8, 0.3, font_size=11, bold=True, color=RGBColor(0xFF,0xFF,0xFF))

    rows = [
        ("Punjabi  (pa)", "Gurmukhi",    "large-v3",       "FLEURS pa_in",      "3000", "55.67%",  "1479 MB"),
        ("Pashto   (ps)", "Nastaliq",    "medium-pashto",  "FLEURS ps_af",      "2000", "38.55%",  " 734 MB"),
        ("Urdu     (ur)", "Nastaliq",    "large-v3",       "FLEURS ur_pk",      "1000", "19.82%",  "1479 MB"),
        ("Nepali   (ne)", "Devanagari",  "large-v3",       "FLEURS ne_np",      "2000", "49.24%",  "1479 MB"),
        ("Mandarin (zh)", "Simplified",  "large-v3",       "FLEURS cmn_hans",   " 400", "16.03%",  "1479 MB"),
        ("Hindi    (hi)", "Devanagari",  "large-v3",       "FLEURS hi_in",      " 600", "19.78%",  "1479 MB"),
        ("Kashmiri (ks)", "Nastaliq",    "large-v3",       "IndicVoices 20k",   "1500", "N/A†",    "1479 MB"),
    ]
    wer_colors = [C_RED, C_GOLD, C_GREEN, C_GOLD, C_GREEN, C_GREEN, C_LGRAY]
    y += 0.4
    for i, (row, wc) in enumerate(zip(rows, wer_colors)):
        bg_col = C_PANEL if i % 2 == 0 else RGBColor(0xF3, 0xE5, 0xF5)
        box(s, 0.3, y, 12.7, 0.35, fill_color=bg_col)
        for j, cell in enumerate(row):
            col = wc if j == 5 else C_WHITE
            bold = j == 5
            txbox(s, cell, cols[j], y+0.04, 1.8, 0.28,
                  font_size=10, bold=bold, color=col)
        y += 0.36

    txbox(s, "Eval WER: 100-sample FLEURS test (FLEURS) / IndicVoices val (ks)  ·  Baseline was turbo model  ·  † Kashmiri: ur-proxy, eval_loss=0.936 at ckpt-1500",
          0.4, 6.95, 12.5, 0.35, font_size=10, italic=True, color=C_LGRAY)

def slide_05_wer_chart(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "WER Comparison")
    slide_number(s, 5)

    txbox(s, "Word Error Rate — Fine-Tuned vs Baseline", 0.4, 0.65, 12.0, 0.6,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.3)

    langs  = ["Mandarin (zh)", "Urdu (ur)",   "Hindi (hi)",   "Pashto (ps)",  "Nepali (ne)", "Punjabi (pa)"]
    ftwer  = [16.03,           19.82,          19.78,          38.55,          49.24,          55.67]
    bswer  = [100.03,          24.44,          30.29,          94.23,          94.55,          105.79]

    bar_max_in = 5.0
    scale = bar_max_in / 100.0
    left_label = 1.5
    bar_left = 2.7
    y = 1.55

    txbox(s, "Baseline WER", bar_left + 0.05, 1.45, 2.5, 0.3,
          font_size=11, color=RGBColor(0xAA, 0xAA, 0xAA), italic=True)
    txbox(s, "Fine-Tuned WER", bar_left + 0.05, 1.65, 2.5, 0.3,
          font_size=11, color=C_GREEN, italic=True)

    y = 1.95
    for lang, ft, bs in zip(langs, ftwer, bswer):
        txbox(s, lang, 0.3, y, left_label, 0.28, font_size=12, color=C_WHITE, align=PP_ALIGN.RIGHT)
        # baseline bar
        bw = bs * scale
        box(s, bar_left, y, bw, 0.14, fill_color=RGBColor(0x55, 0x55, 0x77))
        # finetuned bar
        fw = ft * scale
        col = C_GREEN if ft < 30 else (C_GOLD if ft < 55 else C_RED)
        box(s, bar_left, y + 0.15, fw, 0.14, fill_color=col)
        # labels
        txbox(s, f"{bs}%", bar_left + bw + 0.05, y, 0.8, 0.16,
              font_size=10, color=RGBColor(0xAA, 0xAA, 0xAA))
        txbox(s, f"{ft}%", bar_left + fw + 0.05, y + 0.15, 0.8, 0.16,
              font_size=10, bold=True, color=col)
        y += 0.75

    # improvement callouts on right
    box(s, 8.3, 1.55, 4.6, 5.5, fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1))
    txbox(s, "Improvement Summary", 8.5, 1.65, 4.2, 0.38,
          font_size=14, bold=True, color=C_GOLD)
    improvements = [
        ("Mandarin", "100% → 16.0%", "-84.0 pp"),
        ("Pashto",   "94% → 38.6%",  "-55.7 pp"),
        ("Punjabi",  "106% → 55.7%", "-50.1 pp"),
        ("Nepali",   "95% → 49.2%",  "-45.3 pp"),
        ("Hindi",    "30% → 19.8%",  "-10.5 pp"),
        ("Urdu",     "24% → 19.8%",  "-4.6 pp"),
    ]
    y2 = 2.15
    for lang, prog, delta in improvements:
        box(s, 8.4, y2, 4.3, 0.58, fill_color=RGBColor(0xFA, 0xFA, 0xFA))
        txbox(s, lang, 8.5, y2+0.04, 1.1, 0.26, font_size=12, bold=True, color=C_WHITE)
        txbox(s, prog, 9.6, y2+0.04, 1.8, 0.26, font_size=11, color=C_LGRAY)
        txbox(s, delta, 11.4, y2+0.04, 1.1, 0.26, font_size=12, bold=True, color=C_GREEN, align=PP_ALIGN.RIGHT)
        y2 += 0.63

def slide_06_language_deep(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Language Deep-Dive")
    slide_number(s, 6)

    txbox(s, "Per-Language WER Progression", 0.4, 0.65, 12.0, 0.55,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.28)

    # Three language cards per row — two rows
    cards = [
        ("Punjabi (pa)", "55.67%", C_RED,
         ["Base: whisper-large-v3", "Dataset: FLEURS pa_in (~2,500)", "Steps: 3000",
          "71.6% → 61.3% → 56.7% (train)", "Eval WER: 55.67% vs baseline 105.79%"]),
        ("Urdu (ur)", "19.82%", C_GREEN,
         ["Base: whisper-large-v3", "Dataset: FLEURS ur_pk (2,109)", "Steps: 1000",
          "24.44% baseline → 19.82% eval", "SeamlessM4T: 16.9%"]),
        ("Hindi (hi)", "19.78%", C_GREEN,
         ["Base: whisper-large-v3", "Dataset: FLEURS hi_in (2,120)", "Steps: 600",
          "30.29% baseline → 19.78% eval", "SeamlessM4T: 15.44%"]),
        ("Pashto (ps)", "38.55%", C_GOLD,
         ["Base: pashto-ghag-medium", "Dataset: FLEURS ps_af (~2,000)", "Steps: 2000",
          "94.23% baseline → 38.55% eval", "FT beats SeamlessM4T (44.4%)"]),
        ("Nepali (ne)", "49.24%", C_GOLD,
         ["Base: whisper-large-v3", "Dataset: FLEURS ne_np (3,332)", "Steps: 2000",
          "94.55% baseline → 49.24% eval", "SeamlessM4T: 28.46%"]),
        ("Mandarin (zh)", "16.03%", C_GREEN,
         ["Base: whisper-large-v3", "Dataset: FLEURS cmn_hans (3,246)", "Steps: 400 (best)",
          "100.03% baseline → 16.03% eval", "FT beats SeamlessM4T (100% WER norm issue)"]),
    ]

    positions = [
        (0.3,  1.45), (4.55, 1.45), (8.8,  1.45),
        (0.3,  4.35), (4.55, 4.35), (8.8,  4.35),
    ]
    for (left, top), (lang, wer, col, bullets) in zip(positions, cards):
        box(s, left, top, 4.0, 2.75, fill_color=C_PANEL, line_color=col, line_width=Pt(1.2))
        txbox(s, lang, left+0.12, top+0.08, 2.5, 0.32, font_size=13, bold=True, color=col)
        txbox(s, wer,  left+2.6,  top+0.08, 1.3, 0.32, font_size=18, bold=True, color=col, align=PP_ALIGN.RIGHT)
        y = top + 0.45
        for b in bullets:
            txbox(s, f"• {b}", left+0.12, y, 3.75, 0.28, font_size=10, color=C_LGRAY)
            y += 0.36

def slide_07_pipeline(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Pipeline Integration")
    slide_number(s, 7)

    txbox(s, "VANI 10-Stage Pipeline", 0.4, 0.65, 12.0, 0.55,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.28)

    stages = [
        ("Stage 1",  "VAD",           "Silero — detect speech segments",           C_ACCENT),
        ("Stage 2",  "Preprocessing", "Bandpass 300–3400 Hz + noise reduction",     C_ACCENT),
        ("Stage 3",  "MMS-LID",       "256-language ID — routes to correct model",  C_GOLD),
        ("Stage 4",  "ASR",           "Language-specific fine-tuned Whisper",       C_GOLD),
        ("Stage 5",  "Script Cascade","Arabic-script override for Urdu/Kashmiri",   C_GOLD),
        ("Stage 6",  "Translation",   "NLLB-200 → English (or direct for Punjabi)", C_GREEN),
        ("Stage 7",  "Diarization",   "Speaker separation (up to 4 speakers)",      C_GREEN),
        ("Stage 8",  "Keywords",      "Multilingual keyword/entity detection",       C_GREEN),
        ("Stage 9",  "ISUM",          "Gemma 3:12B — 5W intelligence summary",      C_ACCENT),
        ("Stage 10", "Database",      "SQLite + JSON export + Streamlit UI",         C_ACCENT),
    ]

    y = 1.45
    for i, (num, name, desc, col) in enumerate(stages):
        bg_col = C_PANEL if i % 2 == 0 else RGBColor(0xF3, 0xE5, 0xF5)
        box(s, 0.3, y, 12.7, 0.43, fill_color=bg_col)
        box(s, 0.3, y, 0.06, 0.43, fill_color=col)
        txbox(s, num,  0.45, y+0.07, 0.9,  0.30, font_size=10, bold=True,  color=col)
        txbox(s, name, 1.35, y+0.07, 2.2,  0.30, font_size=12, bold=True,  color=C_WHITE)
        txbox(s, desc, 3.55, y+0.07, 9.3,  0.30, font_size=11,             color=C_LGRAY)
        y += 0.45

def slide_08_key_findings(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Key Findings")
    slide_number(s, 8)

    txbox(s, "Key Findings & Technical Insights", 0.4, 0.65, 12.0, 0.55,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.28)

    findings = [
        ("01", C_GOLD,
         "LoRA r=8 is Sufficient",
         "Only 0.25% of parameters trained yet WER drops 20–52 pp across all languages. Full fine-tuning is unnecessary and would require 40+ GB VRAM."),
        ("02", C_GREEN,
         "FLEURS Generalises to Military Domain",
         "Despite no military-domain audio in training, fine-tuning on FLEURS speech dramatically improves VANI accuracy — domain transfer is real."),
        ("03", C_ACCENT,
         "MMS-LID is Critical for Punjabi",
         "Whisper and FastText both misidentify Punjabi as Hindi. MMS-LID correctly identifies Punjabi with >0.99 confidence, enabling the pa-override rule."),
        ("04", C_RED,
         "fp16 Instability at Low LR",
         "Mandarin training diverged at step ~820 (grad_norm=12.9). Fix: max_grad_norm=0.5 applied to Hindi and all subsequent languages — no recurrence."),
        ("05", C_GOLD,
         "Script-Cascade Prevents Misidentification",
         "Arabic-script detection (>20% Nastaliq chars in transcript) catches Urdu even when MMS-LID confidence is below threshold — critical for noisy radio."),
        ("06", C_RED,
         "CT2 Tokenizer Bug — All Large-v3 Models Translated Instead of Transcribed",
         "ct2-transformers-converter omits tokenizer.json. Whisper-tiny fallback has transcribe=50359 but large-v3 has translate=50359. Every transcribe call was secretly translate. "
         "Fix: copy tokenizer.json from adapter dir into CT2 dir. Now automated in finetune_whisper.py."),
    ]

    y = 1.42
    for num, col, title, detail in findings:
        box(s, 0.3, y, 0.55, 0.85, fill_color=col)
        txbox(s, num, 0.3, y+0.2, 0.55, 0.4, font_size=16, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)
        box(s, 0.9, y, 12.0, 0.85, fill_color=C_PANEL)
        txbox(s, title,  1.05, y+0.05, 11.7, 0.33, font_size=13, bold=True,  color=col)
        txbox(s, detail, 1.05, y+0.38, 11.7, 0.44, font_size=11,             color=C_LGRAY)
        y += 0.97

def slide_09_deployment(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    section_tag(s, "Deployment")
    slide_number(s, 9)

    txbox(s, "Deployment Architecture", 0.4, 0.65, 12.0, 0.55,
          font_size=28, bold=True, color=C_WHITE)
    hline(s, 1.28)

    # Model inventory
    txbox(s, "Deployed Models (CT2 int8)", 0.4, 1.4, 6.2, 0.38,
          font_size=15, bold=True, color=C_GOLD)
    models = [
        ("whisper-large-v3-pa-ct2",    "Punjabi",  "61.3%",  "1479 MB", C_RED),
        ("whisper-medium-pashto-ct2",  "Pashto",   "38.9%",  " 734 MB", C_GOLD),
        ("whisper-large-v3-ur-ct2",    "Urdu",     "22.3%",  "1479 MB", C_GREEN),
        ("whisper-large-v3-ne-ct2",    "Nepali",   "54.3%",  "1479 MB", C_GOLD),
        ("whisper-large-v3-zh-ct2",    "Mandarin", " 8.97%", "1479 MB", C_GREEN),
        ("whisper-large-v3-hi-ct2",    "Hindi",    "23.1%",  "1479 MB", C_GREEN),
        ("whisper-large-v3-ks-ct2",    "Kashmiri", "N/A†",   "1479 MB", C_LGRAY),
    ]
    y = 1.85
    for mname, lang, wer, size, col in models:
        box(s, 0.4, y, 6.0, 0.4, fill_color=C_PANEL)
        box(s, 0.4, y, 0.05, 0.4, fill_color=col)
        txbox(s, mname, 0.55, y+0.06, 3.5, 0.28, font_size=10, color=C_LGRAY)
        txbox(s, lang,  4.1,  y+0.06, 1.0, 0.28, font_size=11, bold=True, color=C_WHITE)
        txbox(s, wer,   5.1,  y+0.06, 0.7, 0.28, font_size=11, bold=True, color=col, align=PP_ALIGN.RIGHT)
        y += 0.42

    # Right — runtime specs
    box(s, 6.9, 1.4, 5.9, 4.9, fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1))
    txbox(s, "Runtime Specifications", 7.1, 1.52, 5.5, 0.38,
          font_size=15, bold=True, color=C_GOLD)
    specs = [
        ("Inference engine", "faster-whisper (CT2)"),
        ("Quantization",     "int8 (CPU + GPU)"),
        ("Device",           "CUDA (RTX 5060 8 GB)"),
        ("Translation",      "NLLB-200 distilled 600M"),
        ("LangID",           "MMS-LID 256-lang + FastText"),
        ("ISUM",             "Gemma 3:12B via Ollama"),
        ("Storage",          "SQLite + JSON"),
        ("Interface",        "Streamlit web UI"),
        ("Network",          "100% offline — no internet"),
    ]
    y2 = 2.02
    for k, v in specs:
        box(s, 7.0, y2, 5.7, 0.4, fill_color=RGBColor(0xFA, 0xFA, 0xFA))
        txbox(s, k, 7.1,  y2+0.06, 2.2, 0.28, font_size=11, color=C_LGRAY)
        txbox(s, v, 9.3,  y2+0.06, 3.2, 0.28, font_size=11, bold=True, color=C_WHITE)
        y2 += 0.42

    txbox(s, "Total deployed model storage: ~10.1 GB   |   Lab upgrade: change device: cuda in config.yaml   |   †Kashmiri: eval_loss metric (no vocab token)",
          0.4, 6.9, 12.5, 0.3, font_size=10, italic=True, color=C_LGRAY)

def slide_10_conclusion(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s, top=0, height=0.08)
    accent_bar(s, top=7.42, height=0.08)
    box(s, 0, 0.08, 0.08, 7.34, fill_color=C_ACCENT)
    slide_number(s, 10)

    txbox(s, "Summary & Next Steps", 0.5, 0.5, 12.0, 0.6,
          font_size=30, bold=True, color=C_WHITE)
    hline(s, 1.22, width_in=12.0, left=0.5)

    # Summary boxes
    summ = [
        (C_GREEN,  "7 Languages Fine-Tuned",
                   "pa · ps · ur · ne · zh · hi · ks\nAll deployed as CT2 int8"),
        (C_GOLD,   "Best Eval WER: 16.03% (Mandarin)",
                   "Urdu 19.82%  ·  Hindi 19.78%\nBeat SeamlessM4T on Pashto & Mandarin"),
        (C_ACCENT, "0.25% Params Trained",
                   "LoRA r=8 is sufficient\nNo full fine-tune needed"),
    ]
    x = 0.5
    for col, title, detail in summ:
        box(s, x, 1.45, 3.9, 1.6, fill_color=C_PANEL, line_color=col, line_width=Pt(1.5))
        box(s, x, 1.45, 3.9, 0.06, fill_color=col)
        txbox(s, title,  x+0.15, 1.58, 3.6, 0.42, font_size=14, bold=True, color=col)
        txbox(s, detail, x+0.15, 2.03, 3.6, 0.95, font_size=12, color=C_LGRAY)
        x += 4.1

    # Next steps
    txbox(s, "Next Steps", 0.5, 3.25, 12.0, 0.4,
          font_size=18, bold=True, color=C_GOLD)
    hline(s, 3.72, width_in=12.0, left=0.5, color=C_GOLD)

    nexts = [
        ("Kashmiri (ks)",    "Done — IndicVoices 20k, 1500 steps, eval_loss=0.936; CT2 deployed; qualitative eval with native audio pending"),
        ("Reduce WER <25%",  "Target: all 6 main languages below 25% WER — more steps, more data, or SeamlessM4T fine-tune comparison"),
        ("Fine-tune SM4T",   "Fine-tune SeamlessM4T on same FLEURS data — compare transcription + translation against fine-tuned Whisper"),
        ("Robustness Eval",  "8 audio degradation conditions (bandpass, AWGN 0–20 dB, PTT clip, MP3 codec)"),
        ("Paper Submission", "VANI paper targeting IJAINN / SLT 2026 — include cross-model eval and SM4T fine-tune comparison"),
    ]
    y = 3.88
    for title, detail in nexts:
        box(s, 0.5, y, 12.3, 0.52, fill_color=C_PANEL)
        box(s, 0.5, y, 0.05, 0.52, fill_color=C_ACCENT)
        txbox(s, title,  0.65, y+0.06, 2.2,  0.38, font_size=12, bold=True, color=C_ACCENT)
        txbox(s, detail, 2.85, y+0.06, 9.9,  0.38, font_size=11,            color=C_LGRAY)
        y += 0.57

    txbox(s, "VANI v2  ·  M.Tech Research  ·  IIT Indore  ·  June 2026",
          0.5, 7.1, 12.0, 0.3, font_size=11, italic=True, color=C_LGRAY, align=PP_ALIGN.CENTER)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    prs = new_prs()
    slide_01_title(prs)
    slide_02_motivation(prs)
    slide_03_methodology(prs)
    slide_04_results_overview(prs)
    slide_05_wer_chart(prs)
    slide_06_language_deep(prs)
    slide_07_pipeline(prs)
    slide_08_key_findings(prs)
    slide_09_deployment(prs)
    slide_10_conclusion(prs)

    OUT_PATH.parent.mkdir(exist_ok=True)
    prs.save(str(OUT_PATH))
    print(f"Saved -> {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()

"""make_demo_video2.py — VANI tactical demo #2: Pashto / Pathankot (with audio).

Commercial-grade Call-of-Duty-style tactical SIGINT walkthrough of a SYNTHETIC,
fictional intercept: a Lashkar-e-Taiba cell planning an attack on Pathankot air
base, which VANI detects and flags CRITICAL. The intercept audio actually plays
(a real Pashto FLEURS clip degraded to radio-intercept quality) with a synced
playback waveform, over a synthesised tactical soundtrack (klaxon, radar pings,
typing ticks, bass stingers).

This is a DEFENSIVE counter-terrorism demonstration with synthetic data; the
transcript is a brief flagged-intercept snippet (target + intent), no operational
detail.

Usage:
    python make_demo_video2.py --stills   # key frames only (fast)
    python make_demo_video2.py            # full render + audio mux
Output: docs/VANI_demo_pathankot.mp4
"""
import argparse
import math
import pathlib
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import soundfile as sf
import imageio_ffmpeg

ROOT = pathlib.Path(__file__).resolve().parent
FF = imageio_ffmpeg.get_ffmpeg_exe()

# ── synthetic scenario data (fictional; defensive demo) ──────────────────────
TRANSCRIPT  = "د پټان کوټ هوايي اډه زموږ هدف دی. لښکر طيبه چمتو ده. بريد نن شپه دی."
TRANSLATION = "Pathankot airbase is our target. Lashkar-e-Taiba is ready. The attack is tonight."
THREAT      = "CRITICAL"
REPORT_ID   = "ISUM-20260721-PTK-014"
LOC_NAME    = "PATHANKOT AFS"
LOC_COORD   = "32.27°N  75.65°E"
ACTOR       = "LASHKAR-E-TAIBA"
PROC_S      = 15.8
FLAG_HI = [("پټان کوټ", "critical"), ("لښکر طيبه", "critical"),
           ("هدف", "critical"), ("بريد", "critical"), ("چمتو", "high")]
TOP_CATS = ["TARGETING", "ATTACK PLANNING", "TERRORIST ORG", "PRE-ATTACK", "AIRBASE"]
ISUM = {
    "who":   "Lashkar-e-Taiba cell (2+ speakers)",
    "what":  "Planned attack on a military airbase",
    "where": "Pathankot Air Force Station, Punjab",
    "when":  "Tonight (imminent)",
}

# ── palette / fonts (shared tactical look) ───────────────────────────────────
BG, PANEL, GRID = "#05080D", "#0A121A", "#0F2A33"
AMBER, GREEN, CYAN, RED, WHITE, DIM = "#FFB000", "#39F07A", "#37C7E6", "#FF3B3B", "#E8F2F2", "#5B7A86"
SEV = {"critical": RED, "high": AMBER, "medium": CYAN, "low": DIM}

def _font(paths):
    for p in paths:
        if pathlib.Path(p).exists():
            return fm.FontProperties(fname=p)
    return fm.FontProperties(family="monospace")
MONO  = _font(["C:/Windows/Fonts/consola.ttf"])
MONOB = _font(["C:/Windows/Fonts/consolab.ttf"])
# Pashto uses Nastaliq/Perso-Arabic script — Nirmala UI is Devanagari-only and
# renders it as tofu. Segoe UI covers Arabic and shapes it correctly (RTL).
DEVA  = _font(["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/tahoma.ttf"])

W, H, FPS = 16.0, 9.0, 24

# ── audio: degrade a real Pashto clip to radio-intercept quality ─────────────
import sys
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
def _load_intercept(sr_out=22050):
    from scipy.signal import butter, sosfilt, resample_poly
    wav, sr = sf.read(str(ROOT / "demo_clips/ps_pashto_1.wav"), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # bandpass 300-3400 Hz (telephone/radio band)
    sos = butter(4, [300/(sr/2), 3400/(sr/2)], btype="band", output="sos")
    wav = sosfilt(sos, wav).astype("float32")
    # light channel noise + mild clip (PTT feel)
    wav += 0.012 * np.random.randn(len(wav)).astype("float32")
    wav = np.tanh(wav * 1.6) * 0.85
    if sr != sr_out:
        wav = resample_poly(wav, sr_out, sr).astype("float32")
    wav /= (np.abs(wav).max() + 1e-9)
    return wav * 0.9, sr_out
INTERCEPT, ASR = _load_intercept()
CLIP_DUR = len(INTERCEPT) / ASR

# waveform envelope for the animation
_n = 700
_e = np.abs(INTERCEPT[:(len(INTERCEPT)//_n)*_n]).reshape(_n, -1).max(axis=1)
WAVE = _e / (_e.max() + 1e-9)

# ── scene timeline ───────────────────────────────────────────────────────────
AUDIO_START = 5.5   # when the intercept plays (video seconds)
SCENES = [
    ("alert",     0.0,  4.0),
    ("playback",  4.0, 16.5),   # audio plays here, waveform + playhead
    ("lang",     16.5, 20.5),
    ("transcript",20.5,29.5),
    ("translate",29.5, 36.5),
    ("threat",   36.5, 42.5),
    ("map",      42.5, 49.5),
    ("network",  49.5, 56.5),
    ("sitrep",   56.5, 63.0),
    ("close",    63.0, 66.0),
]
DUR = SCENES[-1][2]

def scene_at(t):
    for n, a, b in SCENES:
        if a <= t < b:
            return n, (t-a)/(b-a), t-a
    return SCENES[-1][0], 1.0, 0.0
def reached(t, name):
    for n, a, b in SCENES:
        if n == name:
            return t >= a
    return False
def ease(x):
    return 0.5 - 0.5*math.cos(max(0.0, min(1.0, x))*math.pi)

# ── drawing primitives (shared) ──────────────────────────────────────────────
def bgfx(ax, t):
    ax.add_patch(Rectangle((0, 0), W, H, color=BG, zorder=0))
    for x in np.arange(0, W, 0.5):
        ax.plot([x, x], [0, H], color=GRID, lw=0.4, alpha=0.35, zorder=0.5)
    for y in np.arange(0, H, 0.5):
        ax.plot([0, W], [y, y], color=GRID, lw=0.4, alpha=0.35, zorder=0.5)
    sy = (t*2.2) % H
    ax.plot([0, W], [sy, sy], color=CYAN, lw=1.0, alpha=0.10, zorder=0.6)

def brackets(ax, x, y, w, h, color, s=0.28, lw=1.6, alpha=1.0, z=3):
    for cx, cy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        ax.plot([cx, cx+dx*s], [cy, cy], color=color, lw=lw, alpha=alpha, zorder=z)
        ax.plot([cx, cx], [cy, cy+dy*s], color=color, lw=lw, alpha=alpha, zorder=z)

def panel(ax, x, y, w, h, title, accent=CYAN, fill=PANEL, alpha=1.0, base_z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc=fill, ec=accent, lw=1.2, alpha=alpha, zorder=base_z))
    ax.add_patch(Rectangle((x, y+h-0.42), w, 0.42, color=accent, alpha=0.16*alpha, zorder=base_z+0.1))
    ax.text(x+0.18, y+h-0.21, title, color=accent, fontproperties=MONOB, fontsize=12,
            va="center", ha="left", alpha=alpha, zorder=base_z+1)
    brackets(ax, x, y, w, h, accent, alpha=alpha, z=base_z+1)

def header(ax, t):
    ax.add_patch(Rectangle((0, H-0.75), W, 0.75, color="#081119", zorder=1.5))
    ax.plot([0, W], [H-0.75, H-0.75], color=CYAN, lw=1.0, alpha=0.5, zorder=1.6)
    ax.text(0.35, H-0.38, "V A N I", color=CYAN, fontproperties=MONOB, fontsize=18, va="center", zorder=2)
    ax.text(1.9, H-0.38, "// TACTICAL SIGINT ANALYSIS", color=DIM, fontproperties=MONO, fontsize=11, va="center", zorder=2)
    blink = int(t*2) % 2 == 0
    ax.text(W-0.35, H-0.38, "● SECURE · OFFLINE", color=GREEN if blink else DIM,
            fontproperties=MONOB, fontsize=11, va="center", ha="right", zorder=2)
    if reached(t, "threat"):
        pulse = 0.55 + 0.45*abs(math.sin(t*4))
        ax.text(W/2, H-0.38, f"THREAT LEVEL: {THREAT}", color=RED, fontproperties=MONOB,
                fontsize=13, va="center", ha="center", alpha=pulse, zorder=2)

def footer(ax, t):
    ax.add_patch(Rectangle((0, 0), W, 0.5, color="#081119", zorder=1.5))
    ax.text(0.35, 0.25, f"REPORT {REPORT_ID}   ·   SRC intercept_ptk_014.wav   ·   16 kHz MONO   ·   PROC {PROC_S:.1f}s",
            color=DIM, fontproperties=MONO, fontsize=10, va="center", zorder=2)
    ax.text(W-0.35, 0.25, "SYNTHETIC DEMO · CLEARED PERSONNEL ONLY", color=AMBER,
            fontproperties=MONO, fontsize=10, va="center", ha="right", zorder=2)

def typed(full, prog):
    return full[:int(len(full)*max(0.0, min(1.0, prog)))]

def draw_map(ax, x, y, w, h, prog, t):
    panel(ax, x, y, w, h, "TAC-MAP // PUNJAB SECTOR", AMBER)
    cx, cy = x+w*0.55, y+h*0.5
    for r in [0.5, 1.1, 1.7]:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec=GREEN, lw=0.8, alpha=0.25, zorder=2.5))
    ang = t*1.4
    ax.plot([cx, cx+1.7*math.cos(ang)], [cy, cy+1.7*math.sin(ang)], color=GREEN, lw=1.0, alpha=0.5, zorder=2.5)
    ax.plot(cx-1.25, cy-1.0, "s", color=DIM, ms=6, zorder=3)
    ax.text(cx-1.1, cy-1.0, "AMRITSAR", color=DIM, fontproperties=MONO, fontsize=9, va="center", zorder=3)
    if prog > 0.15:
        p = ease((prog-0.15)/0.5)
        rr = 0.55 - 0.4*p
        pulse = 0.6 + 0.4*abs(math.sin(t*5))
        ax.add_patch(Circle((cx, cy), rr, fill=False, ec=RED, lw=2.0, alpha=pulse, zorder=4))
        ax.plot([cx-0.5, cx+0.5], [cy, cy], color=RED, lw=1.2, alpha=pulse, zorder=4)
        ax.plot([cx, cx], [cy-0.5, cy+0.5], color=RED, lw=1.2, alpha=pulse, zorder=4)
        if prog > 0.5:
            ax.text(cx+0.35, cy+0.45, LOC_NAME, color=RED, fontproperties=MONOB, fontsize=12, va="center", zorder=4)
            ax.text(cx+0.35, cy+0.15, LOC_COORD, color=WHITE, fontproperties=MONO, fontsize=9, va="center", zorder=4)
            ax.text(cx-1.55, cy+1.45, ">> TARGET LOCKED", color=RED, fontproperties=MONOB,
                    fontsize=10, va="center", zorder=4, alpha=pulse)

NET = [("INTERCEPT", 0.0, 0.0, CYAN), (ACTOR.split("-")[0]+"-E-TAIBA", -1.7, 0.95, RED),
       ("PATHANKOT AFS", 1.6, 1.0, RED), ("ATTACK", 1.9, -0.05, RED),
       ("AIRBASE", 1.55, -1.05, AMBER), ("CELL / SPKR", -1.8, -0.9, GREEN), ("PASHTO", -1.95, 0.0, DIM)]
def draw_network(ax, x, y, w, h, prog, t):
    panel(ax, x, y, w, h, "NETWORK LINKAGES", CYAN)
    cx, cy = x+w*0.5, y+h*0.46
    sx, sy = w*0.24, h*0.28
    pos = {n: (cx+dx*sx, cy+dy*sy) for (n, dx, dy, _) in NET}
    ecol = {n: c for (n, _, _, c) in NET}
    order = [n for (n, *_ ) in NET[1:]]
    for i, n in enumerate(order):
        ep = ease((prog - i*0.10)/0.18)
        if ep <= 0:
            continue
        x0, y0 = pos["INTERCEPT"]; x1, y1 = pos[n]
        ax.plot([x0, x0+(x1-x0)*ep], [y0, y0+(y1-y0)*ep], color=ecol[n], lw=1.4, alpha=0.7, zorder=3)
    for (n, dx, dy, c) in NET:
        i = 0 if n == "INTERCEPT" else order.index(n)+1
        np_ = ease((prog - i*0.10 + 0.05)/0.18)
        if np_ <= 0:
            continue
        px, py = pos[n]
        rad = (0.16 if n == "INTERCEPT" else 0.12)*np_
        ax.add_patch(Circle((px, py), rad, fc=BG, ec=c, lw=1.8, zorder=4))
        ax.add_patch(Circle((px, py), rad*0.4, fc=c, ec="none", zorder=4.1))
        ax.text(px, py-0.28, n, color=c, fontproperties=MONOB, fontsize=8.0, ha="center", va="top", alpha=np_, zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
def render(ax, t):
    ax.clear(); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    bgfx(ax, t)
    name, p, el = scene_at(t)

    if name == "alert":
        flash = int(t*3) % 2 == 0
        ax.add_patch(Rectangle((0, 0), W, H, color=RED, alpha=0.06 if flash else 0.0, zorder=0.7))
        ax.text(W/2, H*0.62, "● INCOMING TRANSMISSION", color=RED if flash else DIM,
                fontproperties=MONOB, fontsize=30, ha="center", va="center", zorder=3)
        ax.text(W/2, H*0.50, "PRIORITY: FLASH   //   ORIGIN: BORDER SECTOR   //   SECRET",
                color=AMBER, fontproperties=MONO, fontsize=14, ha="center", va="center", zorder=3)
        if p > 0.4:
            ax.text(W/2, H*0.38, "VANI TACTICAL SIGINT  —  ACQUIRING SIGNAL",
                    color=CYAN, fontproperties=MONO, fontsize=13, ha="center", va="center",
                    alpha=ease((p-0.4)/0.5), zorder=3)
        brackets(ax, 1.2, 1.0, W-2.4, H-2.4, RED if flash else DIM, s=0.5, lw=2.2)
        return

    header(ax, t); footer(ax, t)

    if name == "playback":
        panel(ax, 1.2, 2.3, W-2.4, 4.4, ">> INTERCEPT PLAYBACK // PASHTO (ps)", GREEN)
        xs = np.linspace(1.7, W-1.7, len(WAVE)); mid = 4.5
        # audio plays AUDIO_START..AUDIO_START+CLIP_DUR; playhead maps to that window
        a_el = t - AUDIO_START
        ph = max(0.0, min(1.0, a_el / CLIP_DUR)) if a_el >= 0 else 0.0
        nlit = int(len(WAVE) * ph)
        for i in range(1, len(WAVE)):
            hgt = WAVE[i]*1.7
            col = GREEN if i <= nlit else "#1c3a40"
            ax.plot([xs[i], xs[i]], [mid-hgt, mid+hgt], color=col, lw=1.1,
                    alpha=0.9 if i <= nlit else 0.5, zorder=3)
        if 0 <= a_el <= CLIP_DUR:
            phx = xs[min(nlit, len(WAVE)-1)]
            ax.plot([phx, phx], [mid-1.9, mid+1.9], color=CYAN, lw=2.0, alpha=0.9, zorder=4)
            ax.text(1.7, 6.2, ">> PLAYING", color=CYAN, fontproperties=MONOB, fontsize=12, va="center", zorder=3)
        elif a_el < 0:
            ax.text(1.7, 6.2, "BUFFERING…", color=AMBER, fontproperties=MONO, fontsize=11, va="center", zorder=3)
        else:
            ax.text(1.7, 6.2, "VAD: SPEECH CAPTURED", color=GREEN, fontproperties=MONOB, fontsize=12, va="center", zorder=3)
        ax.text(W-1.7, 6.2, f"16 kHz · SECURE · {CLIP_DUR:.1f}s", color=DIM, fontproperties=MONO,
                fontsize=10, va="center", ha="right", zorder=3)
        return

    if name == "lang":
        panel(ax, 3.0, 3.0, W-6.0, 3.0, "LANGUAGE IDENTIFICATION", CYAN)
        spin = "|/-\\"[int(t*8) % 4]
        if p < 0.45:
            ax.text(W/2, 4.5, f"ANALYSING  {spin}", color=AMBER, fontproperties=MONOB, fontsize=22, ha="center", va="center", zorder=3)
        else:
            q = ease((p-0.45)/0.4)
            ax.text(W/2, 4.9, "PASHTO // ps", color=GREEN, fontproperties=MONOB, fontsize=30, ha="center", va="center", alpha=q, zorder=3)
            ax.text(W/2, 4.25, "MMS-LID confidence  0.97", color=WHITE, fontproperties=MONO, fontsize=13, ha="center", va="center", alpha=q, zorder=3)
            ax.text(W/2, 3.7, ">> ASR BACKEND:  SeamlessM4T v2 + ps LoRA  [ONLINE]", color=CYAN,
                    fontproperties=MONOB, fontsize=13, ha="center", va="center", alpha=q, zorder=3)
        return

    TLx, TLy, PW, PH = 0.55, 5.05, 7.4, 3.35
    BLx, BLy = 0.55, 0.75
    TRx, RPW = 8.25, 7.2

    if reached(t, "transcript"):
        panel(ax, TLx, TLy, PW, PH, "TRANSCRIPT // ps (SeamlessM4T)", GREEN)
        tp = p if name == "transcript" else 1.0
        ax.text(TLx+0.25, TLy+PH-1.0, typed(TRANSCRIPT, tp*1.05), color=WHITE, fontproperties=DEVA,
                fontsize=18, va="top", ha="left", wrap=True, zorder=3)
    if reached(t, "translate"):
        panel(ax, BLx, BLy, PW, 3.9, "ENGLISH TRANSLATION // NLLB-200", CYAN)
        tp = p if name == "translate" else 1.0
        ax.text(BLx+0.25, BLy+3.9-1.0, typed(TRANSLATION, tp*1.05), color=WHITE, fontproperties=MONOB,
                fontsize=14, va="top", ha="left", wrap=True, zorder=3)
        if reached(t, "threat"):
            gp = ease(p) if name == "threat" else 1.0
            gx, gy = BLx+0.3, BLy+0.55
            ax.text(gx, gy+0.55, "THREAT ASSESSMENT", color=DIM, fontproperties=MONO, fontsize=10, va="center", zorder=3)
            for i, s in enumerate(["LOW", "MED", "HIGH", "CRIT"]):
                on = gp > (i+1)/4 - 0.01
                col = [GREEN, CYAN, AMBER, RED][i]
                ax.add_patch(Rectangle((gx+i*1.15, gy), 1.0, 0.32, fc=col if on else "#12222a",
                             ec=col, lw=1.0, alpha=1 if on else 0.4, zorder=3))
                ax.text(gx+i*1.15+0.5, gy+0.16, s, color=BG if on else DIM, fontproperties=MONOB,
                        fontsize=9, ha="center", va="center", zorder=3.1)
            for i, (wd, sev) in enumerate(FLAG_HI[:4]):
                if gp > 0.4 + i*0.12:
                    ax.text(gx+5.0, gy+0.5-i*0.30, f">> {wd}", color=SEV.get(sev, AMBER),
                            fontproperties=DEVA, fontsize=12, va="center", zorder=3)
    if reached(t, "map"):
        draw_map(ax, TRx, 4.55, RPW, 3.85, p if name == "map" else 1.0, t)
    if reached(t, "network"):
        draw_network(ax, TRx, 0.75, RPW, 3.55, p if name == "network" else 1.0, t)

    if name in ("sitrep", "close"):
        op = ease(p) if name == "sitrep" else 1.0
        ax.add_patch(Rectangle((0, 0.5), W, H-1.25, color=BG, alpha=0.985*op, zorder=6))
        panel(ax, 2.2, 1.4, W-4.4, 6.0, f"SITREP // {REPORT_ID} // THREAT: {THREAT}", RED,
              fill="#0A0F16", alpha=op, base_z=6.2)
        rows = [("WHO", ISUM["who"]), ("WHAT", ISUM["what"]), ("WHERE", ISUM["where"]),
                ("WHEN", ISUM["when"]), ("ACTOR", ACTOR),
                ("THREAT", f"{THREAT}   ·   " + " / ".join(TOP_CATS[:4]))]
        yy = 6.45
        for i, (k, v) in enumerate(rows):
            if op > 0.15 + i*0.10:
                a = ease((op-0.15-i*0.10)/0.4)
                ax.text(2.7, yy, k, color=AMBER, fontproperties=MONOB, fontsize=13, va="top", alpha=a, zorder=7)
                ax.text(4.6, yy, v[:74], color=WHITE, fontproperties=MONOB, fontsize=12.5, va="top", alpha=a, zorder=7)
            yy -= 0.82
        if name == "close":
            ax.text(W/2, 1.75, f"ANALYSIS COMPLETE · {PROC_S:.1f}s · FULLY OFFLINE",
                    color=GREEN, fontproperties=MONOB, fontsize=15, ha="center", va="center", zorder=7)
    return

# ── tactical soundtrack (synthesised) + speech, mux ──────────────────────────
def build_audio(sr=22050):
    N = int(DUR*sr)
    trk = np.zeros(N, dtype="float32")
    tt = np.arange(N)/sr
    def place(sig, t0):
        i = int(t0*sr); j = min(N, i+len(sig))
        if i < N:
            trk[i:j] += sig[:j-i]
    def tone(f, d, amp=0.3, fade=0.01, kind="sine"):
        n = int(d*sr); x = np.arange(n)/sr
        w = 2*np.pi*f*x
        s = np.sin(w) if kind == "sine" else np.sign(np.sin(w))
        env = np.ones(n); fn = int(fade*sr)
        if fn > 0:
            env[:fn] = np.linspace(0, 1, fn); env[-fn:] = np.linspace(1, 0, fn)
        return (s*env*amp).astype("float32")
    def sweep(f0, f1, d, amp=0.3):
        n = int(d*sr); x = np.arange(n)/sr
        f = np.linspace(f0, f1, n)
        s = np.sin(2*np.pi*np.cumsum(f)/sr)
        env = np.hanning(n)
        return (s*env*amp).astype("float32")
    def noise(d, amp=0.05, lp=True):
        n = int(d*sr); s = np.random.randn(n).astype("float32")
        if lp:
            s = np.convolve(s, np.ones(20)/20, mode="same")
        return s*amp*np.hanning(n)
    # ambience: low drone across whole clip
    drone = (0.02*np.sin(2*np.pi*70*tt) + 0.014*np.sin(2*np.pi*110*tt)
             + 0.02*np.random.randn(N)).astype("float32")
    drone *= np.clip(np.linspace(0, 1, int(1.5*sr)).tolist() + [1.0]*(N-int(1.5*sr)), 0, 1)[:N]
    trk += drone*0.5
    # FLASH klaxon (two-tone) 0.3..3.2
    for k in range(4):
        place(tone(660, 0.35, 0.22), 0.3+k*0.7)
        place(tone(440, 0.35, 0.22), 0.65+k*0.7)
    # intercept speech placed at AUDIO_START (resampled to sr)
    from scipy.signal import resample_poly
    sp = resample_poly(INTERCEPT, sr, ASR).astype("float32") * 0.85
    place(sp, AUDIO_START)
    # radar/UI beeps during lang lock
    place(sweep(300, 900, 0.25, 0.18), 18.5)
    place(tone(900, 0.12, 0.2), 19.6)
    # typing ticks during transcript (20.5..29)
    for i in range(28):
        place(tone(1600, 0.03, 0.06, fade=0.005), 20.8 + i*0.28)
    # threat stinger (bass hit) at 36.5
    place(sweep(180, 40, 0.9, 0.5), 36.5)
    place(tone(55, 0.8, 0.3), 36.6)
    # radar pings during map (42.5..49)
    for tp in [43.2, 44.6, 46.0]:
        place(tone(1200, 0.08, 0.16), tp)
    place(sweep(1400, 300, 0.4, 0.25), 46.6)   # lock
    # network soft blips
    for i in range(6):
        place(tone(700+i*90, 0.05, 0.08), 50.0+i*0.5)
    # sitrep stinger
    place(sweep(200, 60, 0.7, 0.45), 56.5)
    place(tone(60, 0.9, 0.28), 56.6)
    # close tone
    place(tone(880, 0.5, 0.18), 63.2)
    # normalize
    trk = np.tanh(trk*1.1)
    trk /= (np.abs(trk).max()+1e-9); trk *= 0.92
    out = ROOT / "scratch_audio.wav"
    sf.write(str(out), trk, sr)
    return out

def make_fig():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    return fig, fig.add_axes([0, 0, 1, 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    args = ap.parse_args()
    if args.stills:
        out = ROOT/"scratch_frames"; out.mkdir(exist_ok=True)
        fig, ax = make_fig()
        for t in [2.0, 9.0, 18.0, 25.0, 33.0, 40.0, 46.0, 53.0, 60.0, 64.0]:
            render(ax, t); fig.savefig(out/f"t{t:04.1f}.png", dpi=90, facecolor=BG)
        print("stills ->", out); return

    plt.rcParams["animation.ffmpeg_path"] = FF
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, ax = make_fig()
    nframes = int(DUR*FPS)
    silent = ROOT/"scratch_silent.mp4"
    FuncAnimation(fig, lambda i: render(ax, i/FPS), frames=nframes).save(
        str(silent), writer=FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"]))
    print("silent video done; building audio…")
    audio = build_audio()
    final = ROOT/"docs"/"VANI_demo_pathankot.mp4"
    subprocess.run([FF, "-y", "-i", str(silent), "-i", str(audio),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)],
                   check=True, capture_output=True)
    silent.unlink(missing_ok=True); audio.unlink(missing_ok=True)
    print(f"Done -> {final}  ({final.stat().st_size//1024} KB, {DUR:.0f}s + audio)")

if __name__ == "__main__":
    main()

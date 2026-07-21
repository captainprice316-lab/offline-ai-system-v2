"""make_demo_video.py — VANI tactical-SIGINT demo animation (Call of Duty style).

An operator receives a FLASH-priority critical intercept; VANI's tactical HUD
populates in sequence: signal acquisition, language lock, transcript (Hindi),
English translation, tactical map locking onto Srinagar, animated network-linkage
graph, and a CRITICAL threat SITREP. Driven by the real result JSON
(output/01_hi_critical_srinagar_result.json).

Usage:
    python make_demo_video.py --stills     # dump a few key frames as PNGs (fast)
    python make_demo_video.py              # render the full MP4
Output: docs/VANI_demo.mp4  (stills -> scratch_frames/)
"""
import argparse
import io
import json
import math
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent

# ── real intercept data ──────────────────────────────────────────────────────
R = json.load(open(ROOT / "output/01_hi_critical_srinagar_result.json", encoding="utf-8"))
TRANSCRIPT   = R["transcript"]                                   # Devanagari
TRANSLATION  = (R.get("translation") or {}).get("translated_text") or R["translation"]
if isinstance(TRANSLATION, dict):
    TRANSLATION = TRANSLATION.get("translated_text", "")
THREAT       = R["threat_level"]                                # CRITICAL
ISUM         = R.get("isum", {})
KW           = (R.get("keyword_alerts") or {}).get("alerts", [])
TOP_CATS     = [c.upper().replace("_", " ") for c in
                (R.get("keyword_alerts") or {}).get("top_categories", [])[:5]]
PROC_S       = R.get("processing_time_s", 16.7)
REPORT_ID    = R.get("report_id", "ISUM-XXXX")
# flagged transcript keywords (Hindi) with severity
FLAG_HI = []
for a in KW:
    if a.get("matched_in") == "transcript" and a.get("matched_word") not in [f[0] for f in FLAG_HI]:
        FLAG_HI.append((a["matched_word"], a.get("effective_severity", "high")))

# ── palette (tactical) ───────────────────────────────────────────────────────
BG    = "#05080D"
PANEL = "#0A121A"
GRID  = "#0F2A33"
AMBER = "#FFB000"
GREEN = "#39F07A"
CYAN  = "#37C7E6"
RED   = "#FF3B3B"
WHITE = "#E8F2F2"
DIM   = "#5B7A86"
SEV_COLOR = {"critical": RED, "high": AMBER, "medium": CYAN, "low": DIM}

# ── fonts ────────────────────────────────────────────────────────────────────
def _font(paths):
    for p in paths:
        if pathlib.Path(p).exists():
            return fm.FontProperties(fname=p)
    return fm.FontProperties(family="monospace")

MONO = _font(["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"])
MONOB = _font(["C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/courbd.ttf"])
DEVA = _font(["C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/mangal.ttf"])

# ── audio waveform (downsampled envelope) ────────────────────────────────────
try:
    _wav, _sr = sf.read(str(ROOT / "demo_audio/01_hi_critical_srinagar.wav"), dtype="float32")
    if _wav.ndim > 1:
        _wav = _wav.mean(axis=1)
    _n = 600
    _env = np.abs(_wav[: (len(_wav) // _n) * _n]).reshape(_n, -1).max(axis=1)
    WAVE = _env / (_env.max() + 1e-9)
except Exception:
    WAVE = np.abs(np.sin(np.linspace(0, 40, 600))) * (0.4 + 0.6 * np.random.rand(600))

W, H = 16.0, 9.0   # data units (16:9)

# ── scene timeline (seconds) ─────────────────────────────────────────────────
FPS = 24
SCENES = [
    ("alert",      0.0,  4.5),
    ("acquire",    4.5,  9.0),
    ("lang",       9.0, 13.0),
    ("transcript",13.0, 22.0),
    ("translate", 22.0, 30.0),
    ("threat",    30.0, 37.0),
    ("map",       37.0, 44.0),
    ("network",   44.0, 52.0),
    ("sitrep",    52.0, 59.0),
    ("close",     59.0, 63.0),
]
DUR = SCENES[-1][2]


def scene_at(t):
    for name, a, b in SCENES:
        if a <= t < b:
            return name, (t - a) / (b - a), t - a
    return SCENES[-1][0], 1.0, 0.0


def reached(t, name):
    """True once scene `name` has started (for persistent panels)."""
    for n, a, b in SCENES:
        if n == name:
            return t >= a
    return False


def ease(x):
    return 0.5 - 0.5 * math.cos(max(0.0, min(1.0, x)) * math.pi)


# ── drawing primitives ───────────────────────────────────────────────────────
def bg(ax, t):
    ax.add_patch(Rectangle((0, 0), W, H, color=BG, zorder=0))
    # faint grid
    for x in np.arange(0, W, 0.5):
        ax.plot([x, x], [0, H], color=GRID, lw=0.4, alpha=0.35, zorder=0.5)
    for y in np.arange(0, H, 0.5):
        ax.plot([0, W], [y, y], color=GRID, lw=0.4, alpha=0.35, zorder=0.5)
    # scanline sweep
    sy = (t * 2.2) % H
    ax.plot([0, W], [sy, sy], color=CYAN, lw=1.0, alpha=0.10, zorder=0.6)


def corner_brackets(ax, x, y, w, h, color, s=0.28, lw=1.6, alpha=1.0, z=3):
    for cx, cy, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1),
                           (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        ax.plot([cx, cx + dx * s], [cy, cy], color=color, lw=lw, alpha=alpha, zorder=z)
        ax.plot([cx, cx], [cy, cy + dy * s], color=color, lw=lw, alpha=alpha, zorder=z)


def panel(ax, x, y, w, h, title, accent=CYAN, fill=PANEL, alpha=1.0, base_z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc=fill, ec=accent, lw=1.2, alpha=alpha, zorder=base_z))
    ax.add_patch(Rectangle((x, y + h - 0.42), w, 0.42, color=accent, alpha=0.16 * alpha, zorder=base_z+0.1))
    ax.text(x + 0.18, y + h - 0.21, title, color=accent, fontproperties=MONOB,
            fontsize=12, va="center", ha="left", alpha=alpha, zorder=base_z+1)
    corner_brackets(ax, x, y, w, h, accent, alpha=alpha, z=base_z+1)


def header(ax, t):
    ax.add_patch(Rectangle((0, H - 0.75), W, 0.75, color="#081119", zorder=1.5))
    ax.plot([0, W], [H - 0.75, H - 0.75], color=CYAN, lw=1.0, alpha=0.5, zorder=1.6)
    ax.text(0.35, H - 0.38, "V A N I", color=CYAN, fontproperties=MONOB, fontsize=18,
            va="center", zorder=2)
    ax.text(1.9, H - 0.38, "// TACTICAL SIGINT ANALYSIS", color=DIM, fontproperties=MONO,
            fontsize=11, va="center", zorder=2)
    # blinking OFFLINE/SECURE
    blink = (int(t * 2) % 2 == 0)
    ax.text(W - 0.35, H - 0.38, "● SECURE · OFFLINE", color=GREEN if blink else DIM,
            fontproperties=MONOB, fontsize=11, va="center", ha="right", zorder=2)
    # threat strip (appears from threat scene)
    if reached(t, "threat"):
        pulse = 0.55 + 0.45 * abs(math.sin(t * 4))
        ax.text(W / 2, H - 0.38, f"THREAT LEVEL: {THREAT}", color=RED, fontproperties=MONOB,
                fontsize=13, va="center", ha="center", alpha=pulse, zorder=2)


def footer(ax, t):
    ax.add_patch(Rectangle((0, 0), W, 0.5, color="#081119", zorder=1.5))
    msg = f"REPORT {REPORT_ID}   ·   SRC 01_hi_critical_srinagar.wav   ·   16 kHz MONO   ·   PROC {PROC_S:.1f}s"
    ax.text(0.35, 0.25, msg, color=DIM, fontproperties=MONO, fontsize=10, va="center", zorder=2)
    ax.text(W - 0.35, 0.25, "CLEARED PERSONNEL ONLY", color=AMBER, fontproperties=MONO,
            fontsize=10, va="center", ha="right", zorder=2)


def typed(full, prog):
    n = int(len(full) * max(0.0, min(1.0, prog)))
    return full[:n]


# ── map + network (reused across scenes) ─────────────────────────────────────
def draw_map(ax, x, y, w, h, prog, t):
    panel(ax, x, y, w, h, "TAC-MAP // J&K SECTOR", AMBER)
    cx, cy = x + w * 0.55, y + h * 0.5
    # range rings
    for i, r in enumerate([0.5, 1.1, 1.7]):
        ax.add_patch(Circle((cx, cy), r, fill=False, ec=GREEN, lw=0.8, alpha=0.25, zorder=2.5))
    # crosshair sweep
    ang = t * 1.4
    ax.plot([cx, cx + 1.7 * math.cos(ang)], [cy, cy + 1.7 * math.sin(ang)],
            color=GREEN, lw=1.0, alpha=0.5, zorder=2.5)
    # reference nodes
    ax.plot(cx - 1.2, cy - 1.0, "s", color=DIM, ms=6, zorder=3)
    ax.text(cx - 1.05, cy - 1.0, "JAMMU", color=DIM, fontproperties=MONO, fontsize=9, va="center", zorder=3)
    # target lock on Srinagar
    if prog > 0.15:
        p = ease((prog - 0.15) / 0.5)
        rr = 0.55 - 0.4 * p
        pulse = 0.6 + 0.4 * abs(math.sin(t * 5))
        ax.add_patch(Circle((cx, cy), rr, fill=False, ec=RED, lw=2.0, alpha=pulse, zorder=4))
        ax.plot([cx - 0.5, cx + 0.5], [cy, cy], color=RED, lw=1.2, alpha=pulse, zorder=4)
        ax.plot([cx, cx], [cy - 0.5, cy + 0.5], color=RED, lw=1.2, alpha=pulse, zorder=4)
        if prog > 0.5:
            ax.text(cx + 0.35, cy + 0.45, "SRINAGAR", color=RED, fontproperties=MONOB,
                    fontsize=12, va="center", zorder=4)
            ax.text(cx + 0.35, cy + 0.15, "34.08°N  74.80°E", color=WHITE, fontproperties=MONO,
                    fontsize=9, va="center", zorder=4)
            ax.text(cx - 1.55, cy + 1.45, ">> TARGET LOCKED", color=RED, fontproperties=MONOB,
                    fontsize=10, va="center", zorder=4, alpha=pulse)


NET_NODES = [
    ("INTERCEPT",  0.0,  0.0, CYAN),
    ("SRINAGAR",  -1.55, 0.95, RED),
    ("ENEMY ACT", 1.6,  1.0, RED),
    ("ATTACK",    1.85, -0.05, RED),
    ("EXPLOSIVES",1.55, -1.05, AMBER),
    ("SPKR-01",  -1.75,-0.9, GREEN),
    ("HINDI",    -1.9,  0.0, DIM),
]

def draw_network(ax, x, y, w, h, prog, t):
    panel(ax, x, y, w, h, "NETWORK LINKAGES", CYAN)
    cx, cy = x + w * 0.5, y + h * 0.46
    sx, sy = w * 0.24, h * 0.28
    pos = {n: (cx + dx * sx, cy + dy * sy) for (n, dx, dy, _) in NET_NODES}
    ecol = {n: c for (n, _, _, c) in NET_NODES}
    order = [n for (n, *_ ) in NET_NODES[1:]]
    # edges draw in sequentially
    for i, n in enumerate(order):
        ep = ease((prog - i * 0.10) / 0.18)
        if ep <= 0:
            continue
        x0, y0 = pos["INTERCEPT"]; x1, y1 = pos[n]
        xe, ye = x0 + (x1 - x0) * ep, y0 + (y1 - y0) * ep
        ax.plot([x0, xe], [y0, ye], color=ecol[n], lw=1.4, alpha=0.7, zorder=3)
    # nodes pop
    for (n, dx, dy, c) in NET_NODES:
        i = 0 if n == "INTERCEPT" else order.index(n) + 1
        np_ = ease((prog - i * 0.10 + 0.05) / 0.18)
        if np_ <= 0:
            continue
        px, py = pos[n]
        rad = (0.16 if n == "INTERCEPT" else 0.12) * np_
        ax.add_patch(Circle((px, py), rad, fc=BG, ec=c, lw=1.8, zorder=4))
        ax.add_patch(Circle((px, py), rad * 0.4, fc=c, ec="none", zorder=4.1))
        ax.text(px, py - 0.28, n, color=c, fontproperties=MONOB, fontsize=8.5,
                ha="center", va="top", alpha=np_, zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
def render(ax, t):
    ax.clear()
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    bg(ax, t)
    name, p, el = scene_at(t)

    # ---- SCENE: ALERT ----
    if name == "alert":
        flash = int(t * 3) % 2 == 0
        ax.add_patch(Rectangle((0, 0), W, H, color=RED, alpha=0.06 if flash else 0.0, zorder=0.7))
        ax.text(W/2, H*0.62, "● INCOMING TRANSMISSION", color=RED if flash else DIM,
                fontproperties=MONOB, fontsize=30, ha="center", va="center", zorder=3)
        ax.text(W/2, H*0.50, "PRIORITY: FLASH   //   CLASSIFICATION: SECRET",
                color=AMBER, fontproperties=MONO, fontsize=15, ha="center", va="center", zorder=3)
        if p > 0.4:
            ax.text(W/2, H*0.38, "VANI TACTICAL SIGINT  —  INITIALISING",
                    color=CYAN, fontproperties=MONO, fontsize=13, ha="center", va="center",
                    alpha=ease((p-0.4)/0.5), zorder=3)
        corner_brackets(ax, 1.2, 1.0, W-2.4, H-2.4, RED if flash else DIM, s=0.5, lw=2.2)
        return

    header(ax, t); footer(ax, t)

    # ---- SCENE: ACQUIRE (waveform) ----
    if name == "acquire":
        panel(ax, 1.2, 2.3, W-2.4, 4.2, "SIGNAL ACQUISITION", GREEN)
        n = int(len(WAVE) * ease(p))
        xs = np.linspace(1.7, W-1.7, len(WAVE))
        mid = 4.4
        for i in range(1, n):
            hgt = WAVE[i] * 1.7
            ax.plot([xs[i], xs[i]], [mid-hgt, mid+hgt], color=GREEN, lw=1.1, alpha=0.85, zorder=3)
        ax.text(1.7, 6.05, "INTERCEPT ACQUIRED · 16 kHz · SECURE CHANNEL",
                color=WHITE, fontproperties=MONO, fontsize=12, va="center", zorder=3)
        if p > 0.6:
            ax.text(W-1.7, 2.55, "VAD: 3.32s SPEECH DETECTED", color=CYAN, fontproperties=MONO,
                    fontsize=11, va="center", ha="right", alpha=ease((p-0.6)/0.4), zorder=3)
        return

    # ---- SCENE: LANGUAGE LOCK ----
    if name == "lang":
        panel(ax, 3.0, 3.0, W-6.0, 3.0, "LANGUAGE IDENTIFICATION", CYAN)
        spin = "|/-\\"[int(t*8) % 4]
        if p < 0.45:
            ax.text(W/2, 4.5, f"ANALYSING  {spin}", color=AMBER, fontproperties=MONOB,
                    fontsize=22, ha="center", va="center", zorder=3)
        else:
            q = ease((p-0.45)/0.4)
            ax.text(W/2, 4.9, "HINDI // hi", color=GREEN, fontproperties=MONOB,
                    fontsize=30, ha="center", va="center", alpha=q, zorder=3)
            ax.text(W/2, 4.25, "MMS-LID confidence  0.99", color=WHITE, fontproperties=MONO,
                    fontsize=13, ha="center", va="center", alpha=q, zorder=3)
            ax.text(W/2, 3.7, ">> ASR BACKEND:  SeamlessM4T v2  [ONLINE]", color=CYAN,
                    fontproperties=MONOB, fontsize=13, ha="center", va="center", alpha=q, zorder=3)
        return

    # From transcript onward: 4-quadrant tactical dashboard, panels populate over time.
    # top-left transcript, bottom-left translation+threat, top-right map, bottom-right net
    TLx, TLy, PW, PH = 0.55, 5.05, 7.4, 3.35     # top-left
    BLx, BLy         = 0.55, 0.75                 # bottom-left (h=3.9)
    TRx              = 8.25                        # right column x
    RPW             = 7.2

    # --- transcript panel (persistent once reached) ---
    if reached(t, "transcript"):
        panel(ax, TLx, TLy, PW, PH, "TRANSCRIPT // hi (SeamlessM4T)", GREEN)
        tp = p if name == "transcript" else 1.0
        txt = typed(TRANSCRIPT, tp * 1.05)
        ax.text(TLx+0.25, TLy+PH-1.0, txt, color=WHITE, fontproperties=DEVA, fontsize=19,
                va="top", ha="left", wrap=True, zorder=3)
        if name == "transcript" and int(t*3) % 2 == 0 and tp < 1.0:
            pass  # cursor omitted for wrap simplicity

    # --- translation panel ---
    if reached(t, "translate"):
        panel(ax, BLx, BLy, PW, 3.9, "ENGLISH TRANSLATION // NLLB-200", CYAN)
        tp = p if name == "translate" else 1.0
        ax.text(BLx+0.25, BLy+3.9-1.0, typed(TRANSLATION, tp*1.05), color=WHITE,
                fontproperties=MONOB, fontsize=15, va="top", ha="left", wrap=True, zorder=3)
        # threat gauge + keywords (from threat scene)
        if reached(t, "threat"):
            gp = ease(p) if name == "threat" else 1.0
            gx, gy = BLx+0.3, BLy+0.55
            ax.text(gx, gy+0.55, "THREAT ASSESSMENT", color=DIM, fontproperties=MONO,
                    fontsize=10, va="center", zorder=3)
            segs = ["LOW", "MED", "HIGH", "CRIT"]
            for i, s in enumerate(segs):
                on = gp > (i+1)/len(segs) - 0.01
                col = [GREEN, CYAN, AMBER, RED][i]
                ax.add_patch(Rectangle((gx+i*1.15, gy), 1.0, 0.32,
                             fc=col if on else "#12222a", ec=col, lw=1.0, alpha=1 if on else 0.4, zorder=3))
                ax.text(gx+i*1.15+0.5, gy+0.16, s, color=BG if on else DIM,
                        fontproperties=MONOB, fontsize=9, ha="center", va="center", zorder=3.1)
            # flagged keywords
            kx = gx + 5.0
            for i, (wd, sev) in enumerate(FLAG_HI[:4]):
                if gp > 0.4 + i*0.12:
                    ax.text(kx, gy+0.5-i*0.30, f">> {wd}", color=SEV_COLOR.get(sev, AMBER),
                            fontproperties=DEVA, fontsize=12, va="center", zorder=3)

    # --- map panel ---
    if reached(t, "map"):
        mp = p if name == "map" else 1.0
        draw_map(ax, TRx, 4.55, RPW, 3.85, mp, t)

    # --- network panel ---
    if reached(t, "network"):
        npg = p if name == "network" else 1.0
        draw_network(ax, TRx, 0.75, RPW, 3.55, npg, t)

    # --- SITREP overlay ---
    if name in ("sitrep", "close"):
        op = ease(p) if name == "sitrep" else 1.0
        ax.add_patch(Rectangle((0, 0.5), W, H-1.25, color=BG, alpha=0.985*op, zorder=6))
        panel(ax, 2.2, 1.4, W-4.4, 6.0, f"SITREP // {REPORT_ID} // THREAT: {THREAT}", RED,
              fill="#0A0F16", alpha=op, base_z=6.2)
        rows = [
            ("WHO",   str(ISUM.get("who", "Not identified"))),
            ("WHAT",  str(ISUM.get("what", "Attack order; bomb ready."))),
            ("WHERE", str(ISUM.get("where", "Srinagar"))),
            ("WHEN",  str(ISUM.get("when", "Immediate"))),
            ("THREAT", f"{THREAT}   ·   " + " / ".join(TOP_CATS[:4])),
        ]
        yy = 6.35
        for i, (k, v) in enumerate(rows):
            if op > 0.2 + i*0.12:
                a = ease((op-0.2-i*0.12)/0.4)
                ax.text(2.7, yy, k, color=AMBER, fontproperties=MONOB, fontsize=14, va="top", alpha=a, zorder=7)
                ax.text(4.6, yy, v[:78], color=WHITE, fontproperties=MONOB, fontsize=13.5,
                        va="top", alpha=a, zorder=7, wrap=True)
            yy -= 0.95
        if name == "close":
            ax.text(W/2, 1.75, f"ANALYSIS COMPLETE · {PROC_S:.1f}s · FULLY OFFLINE",
                    color=GREEN, fontproperties=MONOB, fontsize=15, ha="center", va="center", zorder=7)
    return


# ══════════════════════════════════════════════════════════════════════════════
def make_fig():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    if args.stills:
        out = ROOT / "scratch_frames"; out.mkdir(exist_ok=True)
        fig, ax = make_fig()
        for t in [2.0, 6.5, 11.5, 18.0, 26.0, 34.0, 41.0, 49.0, 55.0, 61.0]:
            render(ax, t)
            fig.savefig(out / f"t{t:04.1f}.png", dpi=90, facecolor=BG)
        print("stills ->", out)
        return

    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, ax = make_fig()
    nframes = int(DUR * args.fps)
    anim = FuncAnimation(fig, lambda i: render(ax, i / args.fps), frames=nframes, interval=1000/args.fps)
    outp = ROOT / "docs" / "VANI_demo.mp4"
    anim.save(str(outp), writer=FFMpegWriter(fps=args.fps, bitrate=6000,
              codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    print(f"Done -> {outp}  ({outp.stat().st_size//1024} KB, {DUR:.0f}s, {nframes} frames)")


if __name__ == "__main__":
    main()

"""
timeline_module.py — Cross-intercept temporal reconstruction.

Resolves raw temporal strings from when_field (e.g. "0600 hours", "tonight",
"six o'clock") into absolute datetimes anchored to each intercept's capture
timestamp.  Builds a sorted event list and detects temporal clusters — groups
of intercepts that reference the same time window within a configurable margin.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

# ── Time-of-day approximations for named periods ──────────────────────────────
_NAMED_HOUR = {
    "midnight":    0,  "dawn":      5,  "first light": 5,
    "sunrise":     6,  "morning":   8,  "noon":       12,
    "afternoon":  14,  "dusk":      18, "evening":    19,
    "sunset":     19,  "tonight":   21, "night":      22,
    "last light": 19,  "dark":      20,
    # Chinese
    "拂晓": 5, "黎明": 5, "上午": 9, "正午": 12,
    "下午": 14, "黄昏": 18, "傍晚": 18, "夜间": 22, "深夜": 23, "午夜": 0,
}

_DAY_OFFSET = {
    "yesterday": -1, "today": 0, "tonight": 0,
    "tomorrow": 1,  "后天": 2, "昨天": -1, "今天": 0, "明天": 1,
    "昨晚": -1, "今晚": 0, "明晚": 1,
}

_WEEKDAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

# Noise phrases — do not attempt to resolve
_NOISE = re.compile(
    r"^(no specific|not identified|no temporal|no time|"
    r"n/a|none|unknown|not mentioned|temporal references?:\s*$)",
    re.IGNORECASE,
)


def _strip_prefix(s: str) -> str:
    """Remove 'Temporal references: ' prefix if present."""
    s = re.sub(r"(?i)temporal references?:\s*", "", s).strip()
    return s


def _resolve_military_time(token: str, anchor: datetime) -> Optional[datetime]:
    """'0600 hours' → datetime on anchor's date."""
    m = re.match(r"(\d{3,4})\s*(?:hours?|hrs?)", token, re.IGNORECASE)
    if not m:
        return None
    t = m.group(1).zfill(4)
    try:
        h, mn = int(t[:2]), int(t[2:])
        if h > 23 or mn > 59:
            return None
        return anchor.replace(hour=h, minute=mn, second=0, microsecond=0)
    except ValueError:
        return None


def _resolve_clock_time(token: str, anchor: datetime) -> Optional[datetime]:
    """'14:30' or '6:00' → datetime on anchor's date."""
    m = re.match(r"(\d{1,2}):(\d{2})", token)
    if not m:
        return None
    h, mn = int(m.group(1)), int(m.group(2))
    if h > 23 or mn > 59:
        return None
    return anchor.replace(hour=h, minute=mn, second=0, microsecond=0)


def _resolve_named_time(token: str, anchor: datetime) -> Optional[datetime]:
    """'morning', 'tonight', 'dusk' → approximate datetime."""
    key = token.strip().lower()
    if key in _NAMED_HOUR:
        h = _NAMED_HOUR[key]
        day_offset = _DAY_OFFSET.get(key, 0)
        base = anchor + timedelta(days=day_offset)
        return base.replace(hour=h, minute=0, second=0, microsecond=0)
    return None


def _resolve_relative(token: str, anchor: datetime) -> Optional[datetime]:
    """'in 30 minutes', 'in 2 hours' → anchor + delta."""
    m = re.match(r"in\s+(\d+)\s+(minutes?|hours?)", token, re.IGNORECASE)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    delta = timedelta(minutes=n) if unit.startswith("m") else timedelta(hours=n)
    return anchor + delta


def _resolve_day_name(token: str, anchor: datetime) -> Optional[datetime]:
    """'monday', 'friday' → nearest future weekday from anchor."""
    key = token.strip().lower()
    if key not in _WEEKDAYS:
        return None
    target = _WEEKDAYS.index(key)
    diff   = (target - anchor.weekday()) % 7
    if diff == 0:
        diff = 7   # same day → next week
    return (anchor + timedelta(days=diff)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _resolve_written_number_time(token: str, anchor: datetime) -> Optional[datetime]:
    """'six o'clock', 'three o'clock' → approximate datetime."""
    words = {
        "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
        "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
    }
    m = re.match(
        r"(" + "|".join(words) + r")\s+o'?\s*clock",
        token, re.IGNORECASE
    )
    if not m:
        return None
    h = words[m.group(1).lower()]
    # Ambiguous AM/PM — use hour of anchor to guess
    if anchor.hour >= 12 and h < 12:
        h += 12
    return anchor.replace(hour=h % 24, minute=0, second=0, microsecond=0)


def resolve_when_field(
    when_field:   str,
    timestamp_utc: str,
) -> List[Dict]:
    """
    Parse a when_field string and return a list of resolved time references:
      [{
          "raw":        original token string,
          "resolved":   ISO datetime string or None,
          "confidence": float 0-1,
          "method":     how it was resolved,
      }]
    Returns [] if no useful temporal data found.
    """
    if not when_field or _NOISE.match(when_field.strip()):
        return []

    try:
        anchor = datetime.fromisoformat(
            timestamp_utc.replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except Exception:
        return []

    text   = _strip_prefix(when_field)
    tokens = [t.strip() for t in re.split(r"[,;]+", text) if t.strip()]
    results = []

    for tok in tokens:
        if _NOISE.match(tok):
            continue
        resolved = None
        method   = "none"
        conf     = 0.0

        dt = _resolve_military_time(tok, anchor)
        if dt:
            resolved, method, conf = dt, "military-time", 0.90

        if not resolved:
            dt = _resolve_clock_time(tok, anchor)
            if dt:
                resolved, method, conf = dt, "clock-time", 0.90

        if not resolved:
            dt = _resolve_written_number_time(tok, anchor)
            if dt:
                resolved, method, conf = dt, "written-number", 0.65

        if not resolved:
            dt = _resolve_named_time(tok, anchor)
            if dt:
                resolved, method, conf = dt, "named-period", 0.50

        if not resolved:
            dt = _resolve_relative(tok, anchor)
            if dt:
                resolved, method, conf = dt, "relative-delta", 0.75

        if not resolved:
            dt = _resolve_day_name(tok, anchor)
            if dt:
                resolved, method, conf = dt, "day-name", 0.40

        results.append({
            "raw":        tok,
            "resolved":   resolved.isoformat() if resolved else None,
            "confidence": conf,
            "method":     method,
        })

    return [r for r in results if r["method"] != "none"]


def build_timeline(intercepts: List[Dict]) -> List[Dict]:
    """
    Convert a list of intercept dicts (from db.get_all_intercepts or search)
    into a sorted list of timeline events.

    Each event:
      report_id, capture_time (datetime), event_time (datetime),
      event_time_source ("resolved"|"capture"), time_refs, time_confidence,
      threat, language, actors, locations, what, audio_file
    """
    events = []

    for rec in intercepts:
        ts = rec.get("timestamp_utc", "")
        try:
            capture_dt = datetime.fromisoformat(
                ts.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            continue

        when_raw  = rec.get("when_field") or rec.get("when", "") or ""
        time_refs = resolve_when_field(when_raw, ts)

        # Best resolved time: highest-confidence resolved reference
        resolved_refs  = [r for r in time_refs if r["resolved"]]
        best_ref       = max(resolved_refs, key=lambda r: r["confidence"]) \
                         if resolved_refs else None

        if best_ref:
            try:
                event_dt     = datetime.fromisoformat(best_ref["resolved"])
                event_source = "resolved"
                time_conf    = best_ref["confidence"]
            except Exception:
                event_dt     = capture_dt
                event_source = "capture"
                time_conf    = 0.0
        else:
            event_dt     = capture_dt
            event_source = "capture"
            time_conf    = 0.0

        events.append({
            "report_id":         rec.get("report_id", ""),
            "capture_time":      capture_dt,
            "event_time":        event_dt,
            "event_time_source": event_source,
            "time_confidence":   time_conf,
            "time_refs":         [r["raw"] for r in time_refs],
            "threat":            rec.get("threat_level", "CLEAR"),
            "language":          (rec.get("final_language") or "?").upper(),
            "actors":            rec.get("who_field", "") or "",
            "locations":         rec.get("where_field", "") or "",
            "what":              (rec.get("what_field") or rec.get("isum_assessment", "") or "")[:120],
            "audio_file":        rec.get("audio_file", ""),
        })

    events.sort(key=lambda e: e["event_time"])
    return events


def find_temporal_clusters(
    events:         List[Dict],
    window_minutes: int = 60,
) -> List[Dict]:
    """
    Group events whose event_time falls within window_minutes of each other.
    Only considers events with event_time_source == "resolved" for clustering;
    capture-time events are included in the nearest cluster as context.

    Returns list of clusters:
      {
        "cluster_id": int,
        "window_start": datetime,
        "window_end":   datetime,
        "events":       List[Dict],
        "threats":      List[str],
        "actors":       List[str],
      }
    """
    if not events:
        return []

    resolved = [e for e in events if e["event_time_source"] == "resolved"]
    if not resolved:
        return []

    resolved.sort(key=lambda e: e["event_time"])
    clusters = []
    used     = set()

    for i, anchor in enumerate(resolved):
        if anchor["report_id"] in used:
            continue
        window = timedelta(minutes=window_minutes)
        group  = [anchor]
        used.add(anchor["report_id"])

        for j, other in enumerate(resolved):
            if other["report_id"] in used:
                continue
            if abs(other["event_time"] - anchor["event_time"]) <= window:
                group.append(other)
                used.add(other["report_id"])

        threats = list(dict.fromkeys(e["threat"] for e in group))
        actors  = list(dict.fromkeys(
            a.strip()
            for e in group
            for a in re.split(r"[;,]", e["actors"])
            if a.strip() and not re.match(r"(?i)not identified|callsigns", a.strip())
        ))

        clusters.append({
            "cluster_id":   len(clusters) + 1,
            "window_start": min(e["event_time"] for e in group),
            "window_end":   max(e["event_time"] for e in group),
            "events":       group,
            "threats":      threats,
            "actors":       actors[:6],
        })

    clusters.sort(key=lambda c: c["window_start"])
    return clusters


def render_timeline_figure(events: List[Dict]) -> Optional[object]:
    """
    Render a Plotly scatter chart: X = event_time, Y = threat rank,
    sized by time_confidence, coloured by threat, marker shape by source.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not events:
        return None

    THREAT_COLOR = {
        "CRITICAL": "#ff3355", "HIGH": "#ff8c00",
        "MEDIUM":   "#ffaa00", "LOW":  "#88cc00", "CLEAR": "#00ff88",
    }
    THREAT_RANK = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "CLEAR": 1}

    xs, ys, colors, sizes, hovers, symbols = [], [], [], [], [], []

    for e in events:
        xs.append(e["event_time"])
        ys.append(THREAT_RANK.get(e["threat"], 1))
        colors.append(THREAT_COLOR.get(e["threat"], "#8a9aaa"))
        sizes.append(12 + int(e["time_confidence"] * 18))
        symbols.append("circle" if e["event_time_source"] == "resolved" else "circle-open")
        refs_str = ", ".join(e["time_refs"]) if e["time_refs"] else "none"
        hovers.append(
            f"<b>{e['report_id']}</b><br>"
            f"Captured: {e['capture_time'].strftime('%Y-%m-%d %H:%M')}<br>"
            f"Event time: {e['event_time'].strftime('%Y-%m-%d %H:%M')}"
            f" ({e['event_time_source']})<br>"
            f"Threat: {e['threat']}  |  Lang: {e['language']}<br>"
            f"Time refs: {refs_str}<br>"
            f"What: {e['what'][:80]}<br>"
            f"Actors: {e['actors'][:60]}"
        )

    trace = go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            color=colors, size=sizes, symbol=symbols,
            line=dict(width=1.5, color="#0d1117"),
            opacity=0.88,
        ),
        hovertext=hovers,
        hoverinfo="text",
        showlegend=False,
    )

    return go.Figure(
        data=[trace],
        layout=go.Layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#8a9aaa", family="Share Tech Mono, monospace"),
            margin=dict(l=60, r=20, t=20, b=40),
            xaxis=dict(
                title="Event Time",
                showgrid=True, gridcolor="#1a2a3a",
                zeroline=False, color="#8a9aaa",
            ),
            yaxis=dict(
                title="Severity",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["CLEAR", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
                showgrid=True, gridcolor="#1a2a3a",
                zeroline=False, color="#8a9aaa",
                range=[0.5, 5.8],
            ),
            hovermode="closest",
            height=380,
        ),
    )

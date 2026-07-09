"""
ui/streamlit_helpers.py – Reusable Streamlit UI components for VANI
"""

import json
import streamlit as st
import pandas as pd


def render_threat_badge(level: str):
    level = (level or "CLEAR").upper()
    icons = {"CRITICAL":"⬛","HIGH":"🟥","MEDIUM":"🟧","LOW":"🟦","CLEAR":"🟩"}
    st.markdown(
        f'<span class="threat-badge t-{level}">{icons.get(level,"")} {level}</span>',
        unsafe_allow_html=True,
    )


def render_pipeline_metrics(result: dict):
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Language",   result.get("final_language","—").upper())
    m2.metric("Route",      result.get("translation_route","—"))
    m3.metric("Confidence", f"{result.get('route_confidence',0)*100:.0f}%")
    m4.metric("Speech",     f"{result.get('total_speech_sec',0):.1f}s")
    m5.metric("Chunks",     result.get("chunks_created","—"))


def render_segment_timeline(segments: list, max_segments: int = 40):
    for seg in segments[:max_segments]:
        conf = seg.get("confidence", 0)
        cc   = "#00ff88" if conf>0.7 else "#ffaa00" if conf>0.4 else "#ff3355"
        st.markdown(
            f'<div class="seg-row">'
            f'<span class="seg-ts">[{seg.get("start",0):.2f}s–{seg.get("end",0):.2f}s]</span>'
            f'<span style="color:{cc};font-size:0.72rem;min-width:38px">{conf:.2f}</span>'
            f'<span style="color:#e0e0e0">{seg.get("text","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_keyword_table(alerts: list):
    if not alerts:
        return
    df = pd.DataFrame(alerts)
    cols = [c for c in ["severity","category","matched_word",
                         "matched_in","start_sec","end_sec","segment_text"]
            if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def render_isum_fields(isum: dict):
    w1,w2 = st.columns(2)
    with w1:
        _icard("WHO — Actors Identified",    isum.get("who","Not identified."))
        _icard("WHERE — Location Indicators",isum.get("where","Not identified."))
    with w2:
        _icard("WHAT — Activity Detected",   isum.get("what","No activity detected."))
        _icard("WHEN — Temporal Indicators",  isum.get("when","No temporal reference."))


def render_confidence_flags(flags: list):
    if not flags:
        st.markdown(
            '<span class="mono-txt" style="color:var(--accent-green)">'
            '✓ NO FLAGS — HIGH CONFIDENCE</span>',
            unsafe_allow_html=True,
        )
        return
    fc = " ".join([f'<span class="flag-chip">{f}</span>' for f in flags])
    st.markdown(f'⚠ &nbsp;{fc}', unsafe_allow_html=True)


def json_download_button(data: dict, filename: str, label: str = "⬇  Download JSON"):
    st.download_button(
        label,
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name=filename,
        mime="application/json",
    )


def _icard(label: str, value: str):
    st.markdown(
        f'<div class="isum-card"><div class="isum-lbl">{label}</div>'
        f'<div class="isum-val">{value or "—"}</div></div>',
        unsafe_allow_html=True,
    )

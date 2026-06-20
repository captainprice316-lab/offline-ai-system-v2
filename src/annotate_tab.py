"""
annotate_tab.py – Annotation tab content for VANI
Imported and called by app.py to keep app.py manageable.
"""

import json
import streamlit as st
from datetime import datetime, timezone


def render_annotate_tab(db, result, _load_latest_result):
    """Render the [A] ANNOTATE tab. Called from app.py."""

    # ── Helpers (local copies to avoid circular import) ───────────────────────
    def sechdr(label):
        st.markdown(f'<div class="section-hdr">{label}</div>', unsafe_allow_html=True)

    def icard_html(label, value):
        return (f'<div class="isum-card"><div class="isum-lbl">{label}</div>'
                f'<div class="isum-val">{value}</div></div>')

    # ── Load result ───────────────────────────────────────────────────────────
    if not result and callable(_load_latest_result):
        result = _load_latest_result()
        if result:
            st.session_state["last_result"] = result

    # ── Stats banner ──────────────────────────────────────────────────────────
    ann_stats = db.get_annotation_stats()
    sechdr("Training Data Collection")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Annotations",     ann_stats["total"])
    sc2.metric("ASR Corrected",   ann_stats["asr_fixed"])
    sc3.metric("Trans Corrected", ann_stats["trans_fixed"])
    sc4.metric("ISUM Corrected",  ann_stats["isum_fixed"])

    if not result:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:var(--text-secondary)">
            <div class="mono-txt" style="font-size:0.85rem">
                NO INTERCEPT LOADED<br>
                <span style="font-size:0.75rem">
                    Run the pipeline first, or switch to the History tab and reload a result.
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
        return

    report_id = result.get("report_id", "-")
    isum      = result.get("isum", {})
    lang      = (result.get("final_language") or "?").upper()

    st.markdown(
        f'<div class="mono-txt" style="color:#8a9aaa;margin-bottom:0.8rem">'
        f'Annotating: <span style="color:#e8f0e8">{report_id}</span>'
        f' &nbsp;|&nbsp; Audio: <span style="color:#e8f0e8">{result.get("audio_file","-")}</span>'
        f' &nbsp;|&nbsp; Detected Language: <span style="color:#e8f0e8">{lang}</span>'
        f'</div>', unsafe_allow_html=True,
    )

    existing = db.get_annotation(report_id)

    ann_t1, ann_t2, ann_t3, ann_t4, ann_t5 = st.tabs([
        "ASR Transcript", "Translation", "ISUM / 5W Fields",
        "Export Dataset", "Keyword Editor",
    ])

    # ── ASR Transcript ────────────────────────────────────────────────────────
    with ann_t1:
        sechdr(f"Transcript Correction  [{lang}]")
        orig_transcript = result.get("transcript", "")

        st.markdown('<div class="isum-lbl" style="margin-bottom:0.3rem">'
                    'Original Whisper output (read-only)</div>', unsafe_allow_html=True)
        st.text_area("__orig_t", value=orig_transcript, height=110,
                     disabled=True, label_visibility="collapsed", key=f"ann_orig_t_{report_id}")

        st.markdown('<div class="isum-lbl" style="margin-top:0.8rem;margin-bottom:0.3rem">'
                    'Corrected Transcript — fix errors, add missing words</div>',
                    unsafe_allow_html=True)
        default_corr_t = (existing or {}).get("corrected_transcript") or orig_transcript
        corr_t = st.text_area("__corr_t", value=default_corr_t, height=110,
                               label_visibility="collapsed", key=f"ann_corr_t_{report_id}")

        lang_opts = ["hi","pa","ur","ne","doi","ps","zh","my","ks","en","mai","bn","bo","other"]
        default_li = lang_opts.index(lang.lower()) if lang.lower() in lang_opts else len(lang_opts)-1
        corr_lang  = st.selectbox("Correct language (change if misidentified)",
                                   lang_opts, index=default_li, key=f"ann_lang_{report_id}")

        notes_asr = st.text_input("Notes (optional — e.g. heavy noise, multiple speakers)",
                                   value=(existing or {}).get("notes","") or "",
                                   key=f"ann_notes_asr_{report_id}")

        if st.button(">  Save ASR Annotation", type="primary", key=f"save_asr_{report_id}"):
            ann = {
                "report_id":            report_id,
                "annotated_at":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "corrected_transcript": corr_t,
                "transcript_changed":   corr_t.strip() != orig_transcript.strip(),
                "corrected_language":   corr_lang,
                "language_changed":     corr_lang != lang.lower(),
                "notes":                notes_asr,
            }
            if existing:
                for k in ("corrected_translation","translation_changed",
                          "corrected_who","corrected_what","corrected_where",
                          "corrected_when","corrected_assessment",
                          "corrected_threat_level","isum_changed"):
                    if existing.get(k) is not None:
                        ann.setdefault(k, existing[k])
            db.save_annotation(ann)
            st.success(f"ASR annotation saved for {report_id}.")

    # ── Translation ───────────────────────────────────────────────────────────
    with ann_t2:
        sechdr("Translation Correction  [EN]")
        trans_obj  = result.get("translation", {})
        orig_trans = (trans_obj.get("translated_text","") if isinstance(trans_obj,dict)
                      else str(trans_obj))

        st.markdown('<div class="isum-lbl" style="margin-bottom:0.3rem">'
                    'Original translation output (read-only)</div>', unsafe_allow_html=True)
        st.text_area("__orig_tr", value=orig_trans, height=110,
                     disabled=True, label_visibility="collapsed", key=f"ann_orig_tr_{report_id}")

        st.markdown('<div class="isum-lbl" style="margin-top:0.8rem;margin-bottom:0.3rem">'
                    'Corrected English Translation — fix mistranslations, missing context</div>',
                    unsafe_allow_html=True)
        default_corr_tr = (existing or {}).get("corrected_translation") or orig_trans
        corr_tr = st.text_area("__corr_tr", value=default_corr_tr, height=110,
                                label_visibility="collapsed", key=f"ann_corr_tr_{report_id}")

        if st.button(">  Save Translation Annotation", type="primary", key=f"save_trans_{report_id}"):
            ann = {
                "report_id":             report_id,
                "annotated_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "corrected_translation": corr_tr,
                "translation_changed":   corr_tr.strip() != orig_trans.strip(),
            }
            if existing:
                for k in ("corrected_transcript","transcript_changed",
                          "corrected_language","language_changed",
                          "corrected_who","corrected_what","corrected_where",
                          "corrected_when","corrected_assessment",
                          "corrected_threat_level","isum_changed","notes"):
                    if existing.get(k) is not None:
                        ann.setdefault(k, existing[k])
            db.save_annotation(ann)
            st.success(f"Translation annotation saved for {report_id}.")

    # ── ISUM / 5W Fields ──────────────────────────────────────────────────────
    with ann_t3:
        sechdr("ISUM Field Corrections")
        st.markdown(
            '<div class="mono-txt" style="color:var(--text-secondary);font-size:0.78rem;'
            'margin-bottom:0.8rem">Edit any field that is wrong or incomplete. '
            'Leave unchanged if correct.</div>',
            unsafe_allow_html=True,
        )

        w1, w2 = st.columns(2)
        defaults = {
            "who":        (existing or {}).get("corrected_who")        or isum.get("who",""),
            "what":       (existing or {}).get("corrected_what")       or isum.get("what",""),
            "where":      (existing or {}).get("corrected_where")      or isum.get("where",""),
            "when":       (existing or {}).get("corrected_when")       or isum.get("when",""),
            "assessment": (existing or {}).get("corrected_assessment") or isum.get("assessment",""),
        }

        with w1:
            st.markdown('<div class="isum-lbl">WHO — Actors / Callsigns / Units</div>',
                        unsafe_allow_html=True)
            corr_who   = st.text_area("", value=defaults["who"],   height=75, key=f"ann_who_{report_id}")
            st.markdown('<div class="isum-lbl">WHERE — Location / Grid Ref / Direction</div>',
                        unsafe_allow_html=True)
            corr_where = st.text_area("", value=defaults["where"], height=75, key=f"ann_where_{report_id}")
            st.markdown('<div class="isum-lbl">WHEN — Temporal References</div>',
                        unsafe_allow_html=True)
            corr_when  = st.text_area("", value=defaults["when"],  height=75, key=f"ann_when_{report_id}")

        with w2:
            st.markdown('<div class="isum-lbl">WHAT — Activity / Events / Orders</div>',
                        unsafe_allow_html=True)
            corr_what  = st.text_area("", value=defaults["what"],  height=75, key=f"ann_what_{report_id}")
            st.markdown('<div class="isum-lbl">ASSESSMENT — Intelligence Value</div>',
                        unsafe_allow_html=True)
            corr_assessment = st.text_area("", value=defaults["assessment"],
                                            height=115, key=f"ann_assess_{report_id}")

        thr_opts    = ["CLEAR","LOW","MEDIUM","HIGH","CRITICAL"]
        curr_threat = isum.get("threat_level","CLEAR")
        default_thr = thr_opts.index(curr_threat) if curr_threat in thr_opts else 0
        corr_threat = st.selectbox("Corrected Threat Level",
                                    thr_opts, index=default_thr, key=f"ann_threat_{report_id}")

        isum_orig = {k: isum.get(k,"") for k in ("who","what","where","when","assessment")}
        isum_corr = {"who":corr_who,"what":corr_what,"where":corr_where,
                     "when":corr_when,"assessment":corr_assessment}
        isum_changed = (isum_corr != isum_orig or corr_threat != curr_threat)

        if st.button(">  Save ISUM Annotation", type="primary", key="save_isum_ann"):
            ann = {
                "report_id":              report_id,
                "annotated_at":           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "corrected_who":          corr_who,
                "corrected_what":         corr_what,
                "corrected_where":        corr_where,
                "corrected_when":         corr_when,
                "corrected_assessment":   corr_assessment,
                "corrected_threat_level": corr_threat,
                "isum_changed":           isum_changed,
            }
            if existing:
                for k in ("corrected_transcript","transcript_changed",
                          "corrected_translation","translation_changed",
                          "corrected_language","language_changed","notes"):
                    if existing.get(k) is not None:
                        ann.setdefault(k, existing[k])
            db.save_annotation(ann)
            st.success(f"ISUM annotation saved for {report_id}.")

    # ── Export Dataset ────────────────────────────────────────────────────────
    with ann_t4:
        sechdr("Export Training Dataset")

        if ann_stats["total"] == 0:
            st.markdown(
                '<div class="mono-txt" style="color:var(--text-secondary);padding:1rem 0">'
                'No annotations yet. Annotate intercepts in the tabs above to build the training dataset.</div>',
                unsafe_allow_html=True,
            )
        else:
            dataset = db.export_training_data()
            today   = datetime.now(timezone.utc).strftime("%Y%m%d")

            # ── Summary card ──────────────────────────────────────────────────
            lang_rows = "".join(
                f'<div class="seg-row"><span class="seg-ts">{k.upper()}</span>'
                f'<span>{v} samples</span></div>'
                for k, v in ann_stats["by_language"].items()
            )
            _asr_n   = len(dataset["asr"])
            _tran_n  = len(dataset["translation"])
            _isum_n  = len(dataset["isum"])
            st.markdown(
                f'<div class="isum-card">'
                f'<div class="isum-lbl">Dataset Summary</div>'
                f'<div class="seg-row"><span class="seg-ts">Total annotations</span>'
                f'<span style="color:#00ff88;font-weight:700">{ann_stats["total"]}</span></div>'
                f'<div class="seg-row"><span class="seg-ts">ASR corrections</span>'
                f'<span style="color:#00ff88">{_asr_n}</span></div>'
                f'<div class="seg-row"><span class="seg-ts">Translation corrections</span>'
                f'<span style="color:#00aaff">{_tran_n}</span></div>'
                f'<div class="seg-row"><span class="seg-ts">ISUM corrections</span>'
                f'<span style="color:#ffaa00">{_isum_n}</span></div>'
                f'<div class="isum-lbl" style="margin-top:0.8rem">By Language</div>'
                f'{lang_rows}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Full bundle ───────────────────────────────────────────────────
            sechdr("Full Bundle")
            st.download_button(
                "v  Full Training Dataset (JSON)",
                data=json.dumps(dataset, indent=2, ensure_ascii=False),
                file_name=f"vani_training_{today}.json",
                mime="application/json",
                key="dl_training_full",
            )

            # ── Per-split downloads ───────────────────────────────────────────
            sechdr("Per-Split Downloads")
            sp1, sp2, sp3 = st.columns(3)

            with sp1:
                if _asr_n:
                    st.download_button(
                        f"v  ASR ({_asr_n} samples)",
                        data=json.dumps(dataset["asr"], indent=2, ensure_ascii=False),
                        file_name=f"vani_asr_{today}.json",
                        mime="application/json",
                        key="dl_asr_split",
                        help="audio_file + corrected_transcript pairs for Whisper fine-tuning",
                    )
                else:
                    st.markdown(
                        '<div class="mono-txt" style="color:#555f6a;font-size:0.78rem">No ASR corrections yet.</div>',
                        unsafe_allow_html=True,
                    )

            with sp2:
                if _tran_n:
                    st.download_button(
                        f"v  Translation ({_tran_n} samples)",
                        data=json.dumps(dataset["translation"], indent=2, ensure_ascii=False),
                        file_name=f"vani_translation_{today}.json",
                        mime="application/json",
                        key="dl_trans_split",
                        help="source_text + corrected_translation pairs for NLLB / IndicTrans2",
                    )
                else:
                    st.markdown(
                        '<div class="mono-txt" style="color:#555f6a;font-size:0.78rem">No translation corrections yet.</div>',
                        unsafe_allow_html=True,
                    )

            with sp3:
                if _isum_n:
                    st.download_button(
                        f"v  ISUM (JSON, {_isum_n} samples)",
                        data=json.dumps(dataset["isum"], indent=2, ensure_ascii=False),
                        file_name=f"vani_isum_{today}.json",
                        mime="application/json",
                        key="dl_isum_split",
                    )
                else:
                    st.markdown(
                        '<div class="mono-txt" style="color:#555f6a;font-size:0.78rem">No ISUM corrections yet.</div>',
                        unsafe_allow_html=True,
                    )

            # ── Qwen SFT JSONL ────────────────────────────────────────────────
            if _isum_n:
                sechdr("Qwen2.5 SFT Format (JSONL)")
                _sft_lines = []
                for _s in dataset["isum"]:
                    _sft_lines.append(json.dumps({
                        "instruction": (
                            "You are a SIGINT analyst. Generate a structured military intelligence "
                            "summary (ISUM) from this radio intercept.\n\n"
                            f"Language: {(_s.get('language') or 'unknown').upper()}\n"
                            f"Transcript: {(_s.get('transcript') or '')[:500]}\n"
                            f"Translation: {(_s.get('translation') or '')[:500]}"
                        ),
                        "output": json.dumps(_s.get("isum", {}), ensure_ascii=False),
                    }, ensure_ascii=False))
                _sft_jsonl = "\n".join(_sft_lines)
                st.download_button(
                    f"v  Qwen SFT JSONL ({_isum_n} samples)",
                    data=_sft_jsonl,
                    file_name=f"vani_isum_sft_{today}.jsonl",
                    mime="application/jsonlines",
                    key="dl_qwen_sft",
                    help="instruction/output pairs formatted for TRL SFTTrainer or Axolotl",
                )
                st.markdown(
                    '<div class="mono-txt" style="color:var(--text-secondary);font-size:0.73rem;margin-top:0.5rem">'
                    'Load with: <code>datasets.load_dataset("json", data_files="*.jsonl")</code><br>'
                    'Train with: TRL SFTTrainer · Axolotl · LLaMA-Factory · unsloth</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="mono-txt" style="color:var(--text-secondary);font-size:0.73rem;margin-top:1rem">'
                'Target 1,000+ corrections per language before fine-tuning for meaningful gains.</div>',
                unsafe_allow_html=True,
            )

    # ── Keyword Editor ────────────────────────────────────────────────────────
    with ann_t5:
        sechdr("Keyword Dictionary Editor")
        from utils import ROOT as _ROOT
        _kw_path = _ROOT / "alerts" / "keyword_dictionary.json"

        try:
            with open(_kw_path, "r", encoding="utf-8") as _kf:
                _kw_dict = json.load(_kf)
        except Exception as _ke:
            st.error(f"Could not load {_kw_path.name}: {_ke}")
        else:
            _cats    = _kw_dict.get("categories", {})
            _tot_kw  = sum(
                len(_kws)
                for _cd in _cats.values()
                for _kws in _cd.get("keywords", {}).values()
            )
            st.markdown(
                f'<div class="mono-txt" style="color:#8a9aaa;margin-bottom:0.8rem">'
                f'{len(_cats)} categories &nbsp;·&nbsp; {_tot_kw} total keywords &nbsp;·&nbsp; '
                f'{_kw_path.name}</div>',
                unsafe_allow_html=True,
            )

            _sev_opts    = ["critical", "high", "medium", "low"]
            _sev_col_map = {"critical":"#ff3355","high":"#ff6600","medium":"#ffaa00","low":"#00aaff"}

            for _cat_name, _cat_data in list(_cats.items()):
                _sev   = _cat_data.get("severity", "medium")
                _sev_c = _sev_col_map.get(_sev, "#8a9aaa")
                with st.expander(
                    f'{_cat_name}  '
                    f'[{_sev.upper()}]'
                ):
                    _erow1, _erow2 = st.columns([3, 1])
                    with _erow1:
                        st.selectbox(
                            "Severity",
                            _sev_opts,
                            index=_sev_opts.index(_sev) if _sev in _sev_opts else 1,
                            key=f"kw_sev_{_cat_name}",
                        )
                    with _erow2:
                        if st.button(
                            "Delete", key=f"kw_del_{_cat_name}",
                            help=f"Remove '{_cat_name}' category entirely",
                        ):
                            del _kw_dict["categories"][_cat_name]
                            _kw_dict.setdefault("_meta", {})["last_updated"] = \
                                datetime.now(timezone.utc).strftime("%Y-%m")
                            with open(_kw_path, "w", encoding="utf-8") as _wf:
                                json.dump(_kw_dict, _wf, indent=2, ensure_ascii=False)
                            st.success(f"Deleted '{_cat_name}'.")
                            st.rerun()

                    for _lang, _kws in _cat_data.get("keywords", {}).items():
                        st.markdown(
                            f'<div class="isum-lbl" style="margin-top:0.5rem">'
                            f'{_lang.upper()} — {len(_kws)} keywords</div>',
                            unsafe_allow_html=True,
                        )
                        st.text_area(
                            "",
                            value="\n".join(_kws),
                            height=min(140, max(60, len(_kws) * 22)),
                            key=f"kw_{_cat_name}_{_lang}",
                            label_visibility="collapsed",
                            help="One keyword per line. Remove a line to delete that keyword.",
                        )

            # ── Add new category ──────────────────────────────────────────────
            with st.expander("+ Add New Category"):
                _nc_name = st.text_input(
                    "Category ID (lowercase, no spaces)", key="kw_nc_name",
                    placeholder="e.g. ied_vbied",
                )
                _nc_sev  = st.selectbox("Severity", _sev_opts, key="kw_nc_sev")
                _nc_desc = st.text_input("Description (optional)", key="kw_nc_desc")
                _nc_kws  = st.text_area(
                    "English keywords (one per line)", height=80, key="kw_nc_kws"
                )
                if st.button(">  Add Category", key="kw_add_cat_btn"):
                    _nc_id = _nc_name.strip().lower().replace(" ", "_")
                    if not _nc_id:
                        st.error("Enter a category ID.")
                    elif _nc_id in _cats:
                        st.error(f"'{_nc_id}' already exists.")
                    else:
                        _kw_dict["categories"][_nc_id] = {
                            "severity":    _nc_sev,
                            "description": _nc_desc.strip(),
                            "keywords": {
                                "en": [_k.strip() for _k in _nc_kws.splitlines() if _k.strip()]
                            },
                        }
                        _kw_dict.setdefault("_meta", {})["last_updated"] = \
                            datetime.now(timezone.utc).strftime("%Y-%m")
                        with open(_kw_path, "w", encoding="utf-8") as _wf:
                            json.dump(_kw_dict, _wf, indent=2, ensure_ascii=False)
                        st.success(f"Category '{_nc_id}' added. Changes apply on next pipeline run.")
                        st.rerun()

            # ── Save all edits ────────────────────────────────────────────────
            if st.button(">  Save All Changes", type="primary", key="kw_save_all"):
                for _cat_name, _cat_data in _cats.items():
                    _cat_data["severity"] = st.session_state.get(
                        f"kw_sev_{_cat_name}", _cat_data.get("severity")
                    )
                    for _lang in list(_cat_data.get("keywords", {}).keys()):
                        _raw = st.session_state.get(f"kw_{_cat_name}_{_lang}", "")
                        _cat_data["keywords"][_lang] = [
                            _k.strip() for _k in _raw.splitlines() if _k.strip()
                        ]
                _kw_dict.setdefault("_meta", {})["last_updated"] = \
                    datetime.now(timezone.utc).strftime("%Y-%m")
                with open(_kw_path, "w", encoding="utf-8") as _wf:
                    json.dump(_kw_dict, _wf, indent=2, ensure_ascii=False)
                st.success("Dictionary saved. Changes apply on next pipeline run.")

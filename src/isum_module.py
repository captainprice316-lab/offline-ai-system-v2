"""
isum_module.py – Intelligence Summary (ISUM) Generator
--------------------------------------------------------
This module was entirely missing from the original system.
It is the core military value-add of the VANI project.

Generates structured intelligence summaries from processed intercept data,
following a simplified military reporting format (Who / What / Where / When /
Significance).

Two modes:
  1. rule_based (default, works on 8GB CPU) – template-driven NLP
  2. llm        (lab server) – uses a local quantized LLM via llama-cpp-python
                               or transformers (swap in when hardware allows)
"""

from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional
import re

# ── Lazy spaCy NER loader (shared with geo_module) ────────────────────────────
_nlp_isum = None

def _get_nlp():
    global _nlp_isum
    if _nlp_isum is None:
        try:
            import spacy
            _nlp_isum = spacy.load(
                "en_core_web_sm",
                disable=["parser", "lemmatizer", "tagger"],
            )
        except Exception:
            _nlp_isum = False   # mark unavailable so we don't retry
    return _nlp_isum if _nlp_isum else None


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ISUMReport:
    # Header
    report_id:         str
    timestamp_utc:     str
    audio_file:        str
    processing_time_s: float

    # Signal Intelligence fields
    language_detected:  str
    language_confidence: float
    language_uncertain: bool

    # 5W summary
    who:        str    # actors identified
    what:       str    # activity detected
    where:      str    # location indicators
    when:       str    # temporal indicators
    assessment: str    # overall intelligence value

    # Threat
    threat_level:     str    # CRITICAL / HIGH / MEDIUM / LOW / CLEAR
    top_categories:   list
    keyword_alerts:   list

    # Text
    transcript_snippet: str  # first 200 chars for quick reference
    translation_snippet: str

    # Source reliability
    source_confidence_pct: int  # mean ASR segment confidence as 0-100 integer

    # Quality flags
    confidence_flags: list   # list of human-review warnings


# ── Main generator ─────────────────────────────────────────────────────────────

class ISUMGenerator:

    def __init__(self, cfg: dict = None, device: str = "cpu"):
        cfg = cfg or {}
        self.mode          = cfg.get("model", "rule_based")
        self.max_sentences = cfg.get("max_summary_sentences", 4)
        self.device        = device
        self.ollama_url    = cfg.get("ollama_url", "http://localhost:11434").rstrip("/")
        self.ollama_model  = cfg.get("ollama_model", "gemma3:4b")

    def generate(self, pipeline_result: dict, processing_time_s: float = 0.0,
                 qwen_model_path: str = None) -> dict:
        # Try Ollama first (gemma3:4b or configured model)
        if self.mode == "ollama":
            try:
                llm_fields = self._ollama_isum(pipeline_result)
                if llm_fields:
                    report = self._rule_based_isum(pipeline_result, processing_time_s)
                    for k in ("who", "what", "where", "when", "assessment"):
                        if llm_fields.get(k):
                            setattr(report, k, llm_fields[k])
                    result = asdict(report)
                    result["isum_mode"] = "ollama"
                    return result
            except Exception as e:
                import warnings
                warnings.warn(f"Ollama ISUM failed, falling back: {e}")

        # Try Qwen if path provided and mode is not forced rule_based
        if qwen_model_path and self.mode != "rule_based":
            try:
                llm_fields = self._qwen_isum(pipeline_result, qwen_model_path)
                if llm_fields:
                    report = self._rule_based_isum(pipeline_result, processing_time_s)
                    for k in ("who", "what", "where", "when", "assessment"):
                        if llm_fields.get(k):
                            setattr(report, k, llm_fields[k])
                    report.assessment = llm_fields.get("assessment", report.assessment)
                    result = asdict(report)
                    result["isum_mode"] = "llm"
                    return result
            except Exception as e:
                import warnings
                warnings.warn(f"Qwen ISUM failed, falling back to rule-based: {e}")

        report = self._rule_based_isum(pipeline_result, processing_time_s)
        result = asdict(report)
        result["isum_mode"] = "rule_based"
        return result

    # ── Ollama LLM ISUM (gemma3:4b or configured model) ───────────────────────

    def _ollama_isum(self, r: dict) -> dict:
        """Call Ollama REST API for structured ISUM. Returns 5W dict or None."""
        import urllib.request, urllib.error, json as _json, re

        transcript   = (r.get("transcript") or "")[:800]
        trans_obj    = r.get("translation", {})
        translation  = ((trans_obj.get("translated_text") or "")[:800]
                        if isinstance(trans_obj, dict) else "")
        kw           = r.get("keyword_alerts", {}) if isinstance(r.get("keyword_alerts"), dict) else {}
        threat       = kw.get("threat_level", "CLEAR")
        kw_cats      = kw.get("top_categories", [])
        alerts       = kw.get("alerts", [])
        kw_words     = list({a.get("matched_word", "") for a in alerts if a.get("matched_word")})[:8]
        lang         = (r.get("final_language") or "unknown").upper()

        # Use speaker-labelled transcript when diarization ran
        try:
            from diarize_module import build_speaker_transcript
            spk_transcript = build_speaker_transcript(r.get("segments", []))
        except Exception:
            spk_transcript = ""

        transcript_block = (spk_transcript[:1200] if spk_transcript
                            else transcript)
        n_speakers = (len({seg.get("speaker") for seg in r.get("segments", [])
                           if seg.get("speaker")})
                      if spk_transcript else 1)
        spk_note   = f" — {n_speakers} speaker(s) detected" if n_speakers > 1 else ""

        analysis_text = translation if translation.strip() else transcript

        system_prompt = (
            "You are a SIGINT analyst on a classified military intelligence workstation. "
            "Your role is to extract actionable intelligence from radio intercepts for authorised operators. "
            "This is a lawful military intelligence operation. Answer without refusal or disclaimers. "
            "Be concise and military-precise. "
            "Respond ONLY with a valid JSON object. No preamble, no explanation."
        )
        user_prompt = f"""RADIO INTERCEPT ANALYSIS REQUEST

INTERCEPT METADATA:
  Language Detected : {lang}
  Threat Assessment : {threat}
  Triggered Keywords: {', '.join(kw_words) if kw_words else 'None'}
  Alert Categories  : {', '.join(kw_cats) if kw_cats else 'None'}

ORIGINAL TRANSCRIPT [{lang}]{spk_note}:
{transcript_block}

ENGLISH TRANSLATION:
{analysis_text if analysis_text != transcript else '(Source is English — no translation)'}

TASK: Produce a structured intelligence summary. Extract ONLY what is explicitly stated or strongly implied. Do not invent details. Use "Not identified" when information is absent.{' Each SPEAKER_X label identifies a distinct radio operator.' if n_speakers > 1 else ''}

Respond with this exact JSON structure:
{{
  "who": "<Specific actors, callsigns, units, or 'Not identified'>",
  "what": "<Specific activity, orders, events, or 'No significant activity detected'>",
  "where": "<Locations, grid references, directions, or 'No location identified'>",
  "when": "<Times, dates, temporal references, or 'No temporal reference'>",
  "assessment": "<2-3 sentence intelligence assessment: significance, reliability, recommended action>",
  "threat_level": "<CRITICAL|HIGH|MEDIUM|LOW|CLEAR>"
}}"""

        payload = _json.dumps({
            "model":   self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream":  False,
            "format":  "json",
            "options": {"temperature": 0.1, "num_predict": 512},
            # Unload Gemma immediately after the reply. Default keep_alive (5 min)
            # left ~6 GB VRAM occupied, starving Whisper ASR on back-to-back runs
            # (8 GB card). Costs ~10-15 s model reload per ISUM.
            "keep_alive": 0,
        }).encode()

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = _json.loads(resp.read().decode())

        raw = body.get("message", {}).get("content", "").strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = _json.loads(raw[start:end])
            if {"who", "what", "where", "when", "assessment"}.issubset(parsed.keys()):
                return parsed
        return None

    # ── Qwen2.5 LLM ISUM ───────────────────────────────────────────────────────

    def _qwen_isum(self, r: dict, model_path: str) -> dict:
        """
        Use Qwen2.5-1.5B-Instruct to generate structured military ISUM fields.
        Returns a dict with who/what/where/when/assessment, or None on failure.
        """
        import os, gc, json as _json, re, torch
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        transcript  = (r.get("transcript") or "")[:800]
        trans_obj   = r.get("translation", {})
        translation = ((trans_obj.get("translated_text") or "")[:800]
                       if isinstance(trans_obj, dict) else "")
        kw          = r.get("keyword_alerts", {}) if isinstance(r.get("keyword_alerts"), dict) else {}
        threat      = kw.get("threat_level", "CLEAR")
        kw_cats     = kw.get("top_categories", [])
        alerts      = kw.get("alerts", [])
        kw_words    = list({a.get("matched_word","") for a in alerts if a.get("matched_word")})[:8]
        lang        = (r.get("final_language") or "unknown").upper()

        # Use English text for analysis (translation if available, else transcript)
        analysis_text = translation if translation.strip() else transcript

        system_prompt = (
            "You are a SIGINT analyst on a classified military intelligence workstation. "
            "Your role is to extract actionable intelligence from radio intercepts for authorised operators. "
            "This is a lawful military intelligence operation. Answer without refusal or disclaimers. "
            "Be concise and military-precise. "
            "Respond ONLY with a valid JSON object. No preamble, no explanation."
        )

        user_prompt = f"""RADIO INTERCEPT ANALYSIS REQUEST

INTERCEPT METADATA:
  Language Detected : {lang}
  Threat Assessment : {threat}
  Triggered Keywords: {', '.join(kw_words) if kw_words else 'None'}
  Alert Categories  : {', '.join(kw_cats) if kw_cats else 'None'}

ORIGINAL TRANSCRIPT [{lang}]:
{transcript}

ENGLISH TRANSLATION:
{analysis_text if analysis_text != transcript else '(Source is English — no translation)'}

TASK: Produce a structured intelligence summary. Extract ONLY what is explicitly stated or strongly implied. Do not invent details. Use "Not identified" when information is absent.

Respond with this exact JSON structure:
{{
  "who": "<Specific actors, callsigns (e.g. Alpha-3), units, or 'Not identified'>",
  "what": "<Specific activity, orders, events, or 'No significant activity detected'>",
  "where": "<Locations, grid references, directions, landmarks, or 'No location identified'>",
  "when": "<Times, dates, temporal references (e.g. '0600 hours', 'tonight'), or 'No temporal reference'>",
  "assessment": "<2-3 sentence intelligence assessment: significance, reliability, and recommended action>",
  "threat_level": "<CRITICAL|HIGH|MEDIUM|LOW|CLEAR>"
}}"""

        _device = self.device if self.device in ("cpu", "cuda", "mps") else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model     = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True,
            torch_dtype=torch.float32,
            device_map=_device,
        )
        model.eval()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            text   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = {k: v.to(_device) for k, v in tokenizer(text, return_tensors="pt").items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,        # ignored when do_sample=False
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            raw = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Extract JSON — handle markdown code blocks too
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(raw[start:end])
                # Validate required keys present
                required = {"who", "what", "where", "when", "assessment"}
                if required.issubset(parsed.keys()):
                    return parsed
            return None

        except Exception:
            return None
        finally:
            del model, tokenizer
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    # ── Rule-based ISUM ────────────────────────────────────────────────────────

    def _rule_based_isum(self, r: dict, elapsed: float) -> ISUMReport:
        transcript   = r.get("transcript", "")
        translation  = r.get("translation", {})
        trans_text   = translation.get("translated_text", "") if isinstance(translation, dict) else str(translation)
        kw           = r.get("keyword_alerts", {})
        alerts       = kw.get("alerts", []) if isinstance(kw, dict) else []
        threat       = kw.get("threat_level", "CLEAR") if isinstance(kw, dict) else "CLEAR"
        top_cats     = kw.get("top_categories", []) if isinstance(kw, dict) else []
        lang_info    = r.get("final_language", "unknown")
        lang_conf    = r.get("route_confidence", 0.0)
        uncertain    = r.get("uncertain", False)

        # Use translation for English extraction; fall back to transcript
        analysis_text = trans_text if trans_text.strip() else transcript
        is_zh = str(lang_info).lower().startswith("zh") or str(lang_info).lower() == "chinese"

        who   = self._extract_who(analysis_text, alerts, transcript if is_zh else "")
        what  = self._extract_what(analysis_text, alerts, top_cats)
        where = self._extract_where(analysis_text, alerts, transcript if is_zh else "")
        when  = self._extract_when(analysis_text, transcript if is_zh else "")
        assessment = self._make_assessment(threat, top_cats, lang_conf, uncertain)

        flags = self._quality_flags(r, uncertain, lang_conf)

        segs = r.get("segments", [])
        if segs:
            src_conf_pct = round(sum(s.get("confidence", 0) for s in segs) / len(segs) * 100)
        else:
            src_conf_pct = 0

        return ISUMReport(
            report_id           = _gen_report_id(),
            timestamp_utc       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            audio_file          = r.get("audio_file", ""),
            processing_time_s   = round(elapsed, 2),
            language_detected   = lang_info,
            language_confidence = round(lang_conf, 3),
            language_uncertain  = uncertain,
            who                 = who,
            what                = what,
            where               = where,
            when                = when,
            assessment          = assessment,
            threat_level        = threat,
            top_categories      = top_cats[:5],
            keyword_alerts      = alerts[:20],   # cap for readability
            transcript_snippet  = transcript[:200] + ("…" if len(transcript) > 200 else ""),
            translation_snippet = trans_text[:200] + ("…" if len(trans_text) > 200 else ""),
            source_confidence_pct = src_conf_pct,
            confidence_flags    = flags,
        )

    # ── Extraction helpers ─────────────────────────────────────────────────────

    def _extract_who(self, text: str, alerts: list, zh_text: str = "") -> str:
        actors = []

        # Check keyword alerts for actor categories
        enemy_hits = [a for a in alerts if a.get("category") == "enemy_activity"]
        if enemy_hits:
            words = list({a["matched_word"] for a in enemy_hits})[:3]
            actors.append(f"Hostile elements mentioned: {', '.join(words)}")

        # NATO phonetic callsigns: "Alpha 3", "Bravo 2", etc.
        callsigns = re.findall(
            r"\b(?:alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|juliet|"
            r"kilo|lima|mike|november|oscar|papa|quebec|romeo|sierra|tango|uniform|"
            r"victor|whiskey|x-ray|yankee|zulu)\s*\d*\b",
            text, re.IGNORECASE
        )
        if callsigns:
            actors.append(f"Callsigns: {', '.join(set(callsigns[:4]))}")

        # Numeric/alphanumeric callsigns: TF-7, SF-3, B-21, OP-4
        num_callsigns = re.findall(r"\b[A-Z]{1,3}[-\s]\d{1,3}\b", text)
        if num_callsigns:
            actors.append(f"Unit designators: {', '.join(set(num_callsigns[:4]))}")

        # Unit / formation names
        units = re.findall(
            r"\b(?:unit|team|squad|section|platoon|company|battalion|brigade|"
            r"regiment|force|task force|commando|rangers|special forces|SF|QRF|"
            r"strike team|patrol)\s+[\w\d-]+",
            text, re.IGNORECASE
        )
        if units:
            actors.append(f"Units: {', '.join(set(units[:3]))}")

        # Military ranks / titles (English + transliterations + PLA-specific)
        ranks = re.findall(
            r"\b(?:captain|major|colonel|senior colonel|general|lieutenant general|"
            r"major general|brigadier general|lieutenant|commander|officer|"
            r"sergeant|corporal|private|political commissar|commissar|"
            r"chief of staff|havildar|subedar|naik|sepoy|jawan|sipahi|"
            r"sainik|afsar|sahab)\b",
            text, re.IGNORECASE
        )
        if ranks:
            actors.append(f"Ranks/titles: {', '.join(set(ranks[:3]))}")

        # PLA unit structure terms (English translation)
        pla_units = re.findall(
            r"\b(?:theater command|group army|military (?:region|district|zone)|"
            r"border defense (?:regiment|brigade|battalion)|"
            r"combined arms (?:brigade|battalion)|"
            r"rocket force|strategic support|joint logistics)\b",
            text, re.IGNORECASE
        )
        if pla_units:
            actors.append(f"PLA formations: {', '.join(dict.fromkeys(pla_units[:3]))}")

        # PLA 5-digit unit cover designators: "Unit 61398", "61398 Unit"
        pla_codes = re.findall(r"\b(?:unit\s+)?\d{5}\s*(?:unit|force|troops?)?\b", text, re.IGNORECASE)
        pla_codes = [c.strip() for c in pla_codes if re.search(r"\d{5}", c)]
        if pla_codes:
            actors.append(f"PLA unit codes: {', '.join(dict.fromkeys(pla_codes[:3]))}")

        # ── Chinese-script patterns (raw transcript) ───────────────────────────
        if zh_text:
            # Numeric callsigns: 一号/二号/三号 etc. (Number One/Two/Three)
            zh_callsigns = re.findall(
                r"[一二三四五六七八九十百千]+号", zh_text
            )
            if zh_callsigns:
                actors.append(f"Chinese callsigns: {', '.join(dict.fromkeys(zh_callsigns[:4]))}")

            # PLA unit designators: 第72集团军, 第3师, XX旅, etc.
            zh_units = re.findall(
                r"第\d+\s*(?:集团军|战区|军区|军|师|旅|团|营|连|排|班)", zh_text
            )
            zh_units += re.findall(r"\d+\s*(?:部队|集团军|军区|战区)", zh_text)
            if zh_units:
                actors.append(f"PLA units (Chinese): {', '.join(dict.fromkeys(zh_units[:3]))}")

            # PLA ranks in Chinese characters
            zh_ranks = re.findall(
                r"(?:上将|中将|少将|大校|上校|中校|少校|上尉|中尉|少尉|"
                r"上士|中士|下士|上等兵|列兵|政委|司令|参谋长)", zh_text
            )
            if zh_ranks:
                actors.append(f"PLA ranks (Chinese): {', '.join(dict.fromkeys(zh_ranks[:3]))}")

        # Friendly/hostile pronouns as soft indicators
        if re.search(r"\b(?:we are|our forces|our position|our troops)\b", text, re.IGNORECASE):
            actors.append("Friendly forces (self-referenced)")
        if re.search(r"\b(?:they are|their forces|enemy forces|the enemy)\b", text, re.IGNORECASE):
            actors.append("Hostile forces (referenced)")

        # ── NER: PERSON and ORG entities ──────────────────────────────────────
        nlp = _get_nlp()
        if nlp and len(text) >= 10:
            doc = nlp(text[:1000])   # cap to avoid slow processing on long transcripts
            persons = list(dict.fromkeys(
                ent.text.strip() for ent in doc.ents
                if ent.label_ == "PERSON" and len(ent.text.strip()) > 2
            ))
            orgs = list(dict.fromkeys(
                ent.text.strip() for ent in doc.ents
                if ent.label_ == "ORG" and len(ent.text.strip()) > 2
            ))
            if persons:
                actors.append(f"Named persons: {', '.join(persons[:4])}")
            if orgs:
                actors.append(f"Organizations: {', '.join(orgs[:3])}")

        return "; ".join(actors) if actors else "Not identified from intercept."

    def _extract_what(self, text: str, alerts: list, top_cats: list) -> str:
        if not top_cats:
            return "No significant activity detected."

        parts = []
        cat_labels = {
            "attack":          "Attack/fire activity reported",
            "enemy_activity":  "Enemy/hostile presence indicated",
            "movement":        "Troop/unit movement observed",
            "weapons":         "Weapons/equipment referenced",
            "support_request": "Support/reinforcement requested",
            "command":         "Command instructions transmitted",
            "location":        "Location/coordinates exchanged",
            "comms":           "Communication/callsign traffic",
        }
        for cat in top_cats[:3]:
            label = cat_labels.get(cat, cat.replace("_", " ").title())
            parts.append(label)

        return ". ".join(parts) + "."

    def _extract_where(self, text: str, alerts: list, zh_text: str = "") -> str:
        location_alerts = [a for a in alerts if a.get("category") == "location"]
        parts = []

        # Full compass directions (word and abbreviation)
        directions = re.findall(
            r"\b(?:north(?:-?east|-?west)?|south(?:-?east|-?west)?|east|west|"
            r"NE|NW|SE|SW|N|S|E|W)\b",
            text, re.IGNORECASE
        )
        if directions:
            parts.append(f"Direction: {', '.join(set(directions[:4]))}")

        # Grid references: "grid 4523", bare 4–6 digit grids, MGRS "AB1234"
        grids = re.findall(r"\bgrid\s+[\w\d]+\b", text, re.IGNORECASE)
        mgrs  = re.findall(r"\b[A-Z]{2}\s*\d{4,6}\b", text)
        bare  = re.findall(r"\b\d{4,6}\b", text)
        all_grids = grids + mgrs + bare[:2]  # cap bare numerics to avoid noise
        if all_grids:
            parts.append(f"Grid/coords: {', '.join(dict.fromkeys(all_grids[:4]))}")

        # Distance + direction: "5 km north", "200 metres east"
        dist_dir = re.findall(
            r"\b\d+\s*(?:km|kilometer|kilometre|meter|metre|mile|miles)\s*"
            r"(?:north|south|east|west|NE|NW|SE|SW)\b",
            text, re.IGNORECASE
        )
        if dist_dir:
            parts.append(f"Distance/bearing: {', '.join(dist_dir[:3])}")

        # Terrain / infrastructure features followed by a name/identifier
        terrain = re.findall(
            r"\b(?:ridgeline|ridge|valley|nala|river|road|highway|bridge|"
            r"village|town|city|post|bunker|sector|zone|area|hill|peak|"
            r"pass|border|checkpoint|outpost|camp)\s+[\w\d-]+",
            text, re.IGNORECASE
        )
        if terrain:
            parts.append(f"Feature: {', '.join(set(terrain[:3]))}")

        # ── Chinese-script patterns (raw transcript) ───────────────────────────
        if zh_text:
            # Cardinal directions in Chinese characters + compounds
            zh_dirs = re.findall(r"[东西南北][北南东西]?(?:方向?|侧|面)?", zh_text)
            if zh_dirs:
                _dir_map = {"东": "East", "西": "West", "南": "South", "北": "North",
                            "东北": "NE", "西北": "NW", "东南": "SE", "西南": "SW"}
                mapped = [_dir_map.get(d[:2], d) for d in zh_dirs]
                parts.append(f"Direction (Chinese): {', '.join(dict.fromkeys(mapped[:4]))}")

            # Distance in Chinese units: 5公里, 300米, 2里
            zh_dist = re.findall(r"\d+\s*(?:公里|千米|百米|米|里)", zh_text)
            if zh_dist:
                parts.append(f"Distance (Chinese): {', '.join(zh_dist[:3])}")

            # Terrain/location terms in Chinese
            zh_terrain = re.findall(
                r"(?:山脊|山谷|河流|公路|桥梁|村庄|城镇|阵地|哨所|检查站|"
                r"边境|营地|高地|山口|据点|前沿)\s*[\w\d一-鿿]*",
                zh_text
            )
            if zh_terrain:
                parts.append(f"Terrain (Chinese): {', '.join(dict.fromkeys(zh_terrain[:3]))}")

            # Gauss-Krüger / Chinese military grid: X带XXXXXXXX
            zh_grids = re.findall(r"\d+带\d{6,8}", zh_text)
            if zh_grids:
                parts.append(f"Grid (Chinese): {', '.join(zh_grids[:3])}")

        # ── NER: GPE (countries/cities), LOC (geographic), FAC (facilities) ──────
        nlp = _get_nlp()
        if nlp and len(text) >= 10:
            doc = nlp(text[:1000])
            geo_ents = list(dict.fromkeys(
                ent.text.strip() for ent in doc.ents
                if ent.label_ in ("GPE", "LOC", "FAC") and len(ent.text.strip()) > 2
            ))
            if geo_ents:
                parts.append(f"Named locations: {', '.join(geo_ents[:5])}")

        # Fallback: location keyword alerts
        if not parts and location_alerts:
            words = list({a["matched_word"] for a in location_alerts})[:3]
            parts.append(f"Location keywords: {', '.join(words)}")

        return "; ".join(parts) if parts else "No specific location identified."

    def _extract_when(self, text: str, zh_text: str = "") -> str:
        times = []

        # Military time: "0600 hours", "0530 hrs", "2359 hours"
        times += re.findall(r"\b\d{3,4}\s*(?:hours?|hrs?)\b", text, re.IGNORECASE)

        # Clock time: "14:30", "6:00"
        times += re.findall(r"\b\d{1,2}:\d{2}\b", text)

        # H-hour notation: "H-hour", "H+30", "H-15"
        times += re.findall(r"\bH[-+]\s*(?:hour|\d+)\b", text, re.IGNORECASE)

        # Named times of day
        times += re.findall(
            r"\b(?:dawn|dusk|night|midnight|noon|morning|afternoon|evening|"
            r"tonight|today|first light|last light|sunrise|sunset|dark)\b",
            text, re.IGNORECASE
        )

        # Relative time: "in 30 minutes", "in 2 hours"
        times += re.findall(r"\bin\s+\d+\s+(?:minutes?|hours?)\b", text, re.IGNORECASE)

        # After/before event: "after dark", "before sunrise"
        times += re.findall(
            r"\b(?:after|before)\s+(?:dark|sunrise|sunset|dawn|dusk)\b",
            text, re.IGNORECASE
        )

        # Day names
        times += re.findall(
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            text, re.IGNORECASE
        )

        # Date patterns: "12/03", "03-16"
        times += re.findall(r"\b\d{1,2}[/\-]\d{1,2}\b", text)

        # ── Chinese-script time patterns (raw transcript) ──────────────────────
        if zh_text:
            # Hour/minute: 六时, 14时30分, 0600时
            times += re.findall(r"\d{1,4}\s*时(?:\d{1,2}\s*分)?", zh_text)
            # Chinese clock: X点 (X o'clock)
            times += re.findall(r"[零一二三四五六七八九十百千\d]+点(?:半|\d{1,2}分)?", zh_text)
            # Dates: X月X日
            times += re.findall(r"\d{1,2}月\d{1,2}日", zh_text)
            # Today/tonight/tomorrow in Chinese
            times += re.findall(r"(?:今[天晚夜]|明[天晚]|后天|昨[天晚])", zh_text)
            # Named times of day in Chinese
            times += re.findall(
                r"(?:黎明|黄昏|傍晚|夜间|白天|午夜|正午|上午|下午|深夜|拂晓)", zh_text
            )
            # Relative: X分钟后, X小时后
            times += re.findall(r"\d+\s*(?:分钟|小时)(?:后|内|前)", zh_text)

        if times:
            return f"Temporal references: {', '.join(dict.fromkeys(times[:5]))}"
        return "No specific time referenced in intercept."

    def _make_assessment(
        self, threat: str, top_cats: list,
        lang_conf: float, uncertain: bool
    ) -> str:
        parts = []

        # Category-specific intelligence phrases (most specific first)
        cat_phrases = {
            "nuclear_chem_bio": "CBRN/NBC indicators present — escalate to higher command immediately.",
            "attack":           "Active attack or fire mission underway — immediate action required.",
            "enemy_activity":   "Hostile presence or enemy activity confirmed in area of operations.",
            "support_request":  "Fire support or reinforcement requested — assess force disposition.",
            "weapons":          "Weapons, ordnance, or explosive materiel referenced.",
            "movement":         "Troop or unit movement detected — monitor for axis of advance.",
            "command":          "Command-level communication intercepted — assess intent.",
            "location":         "Positional data or coordinates exchanged.",
            "comms":            "Signals/communications traffic — low tactical value.",
        }

        threat_prefix = {
            "CRITICAL": "IMMEDIATE INTELLIGENCE VALUE",
            "HIGH":     "HIGH INTELLIGENCE VALUE",
            "MEDIUM":   "MODERATE INTELLIGENCE VALUE",
            "LOW":      "LOW INTELLIGENCE VALUE",
            "CLEAR":    "NO THREAT INDICATORS detected.",
        }

        prefix = threat_prefix.get(threat, "ASSESSMENT UNAVAILABLE")

        if threat == "CLEAR":
            parts.append(prefix)
        else:
            # Build contextual sentence from top 1–2 categories
            cat_details = [cat_phrases[c] for c in (top_cats or []) if c in cat_phrases]
            if cat_details:
                parts.append(f"{prefix} — {' '.join(cat_details[:2])}")
            else:
                parts.append(f"{prefix} — requires prompt analysis.")

        if uncertain:
            parts.append(
                f"CAUTION: Language identification confidence is low "
                f"({round(lang_conf*100)}%). Translation may be unreliable. "
                f"Recommend human linguist review."
            )
        elif lang_conf < 0.5:
            parts.append("Translation confidence marginal – verify with specialist.")

        return " ".join(parts)

    def _quality_flags(self, r: dict, uncertain: bool, lang_conf: float) -> list:
        flags = []
        if uncertain:
            flags.append("LOW_LANG_CONFIDENCE")
        if lang_conf < 0.50:
            flags.append("TRANSLATION_UNRELIABLE")
        if r.get("whisper_language_probability", 1.0) < 0.60:
            flags.append("ASR_LOW_CONFIDENCE")
        segs = r.get("segments", [])
        if segs:
            avg_conf = sum(s.get("confidence", 0) for s in segs) / len(segs)
            if avg_conf < 0.55:
                flags.append("TRANSCRIPTION_LOW_CONFIDENCE")
        if not r.get("translation", {}).get("success", True):
            flags.append("TRANSLATION_FAILED")
        return flags


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gen_report_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"ISUM-{ts}"

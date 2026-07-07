"""
src/database.py – SQLite transcript storage and retrieval
----------------------------------------------------------
Schema:
  intercepts   – one row per processed audio file
  segments     – timestamped transcript segments linked to intercept
  keyword_alerts – keyword hits linked to intercept
  isums        – intelligence summary linked to intercept

All queries return plain dicts for easy JSON serialisation and
Streamlit display.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as _np
    _TFIDF_AVAILABLE = True
except ImportError:
    _TFIDF_AVAILABLE = False

_ACTOR_CACHE_TTL = 300  # seconds


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS intercepts (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id               TEXT UNIQUE NOT NULL,
    audio_file              TEXT NOT NULL,
    timestamp_utc           TEXT NOT NULL,
    processing_time_s       REAL,
    whisper_language        TEXT,
    whisper_lang_prob       REAL,
    fasttext_language       TEXT,
    fasttext_confidence     REAL,
    final_language          TEXT,
    translation_route       TEXT,
    route_confidence        REAL,
    language_uncertain      INTEGER DEFAULT 0,
    transcript              TEXT,
    translation             TEXT,
    threat_level            TEXT DEFAULT 'CLEAR',
    top_categories          TEXT,   -- JSON array
    confidence_flags        TEXT,   -- JSON array
    total_speech_sec        REAL,
    chunks_created          INTEGER
);

CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intercept_id    INTEGER NOT NULL REFERENCES intercepts(id) ON DELETE CASCADE,
    start_sec       REAL,
    end_sec         REAL,
    text            TEXT,
    confidence      REAL,
    no_speech_prob  REAL,
    chunk_idx       INTEGER
);

CREATE TABLE IF NOT EXISTS keyword_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intercept_id    INTEGER NOT NULL REFERENCES intercepts(id) ON DELETE CASCADE,
    category        TEXT,
    severity        TEXT,
    matched_word    TEXT,
    matched_in      TEXT,
    start_sec       REAL,
    end_sec         REAL,
    segment_text    TEXT
);

CREATE TABLE IF NOT EXISTS isums (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intercept_id    INTEGER NOT NULL REFERENCES intercepts(id) ON DELETE CASCADE,
    report_id       TEXT,
    who_field       TEXT,
    what_field      TEXT,
    where_field     TEXT,
    when_field      TEXT,
    assessment      TEXT,
    threat_level    TEXT,
    transcript_snippet  TEXT,
    translation_snippet TEXT
);

CREATE TABLE IF NOT EXISTS annotations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    intercept_id            INTEGER NOT NULL REFERENCES intercepts(id) ON DELETE CASCADE,
    report_id               TEXT NOT NULL,
    annotated_by            TEXT DEFAULT 'analyst',
    annotated_at            TEXT NOT NULL,

    -- ASR corrections
    corrected_transcript    TEXT,
    transcript_changed      INTEGER DEFAULT 0,

    -- Translation corrections
    corrected_translation   TEXT,
    translation_changed     INTEGER DEFAULT 0,

    -- Language correction
    corrected_language      TEXT,
    language_changed        INTEGER DEFAULT 0,

    -- ISUM corrections
    corrected_who           TEXT,
    corrected_what          TEXT,
    corrected_where         TEXT,
    corrected_when          TEXT,
    corrected_assessment    TEXT,
    corrected_threat_level  TEXT,
    isum_changed            INTEGER DEFAULT 0,

    -- Keyword review
    false_positive_ids      TEXT,   -- JSON array of keyword_alert ids marked wrong
    missed_keywords         TEXT,   -- JSON array of {word, category, severity} analyst added

    -- Quality scores (computed from corrections)
    asr_quality_score       REAL,   -- 1 - WER, if corrected
    notes                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_annotations_intercept ON annotations(intercept_id);
CREATE INDEX IF NOT EXISTS idx_annotations_report    ON annotations(report_id);

CREATE INDEX IF NOT EXISTS idx_intercepts_report_id  ON intercepts(report_id);
CREATE INDEX IF NOT EXISTS idx_intercepts_threat     ON intercepts(threat_level);
CREATE INDEX IF NOT EXISTS idx_intercepts_language   ON intercepts(final_language);
CREATE INDEX IF NOT EXISTS idx_segments_intercept    ON segments(intercept_id);
CREATE INDEX IF NOT EXISTS idx_alerts_intercept      ON keyword_alerts(intercept_id);
CREATE INDEX IF NOT EXISTS idx_alerts_category       ON keyword_alerts(category);

-- Tier 1 auto-metrics saved after every pipeline run
CREATE TABLE IF NOT EXISTS metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    intercept_id     INTEGER REFERENCES intercepts(id) ON DELETE CASCADE,
    report_id        TEXT NOT NULL,
    timestamp_utc    TEXT NOT NULL,
    -- scalar queryable fields
    rtf              REAL,
    rtf_grade        TEXT,
    conf_mean        REAL,
    conf_pct_low     REAL,
    conf_grade       TEXT,
    ensemble_score   REAL,
    ensemble_grade   TEXT,
    isum_score       INTEGER,
    isum_pct         INTEGER,
    mem_peak_mb      REAL,
    vocab_ttr        REAL,
    backtrans_chrf   REAL,
    -- full metrics dict for display
    full_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_report  ON metrics(report_id);
CREATE INDEX IF NOT EXISTS idx_metrics_ts      ON metrics(timestamp_utc);

-- Operator-assigned aliases: resolve a detected callsign/name/place/codeword
-- to a real identity or pseudoname (analyst tradecraft, persists across runs).
CREATE TABLE IF NOT EXISTS aliases (
    term          TEXT PRIMARY KEY COLLATE NOCASE,   -- the detected surface form
    kind          TEXT,                              -- callsign|name|location|codeword|other
    alias         TEXT,                              -- operator-assigned identity/pseudoname
    notes         TEXT,
    updated_utc   TEXT
);

-- FTS5 full-text search index across transcript + translation + ISUM fields.
-- unicode61 tokenizer with diacritic removal handles transliterated Indic text well.
CREATE VIRTUAL TABLE IF NOT EXISTS intercepts_fts USING fts5(
    report_id   UNINDEXED,
    transcript,
    translation,
    isum_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


class TranscriptDB:

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._actor_cache:    Optional[List[dict]] = None
        self._actor_cache_ts: float = 0.0
        self._init_db()

    # ── Init ───────────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            # Migrate: add metrics table if it doesn't exist yet
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intercept_id INTEGER REFERENCES intercepts(id) ON DELETE CASCADE,
                    report_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL,
                    rtf REAL, rtf_grade TEXT,
                    conf_mean REAL, conf_pct_low REAL, conf_grade TEXT,
                    ensemble_score REAL, ensemble_grade TEXT,
                    isum_score INTEGER, isum_pct INTEGER,
                    mem_peak_mb REAL, vocab_ttr REAL, backtrans_chrf REAL,
                    full_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_report ON metrics(report_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_ts     ON metrics(timestamp_utc);
            """)
            # Backfill FTS5 for any records not yet indexed
            try:
                fts_count = conn.execute("SELECT COUNT(*) FROM intercepts_fts").fetchone()[0]
                int_count = conn.execute("SELECT COUNT(*) FROM intercepts").fetchone()[0]
                if int_count > 0 and fts_count < int_count:
                    self._rebuild_fts_index(conn)
            except Exception:
                pass  # FTS5 unavailable on this SQLite build

    def fts5_available(self) -> bool:
        """Return True if this SQLite build includes FTS5."""
        try:
            with self._conn() as conn:
                conn.execute("SELECT count(*) FROM intercepts_fts LIMIT 0")
            return True
        except Exception:
            return False

    def _rebuild_fts_index(self, conn: sqlite3.Connection) -> None:
        """Wipe and repopulate intercepts_fts from the main tables."""
        conn.execute("DELETE FROM intercepts_fts")
        conn.execute("""
            INSERT INTO intercepts_fts(rowid, report_id, transcript, translation, isum_text)
            SELECT i.id,
                   i.report_id,
                   COALESCE(i.transcript, ''),
                   COALESCE(i.translation, ''),
                   COALESCE(s.who_field,'')            || ' ' ||
                   COALESCE(s.what_field,'')           || ' ' ||
                   COALESCE(s.where_field,'')          || ' ' ||
                   COALESCE(s.when_field,'')           || ' ' ||
                   COALESCE(s.assessment,'')           || ' ' ||
                   COALESCE(s.transcript_snippet,'')   || ' ' ||
                   COALESCE(s.translation_snippet,'')
            FROM intercepts i
            LEFT JOIN isums s ON s.intercept_id = i.id
        """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_result(self, result: dict) -> int:
        """
        Save a full pipeline result dict.
        Returns the intercept row id.
        """
        isum      = result.get("isum", {})
        kw        = result.get("keyword_alerts", {})
        trans     = result.get("translation", {})
        trans_txt = trans.get("translated_text", "") if isinstance(trans, dict) else str(trans)

        with self._conn() as conn:
            # ── intercepts table ──────────────────────────────────────────────
            cur = conn.execute("""
                INSERT OR REPLACE INTO intercepts (
                    report_id, audio_file, timestamp_utc, processing_time_s,
                    whisper_language, whisper_lang_prob,
                    fasttext_language, fasttext_confidence,
                    final_language, translation_route, route_confidence,
                    language_uncertain, transcript, translation,
                    threat_level, top_categories, confidence_flags,
                    total_speech_sec, chunks_created
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.get("report_id", ""),
                result.get("audio_file", ""),
                result.get("timestamp_utc", ""),
                result.get("processing_time_s"),
                result.get("whisper_language"),
                result.get("whisper_language_probability"),
                result.get("fasttext_language"),
                result.get("fasttext_confidence"),
                result.get("final_language"),
                result.get("translation_route"),
                result.get("route_confidence"),
                int(result.get("language_uncertain", False)),
                result.get("transcript", ""),
                trans_txt,
                result.get("threat_level", "CLEAR"),
                json.dumps(result.get("top_categories", [])),
                json.dumps(result.get("confidence_flags", [])),
                result.get("total_speech_sec"),
                result.get("chunks_created"),
            ))
            intercept_id = cur.lastrowid

            # ── segments ──────────────────────────────────────────────────────
            for seg in result.get("segments", []):
                conn.execute("""
                    INSERT INTO segments
                    (intercept_id, start_sec, end_sec, text, confidence,
                     no_speech_prob, chunk_idx)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    intercept_id,
                    seg.get("start"),
                    seg.get("end"),
                    seg.get("text", ""),
                    seg.get("confidence"),
                    seg.get("no_speech_prob"),
                    seg.get("chunk_idx"),
                ))

            # ── keyword alerts ────────────────────────────────────────────────
            for alert in kw.get("alerts", []):
                conn.execute("""
                    INSERT INTO keyword_alerts
                    (intercept_id, category, severity, matched_word,
                     matched_in, start_sec, end_sec, segment_text)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    intercept_id,
                    alert.get("category"),
                    alert.get("severity"),
                    alert.get("matched_word"),
                    alert.get("matched_in"),
                    alert.get("start_sec"),
                    alert.get("end_sec"),
                    alert.get("segment_text", ""),
                ))

            # ── isum ──────────────────────────────────────────────────────────
            if isum:
                conn.execute("""
                    INSERT INTO isums
                    (intercept_id, report_id, who_field, what_field,
                     where_field, when_field, assessment, threat_level,
                     transcript_snippet, translation_snippet)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    intercept_id,
                    isum.get("report_id", ""),
                    isum.get("who", ""),
                    isum.get("what", ""),
                    isum.get("where", ""),
                    isum.get("when", ""),
                    isum.get("assessment", ""),
                    isum.get("threat_level", ""),
                    isum.get("transcript_snippet", ""),
                    isum.get("translation_snippet", ""),
                ))

            # ── FTS5 maintenance ──────────────────────────────────────────────
            try:
                # Remove any previous FTS entry for this report_id
                conn.execute(
                    "DELETE FROM intercepts_fts WHERE report_id = ?",
                    (result.get("report_id", ""),)
                )
                isum_text = " ".join(filter(None, [
                    isum.get("who", ""),
                    isum.get("what", ""),
                    isum.get("where", ""),
                    isum.get("when", ""),
                    isum.get("assessment", ""),
                    isum.get("transcript_snippet", ""),
                    isum.get("translation_snippet", ""),
                ])) if isum else ""
                conn.execute(
                    "INSERT INTO intercepts_fts"
                    "(rowid, report_id, transcript, translation, isum_text)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        intercept_id,
                        result.get("report_id", ""),
                        result.get("transcript", ""),
                        trans_txt,
                        isum_text,
                    ),
                )
            except Exception:
                pass  # FTS5 unavailable — silently skip

        self.invalidate_actor_cache()
        return intercept_id

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        keyword:      str,
        language:     Optional[str] = None,
        threat_level: Optional[str] = None,
        limit:        int           = 50,
    ) -> List[dict]:
        """
        Full-text keyword search across transcript + translation + ISUM.
        Optional filters: language, threat_level.
        Returns results ranked by threat severity.
        """
        kw = f"%{keyword.lower()}%"

        query = """
            SELECT
                i.id, i.report_id, i.audio_file, i.timestamp_utc,
                i.final_language, i.threat_level, i.top_categories,
                i.transcript, i.translation, i.confidence_flags,
                i.route_confidence, i.language_uncertain,
                s.report_id   AS isum_report_id,
                s.assessment  AS isum_assessment,
                s.who_field, s.what_field, s.where_field, s.when_field
            FROM intercepts i
            LEFT JOIN isums s ON s.intercept_id = i.id
            WHERE (
                LOWER(i.transcript)  LIKE ?
                OR LOWER(i.translation) LIKE ?
                OR LOWER(s.assessment)  LIKE ?
                OR LOWER(s.what_field)  LIKE ?
            )
        """
        params = [kw, kw, kw, kw]

        if language:
            query  += " AND i.final_language = ?"
            params.append(language)

        if threat_level:
            query  += " AND i.threat_level = ?"
            params.append(threat_level.upper())

        query += """
            ORDER BY
                CASE i.threat_level
                    WHEN 'CRITICAL' THEN 4
                    WHEN 'HIGH'     THEN 3
                    WHEN 'MEDIUM'   THEN 2
                    WHEN 'LOW'      THEN 1
                    ELSE 0
                END DESC,
                i.timestamp_utc DESC
            LIMIT ?
        """
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            d["top_categories"]   = _safe_json(d.get("top_categories"))
            d["confidence_flags"] = _safe_json(d.get("confidence_flags"))
            results.append(d)

        # Batch-fetch matched segments in one query instead of one per row
        if results:
            ids = [d["id"] for d in results]
            placeholders = ",".join("?" * len(ids))
            with self._conn() as conn:
                seg_rows = conn.execute(f"""
                    SELECT intercept_id, start_sec, end_sec, text, confidence
                    FROM segments
                    WHERE intercept_id IN ({placeholders})
                      AND LOWER(text) LIKE ?
                    LIMIT 500
                """, ids + [kw]).fetchall()
            from collections import defaultdict
            segs_by_id: dict = defaultdict(list)
            for s in seg_rows:
                segs_by_id[s["intercept_id"]].append({
                    "start_sec": s["start_sec"], "end_sec": s["end_sec"],
                    "text": s["text"], "confidence": s["confidence"],
                })
            for d in results:
                d["matched_segments"] = segs_by_id[d["id"]][:10]

        return results

    def search_fts(
        self,
        keyword:      str,
        language:     Optional[str] = None,
        threat_level: Optional[str] = None,
        date_from:    Optional[str] = None,   # "YYYY-MM-DD"
        date_to:      Optional[str] = None,   # "YYYY-MM-DD"
        limit:        int           = 50,
    ) -> Tuple[List[dict], dict]:
        """
        FTS5-accelerated full-text search across transcript + translation + ISUM.
        All filters are pushed to SQL — no Python-side post-filtering.
        Falls back to LIKE-based search() if FTS5 is unavailable.
        Returns (results, meta) where meta = {"engine": "fts5"|"like", "elapsed_ms": float}.
        """
        t0 = time.monotonic()

        try:
            # Quote each word so FTS5 treats them as literal terms (implicit AND)
            words = keyword.strip().split()
            if not words:
                return [], {"engine": "fts5", "elapsed_ms": 0.0}
            fts_query = " ".join(f'"{w}"' for w in words)

            query = """
                SELECT
                    i.id, i.report_id, i.audio_file, i.timestamp_utc,
                    i.final_language, i.threat_level, i.top_categories,
                    i.transcript, i.translation, i.confidence_flags,
                    i.route_confidence, i.language_uncertain,
                    s.report_id   AS isum_report_id,
                    s.assessment  AS isum_assessment,
                    s.who_field, s.what_field, s.where_field, s.when_field
                FROM intercepts_fts f
                JOIN intercepts i ON i.id = f.rowid
                LEFT JOIN isums s ON s.intercept_id = i.id
                WHERE intercepts_fts MATCH ?
            """
            params: list = [fts_query]

            if language:
                query  += " AND i.final_language = ?"
                params.append(language)
            if threat_level:
                query  += " AND i.threat_level = ?"
                params.append(threat_level.upper())
            if date_from:
                query  += " AND i.timestamp_utc >= ?"
                params.append(date_from)
            if date_to:
                query  += " AND i.timestamp_utc <= ?"
                params.append(date_to + "T23:59:59Z")

            query += """
                ORDER BY
                    CASE i.threat_level
                        WHEN 'CRITICAL' THEN 4
                        WHEN 'HIGH'     THEN 3
                        WHEN 'MEDIUM'   THEN 2
                        WHEN 'LOW'      THEN 1
                        ELSE 0
                    END DESC,
                    i.timestamp_utc DESC
                LIMIT ?
            """
            params.append(limit)

            with self._conn() as conn:
                rows = conn.execute(query, params).fetchall()

            kw_like = f"%{keyword.lower()}%"
            results = []
            for row in rows:
                d = dict(row)
                d["top_categories"]   = _safe_json(d.get("top_categories"))
                d["confidence_flags"] = _safe_json(d.get("confidence_flags"))
                results.append(d)

            # Batch-fetch matched segments in one query
            if results:
                ids = [d["id"] for d in results]
                placeholders = ",".join("?" * len(ids))
                with self._conn() as conn:
                    seg_rows = conn.execute(f"""
                        SELECT intercept_id, start_sec, end_sec, text, confidence
                        FROM segments
                        WHERE intercept_id IN ({placeholders})
                          AND LOWER(text) LIKE ?
                        LIMIT 500
                    """, ids + [kw_like]).fetchall()
                from collections import defaultdict
                segs_by_id: dict = defaultdict(list)
                for s in seg_rows:
                    segs_by_id[s["intercept_id"]].append({
                        "start_sec": s["start_sec"], "end_sec": s["end_sec"],
                        "text": s["text"], "confidence": s["confidence"],
                    })
                for d in results:
                    d["matched_segments"] = segs_by_id[d["id"]][:10]

            elapsed = round((time.monotonic() - t0) * 1000, 1)
            return results, {"engine": "fts5", "elapsed_ms": elapsed}

        except Exception:
            # FTS5 unavailable or query error — fall back to LIKE search
            results = self.search(keyword, language, threat_level, limit)
            # Apply date filters that LIKE search doesn't handle
            if date_from:
                results = [r for r in results if r.get("timestamp_utc", "") >= date_from]
            if date_to:
                results = [r for r in results
                           if r.get("timestamp_utc", "") <= date_to + "T23:59:59Z"]
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            return results, {"engine": "like", "elapsed_ms": elapsed}

    def semantic_search(
        self,
        query:        str,
        top_k:        int           = 20,
        language:     Optional[str] = None,
        threat_level: Optional[str] = None,
    ) -> Tuple[List[dict], dict]:
        """
        TF-IDF ranked retrieval across translated text + transcript + ISUM assessment.
        Returns results sorted by cosine similarity (highest first).
        Falls back to FTS5 keyword search if sklearn is unavailable.
        Returns (results, meta) in the same format as search_fts().
        """
        if not _TFIDF_AVAILABLE or not query.strip():
            results, meta = self.search_fts(query, language=language,
                                            threat_level=threat_level)
            meta["engine"] = "fts5-fallback"
            return results, meta

        t0 = time.monotonic()

        sql = """
            SELECT i.id, i.report_id, i.audio_file, i.timestamp_utc,
                   i.final_language, i.threat_level, i.top_categories,
                   i.transcript, i.translation, i.confidence_flags,
                   i.route_confidence, i.language_uncertain,
                   s.assessment AS isum_assessment,
                   s.who_field, s.what_field, s.where_field, s.when_field
            FROM intercepts i
            LEFT JOIN isums s ON s.intercept_id = i.id
        """
        params = []
        filters = []
        if language:
            filters.append("i.final_language = ?")
            params.append(language)
        if threat_level:
            filters.append("i.threat_level = ?")
            params.append(threat_level.upper())
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY i.timestamp_utc DESC LIMIT 500"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return [], {"engine": "tfidf", "elapsed_ms": 0.0}

        records = []
        corpus  = []
        for row in rows:
            d = dict(row)
            d["top_categories"]   = _safe_json(d.get("top_categories"))
            d["confidence_flags"] = _safe_json(d.get("confidence_flags"))
            trans = d.get("translation") or ""
            if isinstance(trans, str):
                try:
                    trans = json.loads(trans).get("translated_text", trans)
                except Exception:
                    pass
            doc = " ".join(filter(None, [
                trans,
                d.get("transcript", ""),
                d.get("isum_assessment", ""),
            ]))
            records.append(d)
            corpus.append(doc)

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        try:
            tfidf_matrix = vec.fit_transform(corpus)
            q_vec        = vec.transform([query])
            scores       = cosine_similarity(q_vec, tfidf_matrix).flatten()
        except Exception:
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            return [], {"engine": "tfidf", "elapsed_ms": elapsed}

        top_idx = _np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] < 0.01:
                break
            rec = records[idx].copy()
            rec["similarity_score"] = round(float(scores[idx]), 4)
            rec["matched_segments"] = []
            results.append(rec)

        elapsed = round((time.monotonic() - t0) * 1000, 1)
        return results, {"engine": "tfidf", "elapsed_ms": elapsed}

    def get_all_intercepts(self, limit: int = 100) -> List[dict]:
        """Return recent intercepts for the UI dashboard."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT i.*, s.assessment AS isum_assessment,
                       s.where_field, s.who_field
                FROM intercepts i
                LEFT JOIN isums s ON s.intercept_id = i.id
                ORDER BY i.timestamp_utc DESC
                LIMIT ?
            """, (limit,)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["top_categories"]  = _safe_json(d.get("top_categories"))
            d["confidence_flags"] = _safe_json(d.get("confidence_flags"))
            results.append(d)
        return results

    def get_intercept_by_report_id(self, report_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT i.*, s.who_field, s.what_field, s.where_field,
                       s.when_field, s.assessment AS isum_assessment
                FROM intercepts i
                LEFT JOIN isums s ON s.intercept_id = i.id
                WHERE i.report_id = ?
            """, (report_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["top_categories"]   = _safe_json(d.get("top_categories"))
        d["confidence_flags"] = _safe_json(d.get("confidence_flags"))
        return d

    def invalidate_actor_cache(self) -> None:
        """Force next get_actor_profiles() call to re-query the DB."""
        self._actor_cache = None
        self._actor_cache_ts = 0.0

    def get_actor_profiles(self) -> List[dict]:
        """
        Parse all who_field strings across intercepts and build per-callsign
        actor profiles showing appearances, threat levels, languages and dates.
        Returns list sorted by appearance count descending.
        Result is cached for _ACTOR_CACHE_TTL seconds.
        """
        if (self._actor_cache is not None and
                time.monotonic() - self._actor_cache_ts < _ACTOR_CACHE_TTL):
            return self._actor_cache

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT i.report_id, i.timestamp_utc, i.final_language,
                       i.threat_level, i.audio_file,
                       s.who_field
                FROM intercepts i
                LEFT JOIN isums s ON s.intercept_id = i.id
                WHERE s.who_field IS NOT NULL AND s.who_field != ''
                  AND s.who_field != 'Not identified from intercept.'
                ORDER BY i.timestamp_utc ASC
            """).fetchall()

        # Build actor → appearances index
        from collections import defaultdict
        actors: dict = defaultdict(lambda: {
            "appearances": [],
            "callsign_type": "unknown",
        })

        for row in rows:
            who = (row["who_field"] or "").strip()
            meta = {
                "report_id":   row["report_id"],
                "timestamp":   (row["timestamp_utc"] or "")[:16].replace("T", " "),
                "language":    (row["final_language"] or "?").upper(),
                "threat":      (row["threat_level"] or "CLEAR").upper(),
                "audio_file":  row["audio_file"],
            }
            for segment in who.split(";"):
                segment = segment.strip()
                if not segment:
                    continue
                # Determine type from prefix
                if segment.lower().startswith("callsigns:"):
                    ctype = "callsign"
                    items = segment[len("callsigns:"):].strip()
                elif segment.lower().startswith("unit designators:"):
                    ctype = "unit_designator"
                    items = segment[len("unit designators:"):].strip()
                elif segment.lower().startswith("units:"):
                    ctype = "unit"
                    items = segment[len("units:"):].strip()
                elif segment.lower().startswith("ranks/titles:"):
                    ctype = "rank"
                    items = segment[len("ranks/titles:"):].strip()
                elif "friendly" in segment.lower():
                    ctype = "force_indicator"
                    items = "Friendly forces"
                elif "hostile" in segment.lower() or "enemy" in segment.lower():
                    ctype = "force_indicator"
                    items = "Hostile forces"
                else:
                    ctype = "unknown"
                    items = segment

                for raw in items.split(","):
                    name = raw.strip().strip("-").strip()
                    if len(name) < 2:
                        continue
                    key = name.lower()
                    actors[key]["callsign_type"] = ctype
                    actors[key]["name"] = name
                    actors[key]["appearances"].append(meta)

        # Entity resolution — merge surface variants ("Alpha 3", "Alpha-3", "Alpha Three")
        try:
            from entity_resolver import resolve_entities
            actors = resolve_entities(dict(actors))
        except Exception:
            pass  # resolver unavailable — continue with raw keys

        # Build output list
        thr_priority = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "CLEAR": 4}
        profiles = []
        for key, data in actors.items():
            apps   = data["appearances"]
            threats = [a["threat"] for a in apps]
            langs   = list(dict.fromkeys(a["language"] for a in apps))
            top_threat = sorted(set(threats), key=lambda t: thr_priority.get(t, 5))[0]
            profiles.append({
                "name":          data.get("name", key),
                "aliases":       data.get("aliases", []),
                "canonical_key": data.get("canonical_key", key),
                "callsign_type": data["callsign_type"],
                "count":         len(apps),
                "top_threat":    top_threat,
                "languages":     ", ".join(langs),
                "first_seen":    apps[0]["timestamp"],
                "last_seen":     apps[-1]["timestamp"],
                "report_ids":    [a["report_id"] for a in apps],
                "appearances":   apps,
            })

        profiles.sort(key=lambda p: p["count"], reverse=True)
        self._actor_cache    = profiles
        self._actor_cache_ts = time.monotonic()
        return profiles

    # ── Operator aliases ────────────────────────────────────────────────────────
    def get_aliases(self) -> dict:
        """Return {term_lower: {'kind','alias','notes'}} for all operator aliases."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT term, kind, alias, notes FROM aliases"
            ).fetchall()
        return {
            r["term"].lower(): {"kind": r["kind"], "alias": r["alias"], "notes": r["notes"] or ""}
            for r in rows
        }

    def set_alias(self, term: str, kind: str, alias: str, notes: str = "") -> None:
        """Create/update an operator alias. Empty alias deletes the entry."""
        term = (term or "").strip()
        if not term:
            return
        with self._conn() as conn:
            if not (alias or "").strip():
                conn.execute("DELETE FROM aliases WHERE term = ? COLLATE NOCASE", (term,))
            else:
                conn.execute(
                    "INSERT INTO aliases (term, kind, alias, notes, updated_utc) "
                    "VALUES (?,?,?,?,?) "
                    "ON CONFLICT(term) DO UPDATE SET "
                    "kind=excluded.kind, alias=excluded.alias, "
                    "notes=excluded.notes, updated_utc=excluded.updated_utc",
                    (term, kind, alias.strip(), notes or "",
                     datetime.now(timezone.utc).isoformat()),
                )

    def get_related_intercepts(self, report_id: str, limit: int = 6) -> List[dict]:
        """
        Score all other intercepts against a given report_id across four dimensions:
          - Shared keyword alerts (matched_word overlap)
          - Shared actors (who_field substring tokens)
          - Language + threat alignment
          - Time proximity (within 7 days)
        Returns up to `limit` results sorted by descending score.
        """
        from datetime import datetime
        from collections import defaultdict

        with self._conn() as conn:
            # Fetch source intercept
            src = conn.execute("""
                SELECT i.id, i.report_id, i.timestamp_utc, i.final_language,
                       i.threat_level, s.who_field, s.where_field
                FROM intercepts i
                LEFT JOIN isums s ON s.intercept_id = i.id
                WHERE i.report_id = ?
            """, (report_id,)).fetchone()
            if not src:
                return []

            src_id   = src["id"]
            src_lang = src["final_language"] or ""
            src_thr  = src["threat_level"] or "CLEAR"
            src_who  = src["who_field"] or ""
            src_ts   = src["timestamp_utc"] or ""

            # Source keyword words
            src_kws = {r["matched_word"].lower() for r in conn.execute(
                "SELECT matched_word FROM keyword_alerts WHERE intercept_id=? AND severity IN ('critical','high')",
                (src_id,),
            )}

            # Source actor tokens (words from who_field longer than 2 chars)
            src_actor_tokens = {
                w.lower() for w in src_who.replace(";", " ").replace(",", " ").split()
                if len(w) > 2 and not w.endswith(":")
            }

            # All other intercepts
            others = conn.execute("""
                SELECT i.id, i.report_id, i.timestamp_utc, i.final_language,
                       i.threat_level, i.audio_file,
                       s.who_field, s.where_field, s.assessment
                FROM intercepts i
                LEFT JOIN isums s ON s.intercept_id = i.id
                WHERE i.id != ?
                ORDER BY i.timestamp_utc DESC
                LIMIT 200
            """, (src_id,)).fetchall()

            # Build keyword index for other intercepts
            other_kws: dict = defaultdict(set)
            for r in conn.execute(
                "SELECT intercept_id, matched_word FROM keyword_alerts "
                "WHERE severity IN ('critical','high') AND intercept_id != ?",
                (src_id,),
            ):
                other_kws[r["intercept_id"]].add(r["matched_word"].lower())

        # Parse source timestamp
        def _parse_ts(ts: str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
                try:
                    return datetime.strptime(ts[:19], fmt)
                except Exception:
                    pass
            return None

        src_dt = _parse_ts(src_ts)

        THR_WEIGHT = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "CLEAR": 0}

        scored = []
        for row in others:
            oid    = row["id"]
            score  = 0
            tags   = []

            # ── Shared high/critical keywords ─────────────────────────────────
            shared_kw = src_kws & other_kws.get(oid, set())
            if shared_kw:
                kw_pts = min(len(shared_kw) * 3, 12)
                score += kw_pts
                tags.append(f"keywords: {', '.join(sorted(shared_kw)[:3])}")

            # ── Shared actors ─────────────────────────────────────────────────
            other_who = row["who_field"] or ""
            other_actor_tokens = {
                w.lower() for w in other_who.replace(";", " ").replace(",", " ").split()
                if len(w) > 2 and not w.endswith(":")
            }
            shared_actors = src_actor_tokens & other_actor_tokens
            # Remove generic words
            shared_actors -= {"not", "identified", "from", "intercept",
                               "forces", "friendly", "hostile", "ranks"}
            if shared_actors:
                score += min(len(shared_actors) * 4, 16)
                tags.append(f"actors: {', '.join(sorted(shared_actors)[:3])}")

            # ── Language + threat alignment ───────────────────────────────────
            same_lang = (row["final_language"] or "") == src_lang and src_lang
            same_thr  = (row["threat_level"] or "CLEAR") == src_thr
            if same_lang:
                score += 1; tags.append(f"lang: {src_lang}")
            if same_thr and src_thr not in ("CLEAR", "LOW"):
                score += 2; tags.append(f"threat: {src_thr}")

            # ── Time proximity ─────────────────────────────────────────────────
            other_dt = _parse_ts(row["timestamp_utc"] or "")
            if src_dt and other_dt:
                diff_h = abs((src_dt - other_dt).total_seconds()) / 3600
                if diff_h <= 24:
                    score += 3; tags.append("within 24h")
                elif diff_h <= 72:
                    score += 2; tags.append("within 3d")
                elif diff_h <= 168:
                    score += 1; tags.append("within 7d")

            if score == 0:
                continue

            scored.append({
                "report_id":  row["report_id"],
                "timestamp":  (row["timestamp_utc"] or "")[:16].replace("T", " "),
                "language":   (row["final_language"] or "?").upper(),
                "threat":     (row["threat_level"] or "CLEAR").upper(),
                "audio_file": row["audio_file"],
                "assessment": (row["assessment"] or "")[:120],
                "score":      score,
                "tags":       tags,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def save_metrics(self, report_id: str, metrics: dict) -> None:
        """Persist Tier 1 auto-metrics for a processed intercept."""
        if not metrics:
            return
        from datetime import datetime, timezone
        rtf      = metrics.get("rtf", {}) or {}
        conf     = metrics.get("segment_confidence", {}) or {}
        agree    = metrics.get("model_agreement", {}) or {}
        isum_c   = metrics.get("isum_completeness", {}) or {}
        mem      = metrics.get("memory", {}) or {}
        vocab    = metrics.get("vocab_richness", {}) or {}

        with self._conn() as conn:
            iid = conn.execute(
                "SELECT id FROM intercepts WHERE report_id=?", (report_id,)
            ).fetchone()
            intercept_id = iid[0] if iid else None
            # Upsert — replace if already exists for this report
            conn.execute("DELETE FROM metrics WHERE report_id=?", (report_id,))
            conn.execute("""
                INSERT INTO metrics
                  (intercept_id, report_id, timestamp_utc,
                   rtf, rtf_grade,
                   conf_mean, conf_pct_low, conf_grade,
                   ensemble_score, ensemble_grade,
                   isum_score, isum_pct,
                   mem_peak_mb, vocab_ttr, backtrans_chrf,
                   full_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                intercept_id, report_id,
                datetime.now(timezone.utc).isoformat(),
                rtf.get("value"),   rtf.get("grade"),
                conf.get("mean"),   conf.get("pct_low"),  conf.get("grade"),
                agree.get("ensemble_score"), agree.get("grade"),
                isum_c.get("score"), isum_c.get("pct"),
                mem.get("peak_mb"),
                vocab.get("ttr"),
                metrics.get("backtrans_chrf"),
                json.dumps(metrics, default=str),
            ))

    def get_metrics_history(self, limit: int = 100) -> List[dict]:
        """Return saved metrics rows ordered newest first."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT m.*, i.audio_file, i.final_language, i.threat_level
                FROM metrics m
                LEFT JOIN intercepts i ON i.id = m.intercept_id
                ORDER BY m.timestamp_utc DESC
                LIMIT ?
            """, (limit,)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("full_json"):
                try:
                    d["metrics"] = json.loads(d["full_json"])
                except Exception:
                    d["metrics"] = {}
            results.append(d)
        return results

    def get_stats(self) -> dict:
        """Dashboard statistics."""
        with self._conn() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM intercepts").fetchone()[0]
            threats  = conn.execute("""
                SELECT threat_level, COUNT(*) as cnt
                FROM intercepts GROUP BY threat_level
            """).fetchall()
            langs    = conn.execute("""
                SELECT final_language, COUNT(*) as cnt
                FROM intercepts GROUP BY final_language ORDER BY cnt DESC
            """).fetchall()
            critical = conn.execute("""
                SELECT COUNT(*) FROM intercepts WHERE threat_level='CRITICAL'
            """).fetchone()[0]

        return {
            "total_intercepts": total,
            "critical_count":   critical,
            "by_threat_level":  {r["threat_level"]: r["cnt"] for r in threats},
            "by_language":      {r["final_language"]: r["cnt"] for r in langs},
        }

    # ── Annotation methods ─────────────────────────────────────────────────────

    def save_annotation(self, annotation: dict) -> int:
        """Save an analyst annotation/correction. Returns annotation id."""
        with self._conn() as conn:
            # Resolve intercept_id from report_id
            row = conn.execute(
                "SELECT id FROM intercepts WHERE report_id=?",
                (annotation["report_id"],)
            ).fetchone()
            if not row:
                raise ValueError(f"No intercept found for report_id={annotation['report_id']}")
            intercept_id = row["id"]

            cur = conn.execute("""
                INSERT OR REPLACE INTO annotations (
                    intercept_id, report_id, annotated_by, annotated_at,
                    corrected_transcript, transcript_changed,
                    corrected_translation, translation_changed,
                    corrected_language, language_changed,
                    corrected_who, corrected_what, corrected_where, corrected_when,
                    corrected_assessment, corrected_threat_level, isum_changed,
                    false_positive_ids, missed_keywords, asr_quality_score, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                intercept_id,
                annotation["report_id"],
                annotation.get("annotated_by", "analyst"),
                annotation.get("annotated_at", ""),
                annotation.get("corrected_transcript"),
                int(annotation.get("transcript_changed", False)),
                annotation.get("corrected_translation"),
                int(annotation.get("translation_changed", False)),
                annotation.get("corrected_language"),
                int(annotation.get("language_changed", False)),
                annotation.get("corrected_who"),
                annotation.get("corrected_what"),
                annotation.get("corrected_where"),
                annotation.get("corrected_when"),
                annotation.get("corrected_assessment"),
                annotation.get("corrected_threat_level"),
                int(annotation.get("isum_changed", False)),
                json.dumps(annotation.get("false_positive_ids", [])),
                json.dumps(annotation.get("missed_keywords", [])),
                annotation.get("asr_quality_score"),
                annotation.get("notes"),
            ))
        return cur.lastrowid

    def get_annotation(self, report_id: str) -> Optional[dict]:
        """Get existing annotation for a report, if any."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM annotations WHERE report_id=? ORDER BY id DESC LIMIT 1",
                (report_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["false_positive_ids"] = _safe_json(d.get("false_positive_ids"))
        d["missed_keywords"]    = _safe_json(d.get("missed_keywords"))
        return d

    def get_annotation_stats(self) -> dict:
        """Return annotation statistics for the training data dashboard."""
        with self._conn() as conn:
            total      = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
            asr_fixed  = conn.execute(
                "SELECT COUNT(*) FROM annotations WHERE transcript_changed=1").fetchone()[0]
            trans_fixed = conn.execute(
                "SELECT COUNT(*) FROM annotations WHERE translation_changed=1").fetchone()[0]
            isum_fixed  = conn.execute(
                "SELECT COUNT(*) FROM annotations WHERE isum_changed=1").fetchone()[0]
            by_lang = conn.execute("""
                SELECT i.final_language, COUNT(*) as cnt
                FROM annotations a JOIN intercepts i ON a.intercept_id=i.id
                GROUP BY i.final_language ORDER BY cnt DESC
            """).fetchall()
        return {
            "total":        total,
            "asr_fixed":    asr_fixed,
            "trans_fixed":  trans_fixed,
            "isum_fixed":   isum_fixed,
            "by_language":  {r["final_language"]: r["cnt"] for r in by_lang},
        }

    def export_training_data(self, limit: int = 10000) -> dict:
        """Export all annotations as structured training datasets."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT a.*, i.transcript, i.translation, i.final_language,
                       i.audio_file, i.total_speech_sec,
                       s.transcript_snippet, s.translation_snippet
                FROM annotations a
                JOIN intercepts i ON a.intercept_id = i.id
                LEFT JOIN isums s ON s.intercept_id = i.id
                LIMIT ?
            """, (limit,)).fetchall()

        asr_data, trans_data, isum_data = [], [], []
        for row in rows:
            d = dict(row)
            fp = _safe_json(d.get("false_positive_ids"))
            mk = _safe_json(d.get("missed_keywords"))

            if d.get("transcript_changed"):
                asr_data.append({
                    "audio_file":           d["audio_file"],
                    "language":             d["final_language"],
                    "original_transcript":  d["transcript"],
                    "corrected_transcript": d["corrected_transcript"],
                    "total_speech_sec":     d["total_speech_sec"],
                })

            if d.get("translation_changed"):
                trans_data.append({
                    "source_text":          d["corrected_transcript"] or d["transcript"],
                    "language":             d["corrected_language"] or d["final_language"],
                    "original_translation": d["translation"],
                    "corrected_translation":d["corrected_translation"],
                })

            if d.get("isum_changed"):
                isum_data.append({
                    "transcript":   d["corrected_transcript"] or d["transcript"],
                    "translation":  d["corrected_translation"] or d["translation"],
                    "language":     d["corrected_language"] or d["final_language"],
                    "isum": {
                        "who":        d["corrected_who"],
                        "what":       d["corrected_what"],
                        "where":      d["corrected_where"],
                        "when":       d["corrected_when"],
                        "assessment": d["corrected_assessment"],
                        "threat_level": d["corrected_threat_level"],
                    },
                })

        return {
            "asr":         asr_data,
            "translation": trans_data,
            "isum":        isum_data,
            "total_annotations": len(rows),
        }


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_json(val):
    if val is None:
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []

"""
speaker_store.py — Cross-file speaker identity persistence.

Stores MFCC centroids per speaker voice in SQLite.
On each new pipeline run, new speaker centroids are compared against all
stored centroids via cosine similarity.  A match above COSINE_THRESHOLD
returns the existing VOICE_XXX id; otherwise a new id is minted.

The centroid is updated via exponential moving average (EMA) after every
match to slowly adapt to acoustic drift across recordings.

Threshold choice: MFCC 240-dim cosine is less discriminative than learned
speaker embeddings (x-vector / ECAPA), so the threshold is set conservatively
at 0.88 to minimise false-positive speaker links at the cost of some missed
same-speaker matches.
"""

import json
import sqlite3
import numpy as np
from typing import List, Tuple


COSINE_THRESHOLD = 0.88   # min cosine similarity to declare same speaker
CENTROID_EMA     = 0.15   # weight of new observation when updating centroid

_PALETTE = [
    "#00aaff", "#ff8c00", "#00ff88", "#ff55aa",
    "#aa55ff", "#ffdd00", "#00ddcc", "#ff4444",
    "#88cc00", "#ff77aa", "#55aaff", "#ffaa55",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerStore:
    """
    Persistent cross-file speaker identity store backed by SQLite.
    One table: speaker_voices — one row per unique identified voice.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS speaker_voices (
                    voice_id         TEXT PRIMARY KEY,
                    centroid         BLOB NOT NULL,
                    intercept_ids    TEXT DEFAULT '[]',
                    first_seen       TEXT,
                    last_seen        TEXT,
                    appearance_count INTEGER DEFAULT 1,
                    color            TEXT
                )
            """)

    # ── Public API ─────────────────────────────────────────────────────────────

    def match_or_register(
        self,
        centroid:     np.ndarray,
        intercept_id: str,
        timestamp:    str,
    ) -> Tuple[str, bool]:
        """
        Match centroid against stored voices.

        Returns
        -------
        (voice_id, is_new)
            voice_id : persistent identifier e.g. "VOICE_001"
            is_new   : True if a new voice entry was created
        """
        centroid = centroid.astype(np.float32)

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT voice_id, centroid FROM speaker_voices"
            ).fetchall()

            best_id    = None
            best_score = -1.0
            for row in rows:
                stored = np.frombuffer(bytes(row["centroid"]), dtype=np.float32)
                score  = _cosine(centroid, stored)
                if score > best_score:
                    best_score = score
                    best_id    = row["voice_id"]

            if best_id and best_score >= COSINE_THRESHOLD:
                # ── Match: update centroid via EMA ──────────────────────────
                row    = conn.execute(
                    "SELECT centroid, intercept_ids, appearance_count "
                    "FROM speaker_voices WHERE voice_id = ?",
                    (best_id,),
                ).fetchone()
                stored     = np.frombuffer(bytes(row["centroid"]), dtype=np.float32)
                new_center = (1 - CENTROID_EMA) * stored + CENTROID_EMA * centroid
                ids        = json.loads(row["intercept_ids"] or "[]")
                if intercept_id not in ids:
                    ids.append(intercept_id)
                conn.execute(
                    "UPDATE speaker_voices "
                    "SET centroid=?, intercept_ids=?, last_seen=?, appearance_count=appearance_count+1 "
                    "WHERE voice_id=?",
                    (new_center.tobytes(), json.dumps(ids), timestamp, best_id),
                )
                return best_id, False

            else:
                # ── No match: register new voice ────────────────────────────
                count    = conn.execute(
                    "SELECT COUNT(*) AS n FROM speaker_voices"
                ).fetchone()["n"]
                voice_id = f"VOICE_{count + 1:03d}"
                color    = _PALETTE[count % len(_PALETTE)]
                conn.execute(
                    "INSERT INTO speaker_voices "
                    "(voice_id, centroid, intercept_ids, first_seen, last_seen, appearance_count, color) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?)",
                    (
                        voice_id,
                        centroid.tobytes(),
                        json.dumps([intercept_id]),
                        timestamp,
                        timestamp,
                        color,
                    ),
                )
                return voice_id, True

    def get_all_voices(self) -> List[dict]:
        """Return all registered voices sorted by appearance count."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT voice_id, intercept_ids, first_seen, last_seen,
                       appearance_count, color
                FROM speaker_voices
                ORDER BY appearance_count DESC
            """).fetchall()
        return [
            {
                "voice_id":         row["voice_id"],
                "intercept_ids":    json.loads(row["intercept_ids"] or "[]"),
                "first_seen":       row["first_seen"] or "",
                "last_seen":        row["last_seen"] or "",
                "appearance_count": row["appearance_count"],
                "color":            row["color"] or "#8a9aaa",
            }
            for row in rows
        ]

    def get_color_map(self) -> dict:
        """Return {voice_id: color_hex} for all registered voices."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT voice_id, color FROM speaker_voices"
            ).fetchall()
        return {row["voice_id"]: row["color"] for row in rows}

    def clear(self) -> None:
        """Delete all stored voices (useful for testing / resetting)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM speaker_voices")

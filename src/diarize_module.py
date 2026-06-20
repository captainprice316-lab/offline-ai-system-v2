"""
diarize_module.py – MFCC-based speaker diarization
---------------------------------------------------
Assigns speaker labels (SPEAKER_A, SPEAKER_B, ...) to Whisper transcript
segments using MFCC embeddings + agglomerative clustering.

Zero additional dependencies — uses librosa and scikit-learn which are
already in requirements.txt.
"""

import numpy as np
import librosa
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from typing import List, Dict, Optional

MIN_SEG_DURATION = 0.5   # seconds — skip segments shorter than this for embedding
MFCC_N           = 40
SPEAKER_LABELS   = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_D"]
SPEAKER_COLORS   = {
    "SPEAKER_A": "#00aaff",
    "SPEAKER_B": "#ff8c00",
    "SPEAKER_C": "#00ff88",
    "SPEAKER_D": "#ff55aa",
}


def diarize(
    audio_path:   str,
    segments:     List[Dict],
    max_speakers: int = 4,
) -> List[Dict]:
    """
    Assign speaker labels to transcript segments in-place.

    Parameters
    ----------
    audio_path   : path to preprocessed 16kHz mono WAV
    segments     : list of dicts with 'start', 'end', 'text' keys
    max_speakers : auto-detect up to this many speakers (default 4)

    Returns
    -------
    Same segments list with 'speaker' key added to each entry.
    """
    if not segments:
        return segments

    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        _fallback(segments)
        return segments

    embeddings: List[np.ndarray] = []
    valid_idx:  List[int]        = []

    for i, seg in enumerate(segments):
        start = int(seg.get("start", 0) * sr)
        end   = int(seg.get("end",   0) * sr)
        chunk = audio[start:end]

        if len(chunk) < int(sr * MIN_SEG_DURATION):
            continue

        emb = _mfcc_embedding(chunk, sr)
        if emb is not None:
            embeddings.append(emb)
            valid_idx.append(i)

    if len(embeddings) < 2:
        _fallback(segments)
        return segments

    X = normalize(np.array(embeddings, dtype=np.float32))
    n = _pick_n_speakers(X, max_speakers)

    if n == 1:
        labels = np.zeros(len(X), dtype=int)
    else:
        clust  = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
        labels = clust.fit_predict(X)

    label_map = {valid_idx[j]: int(labels[j]) for j in range(len(valid_idx))}

    # Assign labels to segments that had valid embeddings
    for i, seg in enumerate(segments):
        if i in label_map:
            raw = label_map[i]
            seg["speaker"] = SPEAKER_LABELS[raw] if raw < len(SPEAKER_LABELS) else f"SPEAKER_{raw}"

    # Short segments inherit nearest labelled neighbour
    for i, seg in enumerate(segments):
        if i not in label_map:
            # look left first, then right
            for j in range(i - 1, -1, -1):
                if j in label_map:
                    seg["speaker"] = segments[j]["speaker"]
                    break
            else:
                for j in range(i + 1, len(segments)):
                    if j in label_map:
                        seg["speaker"] = segments[j]["speaker"]
                        break
                else:
                    seg["speaker"] = "SPEAKER_A"

    return segments


def compute_speaker_centroids(
    audio_path: str,
    segments:   List[Dict],
) -> Dict[str, np.ndarray]:
    """
    Compute one mean MFCC embedding per unique speaker label.
    Called after diarize() so that segments already carry 'speaker' keys.
    Returns {speaker_label: centroid_ndarray}. Empty dict on failure.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return {}

    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        start = int(seg.get("start", 0) * sr)
        end   = int(seg.get("end",   0) * sr)
        chunk = audio[start:end]
        if len(chunk) < int(sr * MIN_SEG_DURATION):
            continue
        emb = _mfcc_embedding(chunk, sr)
        if emb is not None:
            buckets[spk].append(emb)

    return {spk: np.mean(embs, axis=0) for spk, embs in buckets.items() if embs}


def build_speaker_transcript(segments: List[Dict]) -> str:
    """
    Format a speaker-labelled transcript string for LLM consumption.
    Collapses consecutive same-speaker segments into one turn.
    Returns empty string if no speaker labels are present.
    """
    if not segments or not segments[0].get("speaker"):
        return ""

    lines:   List[str] = []
    cur_spk: Optional[str] = None
    buf:     List[str] = []

    for seg in segments:
        spk  = seg.get("speaker", "SPEAKER_A")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if spk != cur_spk:
            if buf and cur_spk:
                lines.append(f"{cur_spk}: {' '.join(buf)}")
            cur_spk = spk
            buf     = [text]
        else:
            buf.append(text)

    if buf and cur_spk:
        lines.append(f"{cur_spk}: {' '.join(buf)}")

    return "\n".join(lines)


def speaker_stats(segments: List[Dict]) -> Dict:
    """Return per-speaker word counts and segment counts."""
    stats: Dict[str, Dict] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        if spk not in stats:
            stats[spk] = {"segments": 0, "words": 0}
        stats[spk]["segments"] += 1
        stats[spk]["words"]    += len(seg.get("text", "").split())
    return stats


# ── Internals ──────────────────────────────────────────────────────────────────

def _mfcc_embedding(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_N)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats  = np.vstack([mfcc, delta, delta2])           # (120, T)
        return np.concatenate([feats.mean(axis=1), feats.std(axis=1)])  # (240,)
    except Exception:
        return None


def _pick_n_speakers(X: np.ndarray, max_n: int) -> int:
    if len(X) < 4:
        return min(2, len(X))
    best_score = -1.0
    best_n     = 2
    for n in range(2, min(max_n + 1, len(X))):
        clust  = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
        labels = clust.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_n     = n
    return best_n


def _fallback(segments: List[Dict]) -> None:
    for seg in segments:
        seg["speaker"] = "SPEAKER_A"

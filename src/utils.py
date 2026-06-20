"""
src/utils.py – Shared utilities for VANI pipeline
---------------------------------------------------
Centralises:
  • Project root resolution  (all modules import ROOT from here)
  • Config loading           (single yaml.safe_load with caching)
  • Logger factory           (consistent log format across modules)
  • Memory helpers           (gc + torch cache clear)
  • Timestamp helpers
"""

import gc
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
# utils.py lives in src/; root is one level up
ROOT = Path(__file__).resolve().parent.parent


# ── Config ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config(config_path: str = None) -> dict:
    """Load and cache config.yaml. Call with no args from anywhere."""
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return ROOT / relative


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str = "vani", log_dir: Path = None) -> logging.Logger:
    """
    Get a named logger. Creates file + console handlers on first call.
    Subsequent calls with same name return the cached logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is None:
        cfg = load_config()
        log_dir = ROOT / cfg["paths"]["log_dir"]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vani_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Memory management ─────────────────────────────────────────────────────────

def free_memory(logger=None):
    """Force GC and clear PyTorch cache (CUDA + MPS)."""
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
    if logger:
        logger.debug("Memory freed (gc + torch cache cleared)")


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_report_id(prefix: str = "ISUM") -> str:
    import uuid
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]  # up to ms
    uid = uuid.uuid4().hex[:4].upper()
    return f"{prefix}-{ts}-{uid}"


def elapsed(start: float) -> float:
    return round(time.time() - start, 2)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_dir_wavs(directory: Path):
    """Remove all .wav files from a directory (stale chunks cleanup)."""
    for f in Path(directory).glob("*.wav"):
        try:
            f.unlink()
        except Exception:
            pass

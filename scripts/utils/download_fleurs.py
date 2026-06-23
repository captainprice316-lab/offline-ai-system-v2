#!/usr/bin/env python3
"""
download_fleurs.py — cache all FLEURS subsets needed for the 14-language evaluation.

Run once with internet access; subsequent runs (offline) will use the cache.
    python3 download_fleurs.py
"""

import os
import sys

# Must NOT set HF_DATASETS_OFFLINE here — we need to download
if "HF_DATASETS_OFFLINE" in os.environ:
    del os.environ["HF_DATASETS_OFFLINE"]
if "TRANSFORMERS_OFFLINE" in os.environ:
    del os.environ["TRANSFORMERS_OFFLINE"]

NEW_SUBSETS = [
    ("bn_in", "bn", "Bengali"),
    ("ps_af", "ps", "Pashto"),
    ("fa_ir", "fa", "Persian"),
    ("ar_eg", "ar", "Arabic"),
    ("my_mm", "my", "Burmese"),
    ("sd_in", "sd", "Sindhi"),
    ("tg_tj", "tg", "Tajik"),
    ("uz_uz", "uz", "Uzbek"),
    ("kk_kz", "kk", "Kazakh"),
]

ALREADY_CACHED = ["hi_in", "pa_in", "ur_pk", "ne_np", "cmn_hans_cn"]

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: 'datasets' package not installed — activate the venv first.")

print("Downloading new FLEURS subsets for 14-language evaluation")
print("=" * 60)

ok, failed = [], []
for subset, code, name in NEW_SUBSETS:
    print(f"\n  {name} ({subset}) ...", flush=True)
    try:
        ds = load_dataset("google/fleurs", subset, split="test")
        n = len(ds)
        print(f"    ✓ {n} test samples cached", flush=True)
        ok.append(subset)
    except Exception as e:
        print(f"    ✗ FAILED: {e}", flush=True)
        failed.append(subset)

print("\n" + "=" * 60)
print(f"Downloaded : {len(ok)}/{len(NEW_SUBSETS)} subsets")
if ok:
    print(f"  OK     : {ok}")
if failed:
    print(f"  FAILED : {failed}")
print(f"\nAlready cached: {ALREADY_CACHED}")
print("\nNext: python eval_fleurs.py --subset all")

# -*- coding: utf-8 -*-
"""slim_adapter.py — drop the frozen embedding matrix from a vocab-extended adapter.

Extending the vocabulary (add_ks_chars) makes PEFT ship the whole resized
embedding table inside the adapter: for ks_cloud3 that is 256120 x 1024 fp32 =
1.05 GB, 41.5% of the file, and it is FROZEN — `trainable_token_indices` learns
a separate (21, 1024) delta on top of it. The table is therefore exactly
reproducible by rebuilding the base model and re-running add_kas_token /
add_ks_chars, which the training and eval paths already do before loading an
adapter. Verified byte-identical (max abs diff 0.000000).

Dropping it takes ks_cloud3 from 1.48 GB to ~428 MB with no loss of information
— worth it for upload time, for the project's lean-footprint goal, and for the
next warm start.

Usage:
    python scripts/utils/slim_adapter.py IN_DIR OUT_DIR [--verify]
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

DROP_SUFFIX = "base_layer.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild the model and confirm the dropped tensor is reproducible")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    sd = load_file(str(src / "adapter_model.safetensors"))
    dropped = [k for k in sd if k.endswith(DROP_SUFFIX)]
    if not dropped:
        print("[slim] nothing to drop — adapter carries no embedding table")
    keep = {k: v for k, v in sd.items() if k not in dropped}

    if args.verify and dropped:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        from finetune_seamless import add_kas_token, add_ks_chars, SEAMLESS_DIR
        proc = AutoProcessor.from_pretrained(str(SEAMLESS_DIR))
        m = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            str(SEAMLESS_DIR), torch_dtype=torch.float32)
        add_kas_token(proc, m)
        add_ks_chars(proc, m)
        recon = m.get_input_embeddings().weight.detach()
        for k in dropped:
            saved = sd[k].float()
            d = (saved - recon[:saved.shape[0]]).abs().max().item()
            print(f"[verify] {k}: max abs diff {d:.8f}  "
                  f"{'reproducible' if d < 1e-3 else 'NOT REPRODUCIBLE — do not drop'}")
            if d >= 1e-3:
                raise SystemExit("aborting: dropped tensor is not reproducible")

    save_file(keep, str(dst / "adapter_model.safetensors"))
    for f in src.iterdir():
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy2(f, dst / f.name)

    a = (src / "adapter_model.safetensors").stat().st_size
    b = (dst / "adapter_model.safetensors").stat().st_size
    print(f"[slim] dropped {len(dropped)} tensor(s): {a:,} -> {b:,} bytes "
          f"({100*(a-b)/a:.1f}% smaller)")
    print(f"[slim] wrote {dst}")


if __name__ == "__main__":
    main()

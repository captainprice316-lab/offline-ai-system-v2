# -*- coding: utf-8 -*-
"""training_cost_inventory.py -- recover the real wall-clock cost of every
training run from the checkpoints on disk.

The report quoted "roughly 100 GPU-hours" for the Whisper phase as an estimate.
Every run actually recorded its own timing: HuggingFace Trainer writes
`log_history` into each checkpoint's trainer_state.json, with an `eval_runtime`
on every evaluation entry and `train_runtime` on the final one. Summing those
gives measured hours instead of a guess, and separates TRAINING time from
EVALUATION time -- which matters here, because evaluation turned out to dominate
(see the report's "eval time bottleneck" note).

Reads the highest-step trainer_state.json per run directory. Writes
docs/training_cost_inventory.json.

Usage: python scripts/eval/training_cost_inventory.py
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "training_cost_inventory.json"

ROOTS = [
    (ROOT / "finetune_runs", "whisper"),
    (ROOT / "finetune_runs_seamless", "seamless"),
]


def latest_state(run_dir):
    """The trainer_state.json with the highest step number under run_dir."""
    best, best_step = None, -1
    for p in run_dir.rglob("trainer_state.json"):
        m = re.search(r"checkpoint-(\d+)", str(p))
        step = int(m.group(1)) if m else 0
        if step > best_step:
            best, best_step = p, step
    return best


def checkpoint_span(adapter_dir):
    """Elapsed wall-clock from the first to the last checkpoint written.

    `train_runtime` only lands in the FINAL trainer_state, which these runs do
    not retain, so the trainer cannot tell us how long training took. Checkpoint
    mtimes can. This is an ELAPSED span: it includes any pause, so it is an
    upper bound on compute time, not a measure of it. Reported as such.
    """
    stamps = []
    for d in adapter_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if d.is_dir() and m:
            stamps.append((int(m.group(1)), d.stat().st_mtime))
    if len(stamps) < 2:
        return None
    stamps.sort()
    gaps = [b[1] - a[1] for a, b in zip(stamps, stamps[1:])]
    gaps.sort()
    med = gaps[len(gaps) // 2]
    return {
        "elapsed_seconds": round(stamps[-1][1] - stamps[0][1], 1),
        "n_checkpoints": len(stamps),
        "median_gap_seconds": round(med, 1),
        # gaps far above the median are interruptions, not compute
        "seconds_in_outlier_gaps": round(sum(g for g in gaps if g > 3 * med), 1),
        "first_checkpoint": stamps[0][0],
        "last_checkpoint": stamps[-1][0],
    }


def summarise(state_path):
    st = json.loads(state_path.read_text(encoding="utf-8"))
    hist = st.get("log_history", [])

    eval_s = sum(h["eval_runtime"] for h in hist if "eval_runtime" in h)
    n_evals = sum(1 for h in hist if "eval_runtime" in h)

    # train_runtime appears on the final summary entry; if the run was killed
    # before it was written, fall back to the wall-clock the Trainer tracked.
    train_s = None
    for h in reversed(hist):
        if "train_runtime" in h:
            train_s = h["train_runtime"]
            break

    losses = [h["eval_loss"] for h in hist if "eval_loss" in h]
    wers = [h["eval_wer"] for h in hist if "eval_wer" in h]

    return {
        "steps": st.get("global_step"),
        "epochs": round(st.get("epoch", 0), 3),
        "train_seconds": round(train_s, 1) if train_s is not None else None,
        "eval_seconds": round(eval_s, 1),
        "n_evals": n_evals,
        "total_seconds": (round(train_s + eval_s, 1)
                          if train_s is not None else None),
        "best_eval_loss": round(min(losses), 4) if losses else None,
        "best_eval_wer": round(min(wers), 2) if wers else None,
        "final_eval_loss": round(losses[-1], 4) if losses else None,
        "best_loss_at_final_step": (bool(losses) and
                                    abs(min(losses) - losses[-1]) < 1e-9),
        "source": str(state_path.relative_to(ROOT)).replace("\\", "/"),
    }


out = {}
for base, phase in ROOTS:
    if not base.exists():
        print(f"[skip] {base} not present")
        continue
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        # a run may hold several adapters (pa/adapter, pa/adapter_v2)
        adapters = [d for d in run_dir.iterdir()
                    if d.is_dir() and any(d.rglob("trainer_state.json"))]
        if not adapters:
            adapters = [run_dir] if any(run_dir.rglob("trainer_state.json")) else []
        for ad in adapters:
            sp = latest_state(ad)
            if sp is None:
                continue
            name = run_dir.name if ad == run_dir else f"{run_dir.name}/{ad.name}"
            rec = summarise(sp)
            span = checkpoint_span(ad)
            # A checkpoint span is only believable if the run kept enough
            # checkpoints to cover itself AND the span exceeds the time the
            # trainer says it spent evaluating -- a run cannot have taken less
            # wall-clock than its own evaluations. Every SeamlessM4T run fails
            # this: save_total_limit discarded the early checkpoints and the
            # survivors were copied off the training box, which reset their
            # mtimes. Their timing is not recoverable from disk and must come
            # from the run logs instead.
            if span is not None:
                span["trustworthy"] = (
                    span["n_checkpoints"] >= 5
                    and span["elapsed_seconds"] > rec["eval_seconds"]
                )
            rec["span"] = span
            out[f"{phase}:{name}"] = rec

hdr = (f"{'run':40}{'steps':>7}{'evals':>7}{'eval h':>8}"
       f"{'elapsed h':>11}{'idle h':>8}{'active h':>10}")
print(hdr)
print("-" * len(hdr))
tot_e = tot_active = 0.0
for k, v in out.items():
    eh = v["eval_seconds"] / 3600
    sp = v["span"]
    tot_e += eh
    if sp and sp["trustworthy"]:
        el = sp["elapsed_seconds"] / 3600
        idle = sp["seconds_in_outlier_gaps"] / 3600
        active = max(el - idle, 0.0)
        tot_active += active
        print(f"{k:40}{v['steps'] or 0:>7}{v['n_evals']:>7}{eh:>8.2f}"
              f"{el:>11.2f}{idle:>8.2f}{active:>10.2f}")
    else:
        print(f"{k:40}{v['steps'] or 0:>7}{v['n_evals']:>7}{eh:>8.2f}"
              f"{'n/a':>11}{'n/a':>8}{'n/a':>10}")
print("-" * len(hdr))
print(f"{'TOTAL':40}{'':>7}{'':>7}{tot_e:>8.2f}{'':>11}{'':>8}{tot_active:>10.2f}")
print("  eval hours are trainer-measured for every run; elapsed/active only "
      "where the checkpoint span is trustworthy (see JSON).")

out["_totals"] = {
    "eval_hours": round(tot_e, 2),
    "active_hours": round(tot_active, 2),
    "train_hours_implied": round(max(tot_active - tot_e, 0), 2),
    "n_runs_with_timing": len([k for k in out if not k.startswith("_")]),
    "_caveat": ("Cloud runs (ks_cloud*, ps_cloud, doi_iv*) are absent: only the "
                "final adapter was retrieved from the pod, so no checkpoints "
                "exist locally. Their wall times come from the pod logs and are "
                "listed separately in the report. 'active' = elapsed between "
                "first and last checkpoint MINUS gaps more than 3x the median "
                "inter-checkpoint interval, which are interruptions rather "
                "than compute."),
}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[saved] {OUT.relative_to(ROOT)}")

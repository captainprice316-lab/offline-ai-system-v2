# Kashmiri high-capacity cloud run (`ks_cloud`)

The 8 GB laptop caps us at LoRA **r=32** (1.75% of weights), batch 1. This runs the
**same ks_max2 recipe + 97k/240h combined corpus** at **r=128 (α=256) on all
attn+MLP + trainable `__kas__`**, with a real batch — the capacity the laptop
can't reach. Goal: push Kashmiri L2 WER below ks_max2's **61.9** toward ~50.

**Nothing here touches the laptop.** Training is build-time; the runtime stays
air-gapped. You bring back one small adapter (~200–400 MB) and deploy it locally.

> Honest expectation: this attacks the *capacity* limit, not the *prior* limit
> (SeamlessM4T never pretrained on Kashmiri). If it stalls well above 50, that's
> the empirical case that MMS is "absolutely necessary." If it clears it, great —
> no new backend needed.

---

## 1. Rent a box
Any hourly GPU host (RunPod / Vast.ai / Lambda). Pick a **24–48 GB** card
(RTX 4090 / A10 / A6000 ≈ $0.4–1/hr). Use a **PyTorch CUDA image**. r=128 + batch
fits comfortably; no A100 needed unless you later switch to true full-FT.

## 2. Get the code (not the models/data — those are gitignored)
```bash
git clone <your VANI repo remote> vani && cd vani
# or scp just: finetune_seamless.py  cloud/  scripts/eval/robustness_eval.py
pip install -r cloud/requirements-cloud.txt
huggingface-cli login            # IndicVoices-R may be gated; base model pull needs it too
```

## 3. Build the data on the box (pulls from source — no upload from home)
```bash
python cloud/prep_ks_data.py --out ./ks_data
```
Downloads humair025 + IndicVoices-R (Kashmiri) + OpenSLR-122 **and the base
SeamlessM4T-v2**, rebuilds the combined manifest (dur 2–20 s, eval-leak-deduped vs
the IVR-R test set), and writes `ks_data/env.sh`. Expect ~97k train clips /
~240 h (matches the laptop's `composition.json`).

## 4. Train
```bash
source ks_data/env.sh          # sets KS_COMBINED_DIR, KS_IVR_DIR, VANI_SEAMLESS_DIR, batch
python finetune_seamless.py ks_cloud --steps 8000
```
- Val = the **same 372-clip IVR-R test** → **eval_loss is directly comparable to
  ks_max2's 1.040**; you want it clearly below that.
- Bigger batch: `env.sh` sets `VANI_TRAIN_BS=8 VANI_GRAD_ACCUM=2` (eff batch 16);
  raise `VANI_TRAIN_BS` if VRAM allows. Fewer steps/epoch than the laptop, so
  8k–12k steps ≈ 1–2 epochs.
- **Sweep rank**: edit `lora_r`/`lora_alpha` in the `ks_cloud` config (128→256).
- EarlyStopping patience 3 on eval_loss; checkpoints every 200 steps.

## 5. Bring back the adapter + verify BEFORE deploying
```bash
# download this dir (~200–400 MB) to the laptop's E: tree:
finetune_runs_seamless/ks_cloud/adapter
```
On the laptop, run the **same gate ks_max2 passed** (GPU-free where possible):
```bash
# clean WER (matched decode settings):
python scripts/eval/eval_ks_seamless.py \
    --adapter-dir finetune_runs_seamless/ks_cloud/adapter \
    --min-tok-per-sec 2.5 --no-repeat-ngram 3
# ruler ladder vs ks_max2 + whisper: copy ks_max2_ruler_compare.py, point it at
#   ks_cloud_seamless_hyps.jsonl (change the ks_max2 path), run (CPU).
# degradation sweep: copy ks_max2_degradation_eval.py, set ADAPTER_DIR to ks_cloud.
```
Deploy **only if** it wins the L2 ruler + degradation, same as ks_max2.

## 6. Deploy (one line, like ks_max2)
```yaml
# config.yaml
seamless_adapters:
  ks: finetune_runs_seamless/ks_cloud/adapter
```

## Cost
~6–15 GPU-hours for 1–2 epochs → **~$10–40** total. A day of wall-clock incl. setup.

## Going bigger (true full fine-tune) — not wired, notes only
For maximum capacity: in `train()`, skip `get_peft_model`, unfreeze the model,
drop LR to ~1e-5, add 8-bit AdamW (`bitsandbytes`), use an A100-80GB. Produces a
**whole ~6 GB model** (conflicts with the lean-project goal) — prefer high-rank
LoRA above unless it plateaus.

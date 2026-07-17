# Training chain watcher, stage 4 (armed 2026-07-17). Runs AFTER
# queue_hi_ne_after_ks.ps1 logs "FULL CHAIN COMPLETE":
#   1. ps_bal training — Pashto attempt #3 vs Whisper-medium 38.55:
#      FLEURS x8 oversample + CV cap 10k (fixes ps_cv's domain drift),
#      LoRA r=16 a=32 on q/k/v/out_proj (first capacity increase)
#   2. n=100 held-out eval
#
# Aborts if the upstream chain logs STOPPED/TIMEOUT instead of completing.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ps_bal_after_chain.ps1' -WindowStyle Hidden

param(
    [int]$Steps = 3000,      # ~0.8 epoch of ~30k effective samples at batch 8
    [int]$MaxWaitHours = 36
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ps_bal.log"
$upstream = "$root\logs\queue_hi_ne.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Upstream-State() {
    $all = Get-Content $upstream -Raw -ErrorAction SilentlyContinue
    if ($all -match 'FULL CHAIN COMPLETE') { return 'complete' }
    if ($all -match 'STOPPED|TIMEOUT')     { return 'failed' }
    return 'waiting'
}

# ── Stage 1: wait for the hi/ne chain to finish ──────────────────────────────
Log "chain armed: waiting for FULL CHAIN COMPLETE in queue_hi_ne.log"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Upstream-State) -eq 'waiting') {
    if ((Get-Date) -gt $deadline) {
        Log "TIMEOUT after $MaxWaitHours h - ps_bal NOT started."
        exit 1
    }
    Start-Sleep -Seconds 180
}
if ((Upstream-State) -eq 'failed') {
    Log "upstream chain STOPPED - ps_bal NOT started (GPU state unknown). Manual: python finetune_seamless.py ps_bal --steps $Steps"
    exit 1
}
Log "upstream chain complete"
Start-Sleep -Seconds 60   # VRAM release

# ── Stage 2: ps_bal training ─────────────────────────────────────────────────
Log "launching ps_bal training ($Steps steps)"
$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','finetune_seamless.py','ps_bal','--steps',"$Steps" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ps_bal_train.log" `
    -RedirectStandardError  "$root\logs\ps_bal_train_err.log" `
    -WindowStyle Hidden -PassThru
$p.WaitForExit()
$all = Get-Content "$root\logs\ps_bal_train.log" -Raw -ErrorAction SilentlyContinue
if ($all -notmatch '\[OK\] Adapter saved') {
    Log "ps_bal did NOT save an adapter - STOPPED. Resume: python finetune_seamless.py ps_bal --steps $Steps --resume"
    exit 1
}
Start-Sleep -Seconds 30

# ── Stage 3: n=100 held-out eval ─────────────────────────────────────────────
Log "ps_bal adapter confirmed - running n=100 eval (bars: Whisper 38.55, FLEURS-only adapter 41.30)"
$e = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','scripts\eval\eval_seamless_ft.py','--lang','ps_bal' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ps_bal_eval.log" `
    -RedirectStandardError  "$root\logs\ps_bal_eval_err.log" `
    -WindowStyle Hidden -PassThru
$e.WaitForExit()
Log "ps_bal eval finished (exit $($e.ExitCode)) - ALL QUEUED WORK DONE. Next: report updates + scripts\restore_power.ps1"

# Training chain watcher, stage 5 (armed 2026-07-17). Runs AFTER
# queue_post_ks.ps1 logs "ALL QUEUED WORK DONE":
#   1. ks_r16 training — Kashmiri attempt #3: LoRA r=16 a=32 on q/k/v/out_proj,
#      full IndicVoices data (cap 24k), fresh run (~2 epochs)
#   2. head-to-head eval with decode fixes vs Whisper-ks 74.02
#
# Cancel any time before it fires by stopping this watcher's PID (it only
# polls until the upstream marker appears). Aborts if upstream logs STOPPED.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ks_r16_after_chain.ps1' -WindowStyle Hidden

param(
    [int]$Steps = 6000,     # ~2 epochs of 24k at effective batch 8
    [int]$MaxWaitHours = 36
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ks_r16.log"
$upstream = "$root\logs\queue_post_ks.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Upstream-State() {
    $all = Get-Content $upstream -Raw -ErrorAction SilentlyContinue
    if ($all -match 'ALL QUEUED WORK DONE') { return 'complete' }
    if ($all -match 'STOPPED|TIMEOUT')      { return 'failed' }
    return 'waiting'
}

Log "chain armed: waiting for ALL QUEUED WORK DONE in queue_post_ks.log"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Upstream-State) -eq 'waiting') {
    if ((Get-Date) -gt $deadline) { Log "TIMEOUT after $MaxWaitHours h - ks_r16 NOT started."; exit 1 }
    Start-Sleep -Seconds 180
}
if ((Upstream-State) -eq 'failed') {
    Log "upstream chain STOPPED - ks_r16 NOT started. Manual: python finetune_seamless.py ks_r16 --steps $Steps"
    exit 1
}
Log "upstream chain complete"
Start-Sleep -Seconds 60

Log "launching ks_r16 training ($Steps steps)"
$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','finetune_seamless.py','ks_r16','--steps',"$Steps" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ks_r16_train.log" `
    -RedirectStandardError  "$root\logs\ks_r16_train_err.log" `
    -WindowStyle Hidden -PassThru
$p.WaitForExit()
$all = Get-Content "$root\logs\ks_r16_train.log" -Raw -ErrorAction SilentlyContinue
if ($all -notmatch '\[OK\] Adapter saved') {
    Log "ks_r16 did NOT save an adapter - STOPPED. Resume: python finetune_seamless.py ks_r16 --steps $Steps --resume"
    exit 1
}
Start-Sleep -Seconds 30

Log "ks_r16 adapter confirmed - running head-to-head eval (bar: Whisper-ks 74.02 / 81.46; prior Seamless ks: 92.09)"
$e = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','scripts\eval\eval_ks_seamless.py','--min-tok-per-sec','2.5','--no-repeat-ngram','3','--adapter-dir','finetune_runs_seamless\ks_r16\adapter' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ks_r16_eval.log" `
    -RedirectStandardError  "$root\logs\ks_r16_eval_err.log" `
    -WindowStyle Hidden -PassThru
$e.WaitForExit()
Log "ks_r16 eval finished (exit $($e.ExitCode)) - docs\ks_r16_seamless_results.json. ENTIRE QUEUE DONE. Next: report updates + scripts\restore_power.ps1"

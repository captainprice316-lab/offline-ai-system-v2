# Training chain watcher (armed 2026-07-12, rearmed same evening as a 3-stage chain):
#   1. wait for ps_lv3 Whisper training (detached python) to finish
#   2. launch Kashmiri SeamlessM4T LoRA (custom __kas__ token, fixed labels)
#   3. when ks finishes, retrain Seamless PASHTO with the fixed target-mode labels
#      (the 2026-07 ps adapter was trained against __eng__-prefixed labels — see
#      the label-bug fix in finetune_seamless.py; published 41.22% WER came from it)
#
# Safety at every hop: the next stage starts ONLY if the previous log shows
# "[OK] Adapter saved". On a crash the chain stops and the GPU stays free.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ks_seamless_after_ps.ps1' -WindowStyle Hidden

param(
    [int]$PsPid   = 14324,
    [int]$KsSteps = 2500,   # ~1 epoch of 20k IndicVoices samples at effective batch 8
    [int]$PsSteps = 1000    # same as the original ps Seamless run, for comparability
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ks_seamless.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Start-Training($args_, $outLog, $errLog) {
    $p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
        -ArgumentList (@('-u','finetune_seamless.py') + $args_) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -WindowStyle Hidden -PassThru
    return $p
}

function Wait-Gone($procId) {
    while (Get-Process -Id $procId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 120 }
}

function Saved-Ok($logPath) {
    $tail = Get-Content $logPath -Tail 60 -Raw -ErrorAction SilentlyContinue
    return ($tail -match '\[OK\] Adapter saved')
}

# ── Stage 1: ps_lv3 Whisper ───────────────────────────────────────────────────
Log "chain armed: waiting for ps_lv3 (PID $PsPid) to exit"
Wait-Gone $PsPid
Log "ps_lv3 process gone"
Start-Sleep -Seconds 60   # let CT2 conversion handles close, VRAM release

if (-not (Saved-Ok "$root\logs\ps_lv3_train.log")) {
    Log "ps_lv3 did NOT save an adapter (crash?) - chain STOPPED. Resume: python finetune_whisper.py ps_lv3 --steps 2000 --save-steps 200 --no-cv --resume"
    exit 1
}

# ── Stage 2: Kashmiri Seamless ────────────────────────────────────────────────
Log "ps_lv3 adapter confirmed - launching ks Seamless training ($KsSteps steps)"
$ks = Start-Training @('ks','--steps',"$KsSteps") "$root\logs\ks_seamless_train.log" "$root\logs\ks_seamless_train_err.log"
Log "ks Seamless training started, PID $($ks.Id)"

Wait-Gone $ks.Id
Log "ks Seamless process gone"
Start-Sleep -Seconds 60

if (-not (Saved-Ok "$root\logs\ks_seamless_train.log")) {
    Log "ks Seamless did NOT save an adapter (crash?) - chain STOPPED. Resume: python finetune_seamless.py ks --steps $KsSteps --resume"
    exit 1
}

# ── Stage 3: Pashto Seamless retrain (fixed labels) ──────────────────────────
# Archive the old wrong-label run (adapter + wrong-label Arrow feature caches)
# so the retrain preprocesses from scratch. Same-volume rename = instant.
$oldPs   = "$root\finetune_runs_seamless\ps"
$archive = "$root\finetune_runs_seamless\ps_PRE_LABELFIX_2026-07-12"
if ((Test-Path $oldPs) -and -not (Test-Path $archive)) {
    Rename-Item $oldPs $archive
    Log "archived old ps Seamless run (wrong labels) -> ps_PRE_LABELFIX_2026-07-12"
}

Log "ks adapter confirmed - launching ps Seamless RETRAIN with fixed labels ($PsSteps steps)"
$ps2 = Start-Training @('ps','--steps',"$PsSteps") "$root\logs\ps_seamless_retrain.log" "$root\logs\ps_seamless_retrain_err.log"
Log "ps Seamless retrain started, PID $($ps2.Id) - chain complete; watcher exiting. After it finishes: eval vs old 41.22 (wrong labels) and Whisper-FT 38.55, then scripts\restore_power.ps1"

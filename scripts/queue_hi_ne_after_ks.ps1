# Training chain watcher, stage 3+ (armed 2026-07-17). Runs AFTER
# queue_ks_continue_after_ps_cv.ps1's ks continuation finishes:
#   1. wait for "[OK] Adapter saved" in logs\ks_seamless_continue.log
#   2. ks head-to-head eval WITH the probe decode fixes (min_new_tokens 2.5/s,
#      no_repeat_ngram 3) vs Whisper-ks 74.02
#   3. hi_iv training (FLEURS + IndicVoices-R Hindi, cap 20k) + n=100 eval
#   4. ne_iv training (FLEURS + IndicVoices-R Nepali, cap 20k) + n=100 eval
#
# Safety: each hop requires the previous stage's "[OK] Adapter saved" (whole-log
# scan). If ks was launched but its python died without saving, the chain stops.
# Hard timeout on the initial wait so an upstream-chain failure doesn't leave
# this watcher polling forever.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_hi_ne_after_ks.ps1' -WindowStyle Hidden

param(
    [int]$HiSteps = 3000,   # ~1 epoch of 20k IV + 2.1k FLEURS at effective batch 8
    [int]$NeSteps = 3000,
    [int]$MaxWaitHours = 24
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_hi_ne.log"
$ksLog = "$root\logs\ks_seamless_continue.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Saved-Ok($logPath) {
    $all = Get-Content $logPath -Raw -ErrorAction SilentlyContinue
    return ($all -match '\[OK\] Adapter saved')
}

function Ks-Python-Alive() {
    $ps = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $ps) {
        if ($p.CommandLine -match 'finetune_seamless\.py.+\bks\b') { return $true }
    }
    return $false
}

function Run-Blocking($args_, $outLog, $errLog) {
    $p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
        -ArgumentList (@('-u') + $args_) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    return $p.ExitCode
}

# ── Stage 1: wait for the ks continuation to save ────────────────────────────
Log "chain armed: waiting for ks continuation adapter ($ksLog)"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while (-not (Saved-Ok $ksLog)) {
    if ((Get-Date) -gt $deadline) {
        Log "TIMEOUT after $MaxWaitHours h without a ks adapter - chain STOPPED."
        exit 1
    }
    # crash detection: ks was launched (log exists) but its python is gone
    if ((Test-Path $ksLog) -and -not (Ks-Python-Alive)) {
        Start-Sleep -Seconds 180   # grace for handle flush / restart
        if (-not (Saved-Ok $ksLog) -and -not (Ks-Python-Alive)) {
            Log "ks continuation died without saving (see logs\ks_seamless_continue_err.log) - chain STOPPED. Resume: python finetune_seamless.py ks --steps 7500 --resume"
            exit 1
        }
    }
    Start-Sleep -Seconds 120
}
Log "ks adapter confirmed"
Start-Sleep -Seconds 60   # VRAM release

# ── Stage 2: ks head-to-head eval with probe decode settings ─────────────────
Log "running ks eval (min_new_tokens 2.5/s, no_repeat_ngram 3) vs Whisper 74.02"
$rc = Run-Blocking @('scripts\eval\eval_ks_seamless.py','--min-tok-per-sec','2.5','--no-repeat-ngram','3') `
    "$root\logs\ks_continue_eval.log" "$root\logs\ks_continue_eval_err.log"
Log "ks eval finished (exit $rc) - docs\ks_seamless_results.json"
Start-Sleep -Seconds 30

# ── Stage 3: hi_iv training + eval ───────────────────────────────────────────
Log "launching hi_iv training ($HiSteps steps)"
$rc = Run-Blocking @('finetune_seamless.py','hi_iv','--steps',"$HiSteps") `
    "$root\logs\hi_iv_train.log" "$root\logs\hi_iv_train_err.log"
if (-not (Saved-Ok "$root\logs\hi_iv_train.log")) {
    Log "hi_iv did NOT save an adapter (exit $rc) - chain STOPPED. Resume: python finetune_seamless.py hi_iv --steps $HiSteps --resume"
    exit 1
}
Start-Sleep -Seconds 30
Log "hi_iv adapter confirmed - running n=100 eval (deployed hi adapter: 13.94)"
$rc = Run-Blocking @('scripts\eval\eval_seamless_ft.py','--lang','hi_iv') `
    "$root\logs\hi_iv_eval.log" "$root\logs\hi_iv_eval_err.log"
Log "hi_iv eval finished (exit $rc)"
Start-Sleep -Seconds 30

# ── Stage 4: ne_iv training + eval ───────────────────────────────────────────
Log "launching ne_iv training ($NeSteps steps)"
$rc = Run-Blocking @('finetune_seamless.py','ne_iv','--steps',"$NeSteps") `
    "$root\logs\ne_iv_train.log" "$root\logs\ne_iv_train_err.log"
if (-not (Saved-Ok "$root\logs\ne_iv_train.log")) {
    Log "ne_iv did NOT save an adapter (exit $rc) - chain STOPPED. Resume: python finetune_seamless.py ne_iv --steps $NeSteps --resume"
    exit 1
}
Start-Sleep -Seconds 30
Log "ne_iv adapter confirmed - running n=100 eval (zero-shot ne: 28.46)"
$rc = Run-Blocking @('scripts\eval\eval_seamless_ft.py','--lang','ne_iv') `
    "$root\logs\ne_iv_eval.log" "$root\logs\ne_iv_eval_err.log"
Log "ne_iv eval finished (exit $rc) - FULL CHAIN COMPLETE. Next: report updates (generate_report_pdf.py + generate_finetune_pptx.py) and scripts\restore_power.ps1"

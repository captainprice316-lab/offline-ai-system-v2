# Training chain watcher, post-ks (armed 2026-07-17, REPLACES
# queue_hi_ne_after_ks.ps1 + queue_ps_bal_after_chain.ps1 — user reordered:
# Pashto right after Kashmiri, Hindi and Nepali afterwards):
#   1. wait for "[OK] Adapter saved" in logs\ks_seamless_continue.log
#   2. ks head-to-head eval WITH probe decode fixes vs Whisper-ks 74.02
#   3. ps_bal training (FLEURS x8 + CV cap 10k, LoRA r16 q/k/v/out) + n=100 eval
#      (bars: Whisper-medium 38.55, FLEURS-only adapter 41.30)
#   4. hi_iv training (FLEURS + IndicVoices-R, cap 20k) + n=100 eval (bar 13.94)
#   5. ne_iv training (FLEURS + IndicVoices-R, cap 20k) + n=100 eval (bar 28.46)
#
# Same safety as its predecessors: every hop gated on "[OK] Adapter saved"
# (whole-log scan), ks crash detection, hard timeout on the initial wait.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_post_ks.ps1' -WindowStyle Hidden

param(
    [int]$PsSteps = 3000,
    [int]$HiSteps = 3000,
    [int]$NeSteps = 3000,
    [int]$MaxWaitHours = 24
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_post_ks.log"
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

function Train-And-Eval($name, $steps, $bar) {
    Log "launching $name training ($steps steps)"
    $rc = Run-Blocking @('finetune_seamless.py',"$name",'--steps',"$steps") `
        "$root\logs\${name}_train.log" "$root\logs\${name}_train_err.log"
    if (-not (Saved-Ok "$root\logs\${name}_train.log")) {
        Log "$name did NOT save an adapter (exit $rc) - chain STOPPED. Resume: python finetune_seamless.py $name --steps $steps --resume"
        exit 1
    }
    Start-Sleep -Seconds 30
    Log "$name adapter confirmed - running n=100 eval ($bar)"
    $rc = Run-Blocking @('scripts\eval\eval_seamless_ft.py','--lang',"$name") `
        "$root\logs\${name}_eval.log" "$root\logs\${name}_eval_err.log"
    Log "$name eval finished (exit $rc)"
    Start-Sleep -Seconds 30
}

# ── Stage 1: wait for the ks continuation to save ────────────────────────────
Log "chain armed (reordered: ps_bal -> hi_iv -> ne_iv): waiting for ks adapter"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while (-not (Saved-Ok $ksLog)) {
    if ((Get-Date) -gt $deadline) {
        Log "TIMEOUT after $MaxWaitHours h without a ks adapter - chain STOPPED."
        exit 1
    }
    if ((Test-Path $ksLog) -and -not (Ks-Python-Alive)) {
        Start-Sleep -Seconds 180
        if (-not (Saved-Ok $ksLog) -and -not (Ks-Python-Alive)) {
            Log "ks continuation died without saving - chain STOPPED. Resume: python finetune_seamless.py ks --steps 7500 --resume"
            exit 1
        }
    }
    Start-Sleep -Seconds 120
}
Log "ks adapter confirmed"
Start-Sleep -Seconds 60

# ── Stage 2: ks head-to-head eval with probe decode settings ─────────────────
Log "running ks eval (min_new_tokens 2.5/s, no_repeat_ngram 3) vs Whisper 74.02"
$rc = Run-Blocking @('scripts\eval\eval_ks_seamless.py','--min-tok-per-sec','2.5','--no-repeat-ngram','3') `
    "$root\logs\ks_continue_eval.log" "$root\logs\ks_continue_eval_err.log"
Log "ks eval finished (exit $rc) - docs\ks_seamless_results.json"
Start-Sleep -Seconds 30

# ── Stages 3-5: ps_bal, hi_iv, ne_iv ─────────────────────────────────────────
Train-And-Eval "ps_bal" $PsSteps "bars: Whisper 38.55, FLEURS-only adapter 41.30"
Train-And-Eval "hi_iv"  $HiSteps "bar: deployed hi adapter 13.94"
Train-And-Eval "ne_iv"  $NeSteps "bar: zero-shot ne 28.46"

Log "ALL QUEUED WORK DONE. Next: report updates (generate_report_pdf.py + generate_finetune_pptx.py) and scripts\restore_power.ps1"

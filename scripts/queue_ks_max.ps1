# Kashmiri attempt #4 chain (armed 2026-07-19): ks_max — r=32+MLP LoRA plus a
# TRAINABLE __kas__ embedding row (PEFT trainable_token_indices; every prior
# attempt ran with a frozen urd-init conditioning vector). Full 24k data,
# 7500 steps (~2.5 epochs, early stop patience 3).
#   1. training (~12-13 h)
#   2. head-to-head eval with decode fixes vs Whisper-ks 74.02 / 81.46
#      (prior Seamless best: 88.42 / 88.08)
#   3. restore_power.ps1 (auto, even on failure)
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ks_max.ps1' -WindowStyle Hidden

param([int]$Steps = 7500)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ks_max.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Run-Blocking($args_, $outLog, $errLog) {
    $p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
        -ArgumentList (@('-u') + $args_) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    return $p.ExitCode
}

Log "chain started: ks_max training ($Steps steps)"
$rc = Run-Blocking @('finetune_seamless.py','ks_max','--steps',"$Steps") `
    "$root\logs\ks_max_train.log" "$root\logs\ks_max_train_err.log"
$all = Get-Content "$root\logs\ks_max_train.log" -Raw -ErrorAction SilentlyContinue
if ($all -notmatch '\[OK\] Adapter saved') {
    Log "ks_max did NOT save an adapter (exit $rc) - chain STOPPED. Resume: python finetune_seamless.py ks_max --steps $Steps --resume"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restore_power.ps1" | Out-Null
    Log "power restored after failure"
    exit 1
}
Start-Sleep -Seconds 30

Log "ks_max adapter confirmed - head-to-head eval (bars: Whisper 74.02 / 81.46; prior best 88.42 / 88.08)"
$rc = Run-Blocking @('scripts\eval\eval_ks_seamless.py','--min-tok-per-sec','2.5','--no-repeat-ngram','3','--adapter-dir','finetune_runs_seamless\ks_max\adapter') `
    "$root\logs\ks_max_eval.log" "$root\logs\ks_max_eval_err.log"
Log "eval finished (exit $rc) - docs\ks_max_seamless_results.json"

& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restore_power.ps1" | Out-Null
Log "power restored. ks_max CHAIN COMPLETE"

# Pashto attempt #4 chain (armed 2026-07-18). GPU is free — starts immediately:
#   1. ps_bal2 training (r=32 a=64, q/k/v/out + fc1/fc2 MLP; same balanced data
#      as ps_bal), 4000 steps
#   2. n=100 held-out eval           (bars: Whisper 38.55; ps_bal 39.72)
#   3. decode probe                  (ps_bal best decode was 38.88)
#   4. degradation sweep + score     (rows: seamless_ft_ps_bal2 — wanted for the
#      paper win or lose)
#   5. restore_power.ps1             (training power profile re-applied by the
#      arming session; this puts it back)
#
# Marker-gated after training; a crash stops the chain and still restores power.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ps_bal2.ps1' -WindowStyle Hidden

param([int]$Steps = 4000)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ps_bal2.log"

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

Log "chain started: ps_bal2 training ($Steps steps)"
$rc = Run-Blocking @('finetune_seamless.py','ps_bal2','--steps',"$Steps") `
    "$root\logs\ps_bal2_train.log" "$root\logs\ps_bal2_train_err.log"
$all = Get-Content "$root\logs\ps_bal2_train.log" -Raw -ErrorAction SilentlyContinue
if ($all -notmatch '\[OK\] Adapter saved') {
    Log "ps_bal2 did NOT save an adapter (exit $rc) - chain STOPPED. Resume: python finetune_seamless.py ps_bal2 --steps $Steps --resume"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restore_power.ps1" | Out-Null
    Log "power restored after failure"
    exit 1
}
Start-Sleep -Seconds 30

Log "ps_bal2 adapter confirmed - n=100 eval (bars: Whisper 38.55, ps_bal 39.72)"
$rc = Run-Blocking @('scripts\eval\eval_seamless_ft.py','--lang','ps_bal2') `
    "$root\logs\ps_bal2_eval.log" "$root\logs\ps_bal2_eval_err.log"
Log "eval finished (exit $rc)"
Start-Sleep -Seconds 30

Log "decode probe (ps_bal best was 38.88 @ beam5 lp0.8)"
$rc = Run-Blocking @('scripts\eval\probe_ps_decode.py','--adapter','ps_bal2') `
    "$root\logs\ps_bal2_probe.log" "$root\logs\ps_bal2_probe_err.log"
Log "probe finished (exit $rc) - docs\ps_bal2_decode_probe.json"
Start-Sleep -Seconds 30

Log "degradation sweep (seamless_ft_ps_bal2, 5 cond x 30)"
$rc = Run-Blocking @('scripts\eval\wer_robustness_eval.py','--systems','seamless_ft','--langs','ps','--adapter-name','ps_bal2') `
    "$root\logs\ps_bal2_sweep.log" "$root\logs\ps_bal2_sweep_err.log"
Log "sweep finished (exit $rc) - scoring"
$rc = Run-Blocking @('scripts\eval\score_wer_robustness.py') `
    "$root\logs\ps_bal2_sweep_score.log" "$root\logs\ps_bal2_sweep_score_err.log"
Log "scoring finished (exit $rc)"

& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restore_power.ps1" | Out-Null
Log "power restored. PS_BAL2 CHAIN COMPLETE - all decision data in: docs\seamless_ft_results.json (ps_bal2 row), docs\ps_bal2_decode_probe.json, eval_data\wer_robustness_results.csv"

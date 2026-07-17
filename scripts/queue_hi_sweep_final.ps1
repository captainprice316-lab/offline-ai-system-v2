# Training chain watcher, stage 7 — FINAL (armed 2026-07-17). Runs AFTER
# queue_ps_probe_after_ks_r16.ps1 logs "EVERYTHING DONE":
#   1. hi_iv degradation sweep — seamless_ft with --adapter-name hi_iv, lang hi,
#      5 conditions x 30 clips (~25 min). Rows tagged seamless_ft_hi_iv, directly
#      comparable to the deployed adapter's existing seamless_ft rows and
#      seamless_zs in the same JSONL.
#   2. score_wer_robustness.py -> refreshed eval_data/wer_robustness_results.csv
#
# Deployment gate this answers: hi_iv (clean 12.91 vs deployed 13.94) must win
# the majority of degradation conditions before asr.seamless_adapters swaps.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_hi_sweep_final.ps1' -WindowStyle Hidden

param([int]$MaxWaitHours = 44)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_hi_sweep.log"
$upstream = "$root\logs\queue_ps_probe.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Upstream-State() {
    $all = Get-Content $upstream -Raw -ErrorAction SilentlyContinue
    if ($all -match 'EVERYTHING DONE') { return 'complete' }
    if ($all -match 'STOPPED|TIMEOUT') { return 'failed' }
    return 'waiting'
}

Log "chain armed: waiting for EVERYTHING DONE in queue_ps_probe.log"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Upstream-State) -eq 'waiting') {
    if ((Get-Date) -gt $deadline) { Log "TIMEOUT after $MaxWaitHours h - sweep NOT run."; exit 1 }
    Start-Sleep -Seconds 180
}
if ((Upstream-State) -eq 'failed') {
    Log "upstream STOPPED - sweep NOT run. Manual: python scripts\eval\wer_robustness_eval.py --systems seamless_ft --langs hi --adapter-name hi_iv"
    exit 1
}
Log "upstream complete"
Start-Sleep -Seconds 60

Log "running hi_iv degradation sweep (5 conditions x 30 clips)"
$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','scripts\eval\wer_robustness_eval.py','--systems','seamless_ft','--langs','hi','--adapter-name','hi_iv' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\hi_iv_sweep.log" `
    -RedirectStandardError  "$root\logs\hi_iv_sweep_err.log" `
    -WindowStyle Hidden -PassThru
$p.WaitForExit()
Log "hi_iv sweep finished (exit $($p.ExitCode))"
Start-Sleep -Seconds 30

# ne_iv won clean speech too (24.34 vs ZS 28.46) - same gate before its
# first-ever deployment (bar: the existing seamless_zs ne rows)
Log "running ne_iv degradation sweep (5 conditions x 30 clips)"
$p2 = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','scripts\eval\wer_robustness_eval.py','--systems','seamless_ft','--langs','ne','--adapter-name','ne_iv' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ne_iv_sweep.log" `
    -RedirectStandardError  "$root\logs\ne_iv_sweep_err.log" `
    -WindowStyle Hidden -PassThru
$p2.WaitForExit()
Log "ne_iv sweep finished (exit $($p2.ExitCode)) - scoring"

$s = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','scripts\eval\score_wer_robustness.py' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\hi_iv_sweep_score.log" `
    -RedirectStandardError  "$root\logs\hi_iv_sweep_score_err.log" `
    -WindowStyle Hidden -PassThru
$s.WaitForExit()
Log "scoring finished (exit $($s.ExitCode)) - eval_data\wer_robustness_results.csv refreshed. ABSOLUTELY EVERYTHING FINISHED - reports + scripts\restore_power.ps1 remain."

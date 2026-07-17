# Training chain watcher, stage 6 (armed 2026-07-17). Runs AFTER
# queue_ks_r16_after_chain.ps1 logs "ENTIRE QUEUE DONE":
#   ps_bal decode probe (greedy sanity + beam/length variants) on the same
#   n=100 held-out set — probing whether decode settings bridge the remaining
#   1.17 pp to Whisper-medium's 38.55. ~15-20 min of GPU.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ps_probe_after_ks_r16.ps1' -WindowStyle Hidden

param([int]$MaxWaitHours = 40)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ps_probe.log"
$upstream = "$root\logs\queue_ks_r16.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Upstream-State() {
    $all = Get-Content $upstream -Raw -ErrorAction SilentlyContinue
    if ($all -match 'ENTIRE QUEUE DONE') { return 'complete' }
    if ($all -match 'STOPPED|TIMEOUT')   { return 'failed' }
    return 'waiting'
}

Log "chain armed: waiting for ENTIRE QUEUE DONE in queue_ks_r16.log"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Upstream-State) -eq 'waiting') {
    if ((Get-Date) -gt $deadline) { Log "TIMEOUT after $MaxWaitHours h - probe NOT run."; exit 1 }
    Start-Sleep -Seconds 180
}
if ((Upstream-State) -eq 'failed') {
    Log "upstream STOPPED - probe NOT run. Manual: python scripts\eval\probe_ps_decode.py"
    exit 1
}
Log "upstream complete"
Start-Sleep -Seconds 60

Log "running ps_bal decode probe (bar: Whisper 38.55; ps_bal greedy 39.72)"
$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','scripts\eval\probe_ps_decode.py' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ps_bal_probe.log" `
    -RedirectStandardError  "$root\logs\ps_bal_probe_err.log" `
    -WindowStyle Hidden -PassThru
$p.WaitForExit()
Log "ps_bal decode probe finished (exit $($p.ExitCode)) - docs\ps_bal_decode_probe.json. EVERYTHING DONE - reports + scripts\restore_power.ps1 remain."

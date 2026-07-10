# Queued: Mandarin baseline-vs-Seamless under radio degradation.
#
# The routing sweep compared fine-tuned Whisper vs Seamless. But the corrected n=100
# eval showed the un-fine-tuned large-v3 baseline beats BOTH on clean zh
# (10.99% vs ft 14.22 vs seamless 11.69). VANI's input is noisy radio, so the routing
# call for zh depends on whether that clean-speech win survives degradation.
#
# This waits for the in-flight corrected-eval run to finish (so the GPU is free), then
# sweeps whisper_base on zh across all conditions and adds it to the same hyps file that
# already holds whisper_ft and seamless_zs for zh. score_wer_robustness.py then compares
# all three.
#
# Completion signal for the current run: docs/seamless_ft_results.json reaching 6 langs
# is the last thing scripts/eval/rerun_corrected_evals.ps1 writes.

param(
    [int]$PollSeconds     = 90,
    [int]$MaxWaitMinutes  = 240,
    [int]$Samples         = 30
)

$repo    = 'C:\Users\vis15\offline_ai_system_v2'
$py      = Join-Path $repo 'venv\Scripts\python.exe'
$sentinel = Join-Path $repo 'docs\seamless_ft_results.json'

function Done-Count {
    if (-not (Test-Path $sentinel)) { return 0 }
    try { return (Get-Content $sentinel -Raw | ConvertFrom-Json).Count } catch { return 0 }
}

Write-Host "[queue] waiting for the corrected-eval run (need 6 langs in seamless_ft_results.json)..." -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
while ((Done-Count) -lt 6) {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[queue] gave up after $MaxWaitMinutes min (have $(Done-Count)/6). Not starting." -ForegroundColor Red
        exit 1
    }
    Start-Sleep -Seconds $PollSeconds
}
Write-Host "[queue] prior run complete. GPU free. Starting zh baseline noise sweep." -ForegroundColor Green

# whisper_base on zh, all 5 conditions. Appends to the existing hyps file (resume-safe).
& $py (Join-Path $repo 'scripts\eval\wer_robustness_eval.py') `
    --systems whisper_base --langs zh `
    --conditions clean bandpass awgn_10 awgn_0 codec_mp3 `
    --n $Samples *> (Join-Path $repo 'logs\zh_base_noise.log')
Write-Host "[queue] sweep exit=$LASTEXITCODE" -ForegroundColor Green

# Re-score everything (whisper_ft, seamless_zs, whisper_base) into the CSV + tables.
& $py (Join-Path $repo 'scripts\eval\score_wer_robustness.py') `
    *> (Join-Path $repo 'logs\zh_base_noise_score.log')

Write-Host "[queue] DONE. See logs\zh_base_noise_score.log and eval_data\wer_robustness_results.csv" -ForegroundColor Cyan

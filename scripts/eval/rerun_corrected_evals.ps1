# Re-run the corrected evaluations, one language at a time so progress survives a kill.
#
# compare_all_models.py only writes its results JSON after ALL languages finish, so a
# single interrupted run loses everything. Invoked per-language with --lang, it merges
# into the existing JSON instead -- so each language is durable the moment it completes.
#
# Resumable: languages already present in docs/model_comparison_results.json are skipped.
# Delete that file (or a language's entry) to force a re-run.
#
#   powershell -File scripts\eval\rerun_corrected_evals.ps1
#   powershell -File scripts\eval\rerun_corrected_evals.ps1 -Samples 30   # quick pass

param(
    [int]$Samples = 100,
    [string[]]$Langs = @('pa','ps','ur','ne','zh','hi','ks'),
    [switch]$SkipSeamlessFt
)

$ErrorActionPreference = 'Continue'
$repo   = 'C:\Users\vis15\offline_ai_system_v2'
$py     = Join-Path $repo 'venv\Scripts\python.exe'
$outJson = Join-Path $repo 'docs\model_comparison_results.json'
$logDir = Join-Path $repo 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-DoneLangs {
    if (-not (Test-Path $outJson)) { return @() }
    try { return (Get-Content $outJson -Raw | ConvertFrom-Json | ForEach-Object { $_.lang }) }
    catch { return @() }
}

Write-Host "=== compare_all_models: $Samples samples/lang, TRUE large-v3 baseline ===" -ForegroundColor Cyan
foreach ($lang in $Langs) {
    $done = Get-DoneLangs
    if ($done -contains $lang) {
        Write-Host "[skip] $lang already in results JSON" -ForegroundColor DarkGray
        continue
    }
    $log = Join-Path $logDir "rerun_compare_$lang.log"
    Write-Host "[run ] $lang -> $log" -ForegroundColor Green
    & $py (Join-Path $repo 'scripts\eval\compare_all_models.py') --lang $lang --samples $Samples *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $lang exited $LASTEXITCODE (see $log). Continuing." -ForegroundColor Red
    } else {
        Write-Host "[ok  ] $lang" -ForegroundColor Green
    }
}

if (-not $SkipSeamlessFt) {
    Write-Host "`n=== eval_seamless_ft: $Samples samples/lang ===" -ForegroundColor Cyan
    # No Kashmiri: SeamlessM4T v2 has no kas, and the urd-proxy fails (WER 109%, CER 69%).
    foreach ($lang in @('pa','ps','ur','ne','zh','hi')) {
        $log = Join-Path $logDir "rerun_seamlessft_$lang.log"
        Write-Host "[run ] seamless-ft $lang -> $log" -ForegroundColor Green
        & $py (Join-Path $repo 'scripts\eval\eval_seamless_ft.py') --lang $lang --samples $Samples *> $log
        if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] seamless-ft $lang exited $LASTEXITCODE" -ForegroundColor Red }
    }
}

Write-Host "`n=== ALL DONE ===" -ForegroundColor Cyan
Write-Host "results : docs\model_comparison_results.json  +  docs\seamless_ft_results.json"
Write-Host "raw hyps: eval_data\model_comparison_hyps.jsonl  +  eval_data\seamless_ft_hyps.jsonl"

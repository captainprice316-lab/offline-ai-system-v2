<#
  backup.ps1 - Backup of the irreplaceable VANI data to the external WD My Passport.
  Copies project code + personal folders, and the trained models / LoRA checkpoints
  DIRECTLY from their E: (SN7100) locations. Uses robocopy /E (copy, never delete).

  IMPORTANT: the models/finetune_runs live on E: behind NTFS junctions. Copying the
  project folder with /XJ SKIPS those junctions (so they are not broken/duplicated),
  therefore the E: targets MUST be backed up as their own jobs below - otherwise the
  trained models are silently missed. (This bit us on 2026-07-04.)

  The Passport's drive letter is NOT stable (was F: on 2026-07-04, G: on 2026-07-06),
  so the destination drive is auto-detected by volume label unless -Dest is given.

  Re-downloadable caches (hf_cache, hf_ks_temp) and venv are excluded by default.
#>
param(
    [string]$Dest = "",       # auto-detected from the "My Passport" label if empty
    [switch]$IncludeCaches    # also copy the 328 GB re-downloadable HF caches (E:)
)

$ErrorActionPreference = "Continue"

# ── Resolve destination (auto-detect the external drive letter) ────────────────
if ([string]::IsNullOrWhiteSpace($Dest)) {
    $vol = Get-Volume | Where-Object { $_.FileSystemLabel -eq 'My Passport' -and $_.DriveLetter } | Select-Object -First 1
    if (-not $vol) {
        Write-Host "External 'My Passport' drive not found. Plug it in, or pass -Dest <path>." -ForegroundColor Red
        exit 1
    }
    $Dest = "{0}:\vani_backup_{1}" -f $vol.DriveLetter, (Get-Date -Format 'yyyy-MM-dd')
    Write-Host "Auto-detected external drive $($vol.DriveLetter): ('My Passport')" -ForegroundColor Cyan
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$logDir = Join-Path $Dest "_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# source, destination-subfolder, excluded-dirs
$jobs = @(
    @{ Src="C:\Users\vis15\offline_ai_system_v2"; Name="offline_ai_system_v2"; XD=@("venv","__pycache__",".git") },
    @{ Src="E:\vani_models";                      Name="vani_models";            XD=@() },
    @{ Src="E:\finetune_runs";                    Name="finetune_runs";          XD=@() },
    @{ Src="E:\finetune_runs_seamless";           Name="finetune_runs_seamless"; XD=@() },
    @{ Src="$env:USERPROFILE\Desktop";            Name="Desktop";                XD=@() },
    @{ Src="$env:USERPROFILE\Documents";          Name="Documents";              XD=@() },
    @{ Src="$env:USERPROFILE\Downloads";          Name="Downloads";              XD=@() },
    @{ Src="$env:USERPROFILE\Pictures";           Name="Pictures";               XD=@() }
)
if ($IncludeCaches) {
    $jobs += @{ Src="E:\hf_cache";  Name="hf_cache";  XD=@() }
    $jobs += @{ Src="E:\hf_ks_temp"; Name="hf_ks_temp"; XD=@() }
}

Write-Host "=== VANI backup -> $Dest ===`n" -ForegroundColor Cyan
$results = @()
foreach ($j in $jobs) {
    if (-not (Test-Path $j.Src)) { Write-Host "SKIP (missing): $($j.Src)" -ForegroundColor DarkYellow; continue }
    $dst = Join-Path $Dest $j.Name
    $log = Join-Path $logDir ("log_" + $j.Name + ".txt")
    $xd  = @(); foreach ($d in $j.XD) { $xd += "/XD"; $xd += $d }
    Write-Host "Copying $($j.Src)  ->  $dst" -ForegroundColor Green
    robocopy $j.Src $dst /E /XJ @xd /R:1 /W:1 /MT:16 /NP /NFL /NDL /LOG:$log | Out-Null
    $code = $LASTEXITCODE
    $ok = $code -lt 8   # robocopy: 0-7 success, >=8 = failure
    $results += [pscustomobject]@{ Name=$j.Name; ExitCode=$code; OK=$ok; Log=$log }
    Write-Host ("  -> exit {0}  ({1})" -f $code, ($(if($ok){"OK"}else{"FAILED"}))) -ForegroundColor $(if($ok){"Green"}else{"Red"})
}

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
$results | Format-Table Name, ExitCode, OK -AutoSize

# ── Verify model.bin count (don't trust robocopy exit code alone) ──────────────
$srcBin = (Get-ChildItem 'E:\vani_models','E:\finetune_runs' -Recurse -Force -Filter 'model.bin' -EA SilentlyContinue).Count
$dstBin = (Get-ChildItem (Join-Path $Dest 'vani_models'),(Join-Path $Dest 'finetune_runs') -Recurse -Force -Filter 'model.bin' -EA SilentlyContinue).Count
Write-Host ("model.bin verify:  source E: {0}  vs  backup {1}  -> {2}" -f `
    $srcBin, $dstBin, ($(if($srcBin -eq $dstBin -and $dstBin -gt 0){"MATCH"}else{"MISMATCH - CHECK!"}))) `
    -ForegroundColor $(if($srcBin -eq $dstBin -and $dstBin -gt 0){"Green"}else{"Red"})

$failed = $results | Where-Object { -not $_.OK }
if ($failed -or $srcBin -ne $dstBin) {
    Write-Host "BACKUP INCOMPLETE - check logs in $logDir" -ForegroundColor Red
} else {
    $bytes = (Get-ChildItem $Dest -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum
    Write-Host ("ALL OK. Backup size: {0:N1} GB at {1}" -f ($bytes/1GB), $Dest) -ForegroundColor Green
}

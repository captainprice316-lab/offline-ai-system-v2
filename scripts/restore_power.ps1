# Restore the power settings captured before a long training/eval run.
#
# Reads logs/power_settings_before_training.json, written when sleep was disabled.
# Prefer this over the hardcoded defaults in restore_power_after_training.ps1,
# which assume 30/15 min sleep — not necessarily what this machine was set to.

param(
    [string]$Snapshot = "C:\Users\vis15\offline_ai_system_v2\logs\power_settings_before_training.json"
)

if (-not (Test-Path $Snapshot)) {
    Write-Error "No snapshot at $Snapshot - refusing to guess your power settings."
    exit 1
}

$s = Get-Content $Snapshot -Raw | ConvertFrom-Json

Write-Host "Restoring power settings captured $($s.saved_utc)" -ForegroundColor Cyan
powercfg /change standby-timeout-ac   $s.standby.AC
powercfg /change standby-timeout-dc   $s.standby.DC
powercfg /change hibernate-timeout-ac $s.hibernate.AC
powercfg /change hibernate-timeout-dc $s.hibernate.DC
powercfg /change monitor-timeout-ac   $s.monitor.AC
powercfg /change monitor-timeout-dc   $s.monitor.DC

Write-Host "  sleep      AC=$($s.standby.AC)   DC=$($s.standby.DC)"
Write-Host "  hibernate  AC=$($s.hibernate.AC)   DC=$($s.hibernate.DC)"
Write-Host "  monitor    AC=$($s.monitor.AC)   DC=$($s.monitor.DC)"

# The screensaver lives in HKCU, not in the power scheme, so powercfg cannot restore it.
# It was enabled (ScreenSaveActive=1) before the long runs.
Set-ItemProperty 'HKCU:\Control Panel\Desktop' -Name ScreenSaveActive -Value '1'
rundll32.exe user32.dll,UpdatePerUserSystemParameters
Write-Host "  screensaver re-enabled"

Write-Host "Restored." -ForegroundColor Green
Write-Host "Note: ASUS AsusOptimization / Armoury Crate panel power saving is not managed here." -ForegroundColor DarkYellow

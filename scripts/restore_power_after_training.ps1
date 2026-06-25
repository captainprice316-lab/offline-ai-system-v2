param(
    [string]$WatchLog = "C:\Users\vis15\offline_ai_system_v2\logs\finetune_ps_2000.log",
    [string]$CompletionMarker = "[OK] tokenizer.json copied"
)

Write-Output "[watcher] Monitoring: $WatchLog"
Write-Output "[watcher] Will restore power settings when training completes."

# Default Windows 11 Home power settings
$acMonitor  = 10   # minutes
$dcMonitor  = 5
$acSleep    = 30
$dcSleep    = 15
$acHibernate = 0
$dcHibernate = 180

function Restore-PowerSettings {
    powercfg /change monitor-timeout-ac  $acMonitor
    powercfg /change monitor-timeout-dc  $dcMonitor
    powercfg /change standby-timeout-ac  $acSleep
    powercfg /change standby-timeout-dc  $dcSleep
    powercfg /change hibernate-timeout-ac $acHibernate
    powercfg /change hibernate-timeout-dc $dcHibernate
    Write-Output "[restore] Power settings restored."
    Write-Output "  monitor  AC=$acMonitor min  DC=$dcMonitor min"
    Write-Output "  sleep    AC=$acSleep min   DC=$dcSleep min"
    Write-Output "  hibernate AC=$acHibernate  DC=$dcHibernate min"
}

# Poll every 60 seconds
while ($true) {
    if (Test-Path $WatchLog) {
        $tail = Get-Content $WatchLog -Tail 20 -ErrorAction SilentlyContinue
        if ($tail -match [regex]::Escape($CompletionMarker)) {
            Write-Output "[watcher] Completion marker found. Restoring power settings..."
            Restore-PowerSettings
            break
        }
    }
    Start-Sleep -Seconds 60
}

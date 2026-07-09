# VANI demo auto-startup (NODE-C). Launched by the "VANI_Demo_Startup" scheduled
# task at 07:00. Starts the Streamlit app + the local mock-failover server, then
# logs a health snapshot. Real NODE-A/B live on the partner machines and cannot be
# started from here.
$ErrorActionPreference = "Continue"
$root = "C:\Users\vis15\offline_ai_system_v2"
$py   = Join-Path $root "venv\Scripts\python.exe"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log  = Join-Path $logs "demo_startup.log"
"[{0}] VANI demo startup begin" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append $log

Set-Location $root

# 1. Local mock-failover server (insurance; leave idle until toggled in the GUI)
Start-Process -FilePath $py -WorkingDirectory $root `
  -ArgumentList "integration\mocks\demo_mock_server.py" -WindowStyle Minimized `
  -RedirectStandardOutput (Join-Path $logs "demo_mock.out.log") `
  -RedirectStandardError  (Join-Path $logs "demo_mock.err.log")

# 2. Streamlit app (boots in Auto mode -> auto-detects the real LAN nodes)
Start-Process -FilePath $py -WorkingDirectory $root `
  -ArgumentList "-m streamlit run app.py --server.port 8501 --server.headless true" `
  -WindowStyle Minimized `
  -RedirectStandardOutput (Join-Path $logs "app.out.log") `
  -RedirectStandardError  (Join-Path $logs "app.err.log")

# 3. Give them time to bind, then log a health snapshot
Start-Sleep -Seconds 25
function Probe($url) { try { (Invoke-WebRequest -Uri $url -TimeoutSec 4 -UseBasicParsing).StatusCode } catch { "down" } }
$app   = Probe "http://127.0.0.1:8501"
$mockA = Probe "http://127.0.0.1:8801/health"
$mockB = Probe "http://127.0.0.1:8802/health"
$realA = Probe "http://192.168.10.11:8801/health"
$realB = Probe "http://192.168.10.12:8802/health"
"[{0}] app=$app  mockA=$mockA mockB=$mockB  realA=$realA realB=$realB" -f (Get-Date -Format "HH:mm:ss") | Out-File -Append $log
"  -> open http://localhost:8501 . Real nodes: bring up on partner machines." | Out-File -Append $log

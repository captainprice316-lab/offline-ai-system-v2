@echo off
cd /d "C:\Users\vis15\offline_ai_system_v2"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8
echo [wrapper] ks_max2 degradation start %DATE% %TIME% >> "logs\ks_max2_deg.log"
"venv\Scripts\python.exe" scripts\eval\ks_max2_degradation_eval.py >> "logs\ks_max2_deg.log" 2>&1
echo [wrapper] ks_max2 degradation exited code %ERRORLEVEL% %DATE% %TIME% >> "logs\ks_max2_deg.log"

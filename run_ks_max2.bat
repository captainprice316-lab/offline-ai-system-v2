@echo off
cd /d "C:\Users\vis15\offline_ai_system_v2"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8
echo [wrapper] ks_max2 resume start %DATE% %TIME% >> "logs\ks_max2_train.log"
"venv\Scripts\python.exe" finetune_seamless.py ks_max2 --steps 12000 --resume >> "logs\ks_max2_train.log" 2>&1
echo [wrapper] ks_max2 exited code %ERRORLEVEL% %DATE% %TIME% >> "logs\ks_max2_train.log"

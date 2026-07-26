@echo off
cd /d "C:\Users\vis15\offline_ai_system_v2"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8
echo [wrapper] ks_max2 eval start %DATE% %TIME% >> "logs\ks_max2_eval.log"
"venv\Scripts\python.exe" scripts\eval\eval_ks_seamless.py --adapter-dir "finetune_runs_seamless\ks_max2\adapter" --min-tok-per-sec 2.5 --no-repeat-ngram 3 >> "logs\ks_max2_eval.log" 2>&1
echo [wrapper] ks_max2 eval exited code %ERRORLEVEL% %DATE% %TIME% >> "logs\ks_max2_eval.log"

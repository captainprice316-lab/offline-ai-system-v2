@echo off
cd /d "C:\Users\vis15\offline_ai_system_v2"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set PYTHONIOENCODING=utf-8
rem prevent Intel-Fortran (MKL) "forrtl error (200): window-CLOSE event" aborts
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
echo [wrapper] ps_aug2 start %DATE% %TIME% >> "logs\ps_aug2_train.log"
"venv\Scripts\python.exe" finetune_seamless.py ps_aug2 --steps 5000 --resume >> "logs\ps_aug2_train.log" 2>&1
echo [wrapper] ps_aug2 exited code %ERRORLEVEL% %DATE% %TIME% >> "logs\ps_aug2_train.log"

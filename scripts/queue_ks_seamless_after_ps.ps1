# Waits for the ps_lv3 Whisper training (detached python) to finish, then
# launches the Kashmiri SeamlessM4T LoRA run on the freed GPU.
#
# Safety: ks starts ONLY if ps_lv3's log shows the adapter was saved
# ("[OK] Adapter saved" — printed after training, before CT2 conversion).
# If ps_lv3 crashed, ks is NOT started, so the GPU stays free to --resume ps.
#
# Armed 2026-07-12 for ps PID 14324. Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ks_seamless_after_ps.ps1' -WindowStyle Hidden

param(
    [int]$PsPid = 14324,
    [int]$Steps = 2500   # ~1 epoch of 20k samples at effective batch 8
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ks_seamless.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

Log "queue armed: waiting for ps_lv3 (PID $PsPid) to exit"

while (Get-Process -Id $PsPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 120
}
Log "ps_lv3 process gone"

# settle: let CT2 conversion file handles close and VRAM release
Start-Sleep -Seconds 60

$tail = Get-Content "$root\logs\ps_lv3_train.log" -Tail 60 -Raw -ErrorAction SilentlyContinue
if ($tail -notmatch '\[OK\] Adapter saved') {
    Log "ps_lv3 did NOT save an adapter (crash?) - NOT starting ks. Resume ps with: python finetune_whisper.py ps_lv3 --steps 2000 --save-steps 200 --no-cv --resume"
    exit 1
}
Log "ps_lv3 adapter confirmed saved - launching ks Seamless training ($Steps steps)"

$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','finetune_seamless.py','ks','--steps',"$Steps" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ks_seamless_train.log" `
    -RedirectStandardError  "$root\logs\ks_seamless_train_err.log" `
    -WindowStyle Hidden -PassThru

Log "ks Seamless training started, PID $($p.Id), logs at logs\ks_seamless_train*.log"

# Training chain watcher (armed 2026-07-17):
#   1. wait for the ps_cv Seamless run (FLEURS + CV-20 Pashto, detached python) to finish
#   2. launch the Kashmiri Seamless CONTINUATION: resume from checkpoint-2500 and
#      train to 7500 total steps (= 2 more epochs of the 20k IndicVoices set).
#      Rationale: the 1-epoch adapter's eval_loss was still descending (2.46, no
#      plateau) and the decode probe showed the 129.29 WER is dominated by
#      early-EOS under-generation (min_new_tokens alone: 128.28 -> 94.31 on a
#      50-sample subset). More epochs + decode fixes is the recovery path.
#
# Safety: the ks stage starts ONLY if the ps_cv log shows "[OK] Adapter saved"
# (whole-log scan — a -Tail window falsely stopped the 2026-07-13 chain).
# On a crash the chain stops and the GPU stays free.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','scripts\queue_ks_continue_after_ps_cv.ps1' -WindowStyle Hidden

param(
    [int]$PsCvPid = 20016,
    [int]$KsSteps = 7500    # total steps: 2500 done + 5000 continuation
)

$root = "C:\Users\vis15\offline_ai_system_v2"
$qlog = "$root\logs\queue_ks_continue.log"

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $qlog -Append -Encoding utf8 }

function Saved-Ok($logPath) {
    $all = Get-Content $logPath -Raw -ErrorAction SilentlyContinue
    return ($all -match '\[OK\] Adapter saved')
}

# ── Stage 1: wait for ps_cv ──────────────────────────────────────────────────
Log "chain armed: waiting for ps_cv Seamless (PID $PsCvPid) to exit"
while (Get-Process -Id $PsCvPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 120 }
Log "ps_cv process gone"
Start-Sleep -Seconds 60   # VRAM release, file handles close

if (-not (Saved-Ok "$root\logs\ps_cv_seamless_train.log")) {
    Log "ps_cv did NOT save an adapter (crash?) - chain STOPPED. Resume: python finetune_seamless.py ps_cv --steps 4000 --resume"
    exit 1
}

# ── Stage 2: ks Seamless continuation ────────────────────────────────────────
Log "ps_cv adapter confirmed - launching ks Seamless CONTINUATION (resume -> $KsSteps total steps)"
$ks = Start-Process -FilePath "$root\venv\Scripts\python.exe" `
    -ArgumentList '-u','finetune_seamless.py','ks','--steps',"$KsSteps",'--resume' `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\ks_seamless_continue.log" `
    -RedirectStandardError  "$root\logs\ks_seamless_continue_err.log" `
    -WindowStyle Hidden -PassThru
Log "ks continuation started, PID $($ks.Id) - watcher exiting. After it finishes: eval with min_new_tokens decode (see probe_ks_decode results) vs Whisper-ks 74.02, then ps_cv n=100 eval vs 38.55."

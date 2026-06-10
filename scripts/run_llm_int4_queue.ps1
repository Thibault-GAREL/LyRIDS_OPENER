# File d'attente sequentielle des baselines LLM 7B int4 (un seul GPU 6 Go).
# A lancer DETACHEE (Start-Process) pour survivre a la fermeture de session.
#
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','d:\Loisir\Code_python\LyRIDS_Opener\scripts\run_llm_int4_queue.ps1' -WindowStyle Hidden
#
# Chaque run a --resume : si ca coupe, relancer reprend ou c'etait.
# Ajuste $maxEval / commente des modeles selon le temps dispo (7B genratif = lent).

$ErrorActionPreference = 'Continue'
$py   = 'c:\0-Code_py_temp\pytorch_cuda_env\Scripts\python.exe'
$root = 'd:\Loisir\Code_python\LyRIDS_Opener'
Set-Location $root

$logdir = "C:\0-Code_py_temp\0-log_progress\$(Get-Date -Format 'yyyy-MM-dd')-LyRIDS_Opener-llm_int4"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$ts      = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$master  = Join-Path $logdir "QUEUE_$ts.log"
$maxEval = 200   # phrases test / dataset (modeste car 7B genratif lent)

function Log($m) { Add-Content -Path $master -Value "[$(Get-Date -Format 'HH:mm:ss')] $m" }

# Ordre : du moins cher (1 generation/phrase) au plus cher (UniNER = K gen/phrase).
$models = @('gner_llama','uniner','uniner_def','gollie')

Log "QUEUE LLM int4 START (pid=$PID, max_eval=$maxEval)"
foreach ($m in $models) {
    Log "$m : start"
    & $py -u -m scripts.baselines.run_llm_int4 `
        --model $m --max-eval $maxEval --resume --purge-cache `
        *> (Join-Path $logdir "${m}_$ts.log")
    Log "$m : done (exit=$LASTEXITCODE)"
}
Log "QUEUE LLM int4 DONE"

# File d'attente sequentielle des baselines GPU (un seul GPU ~6 Go -> pas de
# parallele). Concue pour etre lancee DETACHEE via Start-Process, de sorte
# qu'elle survive a la fermeture de la session Claude / du terminal.
#
# Lancement (depuis n'importe ou) :
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','d:\Loisir\Code_python\LyRIDS_Opener\scripts\run_baselines_queue.ps1' -WindowStyle Hidden
#
# Suivi :
#   Get-Content <logdir>\QUEUE_*.log -Wait -Tail 30

$ErrorActionPreference = 'Continue'
$py   = 'c:\0-Code_py_temp\pytorch_cuda_env\Scripts\python.exe'
$root = 'd:\Loisir\Code_python\LyRIDS_Opener'
Set-Location $root

$logdir = "C:\0-Code_py_temp\0-log_progress\$(Get-Date -Format 'yyyy-MM-dd')-LyRIDS_Opener-baselines"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$ts     = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$master = Join-Path $logdir "QUEUE_$ts.log"

function Log($msg) {
    Add-Content -Path $master -Value "[$(Get-Date -Format 'HH:mm:ss')] $msg"
}

Log "QUEUE START (pid=$PID)"

# --- 1) GNER T5-base (13 datasets, 3 axes) - resume si partiel ---
Log "GNER T5-base : start"
& $py -u -m scripts.baselines.run_gner `
    --max-test 1000 --batch-size 16 --timing-sample 64 --resume `
    *> (Join-Path $logdir "gner_$ts.log")
Log "GNER T5-base : done (exit=$LASTEXITCODE)"

# --- 2) GLiNER S (13 datasets) ---
Log "GLiNER S : start"
& $py -u -m scripts.baselines.run_gliner `
    --max-test 1000 --checkpoint urchade/gliner_small-v2.1 `
    --output-dir outputs/results/baselines/gliner_S `
    *> (Join-Path $logdir "gliner_S_$ts.log")
Log "GLiNER S : done (exit=$LASTEXITCODE)"

# --- 3) GLiNER M (13 datasets) ---
Log "GLiNER M : start"
& $py -u -m scripts.baselines.run_gliner `
    --max-test 1000 --checkpoint urchade/gliner_medium-v2.1 `
    --output-dir outputs/results/baselines/gliner_M `
    *> (Join-Path $logdir "gliner_M_$ts.log")
Log "GLiNER M : done (exit=$LASTEXITCODE)"

Log "QUEUE DONE"

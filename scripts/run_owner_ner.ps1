# OWNER end-to-end (mode 'ner' : detection DeBERTa + typing) sur les 13 datasets.
# Demarre APRES la file typing-gold (run_owner_all.ps1) car un seul GPU.
# A lancer DETACHE (Start-Process).
#
# AMI -> metrique ner_test_ami (lue par owner_collect en OWNER-e2e).
# Temps wall-clock -> OWNER_NER_QUEUE.log (cout total MD+typing entrainement+eval).

$ErrorActionPreference = 'Continue'
$ownerPy  = 'D:\conda_envs\owner\python.exe'
$root     = 'D:\Loisir\Code_python\LyRIDS_Opener'
$ownerDir = Join-Path $root 'external\OWNER'
$cfgDir   = Join-Path $ownerDir 'configs\lyrids_ner'
$logdir   = Join-Path $root 'outputs\owner_run'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

$env:MLFLOW_TRACKING_URI    = "file:///D:/Loisir/Code_python/LyRIDS_Opener/external/OWNER/mlruns"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$master      = Join-Path $logdir 'OWNER_NER_QUEUE.log'
$typingQueue = Join-Path $logdir 'OWNER_QUEUE.log'
function Log($m) { Add-Content -Path $master -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }

# Attendre que la file typing-gold soit terminee (GPU libre)
Log "WAIT : file typing-gold en cours..."
while (-not (Select-String -Path $typingQueue -Pattern 'QUEUE OWNER DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
Log "Typing-gold terminee -> GPU libre. Demarrage end-to-end."

$datasets = @(
    'crossner_ai','crossner_literature','crossner_music','crossner_science','crossner_politics',
    'wnut17','conll2003','mit_restaurant','bionlp2004','gum','gentle','fabner','mit_movie'
)

Set-Location $ownerDir
Log "QUEUE OWNER-NER START (pid=$PID, $($datasets.Count) datasets)"
foreach ($ds in $datasets) {
    Log "$ds : start"
    $t0 = Get-Date
    & $ownerPy -u -m owner.main --config-file="$cfgDir\$ds.toml" *> (Join-Path $logdir "ner_$ds.log")
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Log "$ds : done exit=$LASTEXITCODE ${sec}s"
}
Log "QUEUE OWNER-NER DONE"

# OWNER en protocole TRANSFERT (= papier OWNER) : entraine 1x sur conll2003
# (save_finetuned), puis CHARGE + teste chaque cible zero-shot (load_finetuned).
# A lancer DETACHE. Temps wall-clock dans OWNER_TRANSFER_QUEUE.log.
#   - train run  : cout d'entrainement (one-time).
#   - load runs  : INFERENCE par cible (detecter+encoder+cluster) -> base latence.

$ErrorActionPreference = 'Continue'
$ownerPy  = 'D:\conda_envs\owner\python.exe'
$root     = 'D:\Loisir\Code_python\LyRIDS_Opener'
$ownerDir = Join-Path $root 'external\OWNER'
$cfgDir   = Join-Path $ownerDir 'configs\lyrids_ner_transfer'
$logdir   = Join-Path $root 'outputs\owner_run'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$env:MLFLOW_TRACKING_URI     = "file:///D:/Loisir/Code_python/LyRIDS_Opener/external/OWNER/mlruns"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$master = Join-Path $logdir 'OWNER_TRANSFER_QUEUE.log'
function RunOne($name, $cfg) {
    Add-Content $master "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $name : start"
    $t0 = Get-Date
    & $ownerPy -u -m owner.main --config-file="$cfg" *> (Join-Path $logdir "transfer_$name.log")
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $master "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $name : done exit=$LASTEXITCODE ${sec}s"
}

Set-Location $ownerDir
Add-Content $master "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRANSFER START (pid=$PID)"

# 1) ENTRAINEMENT source unique (+ test conll2003 in-domain + SAUVE)
RunOne 'conll2003__train' (Join-Path $cfgDir 'conll2003__train.toml')

# 2) CHARGE + teste les 12 cibles zero-shot (tout sauf conll2003, deja teste au train)
foreach ($ds in @('crossner_ai','crossner_literature','crossner_music','crossner_politics',
                  'crossner_science','wnut17','mit_restaurant','mit_movie','fabner',
                  'bionlp2004','gum','gentle')) {
    RunOne $ds (Join-Path $cfgDir "$ds.toml")
}
Add-Content $master "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRANSFER DONE"

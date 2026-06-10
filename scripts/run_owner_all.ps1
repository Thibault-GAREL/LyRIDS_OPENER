# Lance OWNER (entity_typing) sur les 13 datasets, en DIRECT (python de l'env
# owner, sans mlflow-run), ordre petit->grand pour que les petits finissent en
# premier. A lancer DETACHE (Start-Process) -> survit a la fermeture de session.
#
# Chrono wall-clock par dataset capture dans OWNER_QUEUE.log (= cout total
# entrainement+eval, pour l'axe vitesse/energie du papier). AMI dans mlruns
# (lue ensuite par scripts.baselines.owner_collect).

$ErrorActionPreference = 'Continue'
$ownerPy  = 'D:\conda_envs\owner\python.exe'
$root     = 'D:\Loisir\Code_python\LyRIDS_Opener'
$ownerDir = Join-Path $root 'external\OWNER'
$cfgDir   = Join-Path $ownerDir 'configs\lyrids'
$logdir   = Join-Path $root 'outputs\owner_run'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

$env:MLFLOW_TRACKING_URI   = "file:///D:/Loisir/Code_python/LyRIDS_Opener/external/OWNER/mlruns"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$master = Join-Path $logdir 'OWNER_QUEUE.log'
function Log($m) { Add-Content -Path $master -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }

# Ordre par nb d'entites train croissant (les petits garantis en premier)
$datasets = @(
    'crossner_ai','crossner_literature','crossner_music','crossner_science','crossner_politics',
    'wnut17','conll2003','mit_restaurant','bionlp2004','gum','gentle','fabner','mit_movie'
)

Set-Location $ownerDir
Log "QUEUE OWNER START (pid=$PID, $($datasets.Count) datasets)"
foreach ($ds in $datasets) {
    Log "$ds : start"
    $t0 = Get-Date
    & $ownerPy -u -m owner.main --config-file="$cfgDir\$ds.toml" *> (Join-Path $logdir "$ds.log")
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Log "$ds : done exit=$LASTEXITCODE ${sec}s"
}
Log "QUEUE OWNER DONE"

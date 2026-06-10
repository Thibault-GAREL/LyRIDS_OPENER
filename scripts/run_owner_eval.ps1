# Pipeline complète pour la baseline OWNER (entity_typing) sur nos 13 datasets.
#
# OWNER (github.com/alteca/OWNER, GPLv3) tourne dans SON PROPRE env conda
# (transformers 4.33 / torch 2.0), incompatible avec notre venv (transformers
# 5.1). On l'exécute donc via `mlflow run` (qui crée l'env conda depuis env.yml),
# et notre venv ne fait QUE (a) exporter les données au format OWNER et
# (b) collecter les AMI depuis le file store MLflow.
#
# PRÉREQUIS (à faire UNE fois) :
#   1. Cloner OWNER (déjà fait) : external/OWNER  (gitignoré)
#   2. Avoir conda + mlflow dispo :  conda install -n base -c conda-forge mlflow
#      (le 1er `mlflow run` créera l'env conda 'owner' depuis env.yml, ~plusieurs Go)
#   3. GPU libre (OWNER entraîne un BERT par dataset, 4 epochs) -> lancer quand
#      l'éval int4 est finie.
#
# Lancement :  powershell -ExecutionPolicy Bypass -File scripts\run_owner_eval.ps1

$ErrorActionPreference = 'Continue'
$venvPy = 'c:\0-Code_py_temp\pytorch_cuda_env\Scripts\python.exe'
$root   = 'd:\Loisir\Code_python\LyRIDS_Opener'
Set-Location $root

Write-Host "=== 1/4 Export des 13 datasets au format OWNER ==="
& $venvPy -m scripts.baselines.owner_export

Write-Host "`n=== 2/4 Génération des configs entity_typing ==="
# Pour un setup transfert (comme notre embedder contrastif), ajoute : --train-source conll2003
& $venvPy -m scripts.baselines.owner_make_configs

Write-Host "`n=== 3/4 Runs OWNER (env conda 'owner' via mlflow) ==="
# Toutes les runs loggent dans external/OWNER/mlruns (file store, pas de serveur).
$mlruns = (Join-Path $root 'external\OWNER\mlruns') -replace '\\','/'
$env:MLFLOW_TRACKING_URI = "file:///$mlruns"
Write-Host "MLFLOW_TRACKING_URI = $env:MLFLOW_TRACKING_URI"

$cfgs = Get-ChildItem (Join-Path $root 'external\OWNER\configs\lyrids\*.toml')
foreach ($c in $cfgs) {
    Write-Host "--- OWNER entity_typing : $($c.BaseName) ---"
    mlflow run -e ner -P config_file="$($c.FullName)" (Join-Path $root 'external\OWNER')
    Write-Host "    exit=$LASTEXITCODE"
}

Write-Host "`n=== 4/4 Collecte des AMI -> outputs/results/baselines/owner/ ==="
& $venvPy -m scripts.baselines.owner_collect

Write-Host "`n=== TERMINÉ ==="

# Reprise OWNER : relance UNIQUEMENT les datasets manquants (2 epochs + patchs).
# typing-gold manquant : gum
# end-to-end manquant   : crossner_ai, crossner_literature, mit_restaurant,
#                         gentle, gum, fabner, mit_movie  (ordre petit->grand)
# A lancer DETACHE. Append les temps dans les logs de queue (owner_collect les lit).

$ErrorActionPreference = 'Continue'
$ownerPy  = 'D:\conda_envs\owner\python.exe'
$root     = 'D:\Loisir\Code_python\LyRIDS_Opener'
$ownerDir = Join-Path $root 'external\OWNER'
$logdir   = Join-Path $root 'outputs\owner_run'
$env:MLFLOW_TRACKING_URI    = "file:///D:/Loisir/Code_python/LyRIDS_Opener/external/OWNER/mlruns"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$typLog = Join-Path $logdir 'OWNER_QUEUE.log'
$nerLog = Join-Path $logdir 'OWNER_NER_QUEUE.log'
function RunOne($cfgDir, $log, $ds) {
    $cfg = Join-Path $ownerDir "$cfgDir\$ds.toml"
    Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $ds : start [RESUME]"
    $t0 = Get-Date
    & $ownerPy -u -m owner.main --config-file="$cfg" *> (Join-Path $logdir "resume_${ds}_$($cfgDir.Split('\')[-1]).log")
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $ds : done exit=$LASTEXITCODE ${sec}s [RESUME]"
}

Set-Location $ownerDir
Add-Content $nerLog "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] RESUME START (pid=$PID)"

# 1) typing-gold manquant
RunOne 'configs\lyrids' $typLog 'gum'

# 2) end-to-end manquants (petit -> grand)
foreach ($ds in @('crossner_ai','crossner_literature','mit_restaurant','gentle','gum','fabner','mit_movie')) {
    RunOne 'configs\lyrids_ner' $nerLog $ds
}
Add-Content $nerLog "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] RESUME DONE"

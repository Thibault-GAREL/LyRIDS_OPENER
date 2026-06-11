# Re-run autonome des datasets OWNER qui ont echoue AVANT les correctifs :
#   - typing-gold : bionlp2004, gum  (echec NaN, corrige par le patch loss)
#   - end-to-end  : crossner_ai, crossner_literature (echec md-eval tensor, corrige)
# Demarre APRES la file end-to-end (un seul GPU). A lancer DETACHE.
# Append les temps dans les logs de queue pour que owner_collect les recupere.

$ErrorActionPreference = 'Continue'
$ownerPy  = 'D:\conda_envs\owner\python.exe'
$root     = 'D:\Loisir\Code_python\LyRIDS_Opener'
$ownerDir = Join-Path $root 'external\OWNER'
$logdir   = Join-Path $root 'outputs\owner_run'
$env:MLFLOW_TRACKING_URI    = "file:///D:/Loisir/Code_python/LyRIDS_Opener/external/OWNER/mlruns"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$nerQueue = Join-Path $logdir 'OWNER_NER_QUEUE.log'
$typLog   = Join-Path $logdir 'OWNER_QUEUE.log'

# Attendre la fin de la file end-to-end
while (-not (Select-String -Path $nerQueue -Pattern 'QUEUE OWNER-NER DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 120
}
Add-Content $nerQueue "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] RERUNS START (pid=$PID)"

Set-Location $ownerDir
# (config_dir, log_de_queue, dataset)
$jobs = @(
    @('configs\lyrids',     $typLog,   'bionlp2004'),
    @('configs\lyrids',     $typLog,   'gum'),
    @('configs\lyrids_ner', $nerQueue, 'crossner_ai'),
    @('configs\lyrids_ner', $nerQueue, 'crossner_literature')
)
foreach ($j in $jobs) {
    $cfg = Join-Path $ownerDir "$($j[0])\$($j[2]).toml"
    $log = $j[1]
    Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $($j[2]) : start [RERUN]"
    $t0 = Get-Date
    & $ownerPy -u -m owner.main --config-file="$cfg" *> (Join-Path $logdir "rerun_$($j[2])_$($j[0].Split('\')[-1]).log")
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $($j[2]) : done exit=$LASTEXITCODE ${sec}s [RERUN]"
}
Add-Content $nerQueue "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] RERUNS DONE"

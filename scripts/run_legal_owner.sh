#!/usr/bin/env bash
# =====================================================================
# Baseline OWNER sur les datasets JURIDIQUES, protocole transfert du journal :
# le modele OWNER deja entraine sur conll2003 (checkpoints/transfer_conll2003/100)
# est CHARGE (load_finetuned) et applique zero-shot a chaque cible legale.
# Aucun re-entrainement. Tourne dans l'env conda OWNER (separe du venv).
# Prerequis : configs/lyrids_ner_transfer/{e_ner,lener_br,german_ler_coarse}.toml
#             + data/lyrids/<ds>/test.json (owner_export --legal).
# =====================================================================
set -u
OWNERPY="D:/conda_envs/owner/python.exe"
ROOT="D:/Loisir/Code_python/LyRIDS_Opener"
export MLFLOW_TRACKING_URI="file:///$ROOT/external/OWNER/mlruns"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
LOG="$ROOT/outputs/logs/legal_owner_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/outputs/logs"
cd "$ROOT/external/OWNER"

echo "[$(date +%H:%M:%S)] === OWNER legal (transfer, load_finetuned conll2003) ===" | tee -a "$LOG"
for ds in e_ner lener_br german_ler_coarse; do
  echo "[$(date +%H:%M:%S)] --- OWNER: $ds ---" | tee -a "$LOG"
  "$OWNERPY" -u -m owner.main --config-file="configs/lyrids_ner_transfer/$ds.toml" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] $ds exit=$?" | tee -a "$LOG"
done
echo "[$(date +%H:%M:%S)] === OWNER legal DONE. Collect: python -m scripts.baselines.owner_collect ===" | tee -a "$LOG"

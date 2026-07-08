#!/usr/bin/env bash
# Multi-seed variance study (reviewer: "single random seed").
# For each NEW seed: retrain the full embedder (contrastive Stage-1 + hard-negative
# mining 8k/3ep, the SAME recipe as the paper's selected embedder), then evaluate
# OPENER-Sup (gold + e2e) and OPENER-ZS (gold + e2e) on the full 13-dataset benchmark.
# Seed 42 is the existing run already in the paper, so it is NOT re-run here.
# Models -> outputs/models/seed{S}_*, results -> outputs/results/seed_runs/seed{S}/.
# Run from repo root. GPU does one thing at a time, so seeds run sequentially.
#
# Version "frugale" (le PC doit rester utilisable pendant le run) :
#  - Les evals tournent DATASET PAR DATASET : un process python par dataset,
#    donc la RAM est integralement rendue a l'OS entre datasets (fini le pic 99%)
#    et chaque dataset ecrit son SUMMARY des qu'il finit.
#  - Resume fin : si le SUMMARY d'un dataset existe deja, il est skippe.
#    On peut donc tuer/relancer le script sans rien perdre (les entrainements
#    ont deja leur propre skip via model.safetensors).
#  - Threads CPU limites (OMP/MKL) pour laisser des coeurs au reste du PC.
#  - Lancer le script en priorite BelowNormal (herite par les python enfants) :
#    powershell Start-Process bash -ArgumentList '-lc','bash scripts/run_multiseed.sh >> LOG 2>&1' -WindowStyle Hidden ; puis baisser la priorite.
set -u
PY=/c/0-Code_py_temp/pytorch_cuda_env/Scripts/python.exe
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Mode offline HuggingFace : modeles (GLiNER, Nomic) et datasets sont deja dans
# le cache local ; sans ce flag, une micro-coupure reseau nocturne fait crasher
# les evals au chargement (httpx.ConnectError vu le 2026-07-08 a ~3h du matin).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
DS="crossner_ai crossner_literature crossner_music crossner_politics crossner_science wnut17 mit_restaurant mit_movie fabner bionlp2004 conll2003 gum gentle"
# Seeds à traiter : passer en 1er argument pour n'en faire qu'une (ex: `bash scripts/run_multiseed.sh 7`).
SEEDS="${1:-7 123}"
ts() { date +'%Y-%m-%d %H:%M:%S'; }

# run_eval <module> <outdir> <dataset> [args supplementaires...]
# Un sous-dossier par dataset ; skip si un SUMMARY y existe deja (resume).
run_eval() {
  local module=$1 outdir=$2 dataset=$3
  shift 3
  if ls "$outdir/$dataset"/SUMMARY*.json >/dev/null 2>&1; then
    echo "[$(ts)]     $dataset : SUMMARY present -> skip"
    return 0
  fi
  mkdir -p "$outdir/$dataset"
  $PY -m "$module" --datasets "$dataset" --output-dir "$outdir/$dataset" "$@" \
    || echo "[$(ts)] !! FAIL $module / $dataset"
}

echo "[$(ts)] multi-seed study START | seeds: $SEEDS | datasets: $DS"

for S in $SEEDS; do
  echo "[$(ts)] ========================= SEED $S : START ========================="
  CTR=outputs/models/seed${S}_contrastive
  HARD=outputs/models/seed${S}_hard
  RES=outputs/results/seed_runs/seed${S}
  mkdir -p "$RES"

  # 1/6 contrastive Stage-1 (CoNLL triplets, 3 epochs)
  if [ -f "$CTR/model.safetensors" ]; then
    echo "[$(ts)] [1/6] contrastive already present -> skip"
  else
    echo "[$(ts)] [1/6] contrastive fine-tuning (seed $S)..."
    $PY -m scripts.train_contrastive_embedder --sources conll2003 --epochs 3 \
        --seed $S --output-dir "$CTR" \
      || { echo "[$(ts)] !! FAIL contrastive seed $S -> skip seed"; continue; }
  fi

  # 2/6 hard-negative mining Stage-2 (8000 triplets, 3 epochs, same recipe as paper)
  if [ -f "$HARD/model.safetensors" ]; then
    echo "[$(ts)] [2/6] hard-mining already present -> skip"
  else
    echo "[$(ts)] [2/6] hard-negative mining 8k/3ep (seed $S)..."
    $PY -m scripts.train_contrastive_hard --base-model "$CTR" --output-dir "$HARD" \
        --max-triplets 8000 --epochs 3 --hard-ratio 0.65 --easy-per-dataset 300 \
        --neg-per-error 2 --max-train 2000 --seed $S \
        --cache outputs/cache/seed${S}_hard_triplets.json --from-cache \
      || { echo "[$(ts)] !! FAIL hard-mining seed $S -> skip seed"; continue; }
  fi

  # 3/6 OPENER-Sup, typing on gold mentions
  echo "[$(ts)] [3/6] OPENER-Sup gold (dataset par dataset)..."
  for D in $DS; do
    run_eval scripts.run_balanced_classifiers "$RES/sup_gold" "$D" --embedder "$HARD"
  done

  # 4/6 OPENER-Sup, end-to-end
  echo "[$(ts)] [4/6] OPENER-Sup end-to-end (dataset par dataset)..."
  for D in $DS; do
    run_eval scripts.run_opener_e2e "$RES/sup_e2e" "$D" --embedder "$HARD" --tag s${S}
  done

  # 5/6 OPENER-ZS, typing on gold mentions. IMPORTANT : on passe par le SWEEP
  # (et pas run_opener_zs) car c'est lui qui produit les deux variantes du papier :
  # 'raw (baseline)' = ZS-ind (44.4) et 'ensemble+refine' = ZS-trans (51.4).
  echo "[$(ts)] [5/6] OPENER-ZS gold (sweep, dataset par dataset)..."
  for D in $DS; do
    run_eval scripts.run_opener_zs_sweep "$RES/zs_gold" "$D" --embedder "$HARD"
  done

  # 6/6 OPENER-ZS, end-to-end (with detector fusion)
  echo "[$(ts)] [6/6] OPENER-ZS end-to-end fusion (dataset par dataset)..."
  for D in $DS; do
    run_eval scripts.run_opener_zs_e2e_fusion "$RES/zs_e2e" "$D" --embedder "$HARD"
  done

  echo "[$(ts)] ========================= SEED $S : DONE ========================="
done

echo "[$(ts)] ALL SEEDS DONE. Results under outputs/results/seed_runs/."
echo "[$(ts)] Agregation : $PY -m scripts.aggregate_multiseed --latex"

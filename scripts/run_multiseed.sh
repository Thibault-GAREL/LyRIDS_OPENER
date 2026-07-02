#!/usr/bin/env bash
# Multi-seed variance study (reviewer: "single random seed").
# For each NEW seed: retrain the full embedder (contrastive Stage-1 + hard-negative
# mining 8k/3ep, the SAME recipe as the paper's selected embedder), then evaluate
# OPENER-Sup (gold + e2e) and OPENER-ZS (gold + e2e) on 3 representative datasets.
# Seed 42 is the existing run already in the paper, so it is NOT re-run here.
# Models -> outputs/models/seed{S}_*, results -> outputs/results/seed_runs/seed{S}/.
# Run from repo root. GPU does one thing at a time, so seeds run sequentially.
set -u
PY=/c/0-Code_py_temp/pytorch_cuda_env/Scripts/python.exe
DS="conll2003 crossner_music fabner"
# Seeds à traiter : passer en 1er argument pour n'en faire qu'une (ex: `bash scripts/run_multiseed.sh 7`).
SEEDS="${1:-7 123}"
ts() { date +'%Y-%m-%d %H:%M:%S'; }

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
  echo "[$(ts)] [3/6] OPENER-Sup gold..."
  $PY -m scripts.run_balanced_classifiers --embedder "$HARD" --datasets $DS \
      --output-dir "$RES/sup_gold" || echo "[$(ts)] !! FAIL sup_gold seed $S"

  # 4/6 OPENER-Sup, end-to-end
  echo "[$(ts)] [4/6] OPENER-Sup end-to-end..."
  $PY -m scripts.run_opener_e2e --embedder "$HARD" --datasets $DS \
      --output-dir "$RES/sup_e2e" --tag s${S} || echo "[$(ts)] !! FAIL sup_e2e seed $S"

  # 5/6 OPENER-ZS, typing on gold mentions (prototype + transductive)
  echo "[$(ts)] [5/6] OPENER-ZS gold..."
  $PY -m scripts.run_opener_zs --embedder "$HARD" --datasets $DS \
      --output-dir "$RES/zs_gold" || echo "[$(ts)] !! FAIL zs_gold seed $S"

  # 6/6 OPENER-ZS, end-to-end (with detector fusion)
  echo "[$(ts)] [6/6] OPENER-ZS end-to-end (fusion)..."
  $PY -m scripts.run_opener_zs_e2e_fusion --embedder "$HARD" --datasets $DS \
      --output-dir "$RES/zs_e2e" || echo "[$(ts)] !! FAIL zs_e2e seed $S"

  echo "[$(ts)] ========================= SEED $S : DONE ========================="
done

echo "[$(ts)] ALL SEEDS DONE. Results under outputs/results/seed_runs/."

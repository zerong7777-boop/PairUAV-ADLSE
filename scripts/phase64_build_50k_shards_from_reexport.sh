#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-$UAVM_ROOT/external/reloc3r_pairuav}"
RUN_ROOT="${RUN_ROOT:-$UAVM_ROOT/runs/phase64_correspondence_token_angle_specialist_v1}"

PREDICTIONS_CSV="${PREDICTIONS_CSV:-$RUN_ROOT/g2_50k_rank1_reexport/rank1_predictions.csv}"
CACHE_ROOT="${CACHE_ROOT:-$UAVM_ROOT/official/UAVM_2026/baseline/train_matches_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/g2_50k_rank1_train_shards_topk128}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
TOPK="${TOPK:-128}"
GRID_SIZE="${GRID_SIZE:-8}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

cd "$REPO_ROOT"
"$PYTHON" scripts/phase64_build_token_shards.py \
  --predictions-csv "$PREDICTIONS_CSV" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --shard-size "$SHARD_SIZE" \
  --topk "$TOPK" \
  --grid-size "$GRID_SIZE"

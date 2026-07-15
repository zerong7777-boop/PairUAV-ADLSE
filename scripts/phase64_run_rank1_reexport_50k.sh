#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-$UAVM_ROOT/external/reloc3r_pairuav}"
RUN_ROOT="${RUN_ROOT:-$UAVM_ROOT/runs/phase64_correspondence_token_angle_specialist_v1}"

MANIFEST_JSONL="${MANIFEST_JSONL:-$RUN_ROOT/g2_50k_rank1_reexport_manifest.jsonl}"
CHECKPOINT="${CHECKPOINT:-$UAVM_ROOT/synced_results/reloc3r_official_full_epoch_20260507/checkpoint-final.pth}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-45d7f1d403ff3e2c823667ddfcb900775bfdb4a73afc8ad7c1f7d482aef4ae54}"
IMAGE_ROOT="${IMAGE_ROOT:-$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour}"
JSON_SUBSET_ROOT="${JSON_SUBSET_ROOT:-$RUN_ROOT/g2_50k_rank1_reexport/json_subset}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/g2_50k_rank1_reexport}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-1}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

cd "$REPO_ROOT"
"$PYTHON" scripts/eval_pairuav_manifest_predictions.py \
  --manifest-jsonl "$MANIFEST_JSONL" \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --json-subset-root "$JSON_SUBSET_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --amp "$AMP"

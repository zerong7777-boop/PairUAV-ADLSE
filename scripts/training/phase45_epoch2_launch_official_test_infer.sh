#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/uavm_2026
REPO=$ROOT/external/reloc3r_pairuav
PAIR=$ROOT/official/UAVM_2026/pairUAV
CKPT=$ROOT/runs/reloc3r_official_pairuav/phase45_epoch2_resume_full_v1_20260523_215328/checkpoint-final.pth
EXPECTED_SHA=5681ab612d44dc64c98c82ef5b8b4e36c2bb4b38f6748a469f9c8d78c3894e04
OUT=/root/autodl-tmp/uavm_2026/runs/reloc3r_official_pairuav/official_test_infer_epoch2_pair_v1_20260526_approved
LOG=$OUT/launcher.log

{
  echo "START $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "HOST $(hostname)"
  echo "ROOT $ROOT"
  echo "REPO $REPO"
  echo "PAIR $PAIR"
  echo "CKPT $CKPT"
  echo "OUT $OUT"

  cd "$REPO"
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES=0
  export PATH=/root/miniconda3/envs/uavm5090/bin:/root/miniconda3/bin:$PATH

  ACTUAL_SHA=$(sha256sum "$CKPT" | awk '{print $1}')
  echo "CHECKPOINT_SHA $ACTUAL_SHA"
  if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    echo "ERROR checkpoint SHA mismatch expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
    exit 31
  fi

  JSON_COUNT=$(find "$PAIR/test" -type f -name '*.json' | wc -l)
  IMG_COUNT=$(find "$PAIR/test_tour" -type f | wc -l)
  echo "JSON_COUNT $JSON_COUNT"
  echo "IMG_COUNT $IMG_COUNT"
  if [[ "$JSON_COUNT" -ne 2773116 ]]; then
    echo "ERROR unexpected official test json count: $JSON_COUNT" >&2
    exit 32
  fi

  echo "INFER_START $(date '+%Y-%m-%d %H:%M:%S %Z')"
  /root/miniconda3/envs/uavm5090/bin/python infer_pairuav_with_progress.py \
    --json-root "$PAIR/test" \
    --image-root "$PAIR/test_tour" \
    --checkpoint "$CKPT" \
    --output-dir "$OUT" \
    --split test \
    --batch-size 16 \
    --num-workers 8 \
    --amp 1 \
    --log-every 500

  echo "INFER_DONE $(date '+%Y-%m-%d %H:%M:%S %Z')"
  wc -l "$OUT/result.txt"
  ls -lh "$OUT/result.txt"
} >> "$LOG" 2>&1

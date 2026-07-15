#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

PYTHON_BIN=/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python \
VAL_JSON_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1/eval_json \
MANIFEST=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1/eval_official_manifest.csv \
MODEL_EXPR="Reloc3rRelpose(img_size=512, output_mode='pairuav_selective_correspondence_heading_range', bscr_global_dim=12, bscr_grid_size=4, bscr_topk=16, bscr_hidden_dim=128, bscr_dropout=0.0, bscr_max_residual=0.25)" \
BSCR_FEATURE_MANIFEST=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/features/eval_bscr_features.jsonl \
DIAGNOSTICS_OUTPUT=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/smoke/diagnostics.csv \
BATCH_SIZE=2 \
NUM_WORKERS=2 \
AMP=1 \
bash scripts/eval_pairuav_official_devsplit.sh \
  /media/jgzn/SSD_lexar/RZ/UAVM/runs/reloc3r_official_pairuav/phase26_b_scr_smoke_20260507_053122/checkpoint-final.pth \
  /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/smoke/eval \
  4

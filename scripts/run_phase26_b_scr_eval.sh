#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <bscr_run_dir>" >&2
  exit 2
fi

RUN_DIR="$1"
cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

export PYTHON_BIN=/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python
export MODEL_EXPR="Reloc3rRelpose(img_size=512, output_mode='pairuav_selective_correspondence_heading_range', bscr_global_dim=12, bscr_grid_size=4, bscr_topk=16, bscr_hidden_dim=128, bscr_dropout=0.0, bscr_max_residual=0.25)"
export VAL_JSON_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1/eval_json
export MANIFEST=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1/eval_official_manifest.csv
export BSCR_FEATURE_MANIFEST=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/features/eval_bscr_features.jsonl
export DIAGNOSTICS_OUTPUT=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/eval/b_scr_heading_selective/diagnostics.csv
export BATCH_SIZE=4
export NUM_WORKERS=4

bash scripts/eval_pairuav_official_devsplit.sh \
  "$RUN_DIR/checkpoint-final.pth" \
  /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/eval/b_scr_heading_selective \
  0

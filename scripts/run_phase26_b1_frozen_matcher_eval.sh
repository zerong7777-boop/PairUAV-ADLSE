#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <fusion_run_dir>" >&2
  exit 2
fi

RUN_DIR="$1"
cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

export PYTHON_BIN=/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python
export MODEL_EXPR="Reloc3rRelpose(img_size=512, output_mode='pairuav_frozen_matcher_fusion_heading_range', matcher_feature_dim=13, matcher_hidden_dim=64, matcher_dropout=0.0)"
export VAL_JSON_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1/eval_json
export MANIFEST=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1/eval_official_manifest.csv
export MATCHER_FEATURE_MANIFEST=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_frozen_external_matcher/features/eval_features.jsonl
export BATCH_SIZE=4
export NUM_WORKERS=4

bash scripts/eval_pairuav_official_devsplit.sh \
  "$RUN_DIR/checkpoint-final.pth" \
  /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_frozen_external_matcher/eval/frozen_matcher_fusion \
  0

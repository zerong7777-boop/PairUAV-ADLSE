#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
PYTHONPATH=. /media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python -m unittest tests.test_pairuav_selective_correspondence -v
PYTHONPATH=. /media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python scripts/phase26_b_scr_inspect_inputs.py \
  --cache-root /media/jgzn/SSD_lexar/RZ/UAVM/official/UAVM_2026/baseline/train_matches_data \
  --subset-root /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1 \
  --checkpoint /media/jgzn/SSD_lexar/RZ/UAVM/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth \
  --b1b-reference-record /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_frozen_external_matcher/exp-20260507-phase26-b1-frozen-external-matcher-v1.md \
  --output /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/inputs/input_report.json
PYTHONPATH=. /media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python scripts/phase26_b_scr_build_features.py \
  --cache-root /media/jgzn/SSD_lexar/RZ/UAVM/official/UAVM_2026/baseline/train_matches_data \
  --subset-root /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b1_geometry_local_alignment/subsets/phase11_phase14_overlap_v1 \
  --output-root /media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase26_b_selective_correspondence_reasoning/features
bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase26_b_scr_smoke.env

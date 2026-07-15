#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase26_b_scr_gate_off_bounded.env

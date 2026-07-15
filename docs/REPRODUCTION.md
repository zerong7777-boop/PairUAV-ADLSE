# Reproduction

[中文说明](REPRODUCTION.zh-CN.md)

This document defines the reproducible paths for the PACE PairUAV release.
Use Path A for exact submission-package reproduction. Use Path B only when you
need to regenerate raw official-test predictions from released checkpoints.

## Prerequisites

1. Clone this repository.
2. Create the Python environment:

```bash
conda env create -f environment.yml
conda activate pace
```

3. Download the external asset folder:

```text
Baidu Netdisk: https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
Extraction code: djmv
Asset folder: PairUAV-PACE-release-assets
```

Copy its inner `assets/` and `known/` directories into the repository root.
Hashes are listed in
[../assets/RELEASE_ASSETS.md](../assets/RELEASE_ASSETS.md).

## Path A: Exact Final-Zip Rebuild

This path starts from released raw official-test prediction zips and rebuilds
the final package byte-identically. It is CPU-friendly and does not require
official test images.

Required files:

```text
assets/phase104j_HRofficial_raw_result.zip
assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip
assets/epoch2_pair_official_test_result_20260527_013814.zip
manifests/devsplit_v1_official_metric_manifest.csv
```

Command:

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

Expected final output:

```text
outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
SHA256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

The script validates input hashes before writing outputs.

## Path B: Raw Prediction Regeneration

This path starts from released checkpoints and official PairUAV test data, then
regenerates the three raw prediction zips used by Path A.

Required data:

```text
official PairUAV test JSON directory
official PairUAV test image directory
```

Required checkpoints:

```text
assets/phase104j_paaer_hr_final/checkpoint-final.pth
assets/phase89_h8_midlate_final/checkpoint-final.pth
assets/phase45_epoch2_resume_final/checkpoint-final.pth
```

The exact model-level commands, source snapshot hashes, checkpoint hashes, and
expected raw ZIP hashes are recorded in [A_REPRODUCTION.md](A_REPRODUCTION.md).

## Path C: Training Provenance

Training scripts are retained for audit and continuation, not for
byte-identical checkpoint reproduction. GPU, CUDA, PyTorch, dataloader order,
and filesystem behavior can change exact weights.

Key training entries:

| Target checkpoint | Training script |
|---|---|
| `assets/phase45_epoch2_resume_final/checkpoint-final.pth` | `scripts/training/phase45_epoch2_launch_full.sh` |
| `assets/phase104j_paaer_hr_final/checkpoint-final.pth` | `scripts/training/phase104j_tailw_fulltrain1epoch_fromTail10_lr1e-5_bs4_5090.sh` |
| `assets/phase89_h8_midlate_final/checkpoint-final.pth` | `scripts/training/phase89_mid_late_fulltrain_1epoch_20260618.sh` |
| `assets/phase104i_tail10_init/checkpoint-final.pth` | `scripts/training/phase104i_HR_tailw_fulltrain10k_fromHR50_lr1e-5_bs4.sh` |

Use released checkpoint assets for leaderboard-package reproduction.

## Verification

Recommended checks:

```bash
python -m pytest tests/test_phase104e_paaer_head.py
python -m pytest tests/test_eval_pairuav_manifest_predictions.py
python postprocess/rebuild_final_postprocess.py --help
```

The full historical test suite contains many research-route tests and may
require optional assets. For release verification, Path A hash equality is the
primary check.

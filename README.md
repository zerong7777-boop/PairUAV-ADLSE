# PACE: Polar Axis-Conditioned Estimation for PairUAV 2026

[中文说明](README.zh-CN.md) · [Reproduction](docs/REPRODUCTION.md) · [Compliance](docs/COMPLIANCE.md) · [Method](docs/METHOD.md) · [Results](docs/RESULTS.md)

PACE is a reproducible PairUAV 2026 relative-localization release. Given an
ordered UAV-view image pair, it predicts the polar relation between the two
views: heading and metric range.

The method is built around a simple observation: heading and range share a
pose-regime representation, but their useful evidence, training dynamics, and
failure modes are not identical. PACE therefore combines axis-conditioned
learned predictors with deterministic range fusion and task-aware output
calibration.

## Result

The archived final submission package is:

```text
outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
SHA256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

Known official leaderboard feedback:

| Metric | Value |
|---|---:|
| `final_score` | `0.001874` |
| `distance_rel_error` | `0.001330` |
| `angle_rel_error` | `0.002419` |

These scores are server feedback on hidden official-test labels. The released
repository reproduces the submitted zip byte-identically from public raw
prediction assets and a public train/dev support manifest.

## What Is Included

- Model, dataset, loss, training, inference, and evaluation code under
  `reloc3r/`, `croco/`, and the root entrypoints.
- Deterministic final-package rebuild code under `postprocess/`.
- Run manifests, public support manifest, and asset hash manifests.
- Training scripts for the released checkpoint chain under `scripts/training/`.
- Research diagnostics and mechanism figures used to document the design path.

This repository does **not** include official PairUAV test images, hidden test
labels, or large checkpoint/raw-prediction assets in git. The official data
must be obtained through the competition's allowed channel.

## Release Assets

Large checkpoints and raw prediction zips are hosted externally:

```text
Baidu Netdisk: https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
Extraction code: djmv
Asset folder: PairUAV-PACE-release-assets
```

Copy the inner `assets/` and `known/` directories into the repository root.
Expected paths and SHA256 hashes are listed in
[assets/RELEASE_ASSETS.md](assets/RELEASE_ASSETS.md).

## Quick Start

```bash
conda env create -f environment.yml
conda activate pace

python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

For model-level reproduction from checkpoints, follow
[docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Method Overview

PACE uses three released raw predictors and two deterministic postprocess
modules:

| Component | Role |
|---|---|
| PAAER | Primary protected axis-asymmetric predictor and final heading source. |
| MDHR | Multi-depth heading-readout model retained as complementary distance evidence. |
| MARB | Metric-aware Reloc3r baseline retained as an independent distance source. |
| RSF | Deterministic range stack fusion over PAAER, MDHR, and MARB distances. |
| Task-aware output calibration | Heading lattice snap and support-distance snap using public train/dev support. |

The final range fusion is:

```text
distance = 0.511 * PAAER_distance
         + 0.189 * MDHR_distance
         + 0.300 * MARB_distance
```

Detailed design notes are in [docs/METHOD.md](docs/METHOD.md). The exploration
logic behind the method is in [docs/EXPLORATION_LOGIC.md](docs/EXPLORATION_LOGIC.md).

## Repository Layout

```text
reloc3r/                         model, head, loss, and dataset code
croco/                           vendored Reloc3r/CroCo dependency source
postprocess/                     deterministic final-package rebuild
scripts/training/                released checkpoint training chain
manifests/                       source, inference, and support manifests
assets/                          external asset manifest and expected paths
figures/                         mechanism and qualitative figures
docs/                            method, reproduction, compliance, and evidence notes
research/diagnostics/            archived analysis scripts used for paper evidence
tests/                           smoke and historical mechanism tests
phase91/, phase92/               archived research packages kept top-level for import compatibility
```

The root entrypoints `train.py`, `infer_pairuav_with_progress.py`, and
`eval_pairuav.py` are kept at the top level for backward-compatible scripts.

## Compliance

The released final path uses model predictions, released raw assets, and a
public train/dev support manifest. It does not use hidden official-test labels
or pair-ID lookup. See [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

## Documentation

| Document | Purpose |
|---|---|
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Exact submission rebuild and model-level reproduction paths. |
| [docs/METHOD.md](docs/METHOD.md) | Public method terminology and implementation map. |
| [docs/EXPLORATION_LOGIC.md](docs/EXPLORATION_LOGIC.md) | Route-selection, phenomenon-discovery, and design logic for PACE. |
| [docs/RESULTS.md](docs/RESULTS.md) | Official feedback and internal development signals. |
| [docs/MECHANISM_INSIGHTS.md](docs/MECHANISM_INSIGHTS.md) | Axis-asymmetry evidence and generated figures. |
| [docs/SOURCE_ADAPTATION.md](docs/SOURCE_ADAPTATION.md) | External geometry, matching, solver, and split-fusion evidence. |
| [docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md) | Curated route history and negative evidence. |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | Allowed-signal and hidden-label safety statement. |

## Citation

If this repository is used in a paper or derivative work, cite the repository
and the corresponding PairUAV 2026 task description. Repository citation
metadata is provided in [CITATION.cff](CITATION.cff).

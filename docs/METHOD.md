# Method Notes: PACE

[中文说明](METHOD.zh-CN.md)

This document describes the public method terminology used in this repository.
It is intentionally separate from the reproduction manifests: internal run
identifiers are preserved in file names for hash-stable reproduction, while the
method text below uses paper-facing names.

## System Name

The final public method is called **Polar Axis-Conditioned Estimation (PACE)**.
The paper-facing description treats the final snapping step as downstream
task-aware calibration, not as the central research contribution.

PACE is not a single end-to-end architecture. It is a competition solution that
combines axis-decoupled model predictors with deterministic task-aware output
calibration. This distinction matters: the learned predictors are model
components, while the final leaderboard package also uses a reproducible
calibration step.

## Problem Definition

PairUAV asks for pairwise relative localization from UAV-view image pairs.
Given two UAV-view images, the system predicts a polar relation between the two
views: a heading-like angular output and a distance-like range output. A
natural first formulation is therefore joint polar regression, where one model
predicts both coordinates as a continuous output vector.

PACE uses this formulation as the starting point, but not as the full task
model. The competition metric evaluates heading and distance
separately before combining them, and the annotation space exposes structure
that is not captured by an unconstrained two-dimensional regressor.

## Empirical Phenomena

Development experiments showed three stable phenomena.

First, heading and distance did not fail in lockstep. Some routes improved
heading while damaging distance, and others repaired range-tail behavior while
leaving heading nearly unchanged. This was visible in axis-objective controls,
H-readout sweeps, source-adaptation diagnostics, and tail-robustness runs.

Second, the two axes preferred different evidence. Heading benefited from
richer decoder-depth evidence, geometry-derived signals, and angular
regularity. Distance required a protected metric path and was more sensitive to
large-range tail failures.

Third, final predictions benefited from deterministic task-aware calibration
against public train/dev support. This is a final packaging step, not a claim
that the main learned mechanism is the support projection itself.

## Design Insight

The design view behind PACE is:

```text
polar axis-conditioned estimation with protected metric evidence
```

This does not mean that heading and distance are fully independent latent
factors. The stronger supported claim is that PairUAV contains a shared
pose-regime representation whose coordinates have axis-specific readability,
axis-asymmetric evidence utility, and different tail-risk behavior. A strong
system should therefore protect stable metric evidence, allow axis-specific
readout, and apply output calibration after prediction.

## Method Overview

PACE separates the final prediction problem into three practical questions:

1. Which learned predictor should provide the heading?
2. Which learned predictors should contribute distance evidence?
3. Which output calibration should be applied deterministically at the end?

The released system answers these questions with PAAER heading, RSF distance,
and final output calibration. PAAER and MDHR implement axis-aware learned prediction,
MARB contributes an independent metric-aware distance source, RSF combines
complementary range estimates, and the final calibration maps outputs onto
public train/dev support.

## System Data Flow

The final submitted package is produced by a fixed inference-and-postprocess
graph:

```text
official PairUAV image pairs
        |
        +--> PAAER raw prediction: heading_HR, distance_HR
        +--> MDHR raw prediction:  heading_H8, distance_H8
        +--> MARB raw prediction:  heading_epoch2, distance_epoch2
                                      |
                                      v
range stack:
  heading_e230  = heading_HR
  distance_e230 = 0.511 * distance_HR
                + 0.189 * distance_H8
                + 0.300 * distance_epoch2
                                      |
                                      v
task-aware output calibration:
  heading_final  = nearest 2-degree lattice value, normalized to [-180, 180)
  distance_final = nearest support distance from the train/dev manifest
```

This makes the release reproducible at two levels. Model-level reproduction
regenerates the three raw prediction files from checkpoints. Package-level
reproduction starts from those raw files and deterministically rebuilds the
final `result.zip`.

## Implementation Map

| public name | code / output mode | release asset |
|---|---|---|
| PAAER | `Phase104eProtectedAxisAsymmetricExpertHead`; `pairuav_phase104e_paaer_hard_heading_range` | `assets/phase104j_HRofficial_raw_result.zip` |
| MDHR | H8 mid-late readout; `pairuav_range_h0_heading_mid_late_heading_range` | `assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` |
| MARB | metric-aware Reloc3r baseline checkpoint | `assets/epoch2_pair_official_test_result_20260527_013814.zip` |
| RSF + calibration | `postprocess/rebuild_final_postprocess.py` and `postprocess/phase104j_final_postprocess_manifest.json` | `outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip` |

## Protected Axis-Asymmetric Expert Readout

**Protected Axis-Asymmetric Expert Readout (PAAER)** is the primary predictor.

The design goal is to improve axis-specific readout capacity without destroying
the stable metric path inherited from the base Reloc3r-style predictor. PAAER
therefore keeps a protected anchor path and introduces axis-asymmetric expert
readouts. In practice, heading and distance are allowed to use different
evidence mixtures rather than being forced through a fully shared scalar head.

The released PAAER checkpoint uses the hard PAAER variant. Its range output is
kept on the protected H0/C0 metric path, while its heading output is produced by
an axis-asymmetric expert path:

- the heading expert reads fixed mid and late decoder layers;
- a heading task token attends to that layer bank through the query-bridge path;
- the heading task feature is mapped to a normalized 2D heading vector;
- the range scalar is copied from the protected late metric path.

The implementation also records diagnostic quantities such as the heading
expert/base angular delta, heading layer attention, token entropy, and the
absolute difference between the final range and protected range. In the hard
release path, the final range contract is intentionally zero because the range
prediction is the protected range prediction.

The final official raw prediction named
`phase104j_HRofficial_raw_result.zip` is the PAAER raw output used by PACE.
The corresponding checkpoint asset is
`assets/phase104j_paaer_hr_final/checkpoint-final.pth`.

## Multi-Depth Heading Readout

**Multi-Depth Heading Readout (MDHR)** is a complementary heading-oriented
predictor.

Earlier experiments showed that heading can benefit from decoder evidence at
multiple depths. MDHR keeps the distance path close to a stable late metric
readout, while the heading branch reads a multi-depth set of decoder features.
This makes MDHR useful as a complementary predictor even when it is not used as
the final heading source.

The final MDHR model is the reduced H8 mid-late variant. It preserves the H0
late readout for range, but lets the heading branch read mid and late decoder
features through independent token-to-grid extractors and a fusion MLP. The
larger H8 family also tested early/mid/late readout, but the matched local
control favored the mid-late reduction before full-train promotion. See
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) for the H-readout sweep.

The final official raw prediction named
`phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` is the MDHR raw output
used by PACE.

## Metric-Aware Reloc3r Baseline

**Metric-Aware Reloc3r Baseline (MARB)** is an earlier Reloc3r checkpoint
trained with the PairUAV metric-aware objective.

MARB is not presented as a new architecture. It is retained because it provides
an independent distance estimate with different error correlations from PAAER
and MDHR. In the final system, MARB contributes only through deterministic range
fusion.

This component is intentionally conservative: it anchors the final distance
stack with a Reloc3r-family model trained before the PAAER/MDHR late-stage
changes. Keeping it as a distance-only contributor avoids claiming new
architecture novelty from this baseline while still using its complementary
metric signal.

The final official raw prediction named
`epoch2_pair_official_test_result_20260527_013814.zip` is the MARB raw output
used by PACE.

## Range Stack Fusion

**Range Stack Fusion (RSF)** is the deterministic distance fusion module. It
uses the PAAER heading as the heading source and combines the three distance
predictions as:

```text
distance = 0.511 * PAAER_distance
         + 0.189 * MDHR_distance
         + 0.300 * MARB_distance
```

The weights were selected from development and submission feedback. RSF should
be understood as a competition-system component, not as a claim of a universal
fusion law.

RSF does not alter heading. It carries PAAER heading forward unchanged until the
later task-aware calibration step. This is why the intermediate e230 package
improves distance while keeping the PAAER angle score unchanged.

## Task-Aware Output Calibration

The released code historically names this deterministic final calibration step
**Legal-State Snapping (LSS)**. In paper-facing wording, it should be read as
task-aware output calibration.

The final heading is snapped to the nearest 2-degree lattice value. The final
distance is snapped to the nearest support distance observed in the released
train/dev manifest:

```text
heading  -> nearest 2-degree lattice value
distance -> nearest support distance from gt_distance in train/dev
```

LSS uses only model predictions and the released train/dev split manifest. It
does not use hidden official-test labels. In this repository, the support
manifest is tracked as:

```text
manifests/devsplit_v1_official_metric_manifest.csv
```

The support manifest contains `204120` train/dev rows and `211` unique support
distances in the released manifest used by the final package. The final official
test predictions contain `2773116` rows. The snap is therefore a fixed nearest
neighbor projection onto a public train/dev support set; it is not a selector
trained on hidden official-test labels.

## What PACE Claims

PACE should be read as a reproducible competition solution with the following
design claims:

- heading and distance benefit from axis-aware readout design;
- multi-depth decoder evidence is useful for heading-oriented prediction;
- independent distance predictors can be combined to reduce range error;
- public output support can be applied deterministically without hidden test
  labels;
- the final package can be reconstructed byte-identically from released raw
  predictions and a public support manifest.

PACE does not claim that the final leaderboard score comes from a single
end-to-end model, nor that the postprocess is a general-purpose replacement for
learning better predictors.

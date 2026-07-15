# Compliance And Signal Use

[中文说明](COMPLIANCE.zh-CN.md)

This document states the signal boundary for the released PACE final package.
It is intended to make the challenge-system postprocess auditable.

## Allowed Inputs

The released final package is generated from:

- three released raw official-test prediction zips;
- fixed deterministic postprocess code in `postprocess/`;
- the public train/dev support manifest
  `manifests/devsplit_v1_official_metric_manifest.csv`;
- fixed numeric fusion weights and snapping rules recorded in the manifest and
  source code.

The support manifest contains train/dev labels only. It is used to build a
finite public support set for output calibration.

## Not Used

The released path does not use:

- hidden official-test labels;
- manual inspection of hidden official-test errors;
- pair-ID lookup to assign distance or heading;
- test-set graph optimization, clustering, or cross-row label propagation;
- adaptive selection using official-test leaderboard feedback at inference
  time.

Official leaderboard feedback was used only to record submitted-package scores
and to document final-stage decisions. It is not an input to the released
rebuild script.

## Multi-Model Use

The final package combines PAAER, MDHR, and MARB raw predictions. This is a
fixed deterministic ensemble over released model outputs:

```text
distance = 0.511 * PAAER_distance
         + 0.189 * MDHR_distance
         + 0.300 * MARB_distance
heading  = PAAER_heading
```

The ensemble is a challenge-system component. The main method claim remains
axis-conditioned polar estimation, not that ensembling alone is the scientific
contribution.

## Output Calibration

The code historically calls the final calibration step Legal-State Snapping
(LSS). In paper-facing wording, it is task-aware output calibration:

```text
heading  -> nearest 2-degree lattice value
distance -> nearest support distance observed in train/dev
```

This projection is deterministic and uses only public train/dev support. It is
not trained on hidden official-test labels.

## Reproducibility Boundary

The final `result.zip` can be rebuilt byte-identically through
`postprocess/rebuild_final_postprocess.py` when the released raw zips and
support manifest are present. Full retraining is documented for provenance but
is not expected to reproduce bit-identical checkpoints.

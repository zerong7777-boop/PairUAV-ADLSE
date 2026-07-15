# Source Adaptation And Split-Fusion Evidence

[中文说明](SOURCE_ADAPTATION.zh-CN.md)

This document records the source-adaptation evidence that shaped the final
PairUAV design. It covers external geometry, matching, solver, and structured
split-fusion routes. These runs are design evidence, not official leaderboard
claims unless explicitly marked as official feedback.

Generated summary figure:

```text
figures/fig_source_adaptation_summary.svg
```

Generation script and data:

```text
figures/gen_fig_source_adaptation_summary.py
figures/source_adaptation_summary_data.csv
```

## Evidence Policy

Most results in this document are bounded internal probes. They use labeled
local splits, 2048-row diagnostic subsets, B811 comparable surfaces, or proxy
metrics. They should not be compared directly with hidden-test leaderboard
scores.

Use these results for three claims:

1. External geometry and matching sources can be wired into PairUAV.
2. Heading and distance use these sources differently.
3. Naive source concatenation or frozen-feature transfer is not enough.

Do not use these results to claim that a standalone source-adaptation model
was the final submitted system. The final submitted system is PACE, documented
in [METHOD.md](METHOD.md) and [RESULTS.md](RESULTS.md).

## Single-Source And Solver Adaptation Inventory

| route | adaptation implementation | surface | result | decision |
|---|---|---:|---|---|
| RoMa dense | Official RoMa `roma_outdoor` dense correspondence cache converted to warp/certainty fields. | 2048 rows, 1638/410 train/eval | RoMa Level C: angle MAE `15.3929`, distance MAE `19.4702`; SuperGlue sparse anchor: `89.3941 / 52.3864`. | Positive dense-correspondence source. |
| EfficientLoFTR | Official EfficientLoFTR semi-dense match/confidence field under the same field-regressor contract. | 2048 rows, 1638/410 train/eval | `55.3077 / 39.8429`. | Runnable but dominated by RoMa dense. |
| MASt3R official | Official MASt3R cache with descriptor/match field and pointmap/confidence geometry field. | 2048 rows, 1638/410 train/eval | match field `27.7047 / 39.0629`; geometry field `9.5863 / 33.2327`. | Strong heading/geometry source, weak range source. |
| DUSt3R official | Official DUSt3R pointmap/confidence geometry field. | 2048-row same-subset replay | `11.0434 / 33.1489`. | Useful geometry-heading control, not a primary range route. |
| VGGT direct camera | Official VGGT pose decoding from `pose_enc` to extrinsic/intrinsic matrices. | 2048 rows | `40.6610 / 40.0056`. | Direct camera extraction was weak. |
| VGGT geometry field | Official VGGT `world_points` and confidence field replay. | 2048 rows | `52.5120 / 14.1716`. | Strong distance/range signal, weak heading signal. |
| SAC-Pose frozen head | Official-style SAC-Pose frozen feature migration with a PairUAV head. | 256-row bounded probe | `91.0811 / 32.2643`. | Technical pass, method-quality fail. |
| SAC-Pose relation prior | Structured relation tokens from official SAC-Pose descriptors and keypoints. | 256-row bounded probe | `87.9658 / 59.4438`. | Small heading gain, severe range regression. |
| MADPose | Official MADPose solver with RoMa correspondences and VGGT depth prior, plus train-only linear mapping for sanity ranking. | 1977 rows, 1581/395 train/eval | MADPose shared-focal `70.8120 / 40.1349`; OpenCV point-only control `73.4009 / 36.3710`. | Solver alive, not useful enough under this proxy. |
| FAR SuperGlue / LoFTR | Official FAR-style solver/fusion packet with proxy intrinsics and SuperGlue or LoFTR branches. | bounded solver packet probes | SuperGlue heading proxy median `82.15 deg`; LoFTR FOV sweep improved to about `70-74 deg`, still high. | Executable but not informative enough for training promotion. |
| GRelPose frozen head | Frozen GRelPose/ScanNet-style representation plus a cheap PairUAV head. | val811 bounded surface | best bounded head angle `43.8259`, proxy `0.2472`; native rough diagnostic `91.6852 / 66.9104`. | Feature/task mismatch, not competitive. |

## RoMa / MASt3R / VGGT Hybrid Evidence

The three-source hybrid route joined RoMa, MASt3R, and VGGT on the same 2048
rows with no missing rows, no duplicates, and no label mismatches. It then
trained same-split source-drop variants.

| variant | angle MAE | distance MAE | proxy |
|---|---:|---:|---:|
| RoMa | `12.1377` | `16.1241` | `0.11434` |
| MASt3R | `12.9339` | `11.7043` | `0.09445` |
| VGGT | `11.4752` | `8.0099` | `0.07193` |
| RoMa + MASt3R | `12.2763` | `10.1944` | `0.08507` |
| RoMa + VGGT | `11.4403` | `11.8067` | `0.09081` |
| MASt3R + VGGT | `11.7800` | `7.8554` | `0.07200` |
| RoMa + MASt3R + VGGT | `10.8126` | `8.6817` | `0.07344` |

Interpretation:

- Full three-source concatenation gave the best angle but not the best proxy.
- VGGT alone and MASt3R+VGGT were stronger proxy/range candidates.
- RoMa helped some heading regimes, but naive concatenation could degrade the
  balanced proxy.

The follow-up source-complementarity audit supported the same conclusion. A
four-source oracle reached proxy `0.04588397`, compared with historical VGGT
proxy `0.07192526`. Per-sample winners were distributed across sources:

| winner view | leading sources |
|---|---|
| proxy winner share | MASt3R+VGGT `0.3488`, VGGT `0.2659`, full hybrid `0.2000`, RoMa `0.1854` |
| angle winner share | RoMa `0.3098`, MASt3R+VGGT `0.2683`, VGGT `0.2293`, full hybrid `0.1927` |
| distance winner share | MASt3R+VGGT `0.4122`, VGGT `0.2415`, full hybrid `0.1951`, RoMa `0.1512` |

This is strong mechanism evidence for complementary sources, but the oracle is
not deployable. The tested one-dimensional source-stat predictors were only
partially predictive, so this branch was not promoted as an inference-time
gate.

## Structured Split-Fusion Source Adaptation

The split-fusion line is separate from single-source adaptation. It asks
whether different sources should serve different polar axes:

```text
heading branch <- old field / geometry-heavy source
range branch   <- rich-native / VGGT-style source
```

### Old-Source Parity Audit

The old VGGT/MASt3R/RoMa field sources and rich-native sources were evaluated
under the same 2048-row, 1600/448, 1000-step protocol.

| variant | angle MAE | distance MAE | proxy |
|---|---:|---:|---:|
| old_field_baseline | `1.393762` | `39.247571` | `0.182955` |
| old_roma_mast3r_vggt_concat | `1.328199` | `39.050710` | `0.181712` |
| rich_vggt_only | `2.835736` | `33.045150` | `0.163277` |
| rich_three_source_concat | `2.404302` | `33.637191` | `0.163523` |

The result showed clear axis specialization: old field features were stronger
for heading, while rich VGGT-style features were stronger for distance.

### Split-Fusion Probe

A two-branch `SplitFusionProbeRegressor` then routed heading through old-field
features and distance through rich-native features.

| variant | angle MAE | distance MAE | proxy |
|---|---:|---:|---:|
| split_old_concat_angle_rich_vggt_range | `1.213552` | `5.937480` | `0.033249` |
| split_old_baseline_angle_rich_vggt_range | `1.200125` | `5.590767` | `0.031626` |
| split_old_concat_angle_rich_three_range | `1.330926` | `4.024406` | `0.025360` |
| split_old_baseline_angle_rich_three_range | `1.512513` | `4.331029` | `0.027738` |

This was the first source-adaptation result that was clearly stronger than
both old-field-only and rich-native-only anchors under the same protocol. It
confirmed that source complementarity was actionable when the architecture
respected the heading/range split.

### B811 Axis-Decoupled Replay

The B811 comparable surface then tested a simpler structural question: keep
the base model for distance, but use geometry evidence for heading.

| variant | heading MAE | distance MAE |
|---|---:|---:|
| base only | `8.6087` | `0.9876` |
| geometry only | `1.7769` | `6.9755` |
| axis-decoupled | `1.7769` | `0.9876` |

Direct pairwise comparison on B811 showed:

| statistic | value |
|---|---:|
| geometry heading better | `694 / 811` |
| geometry heading worse | `117 / 811` |
| geometry distance better | `53 / 811` |
| geometry distance worse | `758 / 811` |
| heading-helpful but distance-harmful rows | `492` |

The stricter helpful/neutral/harmful binning used by
[MECHANISM_INSIGHTS.md](MECHANISM_INSIGHTS.md) reports a similar conclusion:
geometry mostly helps heading and mostly harms distance.

### Why It Was Not The Final Submitted Route

Split-fusion was not promoted directly to the final official system for three
practical reasons:

- full official split-fusion inference was expensive because the source path
  required heavy geometry sidecars;
- non-leaky no-op / base-sufficient policies were formally viable but weak;
- D-series cheap student compression did not preserve the strong mechanism
  signal, and one official attempt failed distance parity.

This does not invalidate split-fusion. It means the line is best used as
mechanism evidence and design history in this release.

## Negative And Diagnostic Routes

Several routes are worth documenting precisely because they were real and
negative:

- SAC-Pose connected to PairUAV train/eval but did not provide a useful
  heading/range predictor under frozen or relation-prior migration.
- MADPose and FAR showed that solver pipelines can be executed, but proxy
  intrinsics and weak heading/range signal blocked promotion.
- GRelPose-style frozen representations did not transfer to PairUAV with a
  cheap head.
- Native adapter experiments showed that shallow native summaries were weaker
  than old field tensors. Rich VGGT tensors improved distance but regressed
  heading.
- D-series compression did not disprove the split-fusion mechanism; it only
  rejected a specific cheap residual/student route.

## Paper-Facing Takeaway

The source-adaptation history supports this statement:

```text
PairUAV benefits from external geometric and correspondence evidence, but the
benefit is axis-conditional. Geometry-heavy sources are often useful for
heading and unsafe for distance; range-capable sources can improve metric
prediction while hurting heading. Effective systems should therefore preserve
stable metric evidence and introduce axis-aware readout, fusion, or
postprocessing rather than relying on naive source concatenation.
```

This statement is mechanism-level evidence. The leaderboard-facing claim
remains the reproducible PACE final package.

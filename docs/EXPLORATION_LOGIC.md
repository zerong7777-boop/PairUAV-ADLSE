# Exploration Logic: From Baseline Routes To PACE

[中文说明](EXPLORATION_LOGIC.zh-CN.md)

This document records the paper-facing exploration logic behind **PACE: Polar
Axis-Conditioned Estimation**. It is not a complete run ledger. Its purpose is
to explain why the final repository is organized around axis-conditioned polar
estimation, why the deployed package still contains challenge-specific
postprocess, and how negative routes shaped the method boundary.

Internal run names such as `phase104j` are preserved in file names for
reproducibility. The public method name used in the paper-facing text is PACE.

## 1. Starting From Route Selection

PairUAV can be written as polar relative localization: predict heading and
metric range from a UAV-view image pair. We first treated the task as a route
selection problem rather than as a single architecture tweak.

| route family | question | outcome |
|---|---|---|
| End-to-end RPR / Reloc3r family | Can a Reloc3r-style image-pair regressor learn the official heading/range objective directly? | This became the most reliable host: it supported full official inference, preserved metric range better than geometry-only routes, and had positive hidden-test feedback. |
| Baseline-style field regressor / split-fusion family | Can dense geometry, matching, or source fields be converted into task-specific heading/range regressors? | Strong mechanism evidence; useful for axis asymmetry and source specialization, but not promoted as the final official route. |
| Geometry / solver / matching-aware route | Can explicit correspondences, solvers, or pose pipelines solve the task directly? | Technically runnable but not competitive under bounded PairUAV metrics or too costly for the final full-inference path. |
| Candidate-pool new routes | Can multi-objective, relational, patch-hypothesis, or JEPA-style modules create a stronger late-stage candidate? | Implemented as formal research routes, but they did not complete the required promotion path before the competition deadline. |

The practical conclusion was not that end-to-end regression is universally
superior. It was narrower: under the PairUAV competition constraints, the
direct Reloc3r-family RPR host was the most dependable backbone for official
full-data training and hidden-test inference, while source/geometry routes were
most valuable as mechanism probes.

## 2. Re-examining The Shared-Readout Assumption

DUSt3R/MASt3R/Reloc3r-style systems naturally build shared pair
representations and often use shared or highly coupled readout heads for pose
or geometry prediction. This is a reasonable design for reconstruction-style
objectives, but PairUAV exposes a different target structure: heading and range
are evaluated separately and can react differently to the same evidence source.

Directly porting a shared polar readout therefore risks treating heading and
distance as two interchangeable coordinates of one scalar regression target.
The exploration shifted from "train the shared head harder" to:

```text
How should heading and range interact, and where should their readouts be
allowed to differ, while retaining a stable shared pose-regime representation?
```

## 3. Exploring Interaction Position And Form

The next stage tested where the two axes should share computation and where
they should be protected or specialized.

| exploration line | role in the logic |
|---|---|
| H0-H8 / MDHR readout sweep | Tested late, mid, early, and multi-depth heading/range readouts. The H8 mid-late reduction became **Multi-Depth Heading Readout (MDHR)**. |
| H1 / R1 / HR controls | Opened heading-only, range-only, and full heading+range objectives at the same 2.5k scale. These controls showed that single-axis objectives can create severe negative transfer. |
| PAAER | Introduced a protected metric path and axis-asymmetric expert readout. It became the primary learned predictor in the final package. |
| Tail continuation | Converted the PAAER line from a strong balanced proxy into a more range-tail-robust official candidate. |
| OFFER / PRM / router-style attempts | Explored richer dynamic routing and prediction refinement ideas, but these were not promoted into the final release claim. |

The design lesson was not "hard split the two axes." Phase95 representation
audits rejected a simple hard-subspace story. The supported lesson was:

```text
shared pose-regime representation + coordinate-specific readability +
protected metric evidence
```

## 4. Key Phenomena That Shaped PACE

Four phenomena were especially important because they explain why PACE is not
just a renamed ensemble.

### Axis-Conditioned Evidence Utility

B811 source diagnostics showed that geometry-like evidence was mostly useful
for heading and unsafe for range. Direct pairwise comparison counted geometry
heading better on `694/811` pairs and geometry distance worse on `758/811`
pairs. The stricter binning used by the mechanism summary still showed the
same direction: geometry helpful for heading on `608/811` rows, harmful for
distance on `665/811` rows, with `492` rows in the
`heading_helpful_distance_harmful` regime.

This phenomenon justified protected metric paths and axis-specific readout.

### Training-Time Axis Competence Desynchronization

Across checkpoint sweeps and final-stage feedback, heading and range did not
always peak at the same training state. The most useful distance checkpoint was
often earlier or came from a different model line than the best heading source.
This is why the final system did not simply use one last checkpoint for both
axes. PAAER supplied the final heading, while PAAER/MDHR/MARB all contributed
range evidence through Range Stack Fusion.

### Prediction-Trajectory Instability

Samples whose predictions moved substantially along a checkpoint trajectory
were more likely to become high-error or high-headroom samples. This made
late-stage model selection depend on trajectory behavior and tail diagnostics,
not only on one aggregate validation number. It also explained why PAAER HR50
was not blindly continued: its median range behavior was competitive, but its
high-absolute-range tail was unsafe.

### Shared Pose-Regime Representation With Readout Co-adaptation

Representation probes showed that H8 features encode pose-regime information,
including heading bins and signed/absolute range buckets. However,
subspace-overlap audits did not support non-overlapping heading/range latent
spaces. The useful framing is therefore co-adapted shared representation plus
axis-conditioned readout, not hard factor disentanglement.

## 5. The PACE Design Principle

PACE can be summarized as:

```text
Polar Axis-Conditioned Estimation:
learn a shared pair representation, protect stable metric evidence, and let
heading/range use axis-conditioned readouts and calibrated outputs.
```

The method-level design contains three learned predictor roles:

| component | public role |
|---|---|
| PAAER | Primary protected axis-asymmetric predictor; final heading source. |
| MDHR | Multi-depth heading-readout model retained as complementary range/axis evidence. |
| MARB | Metric-aware Reloc3r baseline retained as an independent distance source. |

The final competition package then adds two deterministic system components:

| component | role |
|---|---|
| RSF | Combines PAAER, MDHR, and MARB distance predictions. |
| Task-aware output calibration | Snaps heading and distance to public train/dev support structures. The code historically calls this LSS. |

## 6. Why The Final Package Includes Postprocess

The final submission was built for a challenge setting, so it includes
postprocess that should be described honestly. Range Stack Fusion gave a
limited but useful official improvement: PAAER raw `0.002514` improved to RSF
e230 `0.002413`, mainly through distance error. The larger final jump came
from task-aware output calibration: RSF e230 plus heading lattice reached
`0.002308`, and the final support-distance calibrated package reached
`0.001874`.

This postprocess is not presented as the central scientific novelty. It is a
reproducible challenge-system component. The main method insight remains
axis-conditioned polar estimation: heading and range share a representation,
but their evidence utility, training dynamics, and tail risks differ enough
that a strong system should model them asymmetrically.

## 7. Claim Boundary

PACE claims:

- PairUAV polar regression benefits from axis-conditioned readout rather than
  a fully homogeneous shared scalar head.
- Geometry/source evidence is useful but axis-asymmetric.
- A protected metric path is important for range stability.
- Complementary range predictors and public-support calibration can improve a
  challenge submission when applied deterministically and reproducibly.

PACE does not claim:

- hidden official-test labels were used;
- pair IDs directly determine distance;
- heading and range are fully independent latent factors;
- the final leaderboard score comes from one single end-to-end model without
  postprocess;
- source-adaptation or geometry-only routes were the final submitted system.

## 8. Reading Map

For implementation details, see [METHOD.md](METHOD.md). For official feedback
and package-level results, see [RESULTS.md](RESULTS.md). For detailed route
tables, see [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). For source
adaptation and split-fusion evidence, see [SOURCE_ADAPTATION.md](SOURCE_ADAPTATION.md).
For mechanism figures and representation evidence, see
[MECHANISM_INSIGHTS.md](MECHANISM_INSIGHTS.md).

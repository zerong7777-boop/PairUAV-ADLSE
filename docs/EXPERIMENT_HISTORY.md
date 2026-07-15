# Experiment History And Internal Results

[中文说明](EXPERIMENT_HISTORY.zh-CN.md)

This document records the main routes that shaped the final release. It is a
curated history, not a complete experiment ledger. The goal is to make the
method design auditable: what was tried, what worked, what failed, and which
claims are only internal validation signals.

## Evidence Levels

| level | meaning | examples |
|---|---|---|
| Official feedback | Hidden-test score returned by the competition server | MARB, MDHR, PAAER raw, RSF/LSS final |
| Local validation | Fixed local split or val811 proxy with labels available | Phase104g/h/i, Phase95 representation audits |
| Mechanism validation | Validation-only diagnostic surfaces, not deployable policies | B811 geometry utility, ADPA experiments |
| Negative / blocker result | A route that was rejected, blocked, or not promoted | checkpoint averaging, global calibration, selector attempts |

Only the first level is official leaderboard evidence. The other levels explain
design decisions and paper-facing mechanism claims.

## Route Summary

| route | design idea | key evidence | outcome |
|---|---|---|---|
| MARB / Reloc3r metric-aware baseline | Train a Reloc3r-style heading/range predictor with the PairUAV official metric-aware objective. | Official hidden-test `final_score=0.003188`, `distance=0.002528`, `angle=0.003849`. | Kept as a baseline and final distance diversity source. |
| External geometry / split-fusion probes | Use geometry-heavy sources to improve polar prediction. | RoMa/MASt3R/DUSt3R/VGGT probes and B811 diagnostics showed geometry/source utility is strongly axis-dependent. | Promoted as mechanism evidence, not as a standalone final predictor. |
| ADPA axis-decoupled composition | Compose geometry-assisted heading with base distance. | ADPA-1 repaired surface: base heading `8.6087`, geometry heading `1.7769`, axis-decoupled heading `1.7769`; distance stays at base `0.9876`. | Supported axis-decoupling on validation surface; not deployed due to control/selectivity issue. |
| MDHR / H8 mid-late | Let heading read multi-depth decoder evidence while preserving stable range path. | Official hidden-test `final_score=0.00246`, improving over MARB by about 22.8%. | Kept as a strong official model and complementary range source. |
| PAAER 2.5k controls | Separate parameter-opening effects from axis-asymmetric readout effects. | HR all-2500 local proxy `0.011680`, better than C0, H8 full, H8 mid-late fair controls. | Promoted PAAER as the main full-loss architecture candidate. |
| PAAER HR50 fulltrain | Scale PAAER full-loss training toward a stronger leaderboard candidate. | Median range error was close to H8, but p95/p99/max tail errors were much worse. | Rejected blind continuation; diagnosed as range-tail robustness failure. |
| PAAER Tail10 | Continue HR50 with tail-weighted range loss. | Local proxy improved from HR50 `0.006376` to Tail10 `0.002934`, beating H8 step50k `0.003412`. | Used as the source line for final PAAER official inference. |
| Phase95 checkpoint averaging | Average H8 intermediate checkpoints or MNR-like variants. | OOF-stable gains over H8 final were absent; best differences were near numerical noise. | Rejected for official promotion. |
| Phase95 low-degree calibration | Apply global range bias/scale and heading offset corrections. | Best OOF method worsened H8 final proxy. | Rejected for official promotion. |
| MTL-GA / PRC-GFC / LG-PHF / FR-JEPA | Explore non-naive multi-objective, relational, patch-hypothesis, and latent-prediction routes. | Implemented and verified as formal lab routes; promotion required matched gates over Tail10/B4 anchors. | Kept as research exploration; not part of the final submitted system. |
| RSF + task-aware calibration | Combine complementary distance sources and snap outputs to public task support structures. | Official feedback improved from RSF e230 `0.002413` to final PACE `0.001874`. | Final challenge-system package. |

## Evidence-To-Design Audit

This table links the curated results above to the public design claims. It is
an audit map, not an additional experiment table.

| design claim | supporting result family | resulting method choice |
|---|---|---|
| Heading and distance should not be forced through one identical evidence path. | ADPA, H-readout sweep, source-adaptation and split-fusion diagnostics. | Axis-aware readout and protected metric paths. |
| Heading benefits from multi-depth decoder evidence. | H8 mid-late matched controls and later official MDHR feedback. | MDHR retained as a strong official predictor and range-diversity source. |
| PAAER is useful but needs range-tail protection. | HR all-2500 control, HR50 tail quantiles, Tail10 improvement. | PAAER/Tail line used for final official inference. |
| Distance benefits from complementary predictors. | MARB, MDHR, and PAAER had different distance error profiles. | RSF uses PAAER, MDHR, and MARB distances. |
| Public output-support structure is a deployable postprocess signal. | Official RSF e230, heading lattice snap, and final support snap feedback. | Task-aware output calibration applied as a deterministic final step. |

## Source Adaptation And Split-Fusion Evidence

External-source experiments were used to test whether geometry, dense matching,
and solver-style priors could improve PairUAV prediction. The complete
inventory is in [SOURCE_ADAPTATION.md](SOURCE_ADAPTATION.md). The high-level
history is:

| family | strongest signal | limitation |
|---|---|---|
| Dense matching sources | RoMa dense improved the same-contract 2048 replay to `15.3929 / 19.4702`. | Still an internal proxy, not a hidden-test system. |
| Geometry-foundation sources | MASt3R and DUSt3R geometry fields gave strong heading signals: `9.5863 / 33.2327` and `11.0434 / 33.1489`. | Distance was weak. |
| VGGT-style range source | VGGT geometry field was weak on heading but stronger on distance: `52.5120 / 14.1716`. | Direct camera extraction was not competitive. |
| Three-source hybrid | VGGT and MASt3R+VGGT were strong proxy/range candidates; full RoMa+MASt3R+VGGT gave the best angle. | Naive concatenation did not consistently improve the balanced proxy. |
| Split-fusion | Two-branch old-field heading plus rich-native range reached `1.330926 / 4.024406 / 0.025360` on the 2048 diagnostic protocol. | Full official inference cost and no-op/selectivity issues blocked final promotion. |
| Solver/frozen-pose routes | MADPose, FAR, SAC-Pose, and GRelPose were technically runnable. | Their bounded PairUAV metrics were not competitive. |

The important design lesson was not that one external source solved PairUAV.
It was that sources specialize by axis. This directly motivated protected
metric paths, axis-specific readout, and the final range/heading postprocess
strategy.

## H-Readout Architecture Sweep

The H-index names were internal Reloc3r-family readout labels, not public
method names. They are included here because MDHR came from this sweep. All
rows below are local `val811` proxy records unless the table explicitly says
otherwise.

Completeness and comparability notes:

- H0-H5/H8 were architecture/readout labels. H6/H7 in earlier notes referred
  to broader research hypotheses, not completed readout implementations, so no
  H6/H7 numeric row is claimed.
- H4/H5 were topology-protected designs, not capability-preserved designs:
  shared encoder/decoder/base features still received gradients from both
  heading and range losses.
- Compare rows only inside the same phase table. Phase83, Phase87, and Phase88
  used different train surfaces or initialization choices.

Internal taxonomy:

| label | implementation intent | recovered metric status |
|---|---|---|
| H0 | shared PairUAV heading/range head | completed local rows |
| H1 | late residual/adapter axis decoupling | completed local rows |
| H2 | mid-level split | completed local row |
| H3 | early split | completed local rows |
| H4 | range-H0 plus heading-H2 | design label; no complete comparable metric row recovered |
| H5 | range-H0 plus heading-H3 | completed local rows |
| H8 | range-H0 plus heading multi-depth decoder readout | completed local rows; later reduced to the MDHR mid-late variant |

Phase83 tested early axis decoupling on a fixed local validation surface:

| setting | steps | angle MAE | range MAE | proxy | note |
|---|---:|---:|---:|---:|---|
| Wtrunk/T2/H0 | 2500 | 4.027444 | 4.010942 | 0.018784 | clean aligned baseline |
| Wtrunk/T2/H0 | 5000 | 3.759863 | 3.306784 | 0.016707 | H0 continuation |
| Wtrunk/T2/H1 | 2500 | 2.333164 | 3.960756 | 0.013982 | late axis decoupling helps Wtrunk |
| Wtrunk/T2/H1 | 5000 | 2.454658 | 3.323074 | 0.013112 | still better than H0 |
| Wtrunk/T2/H1 | 7500 | 2.527736 | 2.790420 | 0.012306 | proxy improves through range |
| Wstrip/T2/H0 | 2500 | 3.141467 | 3.293436 | 0.014964 | unexpectedly strong |
| Wstrip/T2/H0 | 5000 | 2.677615 | 1.776450 | 0.010802 | scales well |
| Wstrip/T2/H0 | 7500 | 2.566347 | 1.431668 | 0.009840 | best Phase83 scaled route |
| Wstrip/T2/H3 | 2500 | 1.889435 | 3.854808 | 0.012549 | strong heading, weaker range |
| Wstrip/T2/H2 | 2500 | 2.277588 | 3.519935 | 0.012993 | also heading-favorable |

Phase87 tested H0/H3/H5/H8 scaling under the Reloc3r-512 backbone-only
initialization and full-model finetuning:

| setting | steps | angle MAE | range MAE | proxy |
|---|---:|---:|---:|---:|
| H0 | 2500 | 7.764222 | 6.604099 | 0.034075 |
| H0 | 10000 | 7.519871 | 2.239851 | 0.025131 |
| H3 | 2500 | 7.474898 | 4.058125 | 0.028449 |
| H3AO | 2500 | 1.667425 | 67.782616 | 0.133008 |
| H5 | 2500 | 3.511572 | 4.315968 | 0.017929 |
| H5 | 5000 | 3.271602 | 5.852879 | 0.020173 |
| H5 | 10000 | 6.061516 | 1.945850 | 0.020523 |
| H8 | 2500 | 3.970750 | 5.872404 | 0.022152 |
| H8 | 5000 | 3.915341 | 2.539262 | 0.015685 |
| H8 | 10000 | 3.013667 | 1.281393 | 0.010798 |
| H8 | 100000 | 1.125074 | 1.144259 | 0.005292 |

Phase87 showed that H8 was not the best 2500-step architecture everywhere, but
it was the first tested H-family route with clear scale-up behavior.

Phase88 then isolated the H8 readout-depth question on its own local surface:

| setting | steps | angle MAE | range MAE | proxy |
|---|---:|---:|---:|---:|
| late-only | 2500 | 2.320584 | 4.892704 | 0.015713 |
| mid-late | 2500 | 1.902788 | 4.297642 | 0.013425 |
| mid-late | 5000 | 1.229486 | 3.664675 | 0.010356 |
| mid-late | 10000 | 1.211283 | 3.432744 | 0.009866 |
| mid-only | 2500 | 1.996787 | 4.257079 | 0.013609 |
| early-late | 2500 | 1.759591 | 4.863244 | 0.014098 |
| early-mid | 2500 | 1.836391 | 5.837262 | 0.016157 |
| full H8 same runner | 2500 | 1.794715 | 5.167480 | 0.014772 |
| decoder-last1-H0 | 2500 | 42.604271 | 5.161415 | 0.128121 |
| decoder-last1-H8 | 2500 | 4.834469 | 4.314959 | 0.021601 |

On the matched 10k same-surface control, the reduced H8 mid-late readout also
beat full early/mid/late H8:

| setting | max-step label | actual train loop | angle MAE | range MAE | proxy |
|---|---:|---|---:|---:|---:|
| H8 mid-late | 10000 | 1 epoch / 5000 iters | 1.211282730 | 3.432743549 | 0.009866082 |
| H8 full | 10000 | 1 epoch / 5000 iters | 1.280849576 | 3.663944244 | 0.010497204 |

This is the direct local evidence that promoted the MDHR mid-late design before
the later full-train run and official hidden-test evaluation.

## Internal Axis-Objective Controls

Phase104g tested all-parameter training under three objectives: heading-only
H1, range-only R1, and full heading+range HR. These were not official
submissions; they were controlled local probes.

| run | angle MAE | range MAE | proxy | interpretation |
|---|---:|---:|---:|---|
| H1 heading-only all-2500 | 1.5958 | 53.6993 | 0.106136 | improves heading but destroys range |
| R1 range-only all-2500 | 122.2311 | 3.6355 | 0.346416 | improves range but destroys heading |
| HR full-loss all-2500 | 1.4985 | 3.9690 | 0.011680 | best balanced PAAER control at this scale |
| H8 mid-late fair 2500 | 1.8194 | 5.2569 | 0.015010 | matched H8-style fair control |

This experiment was important because it showed that simply opening all
parameters is not enough; axis-specific objectives can create severe negative
transfer, while balanced PAAER training can improve the joint proxy.

## Tail Robustness Diagnostic

Phase104h showed why HR50 did not simply dominate H8 mid-late:

| run | range median | p95 | p99 | max |
|---|---:|---:|---:|---:|
| HR50 | 0.4546 | 7.1223 | 42.9710 | 91.2856 |
| H8 step50k | 0.3897 | 1.1979 | 2.1321 | 6.9968 |
| Tail10 | 0.3600 | 1.0446 | 1.2653 | 4.3494 |

The issue was not global scalar calibration. It was high-absolute-range tail
robustness. Tail-weighted continuation directly addressed that failure mode.

## Official Final-Stage Scores

| package | components | final_score | distance_rel_error | angle_rel_error |
|---|---|---:|---:|---:|
| MARB raw | Reloc3r metric-aware baseline | 0.003188 | 0.002528 | 0.003849 |
| MDHR raw | H8 mid-late | 0.002460 | 0.002274 | 0.002646 |
| PAAER raw | PAAER heading + PAAER distance | 0.002514 | 0.002392 | 0.002636 |
| RSF e230 | PAAER heading + fused range | 0.002413 | 0.002191 | 0.002636 |
| RSF e230 + heading lattice | RSF + heading lattice snapping | 0.002308 | 0.002150 | 0.002465 |
| PACE final | RSF + heading lattice + support-distance snapping | 0.001874 | 0.001330 | 0.002419 |

These official numbers are also summarized in [RESULTS.md](RESULTS.md).

## What Was Not Claimed

Several explored routes produced useful information but are not part of the
final method claim:

- checkpoint averaging and global calibration did not produce OOF-stable gains;
- feature-semantics selectors found tiny high-precision slices but insufficient
  recall for deployable routing;
- Phase95 subspace overlap did not support a hard non-overlapping latent
  subspace story;
- LG-PHF/FR-JEPA-style routes were not promoted to the final submission because
  they did not complete the required promotion path before the deadline.

The public release therefore keeps these as design history and negative
evidence, while the reproducible final package remains PACE.

# Research Narrative

[中文说明](RESEARCH_NARRATIVE.zh-CN.md)

This note explains how the final method emerged. It is not a chronological log
of every run; it is a paper-facing reconstruction of the design path from task
diagnosis to the final reproducible submission system.

## Starting Point

PairUAV asks for polar relative localization: a heading-like angular output and
a distance-like range output. The first strong baseline was a metric-aware
Reloc3r model trained on the official PairUAV objective. That model established
the release baseline used here as **Metric-Aware Reloc3r Baseline (MARB)**. Its
official hidden-test feedback was:

| system | final_score | distance_rel_error | angle_rel_error |
|---|---:|---:|---:|
| MARB / Reloc3r full-data one-epoch run | 0.003188 | 0.002528 | 0.003849 |

This baseline was useful but also exposed the core limitation: the best
distance and heading behavior did not always come from the same evidence path.

## Mechanism Diagnosis

Early mechanism probes showed that PairUAV difficulty is not a single scalar
notion of "hardness." Frozen observability proxies produced regime-dependent
angle/range behavior. On control and intermediate splits, heading was harder
than distance; on the stress split, distance became much harder than heading.

Later B811 comparable-surface diagnostics gave a sharper mechanism signal:
geometry-derived evidence was usually helpful for heading but harmful for
distance. On the same 811-row validation surface, geometry was heading-helpful
on 608 rows and distance-harmful on 665 rows. The dominant pair regime was
`heading_helpful_distance_harmful` with 492 rows.

This shifted the research question from "how do we train one better scalar
head?" to "how do we let the two polar axes use different evidence without
destroying the stable metric path?"

## Evidence-To-Design Map

The final design can be read as a sequence of evidence-backed responses rather
than as a late-stage ensemble assembled only for leaderboard feedback:

| observation | supporting evidence | design response |
|---|---|---|
| Heading and distance errors are heterogeneous. | H1/R1/HR controls, H-readout sweeps, and B811 source diagnostics showed that one axis can improve while the other degrades. | Use axis-aware readout instead of a fully shared scalar head. |
| Geometry-like evidence is heading-favorable but distance-risky. | B811 geometry comparison and split-fusion probes showed dominant heading-helpful / distance-harmful behavior. | Protect the metric distance path while allowing richer heading evidence. |
| Multi-depth decoder evidence helps heading. | H8 mid-late outperformed late-only and full early/mid/late variants on matched local controls. | Promote the H8 mid-late design as MDHR. |
| PAAER full-loss scaling exposes range-tail failures. | HR50 had competitive median range error but much worse p95/p99/max tails than H8. | Add tail-weighted continuation before official PAAER inference. |
| Final predictions benefit from public output-support calibration. | Official feedback improved from RSF e230 to heading lattice snapping and then to distance support snapping. | Apply deterministic task-aware calibration after learned prediction and range fusion. |

## From Axis Split To Model Design

The first paper-facing design lesson was axis decoupling. Validation-only
ADPA-style experiments showed that composing geometry heading with base
distance could retain geometry heading gains while preventing geometry distance
damage. The repaired ADPA-1 bounded run reported:

| variant | heading MAE | distance MAE |
|---|---:|---:|
| base only | 8.6087 | 0.9876 |
| geometry only | 1.7769 | 6.9755 |
| axis-decoupled | 1.7769 | 0.9876 |

This was not yet a deployable model, because always using geometry heading still
damaged base-sufficient cases. But it established the correct architectural
direction: preserve the stable distance path while giving heading a richer
readout.

That direction produced two model families used in the final release:

- **Multi-Depth Heading Readout (MDHR):** heading reads multi-depth decoder
  evidence while distance stays close to a stable late metric readout.
- **Protected Axis-Asymmetric Expert Readout (PAAER):** the primary model keeps
  a protected metric anchor and adds axis-asymmetric expert readouts.

## Why Tail Robustness Became Necessary

PAAER full-loss all-parameter training improved the local balanced proxy, but
the 50k fulltrain continuation showed a high-absolute-range tail failure:
median range error was close to H8, while p95/p99/max were much worse. This
diagnosis changed the optimization target from "train longer" to "protect range
tails."

The tail-weighted continuation fixed this failure mode on the same val811
surface:

| run | angle MAE | range MAE | proxy |
|---|---:|---:|---:|
| HR50 fulltrain | 0.7791 | 2.2237 | 0.006376 |
| H8 mid-late step50k | 0.9048 | 0.4744 | 0.003412 |
| PAAER Tail10 | 0.7647 | 0.4278 | 0.002934 |

This result motivated using the PAAER/Tail line for the final official
inference, while retaining MDHR and MARB as complementary predictors.

## Final System Design

The final submitted system, **PACE: Polar Axis-Conditioned Estimation**,
combines the model-design lessons with deterministic competition-system
calibration.

1. PAAER supplies the final heading source.
2. Range Stack Fusion (RSF) combines PAAER, MDHR, and MARB distances.
3. Task-aware output calibration snaps heading and distance to support
   structures observed in the released train/dev split. The implementation is
   historically named LSS in scripts and manifests.

This is why PACE is described as a system rather than as a single neural
architecture. The learned contribution is axis-aware prediction; the final
leaderboard package also uses deterministic public-structure calibration.

## Claim Boundary

The supported claim is not that heading and distance live in completely
separate latent spaces. Phase95 subspace-overlap audits rejected that simple
interpretation. The stronger supported statement is:

```text
PairUAV exposes a shared pose-regime representation whose coordinates have
axis-specific readability and axis-asymmetric error/utility profiles.
```

This motivates protected axis-asymmetric readout, complementary range fusion,
and downstream output calibration, without overstating the mechanism as hard
latent factor independence.

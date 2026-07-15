# Results

[中文说明](RESULTS.zh-CN.md)

This document separates three evidence levels:

1. **Official leaderboard feedback:** scores returned by the competition server
   for submitted packages.
2. **Raw prediction assets:** official-test predictions included for
   reproduction; raw zips do not contain labels or scores.
3. **Internal development proxies:** local validation/proxy measurements used
   during model selection.

## Official Leaderboard Feedback

The table below records the official feedback currently available from the
final submission process. Lower is better. These scores are server feedback,
not values that can be recomputed from the released raw zips because official
test labels are hidden.

| System / package | Components | Postprocess | final_score | distance_rel_error | angle_rel_error | Evidence |
|---|---|---|---:|---:|---:|---|
| PAAER raw | PAAER heading + PAAER distance | none | 0.002514 | 0.002392 | 0.002636 | manual leaderboard feedback |
| PAAER heading + MDHR/PAAER distance mix | PAAER heading + 0.66 PAAER distance + 0.34 MDHR distance | none | 0.002446 | 0.002245 | 0.002646 | manual leaderboard feedback |
| RSF e230 | PAAER heading + RSF distance | none | 0.002413 | 0.002191 | 0.002636 | manual leaderboard feedback |
| RSF e230 + heading lattice | PAAER heading + RSF distance | heading lattice snapping | 0.002308 | 0.002150 | 0.002465 | manual leaderboard feedback |
| PACE final | PAAER heading + RSF distance | heading lattice + support-distance snapping | 0.001874 | 0.001330 | 0.002419 | manual leaderboard feedback; package hash verified in `OPEN_SOURCE_AUDIT_20260701.md` |

The final package itself is reproducible byte-identically through
`postprocess/rebuild_final_postprocess.py`; see
`OPEN_SOURCE_AUDIT_20260701.md` for the hash-level evidence.

## Raw Prediction Assets

The final PACE package is reconstructed from three raw official-test prediction
zips:

| Public name | Asset | Model role | ZIP SHA256 |
|---|---|---|---|
| PAAER raw | `assets/phase104j_HRofficial_raw_result.zip` | Primary axis-decoupled predictor | `294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1` |
| MDHR raw | `assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` | Multi-depth heading-oriented complementary predictor | `b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b` |
| MARB raw | `assets/epoch2_pair_official_test_result_20260527_013814.zip` | Metric-aware Reloc3r baseline predictor | `7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee` |

MDHR and MARB are included as raw assets and checkpoint assets so that the final
system can be reproduced. Their standalone official-test feedback was not
stored in the current public audit file; if that feedback is recovered, it can
be added here without changing the reproduction path.

## Internal Development Signals

Internal proxy metrics were used to decide which model families deserved full
official inference and submission attempts. They are not directly comparable to
the official leaderboard because they use different splits and may use proxy
metrics.

Examples preserved in the runtime notes include:

| Context | Observation |
|---|---|
| 2.5k fair controls | The full-loss HR/PAAER control improved over the H8 and H8 mid-late fair controls in the local proxy. |
| Source adaptation probes | RoMa, MASt3R, DUSt3R, VGGT, SAC-Pose, MADPose, FAR, GRelPose, and split-fusion routes were tested as bounded internal evidence, not official leaderboard systems. |
| Range-tail diagnosis | PAAER full-train variants had competitive median range error but worse high-absolute-range tails before tail-weighted continuation. |
| Tail-weighted continuation | Tail10 improved the local proxy over the H8 step50k anchor and reduced catastrophic range outliers. |
| Final postprocess sweep | RSF weight sweep improved distance error, while heading-lattice and support-distance snapping produced the largest final improvement. |

For source-adaptation details, see [SOURCE_ADAPTATION.md](SOURCE_ADAPTATION.md).
For exact historical details, see the phase status notes in the project runtime
archive. These notes are retained as provenance, not as polished paper tables.

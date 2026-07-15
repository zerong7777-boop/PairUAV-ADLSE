# Open Source Reproduction Audit 2026-07-01

[中文说明](OPEN_SOURCE_AUDIT_20260701.zh-CN.md)

This audit checks whether the public repository plus the release assets listed
in `assets/RELEASE_ASSETS.md` can reproduce the final
`phase104j_snap_e230_Hlat2_Dsupport` submission.

## Scope

Verified:

- Full base source tree is present.
- Final 5090 source snapshots are installed at their runtime target paths.
- Historical training and inference scripts required by the A-path manifest are
  present.
- Official inference manifests and the train/dev support manifest are present.
- The B-path postprocess script rebuilds the final zip byte-identically from
  the three raw official prediction zips.

Not claimed:

- Bit-identical retraining from random initialization or dataloader ordering.
- Redistribution of official PairUAV test images/JSON.
- Inclusion of large checkpoint/raw-prediction assets inside git.

## Static Checks

Manifest-backed checks passed:

```text
source_snapshots_checked: 8
run_scripts_checked: 8
required_files_checked: 14
ast_targets_checked: 10
support_rows_including_header: 204121
errors: []
```

The support manifest hash is:

```text
f61f909a7441ac95c3790725507fb60851f7f10b5f3e6b477e92b5e36a96d580
```

## Final Package Rebuild Evidence

Command class:

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

Observed verification:

```text
hr_zip_sha256: 294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1 OK
hr_result_txt_sha256: 6c7da711caac6e6feb91355f7936a820e29f0c3bb19f4affbdfe195f60a507ad OK
h8_zip_sha256: b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b OK
h8_result_txt_sha256: 50ce60ce71465b7ddf3445e98ccca65576729a16e40d28556cd0678e1c213360 OK
epoch2_zip_sha256: 7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee OK
epoch2_result_txt_sha256: 956711c0c387d5db5999d3c464267e9eb56c4b036688693ec51538fecc4edeba OK
support_manifest_sha256: f61f909a7441ac95c3790725507fb60851f7f10b5f3e6b477e92b5e36a96d580 OK
support_rows: 204120
support_distance_count: 211
e230_rows: 2773116
final_rows: 2773116
e230_result_zip_sha256: 2c4f55a4df297806dd1a851e59b2af08ecc1e35e7361bb5b6e6f4ea9065a2548 OK
final_result_zip_sha256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f OK
known_e230_txt_equal: True
known_final_txt_equal: True
```

## Release Boundary

The source repository is sufficient when combined with:

- Official PairUAV test data for regenerating raw predictions.
- The checkpoint assets listed in `assets/RELEASE_ASSETS.md`.
- Or, for the fast deterministic path, the three raw official prediction zips
  listed in `assets/RELEASE_ASSETS.md`.

The repository alone cannot regenerate the leaderboard package without those
external assets.

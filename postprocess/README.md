# Deterministic Submission Postprocess

[中文说明](README.zh-CN.md)

This folder contains the deterministic postprocess reconstruction for the final
PairUAV submission `phase104j_snap_e230_Hlat2_Dsupport`.

The submission package name contains an internal run identifier for provenance.
In the public PACE release, this is the deterministic final-stage package:
multi-source range stack followed by task-aware heading and distance
calibration.

## Required Assets

Place these files in an asset directory:

```text
phase104j_HRofficial_raw_result.zip
phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip
epoch2_pair_official_test_result_20260527_013814.zip
devsplit_v1_official_metric_manifest.csv
```

The support manifest must match:

```text
f61f909a7441ac95c3790725507fb60851f7f10b5f3e6b477e92b5e36a96d580
```

Input asset hashes:

| Asset | ZIP SHA256 | `result.txt` SHA256 |
|---|---|---|
| `phase104j_HRofficial_raw_result.zip` | `294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1` | `6c7da711caac6e6feb91355f7936a820e29f0c3bb19f4affbdfe195f60a507ad` |
| `phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` | `b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b` | `50ce60ce71465b7ddf3445e98ccca65576729a16e40d28556cd0678e1c213360` |
| `epoch2_pair_official_test_result_20260527_013814.zip` | `7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee` | `956711c0c387d5db5999d3c464267e9eb56c4b036688693ec51538fecc4edeba` |

## Rebuild Command

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

Optional verification against known packages:

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip \
  --known-e230-zip known/phase104j_HRheading_range_stack_sweep_e230.zip \
  --known-final-zip known/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

Expected verification:

```text
verified environment: Python 3.11.9, zlib 1.3.1
support_rows: 204120
support_distance_count: 211
e230_rows: 2773116
final_rows: 2773116
hr_zip_sha256: 294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1 OK
h8_zip_sha256: b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b OK
epoch2_zip_sha256: 7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee OK
e230_result_txt_sha256: ed384ec336cbc4e1e21df6923391e1ce441d6e0e10b9a2ed68e48394e01280d4 OK
e230_result_zip_sha256: 2c4f55a4df297806dd1a851e59b2af08ecc1e35e7361bb5b6e6f4ea9065a2548 OK
final_result_txt_sha256: 32e1bfeca74b64617b0b8f8cc25618ec44c36fe14b4406682c8aa29ecb86a884 OK
final_result_zip_sha256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f OK
```

## Method

The e230 package uses HR heading and a three-source distance stack:

```text
distance = 0.511 * HR + 0.189 * H8 + 0.300 * epoch2
```

The final package then applies:

```text
heading = nearest 2-degree lattice value, normalized to [-180, 180)
distance = nearest train/dev support distance from gt_distance
```

The support set is derived from the released train/dev split manifest, not from
hidden official-test labels.

# Release Assets

[中文说明](RELEASE_ASSETS.zh-CN.md)

The git repository contains code, configs, manifests, and deterministic
postprocess logic. The following large files should be published as release
assets or external downloads.

Some asset names include internal run identifiers. They are kept unchanged so
that hashes, manifests, and reproduction commands remain stable.

## Download

Current external mirror:

```text
Baidu Netdisk: https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
Extraction code: djmv
```

Download the asset folder and copy its contents into the repository root while
preserving the relative paths below. The expected layout is `assets/...` for
checkpoints and raw official prediction zips, plus `known/...` for optional
known output packages.

## Checkpoints

| Relative path | Role | SHA256 |
|---|---|---|
| `assets/reloc3r_official_metric_full_epoch_checkpoint_last/checkpoint-last.pth` | Source checkpoint resumed by the epoch2/rank1 training run | `d08d66fe53fbbcf43ce451e406850c7300a12e5e588879811142d66e67f32c5d` |
| `assets/phase45_epoch2_resume_final/checkpoint-final.pth` | epoch2/rank1 raw official prediction source | `5681ab612d44dc64c98c82ef5b8b4e36c2bb4b38f6748a469f9c8d78c3894e04` |
| `assets/phase104i_tail10_init/checkpoint-final.pth` | initialization checkpoint for phase104j long training | `e7e42848b6e18cb31a713b0b95a9c4e7f6e529d7f9c6b2a3e26882a44bbf2f67` |
| `assets/phase104j_paaer_hr_final/checkpoint-final.pth` | HR/PAAER final raw official prediction source | `7454c0f2b31921cd917c21c3726a7cc1860bef03163c474ed689c19e7af4d01e` |
| `assets/phase89_h8_midlate_final/checkpoint-final.pth` | H8 mid-late raw official prediction source | `cc985ac7671e7d6dc22928df7eb443ac4f95ad4e295333017e019f6da34216c8` |

## Raw Official Prediction Zips

| Relative path | ZIP SHA256 | `result.txt` SHA256 |
|---|---|---|
| `assets/phase104j_HRofficial_raw_result.zip` | `294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1` | `6c7da711caac6e6feb91355f7936a820e29f0c3bb19f4affbdfe195f60a507ad` |
| `assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` | `b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b` | `50ce60ce71465b7ddf3445e98ccca65576729a16e40d28556cd0678e1c213360` |
| `assets/epoch2_pair_official_test_result_20260527_013814.zip` | `7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee` | `956711c0c387d5db5999d3c464267e9eb56c4b036688693ec51538fecc4edeba` |

## Optional Known Output Zips

These are not needed to rebuild the final package, but they are useful for
byte-level verification.

| Relative path | ZIP SHA256 |
|---|---|
| `known/phase104j_HRheading_range_stack_sweep_e230.zip` | `2c4f55a4df297806dd1a851e59b2af08ecc1e35e7361bb5b6e6f4ea9065a2548` |
| `known/phase104j_snap_e230_Hlat2_Dsupport_result.zip` | `3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f` |

## Included Small Asset

`manifests/devsplit_v1_official_metric_manifest.csv` is included in the
repository because it is small enough to track directly.

```text
SHA256: f61f909a7441ac95c3790725507fb60851f7f10b5f3e6b477e92b5e36a96d580
rows including header: 204121
```

## External Competition Data

The official PairUAV test JSON/images are not redistributed here. They are
required only when regenerating the three raw prediction zips from checkpoints.

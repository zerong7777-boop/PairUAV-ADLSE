# Model-Level Reproduction: Source, Checkpoints, Raw Predictions

[中文说明](A_REPRODUCTION.zh-CN.md)

This document covers model-level reproduction for the PACE PairUAV release:
reproduce the raw official predictions from source snapshots and released
checkpoints, then feed those raw predictions to the deterministic postprocess
path.

Historical asset names may contain internal run identifiers such as `phase104j`
or `phase89`. These names are preserved for provenance and hash-stable
reproduction, not as public method names.

The final leaderboard package is not a raw single-model inference result. The
A path produces three raw prediction sources:

```text
HR/PAAER raw prediction
H8 mid-late raw prediction
epoch2/rank1 raw prediction
```

The deterministic postprocess then applies range stacking and task-aware output
calibration.

## Source Snapshot

Start from the `reloc3r_pairuav` codebase. The lab git HEAD observed during
audit was:

```text
90e478bd420942db7bc262bdb574b41a7d026318
```

Use the final 5090 source snapshots below as authoritative for this release,
because the lab working tree changed after the final official inference run.

| Release file | Target path | SHA256 |
|---|---|---|
| `phase104j_5090_current_pose_head.py` | `reloc3r/pose_head.py` | `3581e4b60cbdbc7e1fbc8af1868873b9560589312a7d6680401b72095db1ce1b` |
| `phase104j_5090_current_reloc3r_relpose.py` | `reloc3r/reloc3r_relpose.py` | `e0e6cc1832799a62581ecbc674f31db54a71ba21a91eb1ead8907d0ba0dc5f5a` |
| `phase104j_5090_current_trainable_policy.py` | `reloc3r/trainable_policy.py` | `3c9cd7e950bf13d31c05092b88e2aa1b99b1ec4d422b2722e3dba3a439a7170b` |
| `phase104j_5090_current_loss.py` | `reloc3r/loss.py` | `bab3d9ee945f125676e458dd44c49ed6654f1811139f27b484e9c109428ddbff` |
| `phase104j_5090_current_train.py` | `train.py` | `8b7d788051b4bd67e3272d48cd7eb23a5a621ca55a2c318153db9e12754a3c11` |
| `phase104j_5090_current_eval_pairuav.py` | `eval_pairuav.py` | `64a7b47b23fe553cbe5462098560b482fce7c0a9eed8744a21d438a3249fffdb` |
| `phase104j_5090_current_infer_pairuav_with_progress.py` | `infer_pairuav_with_progress.py` | `d496248db42474c4bb6c8052c0749e7648f42a092fe3d6c109ffb466cf1af30f` |
| `phase104j_5090_current_phase104i_tail_loss.py` | `phase104i_tail_loss.py` | `4b2086aef283928a58ad651c53006556ba944c282dcf6edb9be913337fa3e88a` |

Required source-level checks:

```text
phase104j_5090_current_pose_head.py contains Phase104eProtectedAxisAsymmetricExpertHead
phase104j_5090_current_reloc3r_relpose.py registers pairuav_phase104e_paaer_hard_heading_range
phase104j_5090_current_train.py supports --trainable_policy, --milestone_steps, --step_checkpoint_keep_named
phase104j_5090_current_infer_pairuav_with_progress.py writes result.txt/result.zip/manifest.json
```

## Checkpoint Assets

Publish these checkpoint assets outside git if needed.

| Asset | Role | SHA256 |
|---|---|---|
| `reloc3r_official_metric_full_epoch_checkpoint_last/checkpoint-last.pth` | Source checkpoint resumed by the epoch2/rank1 training run | `d08d66fe53fbbcf43ce451e406850c7300a12e5e588879811142d66e67f32c5d` |
| `phase104j_paaer_hr_final/checkpoint-final.pth` | HR/PAAER final official raw source | `7454c0f2b31921cd917c21c3726a7cc1860bef03163c474ed689c19e7af4d01e` |
| `phase104i_tail10_init/checkpoint-final.pth` | Init checkpoint for phase104j long training | `e7e42848b6e18cb31a713b0b95a9c4e7f6e529d7f9c6b2a3e26882a44bbf2f67` |
| `phase89_h8_midlate_final/checkpoint-final.pth` | H8 mid-late final official raw source | `cc985ac7671e7d6dc22928df7eb443ac4f95ad4e295333017e019f6da34216c8` |
| `phase45_epoch2_resume_final/checkpoint-final.pth` | epoch2/rank1 official raw source | `5681ab612d44dc64c98c82ef5b8b4e36c2bb4b38f6748a469f9c8d78c3894e04` |

All five checkpoint hashes were verified during the open-source audit. The
epoch2 checkpoint size on 5090 was 1.6G.

## Raw Official Inference

Assume the official test files are available as:

```text
OFFICIAL_TEST_JSON=/path/to/UAVM_2026/pairUAV/test
OFFICIAL_TEST_IMAGES=/path/to/UAVM_2026/pairUAV/test_tour
```

Run HR/PAAER:

```bash
python infer_pairuav_with_progress.py \
  --model "Reloc3rRelpose(img_size=512, output_mode='pairuav_phase104e_paaer_hard_heading_range')" \
  --json-root "$OFFICIAL_TEST_JSON" \
  --image-root "$OFFICIAL_TEST_IMAGES" \
  --checkpoint assets/phase104j_paaer_hr_final/checkpoint-final.pth \
  --output-dir outputs/hr_official \
  --split test \
  --resolution "(512,384)" \
  --seed 777 \
  --batch-size 16 \
  --num-workers 8 \
  --amp 1 \
  --zip
```

Expected HR raw output:

```text
rows: 2773116
result.txt SHA256: 6c7da711caac6e6feb91355f7936a820e29f0c3bb19f4affbdfe195f60a507ad
result.zip SHA256: 294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1
```

Run H8 mid-late:

```bash
python infer_pairuav_with_progress.py \
  --model "Reloc3rRelpose(img_size=512, output_mode='pairuav_range_h0_heading_mid_late_heading_range')" \
  --json-root "$OFFICIAL_TEST_JSON" \
  --image-root "$OFFICIAL_TEST_IMAGES" \
  --checkpoint assets/phase89_h8_midlate_final/checkpoint-final.pth \
  --output-dir outputs/h8_official \
  --split test \
  --resolution "(512,384)" \
  --seed 777 \
  --batch-size 16 \
  --num-workers 8 \
  --amp 1 \
  --zip
```

Expected H8 raw output:

```text
rows: 2773116
result.txt SHA256: 50ce60ce71465b7ddf3445e98ccca65576729a16e40d28556cd0678e1c213360
result.zip SHA256: b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b
```

Run epoch2/rank1:

```bash
python infer_pairuav_with_progress.py \
  --model "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')" \
  --json-root "$OFFICIAL_TEST_JSON" \
  --image-root "$OFFICIAL_TEST_IMAGES" \
  --checkpoint assets/phase45_epoch2_resume_final/checkpoint-final.pth \
  --output-dir outputs/epoch2_official \
  --split test \
  --resolution "(512,384)" \
  --seed 777 \
  --batch-size 16 \
  --num-workers 8 \
  --amp 1 \
  --zip
```

Expected epoch2 raw output:

```text
rows: 2773116
result.txt SHA256: 956711c0c387d5db5999d3c464267e9eb56c4b036688693ec51538fecc4edeba
result.zip SHA256: 7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee
```

## Training Recipes

The release includes training scripts for provenance:

```text
phase45_epoch2_launch_full.sh
phase45_epoch2_full.env
phase45_train_pairuav_official_metric_longer_5090.sh
phase45_train_pairuav_full_devsplit.sh
phase104j_tailw_fulltrain1epoch_fromTail10_lr1e-5_bs4_5090.sh
phase104i_HR_tailw_fulltrain10k_fromHR50_lr1e-5_bs4.sh
phase89_mid_late_fulltrain_1epoch_20260618.sh
```

The reloc3r epoch2/rank1 run is a resume run:

```text
source: official_metric_full_epoch_v1_20260506_045442/checkpoint-last.pth
target: phase45_epoch2_resume_full_v1_20260523_215328/checkpoint-final.pth
epochs: 2
lr: 5e-6
batch_size: 8
num_workers: 8
model: Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')
criterion: PairUAVOfficialMetricAwareLoss(...)
```

These scripts are useful to document how the checkpoints were produced. For
leaderboard reproduction, prefer using the released checkpoint assets and the
official inference commands above; full retraining may be nondeterministic
across GPU, CUDA, PyTorch, dataloader, and filesystem ordering.

## Hand Off To B

After generating or downloading the three raw zips, run:

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip outputs/hr_official/result.zip \
  --h8-zip outputs/h8_official/result.zip \
  --epoch2-zip outputs/epoch2_official/result.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

Expected final zip:

```text
3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

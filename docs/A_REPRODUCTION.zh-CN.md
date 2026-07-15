# 模型级复现：源码、Checkpoint 与 Raw Prediction

[English](A_REPRODUCTION.md)

本文档说明 PACE PairUAV release 的模型级复现路径：先从源码快照和已发布 checkpoint 复现 raw official prediction，再将这些 raw prediction 输入确定性后处理路径。

历史资产名称可能包含 `phase104j` 或 `phase89` 等内部运行编号。这些名称仅用于 provenance 和 hash 稳定复现，不作为公开方法名称。

最终榜单提交包不是单个模型的 raw inference 结果。A 路径会产生三个 raw prediction 来源：

```text
HR/PAAER raw prediction
H8 mid-late raw prediction
epoch2/rank1 raw prediction
```

随后确定性后处理会应用 range stacking 和任务感知输出校准。

## 源码快照

从 `reloc3r_pairuav` 代码库开始。审计时 lab git HEAD 为：

```text
90e478bd420942db7bc262bdb574b41a7d026318
```

本 release 以最终 5090 源码快照为准，因为最终 official inference 之后 lab 工作区又发生过变化。

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

必要源码级检查：

```text
phase104j_5090_current_pose_head.py contains Phase104eProtectedAxisAsymmetricExpertHead
phase104j_5090_current_reloc3r_relpose.py registers pairuav_phase104e_paaer_hard_heading_range
phase104j_5090_current_train.py supports --trainable_policy, --milestone_steps, --step_checkpoint_keep_named
phase104j_5090_current_infer_pairuav_with_progress.py writes result.txt/result.zip/manifest.json
```

## Checkpoint 资产

如有需要，请在 git 外发布以下 checkpoint 资产。

| Asset | Role | SHA256 |
|---|---|---|
| `reloc3r_official_metric_full_epoch_checkpoint_last/checkpoint-last.pth` | epoch2/rank1 训练续训所用 source checkpoint | `d08d66fe53fbbcf43ce451e406850c7300a12e5e588879811142d66e67f32c5d` |
| `phase104j_paaer_hr_final/checkpoint-final.pth` | HR/PAAER 最终 official raw source | `7454c0f2b31921cd917c21c3726a7cc1860bef03163c474ed689c19e7af4d01e` |
| `phase104i_tail10_init/checkpoint-final.pth` | phase104j long training 的初始化 checkpoint | `e7e42848b6e18cb31a713b0b95a9c4e7f6e529d7f9c6b2a3e26882a44bbf2f67` |
| `phase89_h8_midlate_final/checkpoint-final.pth` | H8 mid-late 最终 official raw source | `cc985ac7671e7d6dc22928df7eb443ac4f95ad4e295333017e019f6da34216c8` |
| `phase45_epoch2_resume_final/checkpoint-final.pth` | epoch2/rank1 official raw source | `5681ab612d44dc64c98c82ef5b8b4e36c2bb4b38f6748a469f9c8d78c3894e04` |

开源审计中已验证这些 checkpoint hash。5090 上 epoch2 checkpoint 大小约为 1.6G。

## Raw Official Inference

假设 official test 文件位于：

```text
OFFICIAL_TEST_JSON=/path/to/UAVM_2026/pairUAV/test
OFFICIAL_TEST_IMAGES=/path/to/UAVM_2026/pairUAV/test_tour
```

运行 HR/PAAER：

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

期望 HR raw 输出：

```text
rows: 2773116
result.txt SHA256: 6c7da711caac6e6feb91355f7936a820e29f0c3bb19f4affbdfe195f60a507ad
result.zip SHA256: 294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1
```

运行 H8 mid-late：

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

期望 H8 raw 输出：

```text
rows: 2773116
result.txt SHA256: 50ce60ce71465b7ddf3445e98ccca65576729a16e40d28556cd0678e1c213360
result.zip SHA256: b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b
```

运行 epoch2/rank1：

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

期望 epoch2 raw 输出：

```text
rows: 2773116
result.txt SHA256: 956711c0c387d5db5999d3c464267e9eb56c4b036688693ec51538fecc4edeba
result.zip SHA256: 7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee
```

## 训练配方

release 中包含训练脚本，用于 provenance：

```text
phase45_epoch2_launch_full.sh
phase45_epoch2_full.env
phase45_train_pairuav_official_metric_longer_5090.sh
phase45_train_pairuav_full_devsplit.sh
phase104j_tailw_fulltrain1epoch_fromTail10_lr1e-5_bs4_5090.sh
phase104i_HR_tailw_fulltrain10k_fromHR50_lr1e-5_bs4.sh
phase89_mid_late_fulltrain_1epoch_20260618.sh
```

reloc3r epoch2/rank1 run 是续训运行：

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

这些脚本用于说明 checkpoint 的产生方式。若目标是复现榜单提交包，建议使用已发布 checkpoint 资产和上文 official inference 命令；完整重训可能因 GPU、CUDA、PyTorch、dataloader 和文件系统顺序而不具备字节级确定性。

## 交给 B 路径

生成或下载三个 raw zip 后，运行：

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip outputs/hr_official/result.zip \
  --h8-zip outputs/h8_official/result.zip \
  --epoch2-zip outputs/epoch2_official/result.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

期望最终 zip：

```text
3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

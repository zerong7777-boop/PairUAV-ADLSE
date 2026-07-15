# 复现说明

[English](REPRODUCTION.md)

本文档定义 PACE PairUAV release 的可复现路径。推荐用 Path A 字节级重建最终提交包。
只有在需要从 checkpoint 重新生成 official-test raw prediction 时，才使用 Path B。

## 前置准备

1. 克隆本仓库。
2. 创建 Python 环境：

```bash
conda env create -f environment.yml
conda activate pace
```

3. 下载外部资产文件夹：

```text
百度网盘：https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
提取码：djmv
资产文件夹：PairUAV-PACE-release-assets
```

下载后，将其中的 `assets/` 和 `known/` 两个目录复制到仓库根目录。哈希记录见
[../assets/RELEASE_ASSETS.zh-CN.md](../assets/RELEASE_ASSETS.zh-CN.md)。

## Path A：精确重建最终 Zip

这条路径从已发布 raw official-test prediction zips 出发，字节级重建最终提交包。
它对 CPU 友好，不需要 official test images。

所需文件：

```text
assets/phase104j_HRofficial_raw_result.zip
assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip
assets/epoch2_pair_official_test_result_20260527_013814.zip
manifests/devsplit_v1_official_metric_manifest.csv
```

命令：

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

期望最终输出：

```text
outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
SHA256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

脚本会在写出结果前校验输入文件 hash。

## Path B：重新生成 Raw Prediction

这条路径从已发布 checkpoint 和 official PairUAV test 数据出发，重新生成 Path A
所需的三个 raw prediction zip。

所需数据：

```text
official PairUAV test JSON 目录
official PairUAV test 图像目录
```

所需 checkpoint：

```text
assets/phase104j_paaer_hr_final/checkpoint-final.pth
assets/phase89_h8_midlate_final/checkpoint-final.pth
assets/phase45_epoch2_resume_final/checkpoint-final.pth
```

模型级命令、源码快照 hash、checkpoint hash 和期望 raw zip hash 记录在
[A_REPRODUCTION.zh-CN.md](A_REPRODUCTION.zh-CN.md)。

## Path C：训练来源

训练脚本用于审计和继续研究，不保证字节级重建 checkpoint。GPU、CUDA、PyTorch、
dataloader 顺序和文件系统行为都可能改变最终权重。

关键训练入口：

| 目标 checkpoint | 训练脚本 |
|---|---|
| `assets/phase45_epoch2_resume_final/checkpoint-final.pth` | `scripts/training/phase45_epoch2_launch_full.sh` |
| `assets/phase104j_paaer_hr_final/checkpoint-final.pth` | `scripts/training/phase104j_tailw_fulltrain1epoch_fromTail10_lr1e-5_bs4_5090.sh` |
| `assets/phase89_h8_midlate_final/checkpoint-final.pth` | `scripts/training/phase89_mid_late_fulltrain_1epoch_20260618.sh` |
| `assets/phase104i_tail10_init/checkpoint-final.pth` | `scripts/training/phase104i_HR_tailw_fulltrain10k_fromHR50_lr1e-5_bs4.sh` |

榜单提交包复现应使用已发布 checkpoint 资产。

## 验证

推荐检查：

```bash
python -m pytest tests/test_phase104e_paaer_head.py
python -m pytest tests/test_eval_pairuav_manifest_predictions.py
python postprocess/rebuild_final_postprocess.py --help
```

完整历史测试套件包含很多研究路线测试，可能需要可选资产。面向 release 验证，
Path A 的最终 zip hash 一致性是主检查。

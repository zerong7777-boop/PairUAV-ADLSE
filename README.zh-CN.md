# PACE：用于 PairUAV 2026 的极轴条件化估计方法

[English README](README.md) · [复现说明](docs/REPRODUCTION.zh-CN.md) · [合规说明](docs/COMPLIANCE.zh-CN.md) · [方法说明](docs/METHOD.zh-CN.md) · [结果记录](docs/RESULTS.zh-CN.md)

PACE 是一个面向 PairUAV 2026 相对定位任务的可复现 release。给定有序 UAV
视角图像对，系统预测两张图像之间的极坐标关系：heading 和 metric range。

方法出发点很简单：heading 和 range 共享 pose-regime representation，但有效
证据、训练动态和失败模式并不相同。因此 PACE 组合轴条件化学习预测器、确定性
range fusion 和任务感知输出校准，生成最终提交包。

## 结果

归档的最终提交包为：

```text
outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
SHA256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

已知官方榜单反馈：

| Metric | Value |
|---|---:|
| `final_score` | `0.001874` |
| `distance_rel_error` | `0.001330` |
| `angle_rel_error` | `0.002419` |

这些分数来自服务器对 hidden official-test labels 的反馈。仓库可以从公开 raw
prediction 资产和公开 train/dev support manifest 字节级重建最终提交 zip。

## 包含内容

- `reloc3r/`、`croco/` 和根目录入口中的模型、数据集、loss、训练、推理和评测代码。
- `postprocess/` 下的确定性最终提交包重建代码。
- run manifests、公开 support manifest 和资产 hash manifest。
- `scripts/training/` 下的 release checkpoint 训练链脚本。
- 用于说明设计路径的研究诊断脚本和机制图。

本仓库不在 git 中包含 official PairUAV test images、hidden test labels，或大型
checkpoint/raw-prediction 资产。官方数据需要通过比赛允许的渠道获取。

## Release Assets

大 checkpoint 和 raw prediction zip 托管在外部：

```text
百度网盘：https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
提取码：djmv
资产文件夹：PairUAV-PACE-release-assets
```

下载后，将其中的 `assets/` 和 `known/` 两个目录复制到仓库根目录。期望路径和 SHA256 见
[assets/RELEASE_ASSETS.zh-CN.md](assets/RELEASE_ASSETS.zh-CN.md)。

## 快速开始

```bash
conda env create -f environment.yml
conda activate pace

python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

从 checkpoint 重新生成 raw prediction 的模型级复现路径见
[docs/REPRODUCTION.zh-CN.md](docs/REPRODUCTION.zh-CN.md)。

## 方法概览

PACE 使用三个已发布 raw predictor 和两个确定性后处理模块：

| 组件 | 作用 |
|---|---|
| PAAER | 主 protected axis-asymmetric predictor，最终 heading 来源。 |
| MDHR | 多深度 heading-readout 模型，作为互补 distance evidence 保留。 |
| MARB | Metric-aware Reloc3r baseline，作为独立 distance source 保留。 |
| RSF | 对 PAAER、MDHR、MARB 的 distance 做确定性 range stack fusion。 |
| Task-aware output calibration | 使用公开 train/dev support 做 heading lattice snap 和 support-distance snap。 |

最终 range fusion 为：

```text
distance = 0.511 * PAAER_distance
         + 0.189 * MDHR_distance
         + 0.300 * MARB_distance
```

详细方法见 [docs/METHOD.zh-CN.md](docs/METHOD.zh-CN.md)。完整探索逻辑见
[docs/EXPLORATION_LOGIC.zh-CN.md](docs/EXPLORATION_LOGIC.zh-CN.md)。

## 仓库结构

```text
reloc3r/                         模型、head、loss 和 dataset 代码
croco/                           vendored Reloc3r/CroCo 依赖源码
postprocess/                     确定性最终提交包重建脚本
scripts/training/                release checkpoint 训练链
manifests/                       source、inference 和 support manifests
assets/                          外部资产 manifest 和期望路径
figures/                         机制图和定性图
docs/                            方法、复现、合规和证据文档
research/diagnostics/            用于论文证据的归档分析脚本
tests/                           smoke tests 和历史机制测试
phase91/, phase92/               为保持 import 兼容而保留在顶层的归档研究 package
```

根目录入口 `train.py`、`infer_pairuav_with_progress.py` 和 `eval_pairuav.py`
为了兼容已有脚本继续保留在顶层。

## 合规说明

release 最终路径只使用模型预测、已发布 raw assets 和公开 train/dev support
manifest。它不使用 hidden official-test labels，也不做 pair-ID lookup。详见
[docs/COMPLIANCE.zh-CN.md](docs/COMPLIANCE.zh-CN.md)。

## 文档

| 文档 | 用途 |
|---|---|
| [docs/REPRODUCTION.zh-CN.md](docs/REPRODUCTION.zh-CN.md) | 最终提交包重建和模型级复现路径。 |
| [docs/METHOD.zh-CN.md](docs/METHOD.zh-CN.md) | 公开方法命名和实现映射。 |
| [docs/EXPLORATION_LOGIC.zh-CN.md](docs/EXPLORATION_LOGIC.zh-CN.md) | PACE 的路线选择、现象发现和设计逻辑。 |
| [docs/RESULTS.zh-CN.md](docs/RESULTS.zh-CN.md) | 官方反馈和内部开发信号。 |
| [docs/MECHANISM_INSIGHTS.zh-CN.md](docs/MECHANISM_INSIGHTS.zh-CN.md) | 轴非对称证据和机制图。 |
| [docs/SOURCE_ADAPTATION.zh-CN.md](docs/SOURCE_ADAPTATION.zh-CN.md) | 外部几何、匹配、solver 和 split-fusion 证据。 |
| [docs/EXPERIMENT_HISTORY.zh-CN.md](docs/EXPERIMENT_HISTORY.zh-CN.md) | 路线历史和负结果。 |
| [docs/COMPLIANCE.zh-CN.md](docs/COMPLIANCE.zh-CN.md) | 信号来源和 hidden-label 安全说明。 |

## 引用

如果本仓库用于论文或衍生工作，请引用本仓库和对应 PairUAV 2026 任务说明。
仓库引用元数据见 [CITATION.cff](CITATION.cff)。

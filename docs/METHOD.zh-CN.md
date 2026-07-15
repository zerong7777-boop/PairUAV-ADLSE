# 方法说明：PACE

[English](METHOD.md)

本文档说明本仓库采用的公开方法命名。它刻意与复现 manifest 分开：文件名中仍保留内部运行编号，用于资产追踪和 hash 稳定复现；而本文使用面向论文和开源说明的正式名称。

## 系统名称

最终公开方法命名为 **Polar Axis-Conditioned Estimation (PACE)**，中文可写作
**极轴条件化估计方法**。

面向论文和开源说明时，下面会把最终 snapping 步骤表述为下游任务感知输出校准，
而不是把它作为核心研究贡献。

PACE 不是一个单一端到端模型，而是一个面向竞赛提交的系统方案：它把轴解耦的
模型预测器与确定性的任务感知输出校准组合起来。这个边界很重要：学习得到的是
模型预测组件，最终榜单提交包还包含可复现的公开结构校准。

## 问题定义

PairUAV 要求根据 UAV 视角图像对进行相对定位。给定两张 UAV 视角图像，
系统需要预测两者之间的极坐标关系：一个类似 heading 的角度量，以及一个
类似 range 的距离量。最自然的初始表述是联合极坐标回归，即一个模型同时
输出两个连续坐标。

PACE 以这个表述作为起点，但没有把它当作完整任务模型。比赛指标会分别
评估 heading 和 distance 后再组合，并且标注空间本身暴露出无约束二维回归
无法表达的结构。

## 经验现象

开发过程中有三个稳定现象。

第一，heading 和 distance 并不同步失败。有些路线明显改善 heading，却伤害
distance；另一些路线修复 range tail behavior，但 heading 基本不变。这个现象
出现在 axis-objective controls、H-readout sweep、source-adaptation diagnostics
和 tail-robustness runs 中。

第二，两个轴偏好的 evidence 不同。heading 更受益于多深度 decoder evidence、
geometry-derived signals 和角度规律性；distance 更依赖受保护的 metric path，
并且对大 range tail failure 更敏感。

第三，最终预测不应完全视为任意连续值。heading 和 distance 都能从针对
公开 train/dev support 的任务感知校准中获益。

## 设计洞见

PACE 背后的设计视角是：

```text
polar axis-conditioned estimation with protected metric evidence
```

这并不表示 heading 和 distance 是完全独立的 latent factor。更强、也更可靠的
结论是：PairUAV 存在共享 pose-regime representation，但不同坐标具有轴特异
可读性、轴非对称 evidence utility，以及不同的 tail-risk behavior。因此，强系统
应当保护稳定 metric evidence，允许轴特定 readout，并在预测后应用输出校准。

## 方法概览

PACE 将最终预测问题拆成三个实际问题：

1. 哪个学习得到的预测器负责提供 heading？
2. 哪些学习得到的预测器贡献 distance evidence？
3. 最后的公开 support 校准应如何确定性执行？

公开 release 用 PAAER heading、RSF distance 和最终输出校准回答这些问题。
PAAER 和 MDHR 负责轴感知 learned prediction，MARB 提供独立的 metric-aware
distance source，RSF 融合互补 range estimate，最终校准将结果映射到公开
train/dev support 上。

## 系统数据流

最终提交包由固定的推理和后处理图生成：

```text
official PairUAV image pairs
        |
        +--> PAAER raw prediction: heading_HR, distance_HR
        +--> MDHR raw prediction:  heading_H8, distance_H8
        +--> MARB raw prediction:  heading_epoch2, distance_epoch2
                                      |
                                      v
range stack:
  heading_e230  = heading_HR
  distance_e230 = 0.511 * distance_HR
                + 0.189 * distance_H8
                + 0.300 * distance_epoch2
                                      |
                                      v
task-aware output calibration:
  heading_final  = nearest 2-degree lattice value, normalized to [-180, 180)
  distance_final = nearest support distance from the train/dev manifest
```

因此，本仓库支持两层复现：模型级复现从 checkpoint 重新生成三个 raw prediction 文件；提交包级复现从这三个 raw 文件出发，确定性重建最终 `result.zip`。

## 实现映射

| 公开名称 | 代码 / output mode | release 资产 |
|---|---|---|
| PAAER | `Phase104eProtectedAxisAsymmetricExpertHead`; `pairuav_phase104e_paaer_hard_heading_range` | `assets/phase104j_HRofficial_raw_result.zip` |
| MDHR | H8 mid-late readout; `pairuav_range_h0_heading_mid_late_heading_range` | `assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` |
| MARB | metric-aware Reloc3r baseline checkpoint | `assets/epoch2_pair_official_test_result_20260527_013814.zip` |
| RSF + calibration | `postprocess/rebuild_final_postprocess.py` 和 `postprocess/phase104j_final_postprocess_manifest.json` | `outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip` |

## Protected Axis-Asymmetric Expert Readout

**Protected Axis-Asymmetric Expert Readout (PAAER)** 是主预测器。

它的设计目标是在不破坏基础 Reloc3r 风格预测器中稳定 metric 路径的前提下，提高轴特定 readout 的表达能力。因此，PAAER 保留受保护的 anchor path，并引入轴非对称 expert readout。实践中，heading 和 distance 可以使用不同的证据混合，而不是被迫共享同一个标量 head。

公开 checkpoint 使用的是 hard PAAER 变体。它的 range 输出保留在受保护的 H0/C0 metric path 上，而 heading 输出由轴非对称 expert path 产生：

- heading expert 读取固定的 mid 和 late decoder layers；
- heading task token 通过 query-bridge path 访问这组 layer bank；
- heading task feature 被映射为归一化二维 heading vector；
- range scalar 直接来自受保护的 late metric path。

实现中还会记录 heading expert/base angular delta、heading layer attention、token entropy，以及 final range 与 protected range 的绝对差等诊断量。在 hard release path 中，final range contract 有意为零，因为 range prediction 就是 protected range prediction。

最终官方 raw prediction `phase104j_HRofficial_raw_result.zip` 是 PACE 使用的 PAAER raw 输出。对应 checkpoint 资产为：

```text
assets/phase104j_paaer_hr_final/checkpoint-final.pth
```

## Multi-Depth Heading Readout

**Multi-Depth Heading Readout (MDHR)** 是互补的 heading-oriented 预测器。

早期实验显示，heading 可以从多层 decoder evidence 中获益。MDHR 让 distance 路径尽量贴近稳定的 late metric readout，同时让 heading 分支读取多深度 decoder 特征。因此，即使 MDHR 没有作为最终 heading 来源，它仍然能作为互补预测器提供价值。

最终 MDHR 模型是收缩后的 H8 mid-late 变体。它为 range 保留 H0 late readout，但让 heading 分支通过独立 token-to-grid extractor 和 fusion MLP 读取 mid 与 late decoder features。更大的 H8 family 也测试过 early/mid/late readout，但 matched local control 在 full-train 晋级前支持 mid-late 收缩。H-readout sweep 见 [EXPERIMENT_HISTORY.zh-CN.md](EXPERIMENT_HISTORY.zh-CN.md)。

最终官方 raw prediction `phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` 是 PACE 使用的 MDHR raw 输出。

## Metric-Aware Reloc3r Baseline

**Metric-Aware Reloc3r Baseline (MARB)** 是一个较早的 Reloc3r checkpoint，使用 PairUAV metric-aware objective 训练。

MARB 不作为新架构提出。它被保留是因为它提供了与 PAAER、MDHR 误差相关性不同的独立 distance 估计。最终系统中，MARB 只通过确定性 range fusion 贡献结果。

这个组件是刻意保守的：它用一个早于 PAAER/MDHR 后期改造的 Reloc3r-family 模型，为最终 distance stack 提供锚点。把它限定为 distance-only contributor，可以避免把 baseline 包装成新结构贡献，同时利用它互补的 metric signal。

最终官方 raw prediction `epoch2_pair_official_test_result_20260527_013814.zip` 是 PACE 使用的 MARB raw 输出。

## Range Stack Fusion

**Range Stack Fusion (RSF)** 是确定性的 distance fusion 模块。它使用 PAAER heading 作为 heading 来源，并按以下方式组合三个 distance 预测：

```text
distance = 0.511 * PAAER_distance
         + 0.189 * MDHR_distance
         + 0.300 * MARB_distance
```

权重来自开发过程和提交反馈。RSF 应理解为竞赛系统组件，而不是一个通用融合定律。

RSF 不修改 heading。它会把 PAAER heading 原样传到后续任务感知校准步骤。因此，中间的 e230 包主要改善 distance，同时保持 PAAER angle score 不变。

## Task-Aware Output Calibration

release 代码中历史上把这个最终确定性校准步骤命名为 **Legal-State Snapping (LSS)**。面向论文和开源说明时，更准确的表述是任务感知输出校准。

最终 heading 会 snap 到最近的 2 度 lattice 值。最终 distance 会 snap 到已发布 train/dev manifest 中观测到的最近 support distance：

```text
heading  -> nearest 2-degree lattice value
distance -> nearest support distance from gt_distance in train/dev
```

LSS 实现只使用模型预测和已发布的 train/dev split manifest，不使用隐藏 official-test label。本仓库中 support manifest 的路径为：

```text
manifests/devsplit_v1_official_metric_manifest.csv
```

最终包使用的 support manifest 包含 `204120` 个 train/dev rows 和 `211` 个 unique support distances。最终 official test prediction 包含 `2773116` 行。因此，snap 是投影到公开 train/dev support set 的固定最近邻操作，不是用 hidden official-test label 训练出来的 selector。

## PACE 的主张边界

PACE 应被理解为一个可复现的竞赛系统方案，其设计主张包括：

- heading 和 distance 受益于 axis-aware readout design；
- 多深度 decoder evidence 对 heading-oriented prediction 有帮助；
- 独立 distance predictor 的组合可以降低 range error；
- 公开 output support 可以在不使用隐藏 test label 的前提下确定性应用；
- 最终提交包可以从公开 raw prediction 和公开 support manifest 字节级重建。

PACE 不主张最终榜单分数来自单个端到端模型，也不主张后处理可以替代更强的预测器学习。

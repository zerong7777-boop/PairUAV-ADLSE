# 合规与信号使用说明

[English](COMPLIANCE.md)

本文档说明 PACE 最终 release 包的信号边界，目的是让挑战赛后处理路径可审计。

## 允许输入

release 最终包由以下内容生成：

- 三个已发布 raw official-test prediction zip；
- `postprocess/` 中的固定确定性后处理代码；
- 公开 train/dev support manifest：
  `manifests/devsplit_v1_official_metric_manifest.csv`；
- manifest 和源码中记录的固定融合权重与 snapping 规则。

support manifest 只包含 train/dev labels。它用于构造有限的公开 support set，供输出
校准使用。

## 未使用内容

release 路径不使用：

- hidden official-test labels；
- hidden official-test errors 的人工检查；
- 通过 pair ID lookup 直接指定 distance 或 heading；
- test-set graph optimization、clustering 或 cross-row label propagation；
- inference 阶段基于 official-test leaderboard feedback 的自适应选择。

官方榜单反馈只用于记录已提交包的分数和说明最终阶段决策，不是 release rebuild
脚本的输入。

## 多模型使用

最终包组合 PAAER、MDHR 和 MARB 的 raw predictions。这是对已发布模型输出做的固定
确定性 ensemble：

```text
distance = 0.511 * PAAER_distance
         + 0.189 * MDHR_distance
         + 0.300 * MARB_distance
heading  = PAAER_heading
```

ensemble 是挑战赛系统组件。主要方法主张仍然是 axis-conditioned polar estimation，
不是把多模型集成本身作为核心科学贡献。

## 输出校准

代码历史上把最终校准步骤称为 Legal-State Snapping (LSS)。面向论文表述时，更准确
的说法是 task-aware output calibration：

```text
heading  -> nearest 2-degree lattice value
distance -> nearest support distance observed in train/dev
```

这个投影是确定性的，只使用公开 train/dev support，不使用 hidden official-test labels
训练。

## 复现边界

当已发布 raw zips 和 support manifest 存在时，最终 `result.zip` 可以通过
`postprocess/rebuild_final_postprocess.py` 字节级重建。完整重训路径用于说明
provenance，但不保证得到 bit-identical checkpoints。

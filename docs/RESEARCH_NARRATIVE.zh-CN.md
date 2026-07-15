# 研究叙事

[English](RESEARCH_NARRATIVE.md)

本文说明最终方法是如何形成的。它不是逐条运行日志，而是面向论文和开源读者，对“从任务诊断到最终可复现提交系统”的设计历程重构。

## 起点

PairUAV 要求预测极坐标相对定位结果：一个类似 heading 的角度输出，以及一个类似 range 的距离输出。第一个强基线是使用官方 PairUAV objective 训练的 metric-aware Reloc3r 模型。本仓库将它作为 **Metric-Aware Reloc3r Baseline (MARB)**。它的 official hidden-test 反馈为：

| 系统 | final_score | distance_rel_error | angle_rel_error |
|---|---:|---:|---:|
| MARB / Reloc3r full-data one-epoch run | 0.003188 | 0.002528 | 0.003849 |

这个基线有价值，但也暴露了核心限制：最好的 distance 行为和最好的 heading 行为不总是来自同一条 evidence path。

## 机制诊断

早期机制实验显示，PairUAV 难度不是一个单一标量的“hardness”。冻结 observability proxy 在不同 regime 下表现出不同的 angle/range 行为。control 和 intermediate split 中，heading 比 distance 更难；stress split 中，distance 反而明显更难。

后续 B811 comparable-surface 诊断给出了更清晰的机制信号：几何来源 evidence 通常有利于 heading，但有害于 distance。在同一个 811 行验证 surface 上，geometry 对 heading 有帮助的样本为 608 行，对 distance 有害的样本为 665 行。主导 pair regime 是 `heading_helpful_distance_harmful`，共有 492 行。

因此研究问题从“如何训练一个更好的共享标量 head”转向“如何允许两个极坐标轴使用不同 evidence，同时不破坏稳定 metric path”。

## 证据到设计的映射

最终方法可以理解为一组由证据驱动的设计响应，而不是只为了榜单反馈临时拼接出的 late-stage ensemble：

| 观察 | 支撑证据 | 设计响应 |
|---|---|---|
| Heading 和 distance 的错误具有异质性。 | H1/R1/HR 控制实验、H-readout sweep 和 B811 source diagnostics 都显示，一个轴改善时另一个轴可能退化。 | 使用 axis-aware readout，而不是完全共享的标量 head。 |
| 几何类 evidence 倾向于帮助 heading，但对 distance 有风险。 | B811 geometry comparison 和 split-fusion probes 显示主导现象是 heading-helpful / distance-harmful。 | 保护 metric distance path，同时允许 heading 使用更丰富 evidence。 |
| 多深度 decoder evidence 对 heading 有帮助。 | H8 mid-late 在 matched local controls 中优于 late-only 和 full early/mid/late 变体。 | 将 H8 mid-late 提升为 MDHR。 |
| PAAER full-loss scaling 暴露出 range-tail failure。 | HR50 的 median range error 接近 H8，但 p95/p99/max tail 明显更差。 | 在 official PAAER inference 前加入 tail-weighted continuation。 |
| 最终预测受益于公开 output-support 校准。 | 官方反馈从 RSF e230 到 heading lattice snapping，再到 distance support snapping 连续改善。 | 在 learned prediction 和 range fusion 之后执行确定性任务感知校准。 |

## 从轴拆分到模型设计

第一个面向论文的设计结论是 axis decoupling。验证-only 的 ADPA 系列实验显示，把 geometry heading 与 base distance 组合，可以保留 geometry heading 的收益，同时阻止 geometry distance 的损害。修复后的 ADPA-1 bounded run 结果为：

| variant | heading MAE | distance MAE |
|---|---:|---:|
| base only | 8.6087 | 0.9876 |
| geometry only | 1.7769 | 6.9755 |
| axis-decoupled | 1.7769 | 0.9876 |

这还不是可部署模型，因为总是使用 geometry heading 仍会伤害 base-sufficient cases。但它确立了正确的架构方向：保护稳定 distance path，同时给 heading 更丰富的 readout。

这一路线最终形成了 release 中的两个模型族：

- **Multi-Depth Heading Readout (MDHR)：** heading 读取多深度 decoder evidence，distance 保持接近稳定 late metric readout。
- **Protected Axis-Asymmetric Expert Readout (PAAER)：** 主模型保留受保护的 metric anchor，并加入轴非对称 expert readout。

## 为什么需要尾部鲁棒性

PAAER full-loss 全参数训练改善了本地 balanced proxy，但 50k fulltrain continuation 暴露出高绝对距离尾部失败：median range error 接近 H8，但 p95/p99/max 明显更差。因此优化目标从“继续训练更久”变成了“保护 range tail”。

tail-weighted continuation 在同一个 val811 surface 上修复了这个失败模式：

| run | angle MAE | range MAE | proxy |
|---|---:|---:|---:|
| HR50 fulltrain | 0.7791 | 2.2237 | 0.006376 |
| H8 mid-late step50k | 0.9048 | 0.4744 | 0.003412 |
| PAAER Tail10 | 0.7647 | 0.4278 | 0.002934 |

这个结果推动我们使用 PAAER/Tail 线做最终 official inference，同时保留 MDHR 和 MARB 作为互补预测器。

## 最终系统设计

最终提交系统 **PACE: Polar Axis-Conditioned Estimation（极轴条件化估计）**
把模型设计经验和确定性的竞赛系统校准结合起来。

1. PAAER 提供最终 heading 来源。
2. Range Stack Fusion (RSF) 组合 PAAER、MDHR 和 MARB 的 distance。
3. 任务感知输出校准将 heading 和 distance snap 到已发布 train/dev split 中观测到的 support 结构；脚本和 manifest 中仍保留历史实现名 LSS。

这就是为什么 PACE 被描述为一个系统，而不是一个单一神经网络架构。学习部分的贡献是 axis-aware prediction；最终榜单包还使用了确定性的公开结构校准。

## 主张边界

目前支持的主张不是“heading 和 distance 位于完全分离的 latent space”。Phase95 subspace-overlap audit 否定了这种简单解释。更准确、证据更充分的说法是：

```text
PairUAV 存在共享 pose-regime representation，其中不同坐标具有轴特异可读性，
并呈现轴非对称的误差与效用形态。
```

这个结论足以支持 protected axis-asymmetric readout、互补 range fusion 和下游输出校准，但不会把机制夸大成硬性的 latent factor independence。

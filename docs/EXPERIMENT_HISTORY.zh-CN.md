# 实验历程与内部结果

[English](EXPERIMENT_HISTORY.md)

本文档记录影响最终 release 的主要路线。它不是完整实验台账，而是经过筛选的研究历史，用于说明方法设计为什么走到现在：尝试过什么、什么有效、什么失败，以及哪些主张只属于内部验证信号。

## 证据层级

| 层级 | 含义 | 例子 |
|---|---|---|
| 官方反馈 | 比赛服务器返回的 hidden-test 分数 | MARB、MDHR、PAAER raw、RSF/LSS final |
| 本地验证 | 固定本地 split 或 val811 proxy，有标签可用 | Phase104g/h/i、Phase95 表征审计 |
| 机制验证 | validation-only 诊断 surface，不是可部署 policy | B811 geometry utility、ADPA experiments |
| 负结果 / blocker | 被否定、被阻塞或未晋级的路线 | checkpoint averaging、global calibration、selector attempts |

只有第一层是官方榜单证据。其它层用于解释设计决策和面向论文的机制主张。

## 路线总览

| 路线 | 设计想法 | 关键证据 | 结论 |
|---|---|---|---|
| MARB / Reloc3r metric-aware baseline | 使用 PairUAV official metric-aware objective 训练 Reloc3r 风格 heading/range predictor。 | Official hidden-test `final_score=0.003188`, `distance=0.002528`, `angle=0.003849`。 | 保留为基线和最终 distance diversity 来源。 |
| External geometry / split-fusion probes | 使用几何重的来源提升极坐标预测。 | RoMa/MASt3R/DUSt3R/VGGT probes 和 B811 诊断显示 geometry/source utility 具有强轴依赖性。 | 晋级为机制证据，不作为最终 standalone predictor。 |
| ADPA axis-decoupled composition | 组合 geometry-assisted heading 与 base distance。 | ADPA-1 repaired surface：base heading `8.6087`，geometry heading `1.7769`，axis-decoupled heading `1.7769`；distance 保持 base `0.9876`。 | 支持 validation surface 上的 axis-decoupling；因 control/selectivity 问题未部署。 |
| MDHR / H8 mid-late | 让 heading 读取多深度 decoder evidence，同时保护稳定 range path。 | Official hidden-test `final_score=0.00246`，相对 MARB 提升约 22.8%。 | 保留为强 official 模型和互补 range 来源。 |
| PAAER 2.5k controls | 区分开放参数效应和 axis-asymmetric readout 效应。 | HR all-2500 本地 proxy `0.011680`，优于 C0、H8 full、H8 mid-late fair controls。 | 将 PAAER 晋级为主 full-loss architecture candidate。 |
| PAAER HR50 fulltrain | 将 PAAER full-loss training 扩大到更接近榜单的规模。 | Median range error 接近 H8，但 p95/p99/max tail errors 明显更差。 | 否定盲目长训；诊断为 range-tail robustness failure。 |
| PAAER Tail10 | 从 HR50 使用 tail-weighted range loss 继续训练。 | 本地 proxy 从 HR50 `0.006376` 改善到 Tail10 `0.002934`，优于 H8 step50k `0.003412`。 | 作为最终 PAAER official inference 的来源路线。 |
| Phase95 checkpoint averaging | 对 H8 中间 checkpoint 或 MNR-like 变体做平均。 | 没有 OOF-stable 的 H8 final 增益；最好差异接近数值噪声。 | 不晋级 official。 |
| Phase95 low-degree calibration | 应用全局 range bias/scale 和 heading offset 修正。 | best OOF method 反而使 H8 final proxy 变差。 | 不晋级 official。 |
| MTL-GA / PRC-GFC / LG-PHF / FR-JEPA | 探索非 naive 的多目标、关系耦合、patch hypothesis 和 latent prediction 路线。 | 已实现并通过 formal lab route 验证；晋级要求在 Tail10/B4 anchor 上通过 matched gates。 | 保留为研究探索，不属于最终提交系统。 |
| RSF + task-aware calibration | 融合互补 distance 来源，并 snap 到公开任务 support structure。 | 官方反馈从 RSF e230 `0.002413` 改善到最终 PACE `0.001874`。 | 最终挑战赛系统包。 |

## 证据到设计的审计表

下表把上面的精选结果和公开设计主张对应起来。它是 audit map，不是新增实验表。

| 设计主张 | 支撑结果族 | 对应方法选择 |
|---|---|---|
| Heading 和 distance 不应被迫使用完全相同的 evidence path。 | ADPA、H-readout sweep、source-adaptation 和 split-fusion diagnostics。 | Axis-aware readout 和 protected metric paths。 |
| Heading 受益于多深度 decoder evidence。 | H8 mid-late matched controls 和后续官方 MDHR 反馈。 | MDHR 保留为强 official predictor 和 range-diversity source。 |
| PAAER 有价值，但需要 range-tail protection。 | HR all-2500 control、HR50 tail quantiles、Tail10 improvement。 | PAAER/Tail 线用于最终 official inference。 |
| Distance 受益于互补 predictor。 | MARB、MDHR 和 PAAER 呈现不同 distance error profiles。 | RSF 使用 PAAER、MDHR 和 MARB distances。 |
| 合法输出结构是可部署的后处理信号。 | 官方 RSF e230、heading lattice snap 和最终 support snap feedback。 | LSS 作为确定性 final calibration。 |

## Source Adaptation 与 Split-Fusion 证据

外部 source 实验用于测试几何、dense matching 和 solver-style prior 是否能改善 PairUAV 预测。完整盘点见 [SOURCE_ADAPTATION.zh-CN.md](SOURCE_ADAPTATION.zh-CN.md)。高层历史如下：

| family | strongest signal | limitation |
|---|---|---|
| Dense matching sources | RoMa dense 在 same-contract 2048 replay 上达到 `15.3929 / 19.4702`。 | 仍是内部 proxy，不是 hidden-test system。 |
| Geometry-foundation sources | MASt3R 和 DUSt3R geometry fields 给出强 heading signal：`9.5863 / 33.2327` 和 `11.0434 / 33.1489`。 | distance 较弱。 |
| VGGT-style range source | VGGT geometry field heading 弱但 distance 较强：`52.5120 / 14.1716`。 | direct camera extraction 不具备竞争力。 |
| Three-source hybrid | VGGT 和 MASt3R+VGGT 是强 proxy/range candidate；full RoMa+MASt3R+VGGT 的 angle 最好。 | naive concatenation 没有稳定改善 balanced proxy。 |
| Split-fusion | 两分支 old-field heading + rich-native range 在 2048 diagnostic protocol 上达到 `1.330926 / 4.024406 / 0.025360`。 | full official inference 成本和 no-op/selectivity 问题阻止最终晋级。 |
| Solver/frozen-pose routes | MADPose、FAR、SAC-Pose 和 GRelPose 都能技术运行。 | bounded PairUAV 指标不具备竞争力。 |

重要设计教训不是某个 external source 单独解决了 PairUAV，而是 source 按轴专门化。这直接推动了 protected metric path、axis-specific readout，以及最终 range/heading 后处理策略。

## H-Readout 架构扫描

H-index 名称是内部 Reloc3r-family readout 编号，不是公开方法名。这里保留这批结果，是因为 MDHR 就是从这条 sweep 里收敛出来的。除非表格明确说明，下面所有结果都是本地 `val811` proxy 记录。

完整性和可比性说明：

- H0-H5/H8 是架构/readout 标签。早期笔记里的 H6/H7 指的是更宏观的研究假设，不是已完成的 readout 实现，因此这里不声明 H6/H7 数值结果。
- H4/H5 是 topology-protected 设计，不是 capability-preserved 设计：共享 encoder/decoder/base features 仍然会同时接收 heading 和 range loss 的梯度。
- 只应在同一 phase 表内横向比较。Phase83、Phase87、Phase88 的训练 surface 或初始化选择不同。

内部 taxonomy：

| 标签 | 实现意图 | 已恢复指标状态 |
|---|---|---|
| H0 | 共享 PairUAV heading/range head | 有完整本地结果 |
| H1 | late residual/adapter 轴解耦 | 有完整本地结果 |
| H2 | mid-level split | 有本地结果 |
| H3 | early split | 有完整本地结果 |
| H4 | range-H0 加 heading-H2 | 设计标签；未恢复完整可比指标行 |
| H5 | range-H0 加 heading-H3 | 有完整本地结果 |
| H8 | range-H0 加 heading 多深度 decoder readout | 有完整本地结果；后续收缩为 MDHR mid-late 变体 |

Phase83 在固定本地验证 surface 上测试了早期轴解耦：

| setting | steps | angle MAE | range MAE | proxy | note |
|---|---:|---:|---:|---:|---|
| Wtrunk/T2/H0 | 2500 | 4.027444 | 4.010942 | 0.018784 | clean aligned baseline |
| Wtrunk/T2/H0 | 5000 | 3.759863 | 3.306784 | 0.016707 | H0 continuation |
| Wtrunk/T2/H1 | 2500 | 2.333164 | 3.960756 | 0.013982 | late axis decoupling helps Wtrunk |
| Wtrunk/T2/H1 | 5000 | 2.454658 | 3.323074 | 0.013112 | still better than H0 |
| Wtrunk/T2/H1 | 7500 | 2.527736 | 2.790420 | 0.012306 | proxy improves through range |
| Wstrip/T2/H0 | 2500 | 3.141467 | 3.293436 | 0.014964 | unexpectedly strong |
| Wstrip/T2/H0 | 5000 | 2.677615 | 1.776450 | 0.010802 | scales well |
| Wstrip/T2/H0 | 7500 | 2.566347 | 1.431668 | 0.009840 | best Phase83 scaled route |
| Wstrip/T2/H3 | 2500 | 1.889435 | 3.854808 | 0.012549 | strong heading, weaker range |
| Wstrip/T2/H2 | 2500 | 2.277588 | 3.519935 | 0.012993 | also heading-favorable |

Phase87 在 Reloc3r-512 backbone-only 初始化和 full-model finetune 设置下测试了 H0/H3/H5/H8 的 scale line：

| setting | steps | angle MAE | range MAE | proxy |
|---|---:|---:|---:|---:|
| H0 | 2500 | 7.764222 | 6.604099 | 0.034075 |
| H0 | 10000 | 7.519871 | 2.239851 | 0.025131 |
| H3 | 2500 | 7.474898 | 4.058125 | 0.028449 |
| H3AO | 2500 | 1.667425 | 67.782616 | 0.133008 |
| H5 | 2500 | 3.511572 | 4.315968 | 0.017929 |
| H5 | 5000 | 3.271602 | 5.852879 | 0.020173 |
| H5 | 10000 | 6.061516 | 1.945850 | 0.020523 |
| H8 | 2500 | 3.970750 | 5.872404 | 0.022152 |
| H8 | 5000 | 3.915341 | 2.539262 | 0.015685 |
| H8 | 10000 | 3.013667 | 1.281393 | 0.010798 |
| H8 | 100000 | 1.125074 | 1.144259 | 0.005292 |

Phase87 说明 H8 并不是每个 2500-step 短训矩阵里的最优结构，但它是第一条出现明确 scale-up 行为的 H-family 路线。

Phase88 进一步隔离了 H8 readout-depth 问题：

| setting | steps | angle MAE | range MAE | proxy |
|---|---:|---:|---:|---:|
| late-only | 2500 | 2.320584 | 4.892704 | 0.015713 |
| mid-late | 2500 | 1.902788 | 4.297642 | 0.013425 |
| mid-late | 5000 | 1.229486 | 3.664675 | 0.010356 |
| mid-late | 10000 | 1.211283 | 3.432744 | 0.009866 |
| mid-only | 2500 | 1.996787 | 4.257079 | 0.013609 |
| early-late | 2500 | 1.759591 | 4.863244 | 0.014098 |
| early-mid | 2500 | 1.836391 | 5.837262 | 0.016157 |
| full H8 same runner | 2500 | 1.794715 | 5.167480 | 0.014772 |
| decoder-last1-H0 | 2500 | 42.604271 | 5.161415 | 0.128121 |
| decoder-last1-H8 | 2500 | 4.834469 | 4.314959 | 0.021601 |

在 matched 10k same-surface control 上，收缩后的 H8 mid-late readout 也优于完整 early/mid/late H8：

| setting | max-step label | actual train loop | angle MAE | range MAE | proxy |
|---|---:|---|---:|---:|---:|
| H8 mid-late | 10000 | 1 epoch / 5000 iters | 1.211282730 | 3.432743549 | 0.009866082 |
| H8 full | 10000 | 1 epoch / 5000 iters | 1.280849576 | 3.663944244 | 0.010497204 |

这是 MDHR mid-late 设计在后续 full-train 和 official hidden-test 之前被晋级的直接本地证据。

## 内部 Axis-Objective 控制实验

Phase104g 测试了三种全参数训练目标：heading-only H1、range-only R1，以及完整 heading+range HR。这些不是 official submission，而是受控本地 probe。

| run | angle MAE | range MAE | proxy | interpretation |
|---|---:|---:|---:|---|
| H1 heading-only all-2500 | 1.5958 | 53.6993 | 0.106136 | heading 改善，但 range 崩溃 |
| R1 range-only all-2500 | 122.2311 | 3.6355 | 0.346416 | range 改善，但 heading 崩溃 |
| HR full-loss all-2500 | 1.4985 | 3.9690 | 0.011680 | 该规模下最好的 balanced PAAER control |
| H8 mid-late fair 2500 | 1.8194 | 5.2569 | 0.015010 | 匹配规模的 H8-style fair control |

这个实验重要在于：简单开放全部参数并不够；轴特定目标可能产生严重 negative transfer，而 balanced PAAER training 可以改善 joint proxy。

## Tail Robustness 诊断

Phase104h 解释了为什么 HR50 没有直接压过 H8 mid-late：

| run | range median | p95 | p99 | max |
|---|---:|---:|---:|---:|
| HR50 | 0.4546 | 7.1223 | 42.9710 | 91.2856 |
| H8 step50k | 0.3897 | 1.1979 | 2.1321 | 6.9968 |
| Tail10 | 0.3600 | 1.0446 | 1.2653 | 4.3494 |

问题不是全局标量校准，而是 high-absolute-range tail robustness。tail-weighted continuation 直接针对这个失败模式。

## 官方最终阶段分数

| package | components | final_score | distance_rel_error | angle_rel_error |
|---|---|---:|---:|---:|
| MARB raw | Reloc3r metric-aware baseline | 0.003188 | 0.002528 | 0.003849 |
| MDHR raw | H8 mid-late | 0.002460 | 0.002274 | 0.002646 |
| PAAER raw | PAAER heading + PAAER distance | 0.002514 | 0.002392 | 0.002636 |
| RSF e230 | PAAER heading + fused range | 0.002413 | 0.002191 | 0.002636 |
| RSF e230 + heading lattice | RSF + heading lattice snapping | 0.002308 | 0.002150 | 0.002465 |
| PACE final | RSF + heading lattice + support-distance snapping | 0.001874 | 0.001330 | 0.002419 |

这些官方数值也汇总在 [RESULTS.zh-CN.md](RESULTS.zh-CN.md) 中。

## 没有主张什么

若干探索路线提供了有用信息，但不是最终方法主张的一部分：

- checkpoint averaging 和 global calibration 没有产生 OOF-stable gain；
- feature-semantics selector 找到很小的高精度 slice，但 recall 不足以支持可部署 routing；
- Phase95 subspace overlap 不支持硬性的 non-overlapping latent subspace 叙事；
- LG-PHF/FR-JEPA 类路线在截止前没有完成 required promotion path，因此未进入最终提交。

公开 release 因此将这些内容作为设计历史和负结果保留，而可复现最终包仍然是 PACE。

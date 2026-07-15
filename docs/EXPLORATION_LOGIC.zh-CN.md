# 探索逻辑：从 Baseline 路线到 PACE

[English](EXPLORATION_LOGIC.md)

本文档记录 **PACE: Polar Axis-Conditioned Estimation（极轴条件化估计）**
背后的论文视角探索逻辑。它不是完整实验流水账，而是解释为什么最终仓库围绕
轴条件化极坐标估计组织，为什么最终包仍包含挑战赛后处理，以及负结果如何限定
方法主张边界。

`phase104j` 等内部运行名会继续保留在部分文件名中用于复现。面向论文和公开说明时，
本仓库采用的正式方法名是 PACE。

## 1. 从路线选择出发

PairUAV 可以写成极坐标相对定位任务：给定 UAV 视角图像对，预测 heading 和
metric range。最初我们不是只改一个 head，而是把它当作路线选择问题。

| 路线族 | 核心问题 | 结果 |
|---|---|---|
| End-to-end RPR / Reloc3r family | Reloc3r 风格的图像对回归器能否直接学习官方 heading/range objective？ | 成为最可靠的宿主路线：支持完整 official inference，比纯几何路线更能保持 metric range，并有正向 hidden-test 反馈。 |
| Baseline-style field regressor / split-fusion family | 能否把 dense geometry、matching 或 source field 转成 task-specific heading/range regressor？ | 提供了很强的机制证据，尤其是轴非对称和 source specialization，但没有作为最终 official route 晋级。 |
| Geometry / solver / matching-aware route | 显式 correspondence、solver 或 pose pipeline 能否直接解决任务？ | 技术上可运行，但 bounded PairUAV metric 不够竞争，或完整 official inference 成本过高。 |
| Candidate-pool new routes | 多目标、relational、patch-hypothesis、JEPA 风格模块能否形成更强后期候选？ | 作为正式研究路线实现过，但在比赛截止前没有完成晋级到最终提交所需的验证链。 |

这里的结论不是“端到端回归在所有问题上都更好”，而是更窄的经验判断：在 PairUAV
比赛约束下，直接的 Reloc3r-family RPR 宿主最适合做完整训练和 hidden-test
推理；source/geometry 路线则更适合作为机制探针。

## 2. 重新审视共享 Readout 假设

DUSt3R、MASt3R、Reloc3r 风格系统自然会构造共享 pair representation，并常用
共享或高度耦合的 readout head 做 pose 或 geometry prediction。对于重建类目标，
这是合理设计；但 PairUAV 的目标结构不同：heading 和 range 分别评估，而且会对
同一 evidence source 产生不同反应。

如果直接搬用共享极坐标 readout，就容易把 heading 和 distance 当作同一个标量
回归目标的两个可互换坐标。于是问题从“把共享 head 训练得更强”转向：

```text
heading 和 range 应该在哪里交互、以什么形式交互；
又应该在哪里允许 readout 分化，同时保留稳定共享 pose-regime representation？
```

## 3. 探索交互位置与交互形式

下一阶段围绕“两个轴在哪里共享、在哪里保护或专门化”展开。

| 探索线 | 在逻辑链中的作用 |
|---|---|
| H0-H8 / MDHR readout sweep | 测试 late、mid、early、多深度 heading/range readout。H8 mid-late 收缩版最终成为 **Multi-Depth Heading Readout (MDHR)**。 |
| H1 / R1 / HR controls | 在同样 2.5k 规模下测试 heading-only、range-only、full heading+range objective。结果显示单轴 objective 会造成严重 negative transfer。 |
| PAAER | 引入受保护 metric path 和轴非对称 expert readout，成为最终包里的主学习预测器。 |
| Tail continuation | 把 PAAER 从一个本地 balanced proxy 较强的候选，修正为更关注 range tail 的 official candidate。 |
| OFFER / PRM / router-style attempts | 探索更丰富的动态路由和 prediction refinement，但没有进入最终 release 主张。 |

这里的设计结论不是“把两个轴硬切开”。Phase95 表征审计否定了简单的 hard-subspace
故事。更稳妥的结论是：

```text
shared pose-regime representation + coordinate-specific readability +
protected metric evidence
```

## 4. 形成 PACE 的关键现象

四个现象最重要，因为它们解释了 PACE 不是简单改名的 ensemble。

### 轴条件化 Evidence Utility

B811 source diagnostics 显示，几何类 evidence 通常有利于 heading，但对 range
不安全。直接 pairwise comparison 中，geometry heading 更好的样本为 `694/811`，
geometry distance 更差的样本为 `758/811`。机制图使用的更严格分箱也给出同一方向：
geometry 对 heading helpful 的样本为 `608/811`，对 distance harmful 的样本为
`665/811`，其中 `492` 行落在 `heading_helpful_distance_harmful` regime。

这个现象直接支持 protected metric path 和 axis-specific readout。

### 训练时间上的轴向能力不同步

在 checkpoint sweep 和最终阶段反馈中，heading 与 range 并不总在同一个训练状态
达到最好。最有用的 distance checkpoint 往往更早，或者来自不同模型线。这解释了
为什么最终系统没有简单使用一个 last checkpoint 同时输出两个轴：PAAER 提供最终
heading，而 PAAER、MDHR、MARB 都通过 Range Stack Fusion 贡献 range evidence。

### Prediction Trajectory Instability

如果某个样本在 checkpoint 轨迹中 prediction 摆动很大，它更可能成为 final
top-error 或 high-headroom 样本。因此后期模型选择不能只看单个 aggregate validation
number，还要看 trajectory behavior 和 tail diagnostics。这也解释了为什么 PAAER
HR50 没有盲目继续训练：它的 median range behavior 有竞争力，但高绝对距离尾部不安全。

### 共享 Pose-Regime Representation 与 Readout 协同适配

表征 probe 显示 H8 feature 编码了 pose-regime 信息，包括 heading bin、signed/absolute
range bucket。但 subspace-overlap audit 不支持 heading/range 处于非重叠 latent
space 的强说法。因此更准确的框架是共享表征加轴条件化 readout，而不是 hard factor
disentanglement。

## 5. PACE 的设计原则

PACE 可以概括为：

```text
Polar Axis-Conditioned Estimation:
学习共享图像对表征，保护稳定 metric evidence，并让 heading/range 使用
轴条件化 readout 与输出校准。
```

方法层面包含三个学习预测器角色：

| 组件 | 公开角色 |
|---|---|
| PAAER | 主 protected axis-asymmetric predictor；最终 heading 来源。 |
| MDHR | 多深度 heading-readout 模型，作为互补 range/axis evidence 保留。 |
| MARB | Metric-aware Reloc3r baseline，作为独立 distance source 保留。 |

最终挑战赛包还包含两个确定性系统组件：

| 组件 | 作用 |
|---|---|
| RSF | 融合 PAAER、MDHR、MARB 的 distance prediction。 |
| Task-aware output calibration | 将 heading 和 distance snap 到公开 train/dev support structure；代码中历史实现名为 LSS。 |

## 6. 为什么最终包包含后处理

最终提交面向挑战赛，因此包含后处理，需要如实描述。Range Stack Fusion 带来有限但
有效的 official 改善：PAAER raw `0.002514` 提升到 RSF e230 `0.002413`，主要来自
distance error。更大的最终提升来自任务感知输出校准：RSF e230 加 heading lattice
达到 `0.002308`，最终 support-distance calibrated package 达到 `0.001874`。

这部分后处理不是本文希望拔高的核心科学贡献，而是可复现的挑战赛系统组件。主要方法
洞见仍然是 axis-conditioned polar estimation：heading 和 range 共享表征，但它们的
evidence utility、training dynamics 和 tail risk 足够不同，因此强系统应当非对称地
建模它们。

## 7. 主张边界

PACE 主张：

- PairUAV 极坐标回归受益于 axis-conditioned readout，而不是完全同质的共享标量 head；
- geometry/source evidence 有价值，但具有明显轴非对称性；
- protected metric path 对 range stability 很重要；
- 互补 range predictor 和 public-support calibration 可以在确定、可复现的前提下改善挑战赛提交。

PACE 不主张：

- 使用 hidden official-test label；
- pair ID 可以直接决定 distance；
- heading 和 range 是完全独立 latent factor；
- 最终榜单分数来自一个没有后处理的单一端到端模型；
- source-adaptation 或 geometry-only 路线就是最终提交系统。

## 8. 阅读路径

实现细节见 [METHOD.zh-CN.md](METHOD.zh-CN.md)。官方反馈和提交包结果见
[RESULTS.zh-CN.md](RESULTS.zh-CN.md)。详细路线表见
[EXPERIMENT_HISTORY.zh-CN.md](EXPERIMENT_HISTORY.zh-CN.md)。source adaptation 和
split-fusion 证据见 [SOURCE_ADAPTATION.zh-CN.md](SOURCE_ADAPTATION.zh-CN.md)。
机制图和表征证据见 [MECHANISM_INSIGHTS.zh-CN.md](MECHANISM_INSIGHTS.zh-CN.md)。

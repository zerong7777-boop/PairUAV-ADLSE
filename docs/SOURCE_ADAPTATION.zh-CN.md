# Source Adaptation 与 Split-Fusion 证据

[English](SOURCE_ADAPTATION.md)

本文记录影响最终 PairUAV 设计的 source adaptation 证据，包括外部几何模型、匹配模型、solver 路线，以及结构化 split-fusion 路线。除非明确标注为官方反馈，否则这些结果都是设计证据，不是 official leaderboard claim。

生成的总结图：

```text
figures/fig_source_adaptation_summary.svg
```

生成脚本和数据：

```text
figures/gen_fig_source_adaptation_summary.py
figures/source_adaptation_summary_data.csv
```

## 证据口径

本文大多数结果是 bounded internal probe。它们使用本地有标签 split、2048-row 诊断子集、B811 comparable surface 或 proxy metric。因此不能和 hidden-test leaderboard 分数直接比较。

这些结果可以支持三类主张：

1. 外部几何和匹配 source 可以真实接入 PairUAV。
2. Heading 和 distance 对这些 source 的使用方式不同。
3. naive source concatenation 或 frozen-feature transfer 不足以解决问题。

不要用这些结果主张某个 standalone source-adaptation 模型就是最终提交系统。最终提交系统是 PACE，见 [METHOD.zh-CN.md](METHOD.zh-CN.md) 和 [RESULTS.zh-CN.md](RESULTS.zh-CN.md)。

## 单源与 Solver Adaptation 总表

| 路线 | adaptation 实现 | surface | 结果 | 决策 |
|---|---|---:|---|---|
| RoMa dense | 官方 RoMa `roma_outdoor` dense correspondence cache 转成 warp/certainty field。 | 2048 rows, 1638/410 train/eval | RoMa Level C: angle MAE `15.3929`, distance MAE `19.4702`; SuperGlue sparse anchor: `89.3941 / 52.3864`。 | 正向 dense-correspondence source。 |
| EfficientLoFTR | 官方 EfficientLoFTR semi-dense match/confidence field，使用同一 field-regressor contract。 | 2048 rows, 1638/410 train/eval | `55.3077 / 39.8429`。 | 跑通但弱于 RoMa dense。 |
| MASt3R official | 官方 MASt3R cache，包含 descriptor/match field 和 pointmap/confidence geometry field。 | 2048 rows, 1638/410 train/eval | match field `27.7047 / 39.0629`; geometry field `9.5863 / 33.2327`。 | heading/geometry source 强，range source 弱。 |
| DUSt3R official | 官方 DUSt3R pointmap/confidence geometry field。 | 2048-row same-subset replay | `11.0434 / 33.1489`。 | 有用的 geometry-heading control，不是主 range 路线。 |
| VGGT direct camera | 官方 VGGT `pose_enc` 解码为 extrinsic/intrinsic matrix。 | 2048 rows | `40.6610 / 40.0056`。 | direct camera extraction 较弱。 |
| VGGT geometry field | 官方 VGGT `world_points` 和 confidence field replay。 | 2048 rows | `52.5120 / 14.1716`。 | 强 distance/range signal，弱 heading signal。 |
| SAC-Pose frozen head | 官方风格 SAC-Pose frozen feature migration，加 PairUAV head。 | 256-row bounded probe | `91.0811 / 32.2643`。 | 技术跑通，方法质量失败。 |
| SAC-Pose relation prior | 使用官方 SAC-Pose descriptor/keypoint 构造 structured relation token。 | 256-row bounded probe | `87.9658 / 59.4438`。 | heading 小幅改善，range 严重退化。 |
| MADPose | 官方 MADPose solver，输入 RoMa correspondences 和 VGGT depth prior，再用 train-only linear mapping 做 sanity ranking。 | 1977 rows, 1581/395 train/eval | MADPose shared-focal `70.8120 / 40.1349`; OpenCV point-only control `73.4009 / 36.3710`。 | solver 可运行，但 proxy 下不够有用。 |
| FAR SuperGlue / LoFTR | 官方 FAR-style solver/fusion packet，使用 proxy intrinsics 和 SuperGlue/LoFTR branch。 | bounded solver packet probes | SuperGlue heading proxy median `82.15 deg`; LoFTR FOV sweep 改善到约 `70-74 deg`，仍然很高。 | 可执行，但不足以晋级训练。 |
| GRelPose frozen head | Frozen GRelPose/ScanNet-style representation 加 cheap PairUAV head。 | val811 bounded surface | best bounded head angle `43.8259`, proxy `0.2472`; native rough diagnostic `91.6852 / 66.9104`。 | feature/task mismatch，不具备竞争力。 |

## RoMa / MASt3R / VGGT Hybrid 证据

三源 hybrid 路线在同一批 2048 rows 上对齐 RoMa、MASt3R 和 VGGT，missing rows、duplicates 和 label mismatches 都为 0。随后在相同 split 上训练 source-drop variants。

| variant | angle MAE | distance MAE | proxy |
|---|---:|---:|---:|
| RoMa | `12.1377` | `16.1241` | `0.11434` |
| MASt3R | `12.9339` | `11.7043` | `0.09445` |
| VGGT | `11.4752` | `8.0099` | `0.07193` |
| RoMa + MASt3R | `12.2763` | `10.1944` | `0.08507` |
| RoMa + VGGT | `11.4403` | `11.8067` | `0.09081` |
| MASt3R + VGGT | `11.7800` | `7.8554` | `0.07200` |
| RoMa + MASt3R + VGGT | `10.8126` | `8.6817` | `0.07344` |

解释：

- full three-source concatenation 的 angle 最好，但 proxy 不是最好。
- VGGT 单源和 MASt3R+VGGT 是更强的 proxy/range candidate。
- RoMa 对某些 heading regime 有帮助，但 naive concatenation 可能损害 balanced proxy。

后续 source-complementarity audit 支持同一结论。四源 oracle proxy 达到 `0.04588397`，而 historical VGGT proxy 为 `0.07192526`。per-sample winner 分布在多个 source 上：

| winner view | leading sources |
|---|---|
| proxy winner share | MASt3R+VGGT `0.3488`, VGGT `0.2659`, full hybrid `0.2000`, RoMa `0.1854` |
| angle winner share | RoMa `0.3098`, MASt3R+VGGT `0.2683`, VGGT `0.2293`, full hybrid `0.1927` |
| distance winner share | MASt3R+VGGT `0.4122`, VGGT `0.2415`, full hybrid `0.1951`, RoMa `0.1512` |

这说明 source 互补性真实存在，但 oracle 不可部署。当时测试的一维 source-stat predictor 只具备部分可预测性，因此这条线没有晋级为 inference-time gate。

## 结构化 Split-Fusion Source Adaptation

Split-fusion 线不同于单源适配。它问的是：不同 source 是否应该服务不同的 polar axis：

```text
heading branch <- old field / geometry-heavy source
range branch   <- rich-native / VGGT-style source
```

### Old-Source Parity Audit

old VGGT/MASt3R/RoMa field sources 和 rich-native sources 在同一 2048-row、1600/448、1000-step 协议下评估。

| variant | angle MAE | distance MAE | proxy |
|---|---:|---:|---:|
| old_field_baseline | `1.393762` | `39.247571` | `0.182955` |
| old_roma_mast3r_vggt_concat | `1.328199` | `39.050710` | `0.181712` |
| rich_vggt_only | `2.835736` | `33.045150` | `0.163277` |
| rich_three_source_concat | `2.404302` | `33.637191` | `0.163523` |

结果显示出明确的轴分工：old field features 对 heading 更强，rich VGGT-style features 对 distance 更强。

### Split-Fusion Probe

随后使用两分支 `SplitFusionProbeRegressor`：heading 走 old-field features，distance 走 rich-native features。

| variant | angle MAE | distance MAE | proxy |
|---|---:|---:|---:|
| split_old_concat_angle_rich_vggt_range | `1.213552` | `5.937480` | `0.033249` |
| split_old_baseline_angle_rich_vggt_range | `1.200125` | `5.590767` | `0.031626` |
| split_old_concat_angle_rich_three_range | `1.330926` | `4.024406` | `0.025360` |
| split_old_baseline_angle_rich_three_range | `1.512513` | `4.331029` | `0.027738` |

这是第一组在同一协议下明确强于 old-field-only 和 rich-native-only anchor 的 source-adaptation 结果。它说明 source complementarity 不是只能在 oracle 中看到；只要结构尊重 heading/range split，互补性就能转化为性能。

### B811 Axis-Decoupled Replay

B811 comparable surface 进一步测试一个更简单的结构问题：distance 保留 base model，heading 使用 geometry evidence。

| variant | heading MAE | distance MAE |
|---|---:|---:|
| base only | `8.6087` | `0.9876` |
| geometry only | `1.7769` | `6.9755` |
| axis-decoupled | `1.7769` | `0.9876` |

B811 上的直接 pairwise comparison 显示：

| statistic | value |
|---|---:|
| geometry heading better | `694 / 811` |
| geometry heading worse | `117 / 811` |
| geometry distance better | `53 / 811` |
| geometry distance worse | `758 / 811` |
| heading-helpful but distance-harmful rows | `492` |

[MECHANISM_INSIGHTS.zh-CN.md](MECHANISM_INSIGHTS.zh-CN.md) 中使用的 stricter helpful/neutral/harmful binning 得到相同方向的结论：geometry 多数情况下帮助 heading，但多数情况下伤害 distance。

### 为什么它没有成为最终提交路线

split-fusion 没有直接晋级最终 official system，主要有三个实际原因：

- full official split-fusion inference 成本高，因为 source path 需要 heavy geometry sidecars；
- non-leaky no-op / base-sufficient policy 虽然形式上可行，但提升较弱；
- D-series cheap student compression 没有保留强机制信号，并且一次 official attempt 出现 distance parity failure。

这不否定 split-fusion。它说明这条线在本 release 中更适合作为机制证据和设计历史。

## 负结果与诊断路线

下面这些路线值得记录，正是因为它们是真实尝试过的负结果：

- SAC-Pose 可以接入 PairUAV train/eval，但 frozen 或 relation-prior migration 都没有得到有用的 heading/range predictor。
- MADPose 和 FAR 说明 solver pipeline 可执行，但 proxy intrinsics 和弱 heading/range signal 阻碍晋级。
- GRelPose-style frozen representation 加 cheap head 无法有效迁移到 PairUAV。
- native adapter 实验证明 shallow native summary 弱于 old field tensors；rich VGGT tensors 可以改善 distance，但会伤害 heading。
- D-series compression 没有否定 split-fusion 机制，只是否定了某个 cheap residual/student route。

## 面向论文的结论

source-adaptation 历史支持下面这个表述：

```text
PairUAV benefits from external geometric and correspondence evidence, but the
benefit is axis-conditional. Geometry-heavy sources are often useful for
heading and unsafe for distance; range-capable sources can improve metric
prediction while hurting heading. Effective systems should therefore preserve
stable metric evidence and introduce axis-aware readout, fusion, or
postprocessing rather than relying on naive source concatenation.
```

这个表述属于机制层证据。面向榜单的主张仍然是可复现的 PACE 最终提交包。

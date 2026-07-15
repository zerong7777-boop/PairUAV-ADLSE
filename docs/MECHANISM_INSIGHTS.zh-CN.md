# 机制洞察与表征证据

[English](MECHANISM_INSIGHTS.md)

本文汇总最终设计背后的机制证据。核心结论是轴非对称，而不是硬性的因子独立。

生成的总结图：

```text
figures/fig_mechanism_summary.svg
figures/fig_source_adaptation_summary.svg
figures/fig_val_qualitative_cases.svg
figures/fig_val_axis_gradcam.png
figures/fig_val_axis_gradcam.svg
figures/fig_val_patch_occlusion_axis_sensitivity.png
figures/fig_val_patch_occlusion_axis_sensitivity.svg
figures/fig_axis_conflict_quadrant.svg
figures/fig_regime_conditioned_axis_gain.svg
figures/fig_representation_probe_summary.svg
figures/fig_range_tail_failure_quantiles.svg
```

生成脚本：

```text
figures/gen_fig_mechanism_summary.py
figures/mechanism_summary_data.csv
figures/gen_fig_source_adaptation_summary.py
figures/source_adaptation_summary_data.csv
figures/gen_fig_val_qualitative_cases.py
figures/val_qualitative_cases.csv
figures/gen_fig_val_axis_gradcam.py
figures/val_axis_gradcam_summary.csv
figures/val_axis_gradcam_summary.json
figures/gen_fig_val_patch_occlusion_axis_sensitivity.py
figures/val_patch_occlusion_axis_sensitivity_summary.csv
figures/val_patch_occlusion_axis_sensitivity_summary.json
figures/gen_fig_axis_mechanism_deep_dive.py
figures/b811_base_geometry_surface.csv
figures/b811_anchor_val_predictions.csv
figures/split_fusion_train_val_predictions.csv
figures/axis_conflict_quadrant_summary.csv
figures/regime_conditioned_axis_gain.csv
figures/representation_probe_summary.csv
figures/range_tail_failure_quantiles.csv
```

更完整的外部 source adaptation 和 split-fusion 结果盘点见 [SOURCE_ADAPTATION.zh-CN.md](SOURCE_ADAPTATION.zh-CN.md)。

## 如何阅读这些证据

本文中的图和表都围绕一个机制问题组织：heading 和 distance 到底只是同一个
回归目标的两个可互换坐标，还是呈现出异质 evidence 与失败模式？当前证据支持
第二种理解，但需要保留边界：这里不主张两个轴是完全独立的 latent factor。

| 证据类型 | 产物 | 机制作用 |
|---|---|---|
| Regime-level difficulty | information-solvability split probes | 显示 heading-vs-distance 难度会随 split regime 改变 |
| Pair-level source conflict | B811 axis-conflict quadrant | 显示同一个 pair 上 geometry 可能帮助 heading、伤害 distance |
| Image-space evidence use | GradCAM 和 patch-occlusion 图 | 显示同一个 H8 模型在两个轴上可能使用不同 target-view 区域 |
| Validation composition | ADPA axis-decoupled surface | 显示 heading 和 distance 可以从不同 evidence stream 重新组合 |
| Representation audits | Phase95 probe summaries | 支持共享 pose-regime readability，但不支持 hard subspace independence |
| Range-tail diagnostics | HR50/H8/Tail10 quantiles | 显示 tail robustness 是 range-specific optimization 问题 |

## 洞察 1：Heading 和 Distance 在不同 Regime 下失败方式不同

information-solvability probing 使用冻结 match-observability proxy，在多个 split family 上比较行为。结果显示 heading 和 distance 没有固定的难度顺序：

| split | role | heading minus range mean | heading over range mean | reading |
|---|---|---:|---:|---|
| `group_control_split_v1` | control | 10.45 | 1.34 | heading 比 distance 更难 |
| `pair_name_heldout_split_v1` | intermediate | 9.38 | 1.30 | heading 比 distance 更难 |
| `delta_heldout_split_v1` | prior-break | 2.36 | 1.07 | heading 和 distance 接近耦合 |
| `extreme_delta_split_v1` | stress | -28.96 | 0.65 | distance 明显比 heading 更难 |

这支持一种 regime-dependent polar-axis 视角：单一全局 hardness score 不足以解释 PairUAV 行为。

## 洞察 2：Geometry Utility 具有轴非对称性

B811 comparable-surface diagnostic 在同一批验证 pair 上比较 base 和 geometry outcome。geometry source 通常有利于 heading，但有害于 distance：

直接 lower-error comparison 中，geometry heading 在 `694/811` 个 pair 上更好，geometry distance 在 `758/811` 个 pair 上更差。下表使用机制总结图中的 stricter helpful/neutral/harmful binning。

| axis | geometry helpful | neutral/tie | geometry harmful | dominant behavior |
|---|---:|---:|---:|---|
| heading | 608 | 135 | 68 | geometry helps |
| distance | 10 | 136 | 665 | geometry harms |

Pair-level regime counts 进一步显示：

| pair regime | count |
|---|---:|
| heading_helpful_distance_harmful | 492 |
| base_sufficient_candidate | 189 |
| joint_geometry_helpful | 117 |
| joint_geometry_harmful | 13 |

这个结果直接推动了 axis-decoupled composition 和 protected metric path。

完整 B811 axis-conflict quadrant 更直接地展示了这个现象：

![Axis-conflict quadrant](../figures/fig_axis_conflict_quadrant.svg)

如果使用直接 gain 定义 `base error - geometry error`，`645/811` 个 pair 落在
“geometry 帮助 heading、伤害 range”的主导象限；只有 `49/811` 是两个轴同时改善，
只有 `4/811` 是 range 改善但 heading 变差。这个证据比平均指标更强，因为 trade-off
发生在 pair 级别。

regime-conditioned 视角进一步说明，这种 gain 与真实 heading 和绝对 range regime 有关：

![Regime-conditioned axis gain](../figures/fig_regime_conditioned_axis_gain.svg)

两张热图使用同一批 B811 样本，但分别聚合 heading gain 和 range gain。可以看到不同
pose regime 下两个轴的 gain 符号和幅度并不一致。这支持 regime-dependent polar-axis
视角，而不是单一全局 geometry-quality score。

release 中也加入了一个可复现的 val 定性样例 manifest。仓库中已保存的 SVG 是自包含版本，
内嵌了选中的验证图像；如果要从 manifest 重新生成，需要提供官方 `train_tour` 路径：

```bash
python figures/gen_fig_val_qualitative_cases.py \
  --manifest figures/val_qualitative_cases.csv \
  --dataset-root /path/to/pairUAV/train_tour \
  --output figures/fig_val_qualitative_cases.svg \
  --embed-images
```

为了给出图像空间层面的证据，release 额外提供了基于 H8 mid-late checkpoint 的
target-aligned output-gradient activation map。对每个验证 pair，脚本分别对
target-aligned heading output 和 signed distance output 在 target view 上反传，
得到 heading heatmap、distance heatmap，以及两者的绝对差异热图。相同脚本也可以通过
`--gradient-target loss` 生成 loss-gradient 版本。

![验证图像 target-aligned output-gradient 热图](../figures/fig_val_axis_gradcam.png)

```bash
python figures/gen_fig_val_axis_gradcam.py \
  --repo-root /path/to/reloc3r_pairuav \
  --json-root /path/to/val_json \
  --image-root /path/to/pairUAV/train_tour \
  --checkpoint /path/to/H8_mid_late/checkpoint-final.pth \
  --case-manifest figures/val_qualitative_cases.csv \
  --gradient-target target_aligned_output \
  --output-png figures/fig_val_axis_gradcam.png \
  --output-svg figures/fig_val_axis_gradcam.svg \
  --summary-csv figures/val_axis_gradcam_summary.csv \
  --summary-json figures/val_axis_gradcam_summary.json
```

在仓库中保存的 4 个验证样例上，heading 与 distance 热图的 top-20% 激活区域平均重叠率为
`0.241`，样例之间从几乎不重叠（`0.001`）到部分重叠（`0.543`）都有；归一化热图的平均绝对差异为
`0.540`。这说明同一个 H8 模型在两个轴上使用的图像证据并不完全相同。这里的证据类型应表述为
gradient-based spatial evidence，而不是严格的因果遮挡证明。

release 还加入了更强的 perturbation-style 可视化。patch-occlusion 脚本不是反传梯度，
而是每次遮挡 target view 的一个 patch，然后分别测量 H8 heading output 和 range output
发生了多大变化：

![验证图像 patch-occlusion 轴敏感性](../figures/fig_val_patch_occlusion_axis_sensitivity.png)

```bash
python figures/gen_fig_val_patch_occlusion_axis_sensitivity.py \
  --repo-root /path/to/reloc3r_pairuav \
  --json-root /path/to/val_json \
  --image-root /path/to/pairUAV/train_tour \
  --checkpoint /path/to/H8_mid_late/checkpoint-final.pth \
  --case-manifest figures/val_qualitative_cases.csv \
  --case-groups axis_conflict,joint_helpful \
  --grid 4 \
  --mask-mode zero \
  --device cpu \
  --output-png figures/fig_val_patch_occlusion_axis_sensitivity.png \
  --output-svg figures/fig_val_patch_occlusion_axis_sensitivity.svg \
  --summary-csv figures/val_patch_occlusion_axis_sensitivity_summary.csv \
  --summary-json figures/val_patch_occlusion_axis_sensitivity_summary.json
```

在仓库中保存的 4 个 occlusion 样例上，heading/range top-20% 敏感区域平均重叠率为
`0.360`，从 `0.000` 到 `0.707` 都有。它仍然不是完整因果证明，但比纯梯度图更少依赖局部梯度。

split-fusion 前置实验在 source-adaptation 协议下显示了同样的轴分工：

| probe | heading source | distance source | angle MAE | distance MAE | proxy |
|---|---|---|---:|---:|---:|
| old-field baseline | old field | old field | 1.393762 | 39.247571 | 0.182955 |
| rich VGGT only | rich VGGT | rich VGGT | 2.835736 | 33.045150 | 0.163277 |
| split-fusion | old-field concat | rich three-source | 1.330926 | 4.024406 | 0.025360 |

这条线没有作为最终提交路线，但它是重要机制证据：当 heading 和 distance 被路由到不同 evidence stream 时，source utility 明显更好。

## 洞察 3：Axis-Decoupled Composition 在验证 Surface 上成立

ADPA-1 修复 prediction-surface contract blocker 后，测试了一个简单轴组合：

```text
heading = geometry-assisted heading
distance = base distance
```

验证结果为：

| variant | heading MAE | distance MAE |
|---|---:|---:|
| base only | 8.6087 | 0.9876 |
| geometry only | 1.7769 | 6.9755 |
| axis-decoupled | 1.7769 | 0.9876 |

这支持设计原则，但它本身不是可部署 selector。base-sufficient slice 仍然需要 control protection，因为 geometry heading 并不总是更好。

## 洞察 4：表征证据支持共享 Pose-Regime 可读性

Phase95 representation audits 测试 H8 representation 是否编码了非 shuffle artifact 的 pose-regime 信息。

H8 final val811 true-vs-shuffle gaps：

| regime label | true score | shuffle mean | true-minus-shuffle |
|---|---:|---:|---:|
| heading_8bin | 0.608408 | -0.000625 | 0.609033 |
| range_abs_bucket | 0.253464 | -0.004357 | 0.257820 |
| range_signed_bucket | 0.578852 | -0.010970 | 0.589821 |

同样现象在 train811 上也稳定存在。H8 final 相比 Wbounded H8 明显增强了 heading-regime structure，而 signed-range structure 在 Wbounded H8 中已经较强。

![Representation probe summary](../figures/fig_representation_probe_summary.svg)

但是，Phase95-R2 subspace-overlap **不支持** 更强的“heading 和 range 位于非重叠线性子空间”说法。测试指标中的 cross/self ratio 大于 1。因此正确结论是：

```text
shared pose-regime representation with coordinate-specific readability
```

而不是：

```text
hard heading/range latent disentanglement
```

## 洞察 5：Tail Robustness 是单独的 Range 机制

PAAER HR50 主要失败在 high-absolute-range tail，而不是全局 range calibration：

| run | range median | p95 | p99 | max |
|---|---:|---:|---:|---:|
| HR50 | 0.4546 | 7.1223 | 42.9710 | 91.2856 |
| H8 step50k | 0.3897 | 1.1979 | 2.1321 | 6.9968 |
| Tail10 | 0.3600 | 1.0446 | 1.2653 | 4.3494 |

![Range-tail failure quantiles](../figures/fig_range_tail_failure_quantiles.svg)

这就是为什么最终路线不是简单继续长训，而是引入 tail-weighted range training，并最终使用确定性 range stacking。

## 面向论文的解释

本 release 支持下面的机制表述：

```text
PairUAV polar localization contains a shared pose-regime representation, but
heading and distance expose different evidence utility, optimization behavior,
and tail-risk profiles. A strong system should preserve stable metric evidence
while allowing axis-specific readout and calibration.
```

本 release 不主张：

- 使用隐藏 official-test label 进行后处理；
- pair ID 可以直接决定 distance；
- heading 和 distance 是完全独立的 latent factors；
- 最终分数来自一个不含确定性后处理的单一端到端模型。

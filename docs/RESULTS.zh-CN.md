# 结果记录

[English](RESULTS.md)

本文档区分三类证据：

1. **官方榜单反馈：** 比赛服务器对提交包返回的分数。
2. **Raw prediction 资产：** 为复现提供的 official-test 预测文件；raw zip 不包含标签或分数。
3. **内部开发代理指标：** 模型选择过程中使用的本地验证或 proxy measurement。

## 官方榜单反馈

下表记录最终提交过程中目前可用的官方反馈。数值越低越好。这些分数来自服务器反馈；由于 official test label 隐藏，不能从公开 raw zip 重新计算这些分数。

| 系统 / 提交包 | 组件 | 后处理 | final_score | distance_rel_error | angle_rel_error | 证据 |
|---|---|---|---:|---:|---:|---|
| PAAER raw | PAAER heading + PAAER distance | 无 | 0.002514 | 0.002392 | 0.002636 | 手动记录的榜单反馈 |
| PAAER heading + MDHR/PAAER distance mix | PAAER heading + 0.66 PAAER distance + 0.34 MDHR distance | 无 | 0.002446 | 0.002245 | 0.002646 | 手动记录的榜单反馈 |
| RSF e230 | PAAER heading + RSF distance | 无 | 0.002413 | 0.002191 | 0.002636 | 手动记录的榜单反馈 |
| RSF e230 + heading lattice | PAAER heading + RSF distance | heading lattice snapping | 0.002308 | 0.002150 | 0.002465 | 手动记录的榜单反馈 |
| PACE final | PAAER heading + RSF distance | heading lattice + support-distance snapping | 0.001874 | 0.001330 | 0.002419 | 手动记录的榜单反馈；包 hash 已在 `OPEN_SOURCE_AUDIT_20260701.md` 中验证 |

最终提交包本身可以通过 `postprocess/rebuild_final_postprocess.py` 字节级一致地重建；hash 级证据见 `OPEN_SOURCE_AUDIT_20260701.md`。

## Raw Prediction 资产

最终 PACE 提交包由三个 raw official-test prediction zip 重建：

| 公开名称 | 资产 | 模型角色 | ZIP SHA256 |
|---|---|---|---|
| PAAER raw | `assets/phase104j_HRofficial_raw_result.zip` | 主轴解耦预测器 | `294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1` |
| MDHR raw | `assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` | 多深度 heading-oriented 互补预测器 | `b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b` |
| MARB raw | `assets/epoch2_pair_official_test_result_20260527_013814.zip` | Metric-aware Reloc3r baseline 预测器 | `7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee` |

MDHR 和 MARB 同时作为 raw asset 和 checkpoint asset 发布，用于复现最终系统。它们的单模型 official-test 反馈未记录在当前 public audit 文件中；如果后续恢复这些反馈，可以补充到本文档，不影响复现路径。

## 内部开发信号

内部 proxy metric 用于判断哪些模型族值得进行完整 official inference 和提交尝试。它们使用不同 split，也可能使用 proxy 指标，因此不能与官方榜单直接比较。

运行记录中保留的例子包括：

| 场景 | 观察 |
|---|---|
| 2.5k fair controls | full-loss HR/PAAER control 在本地 proxy 上优于 H8 和 H8 mid-late fair controls。 |
| Source adaptation probes | RoMa、MASt3R、DUSt3R、VGGT、SAC-Pose、MADPose、FAR、GRelPose 和 split-fusion 路线都作为 bounded internal evidence 测试过，不是 official leaderboard system。 |
| Range-tail diagnosis | PAAER full-train 变体的中位 range error 有竞争力，但在 tail-weighted continuation 前，高绝对距离尾部表现更差。 |
| Tail-weighted continuation | Tail10 相比 H8 step50k anchor 改善了本地 proxy，并减少了灾难性 range outlier。 |
| Final postprocess sweep | RSF weight sweep 改善了 distance error；heading-lattice 和 support-distance snapping 带来了最大的最终提升。 |

source-adaptation 细节见 [SOURCE_ADAPTATION.zh-CN.md](SOURCE_ADAPTATION.zh-CN.md)。精确历史细节见项目 runtime archive 中的 phase status notes。这些记录作为 provenance 保留，不作为正式论文表格。

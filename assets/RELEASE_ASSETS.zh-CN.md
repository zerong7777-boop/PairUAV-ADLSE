# Release Assets

[English](RELEASE_ASSETS.md)

git 仓库中包含代码、配置、manifest 和确定性后处理逻辑。以下大文件应作为 release assets 或外部下载发布。

部分资产名称包含内部运行编号。为了保持 hash、manifest 和复现命令稳定，这些名称保持不变。

## 下载

当前外部镜像：

```text
百度网盘：https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
提取码：djmv
```

下载资产文件夹后，将其中内容按原目录结构复制到仓库根目录。checkpoint 和 raw official prediction zip 的期望路径位于 `assets/...`，可选已知输出包位于 `known/...`。

## Checkpoints

| 相对路径 | 用途 | SHA256 |
|---|---|---|
| `assets/reloc3r_official_metric_full_epoch_checkpoint_last/checkpoint-last.pth` | epoch2/rank1 训练续训所用 source checkpoint | `d08d66fe53fbbcf43ce451e406850c7300a12e5e588879811142d66e67f32c5d` |
| `assets/phase45_epoch2_resume_final/checkpoint-final.pth` | epoch2/rank1 raw official prediction 来源 | `5681ab612d44dc64c98c82ef5b8b4e36c2bb4b38f6748a469f9c8d78c3894e04` |
| `assets/phase104i_tail10_init/checkpoint-final.pth` | phase104j long training 的初始化 checkpoint | `e7e42848b6e18cb31a713b0b95a9c4e7f6e529d7f9c6b2a3e26882a44bbf2f67` |
| `assets/phase104j_paaer_hr_final/checkpoint-final.pth` | HR/PAAER final raw official prediction 来源 | `7454c0f2b31921cd917c21c3726a7cc1860bef03163c474ed689c19e7af4d01e` |
| `assets/phase89_h8_midlate_final/checkpoint-final.pth` | H8 mid-late raw official prediction 来源 | `cc985ac7671e7d6dc22928df7eb443ac4f95ad4e295333017e019f6da34216c8` |

## Raw Official Prediction Zips

| 相对路径 | ZIP SHA256 | `result.txt` SHA256 |
|---|---|---|
| `assets/phase104j_HRofficial_raw_result.zip` | `294c1b96f352ed0c514423c943cc86b785db0ec86af696e99878ec17516615b1` | `6c7da711caac6e6feb91355f7936a820e29f0c3bb19f4affbdfe195f60a507ad` |
| `assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip` | `b7c7347b5db8d471132e480d810086e187f27ff0249289ba8fdd4f27e4d6444b` | `50ce60ce71465b7ddf3445e98ccca65576729a16e40d28556cd0678e1c213360` |
| `assets/epoch2_pair_official_test_result_20260527_013814.zip` | `7fd5fbc532fbc58dc70940ef84739ea552064971b547feb584f2e09d7e3cbcee` | `956711c0c387d5db5999d3c464267e9eb56c4b036688693ec51538fecc4edeba` |

## 可选已知输出包

这些文件不是重建最终包所必需的，但对字节级验证有帮助。

| 相对路径 | ZIP SHA256 |
|---|---|
| `known/phase104j_HRheading_range_stack_sweep_e230.zip` | `2c4f55a4df297806dd1a851e59b2af08ecc1e35e7361bb5b6e6f4ea9065a2548` |
| `known/phase104j_snap_e230_Hlat2_Dsupport_result.zip` | `3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f` |

## 已包含的小资产

`manifests/devsplit_v1_official_metric_manifest.csv` 足够小，因此直接纳入仓库。

```text
SHA256: f61f909a7441ac95c3790725507fb60851f7f10b5f3e6b477e92b5e36a96d580
rows including header: 204121
```

## 外部比赛数据

official PairUAV test JSON/images 不在本仓库中重新分发。只有从 checkpoint 重新生成三个 raw prediction zip 时才需要它们。

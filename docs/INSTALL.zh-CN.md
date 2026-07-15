# 安装说明

[English version](INSTALL.md)

本仓库有两条运行路径：

1. **仅后处理复现：** 从已发布的 raw prediction zip 重建最终提交包。这条路径
   对 CPU 友好，不需要 PyTorch 或 CUDA。
2. **模型推理 / 训练复现：** 从已发布 checkpoint 重新生成 raw prediction，或
   继续训练。这条路径需要 CUDA-capable Linux 环境，以及与 GPU/driver 匹配的
   PyTorch 构建。

`scripts/` 和 `scripts/training/` 里的多数脚本默认面向 Linux 路径和 Bash。
确定性后处理脚本是普通 Python 脚本，是验证 release 的最轻量入口。

## 推荐 Conda 环境

在仓库根目录运行：

```bash
conda env create -f environment.yml
conda activate pace
```

`environment.yml` 有意不固定 PyTorch/CUDA 构建。PyTorch 二进制兼容性取决于
NVIDIA driver、CUDA runtime 和 GPU 架构。请根据本机环境，使用 PyTorch 官方
安装选择器给出的命令单独安装 PyTorch。

在 CUDA 12.x Linux 机器上，这一步通常类似：

```bash
pip install torch torchvision --index-url <pytorch-cuda-wheel-index>
```

如果使用 Blackwell/RTX 50 系列 GPU，需要选择明确支持该架构的 PyTorch 构建。
如果模型可以 import 但 CUDA kernel 运行失败，首先检查 PyTorch/CUDA/GPU 兼容性。

## Pip 备选方式

如果希望自行管理环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -r requirements_optional.txt
```

然后再安装与本机 CUDA 环境匹配的 PyTorch。

## 可选 CroCo CUDA 扩展

仓库中 vendored CroCo 依赖包含 CUDA rotary-position 扩展，位于
`croco/models/curope/`。部分路径不重新编译也能运行，但完整 GPU 推理建议在目标
环境中重新编译：

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../..
```

如果编译失败，检查 `nvcc`、`CUDA_HOME` 和 PyTorch/CUDA 版本匹配。仅后处理复现
不需要这个扩展。

## Release Assets

大 checkpoint 和 raw official prediction zip 不放入 git。请从百度网盘下载
release asset 文件夹：

```text
https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
提取码：djmv
```

下载后，将其中的 `assets/` 和 `known/` 两个目录复制到仓库根目录，并保持相对
路径不变。期望路径和 SHA256 见
[assets/RELEASE_ASSETS.zh-CN.md](../assets/RELEASE_ASSETS.zh-CN.md)。

official PairUAV test JSON/images 不在本仓库中重新分发。只有从 checkpoint 重新
生成 raw prediction zip 时才需要这些官方测试数据。

## 快速验证

安装基础 Python 依赖并放好已发布 raw zip 后，运行确定性重建：

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

期望最终输出：

```text
outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
SHA256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

模型级复现需要先安装 PyTorch 并准备 official PairUAV test 数据，然后按
[A_REPRODUCTION.zh-CN.md](A_REPRODUCTION.zh-CN.md) 执行。

## 常见问题

- **`ModuleNotFoundError: torch`：** 需要单独安装与 CUDA 环境匹配的 PyTorch。
  `environment.yml` 不固定 PyTorch。
- **CUDA 扩展编译失败：** 检查 `nvcc`、`CUDA_HOME` 和 PyTorch/CUDA 兼容性。
  仅后处理路径不需要该扩展。
- **缺少 checkpoint 或 raw zip：** 下载 `PairUAV-PACE-release-assets`，并保持
  文档记录的 `assets/...` 和 `known/...` 路径结构。
- **official test 数据路径错误：** official test JSON/images 不包含在本仓库中，
  需要按比赛允许渠道另行准备。
- **重训后 checkpoint 与发布权重不完全一致：** 这是预期现象。榜单提交包复现应
  使用已发布 checkpoint。

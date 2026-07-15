# Installation

[中文说明](INSTALL.zh-CN.md)

This repository has two execution paths:

1. **Postprocess-only reproduction:** rebuild the final submission package from
   released raw prediction zips. This path is CPU-friendly and does not require
   PyTorch or CUDA.
2. **Model inference / training reproduction:** regenerate raw predictions or
   continue training from released checkpoints. This path requires a CUDA-capable
   Linux environment and a PyTorch build matched to your GPU and driver.

Most shell scripts in `scripts/` and `scripts/training/` assume Linux paths and
Bash. The deterministic postprocess script is regular Python and is the easiest
entry point for verifying the release.

## Recommended Conda Environment

Create the base environment from the repository root:

```bash
conda env create -f environment.yml
conda activate pace
```

`environment.yml` intentionally does not pin a PyTorch/CUDA build. PyTorch
binary compatibility depends on your NVIDIA driver, CUDA runtime, and GPU
architecture. Install PyTorch separately using the command recommended by the
official PyTorch selector for your machine.

For example, on a CUDA 12.x Linux machine this step usually has the form:

```bash
pip install torch torchvision --index-url <pytorch-cuda-wheel-index>
```

For Blackwell/RTX 50-series GPUs, use a PyTorch build that explicitly supports
that architecture. If the model imports but CUDA kernels fail at runtime, the
first thing to check is the PyTorch/CUDA/GPU compatibility matrix.

## Pip Fallback

If you prefer to manage the environment yourself:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -r requirements_optional.txt
```

Then install a matching PyTorch build for your CUDA setup.

## Optional CroCo CUDA Extension

The vendored CroCo dependency includes a CUDA rotary-position extension under
`croco/models/curope/`. Some model paths can run without rebuilding it, but full
GPU inference is more reliable when the extension is compiled in the target
environment:

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../..
```

If this step fails, verify that `nvcc` is available, `CUDA_HOME` points to the
right toolkit, and the installed PyTorch build matches the CUDA toolkit. The
postprocess-only reproduction path does not need this extension.

## Release Assets

Large checkpoints and raw official prediction zips are not stored in git.
Download the release asset folder from Baidu Netdisk:

```text
https://pan.baidu.com/s/1k1QGg6KObLIVikCubvG5BA?pwd=djmv
Extraction code: djmv
```

Copy the inner `assets/` and `known/` directories into the repository root while
preserving relative paths. The expected asset paths and SHA256 hashes are listed
in [assets/RELEASE_ASSETS.md](../assets/RELEASE_ASSETS.md).

The official PairUAV test JSON/images are not redistributed here. They are only
needed when regenerating raw prediction zips from checkpoints.

## Quick Verification

After installing the base Python dependencies and placing the released raw zips,
run the deterministic rebuild:

```bash
python postprocess/rebuild_final_postprocess.py \
  --hr-zip assets/phase104j_HRofficial_raw_result.zip \
  --h8-zip assets/phase90_phase89_h8_midlate_fulltrain_1epoch_result.zip \
  --epoch2-zip assets/epoch2_pair_official_test_result_20260527_013814.zip \
  --support-manifest manifests/devsplit_v1_official_metric_manifest.csv \
  --e230-output-zip outputs/phase104j_HRheading_range_stack_sweep_e230.zip \
  --final-output-zip outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
```

Expected final output:

```text
outputs/phase104j_snap_e230_Hlat2_Dsupport_result.zip
SHA256: 3e53085ca02a8441d247c3aa96cfeee9d0e3dc5944431c9c56fc842b01fe8a0f
```

For model-level reproduction, follow [A_REPRODUCTION.md](A_REPRODUCTION.md)
after installing PyTorch and preparing the official PairUAV test data.

## Common Issues

- **`ModuleNotFoundError: torch`:** install a PyTorch build matching your CUDA
  setup. It is not pinned in `environment.yml`.
- **CUDA extension build failure:** check `nvcc`, `CUDA_HOME`, and PyTorch/CUDA
  compatibility. You can still run the postprocess-only path without the
  extension.
- **Missing checkpoint or raw zip:** download `PairUAV-PACE-release-assets`
  and preserve the documented `assets/...` and `known/...` layout.
- **Official test data path errors:** the competition test JSON/images are not
  included in this repository and must be supplied separately for raw inference.
- **Byte-level checkpoint differences after retraining:** this is expected.
  Use released checkpoints for leaderboard-package reproduction.

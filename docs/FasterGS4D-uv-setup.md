# Running FasterGS4D with `uv` on a Blackwell GPU / glibc ≥ 2.41

The upstream instructions assume Conda plus a system-wide CUDA 12.8 SDK. This note records a
working setup that uses `uv` for the Python environment and a **local, root-free CUDA
toolchain**, on a machine where the stock instructions cannot work.

Reference machine: RTX 5060 Ti (`sm_120`), Ubuntu with glibc 2.43, gcc 13, driver 595.71
(CUDA 13.2), system `nvcc` 12.4, no Conda.

## Why the stock recipe fails here

1. **`sm_120` needs nvcc ≥ 12.8.** The distro `nvcc` is 12.4, whose newest target is `sm_90a`,
   so nothing it produces will run on Blackwell.
2. **glibc ≥ 2.41 collides with CUDA ≤ 13.0 headers.** glibc added the C23 functions
   `cospi`/`sinpi`/`rsqrt` to `<math.h>`, declared `noexcept(true)`. CUDA's
   `crt/math_functions.h` declares the same names without an exception specification, so every
   `.cu` file fails with *"exception specification is incompatible with that of previous
   function"*. CUDA 13.0 fixes `cospi`/`sinpi` but not `rsqrt`; **CUDA 13.2 is the first
   release that compiles cleanly against glibc 2.43.**
3. **A CUDA 13 toolchain needs a CUDA 13 PyTorch.** `torch.utils.cpp_extension` refuses to
   build when the nvcc major version differs from the one PyTorch was built with, so the
   pinned `torch==2.8.0+cu128` has to become `torch==2.9.1+cu130` (a minor-version skew of
   13.2 vs 13.0 is only a warning).
4. **CCCL 3.0 removed `cub::Max`.** CUDA 13 ships CCCL 3, so
   `FasterGS4DCudaBackend/.../rasterization/include/kernels_forward.cuh` no longer compiles.
   See [the patch](#cccl-30-patch) below.

## Layout

Everything lives under the NeRFICG checkout; nothing is installed system-wide.

```
nerficg/
├── .toolchain/
│   ├── bin/                 # gcc -> gcc-13, g++ -> g++-13 shims (nvcc calls them unversioned)
│   └── cuda-13.2/           # nvcc + crt + nvvm + nvptxcompiler + cudart + cccl, ~600 MB
├── .venv/                   # uv-managed, Python 3.11
├── env.sh                   # exports CUDA_HOME, PATH, CC/CXX/CUDAHOSTCXX, TORCH_CUDA_ARCH_LIST
├── pyproject.toml           # replaces environments/py311_cu128.yaml
└── src/Methods/FasterGS4D/  # the FasterGS4D branch of this repository
```

`docs/setup_fastergs4d_uv.sh` in this repository performs the whole setup end to end.

The CUDA components are pulled from NVIDIA's redistributable archives
(`https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.1.json`), which need
no installer and no root: `cuda_nvcc`, `cuda_crt`, `libnvvm`, `libnvptxcompiler` (the compiler
and its device backend), `cuda_cudart` (runtime headers/stubs) and `cuda_cccl` (Thrust/CUB).
Note that `nvidia-cuda-nvcc-cu12` on PyPI is **not** an alternative — it ships `ptxas` and
`nvvm` but no `nvcc` driver binary.

## Python environment

`uv` replaces the Conda environment file. The dependency list is the one from
`environments/py311_cu128.yaml`, with `torch` and `torchvision` pinned to the cu130 index:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
```

`pip` is added as an explicit dependency because NeRFICG's `scripts/install.py` shells out to
`pip install <extension_dir> --no-build-isolation` to build the CUDA extensions, and a `uv`
virtualenv has no `pip` by default.

## CCCL 3.0 patch

`cub::Max` was removed in CCCL 3.0. `cuda::maximum<>` is the replacement and has existed since
CCCL 2.2 (CUDA 12.3), so the change is backward compatible with the CUDA 12.8 setup upstream
targets:

```diff
 #include <cooperative_groups.h>
+#include <cuda/functional>  // cuda::maximum; cub::Max was removed in CCCL 3.0 (CUDA 13)
 namespace cg = cooperative_groups;
@@
-        n_processed_and_used = BlockReduce(temp_storage).Reduce(n_processed_and_used, cub::Max());
+        n_processed_and_used = BlockReduce(temp_storage).Reduce(n_processed_and_used, cuda::maximum<>{});
```

Applied to the `FasterGS4D` branch as `docs/patches/fastergs4d-cccl3.patch`.

## Dataset

`src/Methods/FasterGS4D/fastergs4d_mutant.yaml` targets `dataset/dnerf/mutant`, i.e. the
synthetic monocular dynamic scenes from [D-NeRF](https://github.com/albertpumarola/D-NeRF).
These are self-calibrated (poses ship with the scene) so no COLMAP run is needed.

## Training

```shell
cd nerficg
source env.sh
python ./scripts/train.py -c configs/fastergs4d_mutant.yaml
```

Set `TRAINING.GUI.ACTIVATE: false` in the config for headless runs.

## Verified run

`configs/fastergs4d_mutant.yaml` (the branch's own sample config, GUI disabled), 30 000
iterations on the D-NeRF `mutant` scene, RTX 5060 Ti — while a second job occupied 10.8 GB of
VRAM and most of the GPU:

| | |
|---|---|
| Test PSNR / SSIM / LPIPS | 37.82 / 0.986 / 0.020 (20 views) |
| Gaussians | 100 000 → 198 049 |
| Peak VRAM | 1.84 GiB allocated (2.04 GiB reserved) |
| Training time | 30:14 for 30 000 iterations (60.5 ms/iter) |

The per-iteration time reflects heavy GPU contention, not the method's throughput.

## Viewing the result

All paths are relative to the training output directory
(`output/FasterGS4D/<config>_<timestamp>/`).

**Still renders.** Training already writes `test_30000/rgb` (model) next to `test_30000/rgb_gt`
(ground truth) — open them in any image viewer.

**Novel trajectories.** `scripts/inference.py -d <output_dir> -s <trajectory>` renders any
trajectory from `src/Visual/Trajectories/`: `bullet_time` (lemniscate orbit with time
replaying), `spiral_path`, `fixed_view`, `ellipse_path`, `fancy_zoom`, `novel_view`,
`stabilized_path`. Frames land in `inference/<trajectory>_<iterations>/rgb`.

Note that on D-NeRF scenes the synthesised trajectories roll the camera: they derive an
up-axis from the training poses, which are spread over a full hemisphere. The rendering is
correct, the framing is not. `docs/time_sweep.py` avoids this by reusing a dataset camera
verbatim and only advancing the timestamp, which is usually what you want for inspecting a
dynamic reconstruction:

```shell
cd nerficg && source env.sh
python <this_repo>/docs/time_sweep.py output/FasterGS4D/<run> <test_view_index> <n_frames>
```

**Video.** There is no ffmpeg dependency in this environment; `docs/make_video.py` turns a
frame directory into an `.mp4` (OpenCV) plus a half-resolution `.gif` (Pillow):

```shell
python <this_repo>/docs/make_video.py <frame_dir> <output_stem> <fps>
```

**Interactive.** `python ./scripts/gui.py` opens the NeRFICG viewer, where the model can be
flown around freely and scrubbed through time. It needs a desktop session — run it from a
terminal on the machine's display, not over a plain SSH connection.

**Other tools.** `scripts/convert_to_ply.py` exports the Gaussians for external viewers,
though a .ply carries only a static frame of a 4D model.

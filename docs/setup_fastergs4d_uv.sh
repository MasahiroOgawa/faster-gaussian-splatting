#!/usr/bin/env bash
# Set up the FasterGS4D variant (dynamic / 4D Gaussian Splatting) on a machine with
#   - no conda, no root, a uv-managed Python environment
#   - an NVIDIA Blackwell GPU (sm_120, e.g. RTX 50xx)
#   - glibc >= 2.41
#
# The upstream README assumes conda + a system CUDA 12.8 SDK. Two things break that here:
#   1. sm_120 needs nvcc >= 12.8, and distro packages often ship 12.4.
#   2. glibc >= 2.41 declares the C23 math functions cospi/sinpi/rsqrt, which collide with
#      the same declarations in every CUDA <= 13.0 <crt/math_functions.h>. CUDA 13.2 is the
#      first release whose headers compile against such a glibc, so we pair it with a
#      PyTorch cu130 wheel (same CUDA major version).
#
# Everything lands inside $NERFICG_ROOT; nothing is installed system-wide.
set -euo pipefail

NERFICG_ROOT="${NERFICG_ROOT:-$HOME/proj/study/nerficg}"
CUDA_VERSION="${CUDA_VERSION:-13.2.1}"     # CUDA redistributable manifest to pull nvcc from
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"    # newest torch with a cu130 wheel for Python 3.11
HOST_GCC="${HOST_GCC:-13}"                 # major version of the gcc/g++ to use as nvcc host compiler

# ---------------------------------------------------------------- framework + method
if [[ ! -d $NERFICG_ROOT ]]; then
  git clone https://github.com/nerficg-project/nerficg.git --recursive "$NERFICG_ROOT"
fi
if [[ ! -d $NERFICG_ROOT/src/Methods/FasterGS4D ]]; then
  git clone --single-branch -b FasterGS4D \
    https://github.com/nerficg-project/faster-gaussian-splatting.git \
    "$NERFICG_ROOT/src/Methods/FasterGS4D"
fi
cd "$NERFICG_ROOT"

# ---------------------------------------------------------------- host compiler shims
# nvcc invokes plain `gcc`/`g++`; distros that ship only versioned binaries need symlinks.
mkdir -p .toolchain/bin
for tool in gcc g++ gcc-ar gcc-nm gcc-ranlib; do
  src=/usr/bin/${tool}-${HOST_GCC}
  [[ -x $src ]] && ln -sfn "$src" ".toolchain/bin/${tool}"
done
ln -sfn /usr/bin/gcc-${HOST_GCC} .toolchain/bin/cc
ln -sfn /usr/bin/g++-${HOST_GCC} .toolchain/bin/c++

# ---------------------------------------------------------------- local CUDA toolkit
# Only the components needed to build PyTorch CUDA extensions: the compiler driver and its
# device backend (cuda_nvcc + cuda_crt + libnvvm + libnvptxcompiler), the runtime headers
# and stubs (cuda_cudart), and Thrust/CUB (cuda_cccl). ~600 MB unpacked, no root required.
CUDA_HOME="$NERFICG_ROOT/.toolchain/cuda-${CUDA_VERSION%.*}"
if [[ ! -x $CUDA_HOME/bin/nvcc ]]; then
  mkdir -p "$CUDA_HOME"
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  base=https://developer.download.nvidia.com/compute/cuda/redist
  curl -fsSL -o "$tmp/redist.json" "$base/redistrib_${CUDA_VERSION}.json"
  python3 - "$tmp/redist.json" > "$tmp/paths.txt" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
for component in ['cuda_nvcc', 'cuda_crt', 'libnvvm', 'libnvptxcompiler', 'cuda_cudart', 'cuda_cccl']:
    print(manifest[component]['linux-x86_64']['relative_path'])
PY
  while read -r path; do
    archive=$(basename "$path")
    curl -fsSL -o "$tmp/$archive" "$base/$path"
    tar xf "$tmp/$archive" -C "$tmp"
    cp -a "$tmp/${archive%.tar.xz}/." "$CUDA_HOME/"
  done < "$tmp/paths.txt"
  [[ -e $CUDA_HOME/lib64 ]] || ln -sfn lib "$CUDA_HOME/lib64"  # torch looks for lib64
fi

# ---------------------------------------------------------------- python environment
# Replaces environments/py311_cu128.yaml. `pip` is in the list because NeRFICG's
# scripts/install.py shells out to it to build the method's CUDA extensions.
if [[ ! -f pyproject.toml ]]; then
  cat > pyproject.toml <<'TOML'
[project]
name = "nerficg-workspace"
version = "2.0.0"
description = "NeRFICG framework environment (uv-managed, replaces the conda environments/*.yaml)"
requires-python = "==3.11.*"
dependencies = []

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
TOML
fi
uv venv --python 3.11
uv add "torch==${TORCH_VERSION}" torchvision numpy tqdm natsort pyyaml munch tabulate wandb \
       opencv-python kornia torchmetrics lpips einops "setuptools==80.10.2" plyfile matplotlib \
       timm plotly pillow jax pyproj scikit-learn pycolmap \
       "cuda-python==12.8.*" PySDL3 numpy-quaternion platformdirs imgui-bundle pyopengl pip

# ---------------------------------------------------------------- toolchain env file
# uv owns Python and the venv, not the CUDA SDK; these settings are handed to each command
# with `uv run --env-file .env`. uv sets PATH itself and ignores any PATH set here, so every
# tool is named absolutely: torch calls $CUDA_HOME/bin/nvcc and forwards $CC to it via -ccbin.
cat > .env <<EOF
CUDA_HOME=$CUDA_HOME
CUDA_PATH=$CUDA_HOME
LD_LIBRARY_PATH=$CUDA_HOME/lib64
CC=$NERFICG_ROOT/.toolchain/bin/gcc
CXX=$NERFICG_ROOT/.toolchain/bin/g++
CUDAHOSTCXX=$NERFICG_ROOT/.toolchain/bin/g++
TORCH_CUDA_ARCH_LIST=12.0
MAX_JOBS=8
EOF

# ---------------------------------------------------------------- build CUDA extensions
uv run --env-file .env python ./scripts/install.py -m FasterGS4D

echo "Done. Train with:"
echo "  cd $NERFICG_ROOT && uv run --env-file .env python ./scripts/train.py -c configs/<config>.yaml"

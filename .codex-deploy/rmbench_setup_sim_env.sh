#!/usr/bin/env bash
set -euo pipefail

RMBENCH_ROOT=/workspace/RMBench
SIM_ENV=${RMBENCH_ROOT}/.venv-sim
BASE_PYTHON=/workspace/MIKASA-Robo/.venv/bin/python
CUROBO_ROOT=${RMBENCH_ROOT}/third_party/curobo
UV_CACHE_DIR=/workspace/uv_cache

test -x "${BASE_PYTHON}"
test -f "${CUROBO_ROOT}/setup.py"

if [[ ! -x "${SIM_ENV}/bin/python" ]]; then
    uv venv --python "${BASE_PYTHON}" "${SIM_ENV}"
fi

site_packages=$("${SIM_ENV}/bin/python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')
install -m 0644 /opt/supervisor-scripts/rmbench_mikasa_base.pth \
    "${site_packages}/rmbench_mikasa_base.pth"

# Install only simulator-specific overrides in the overlay.  The shared base
# supplies the Blackwell-capable Torch 2.8/CUDA 12.8 stack without being
# modified by RMBench's incompatible pinned requirements.
uv pip install --python "${SIM_ENV}/bin/python" --no-deps \
    'mplib==0.2.1' \
    'open3d==0.19.0' \
    'numpy-quaternion==2024.0.13' \
    'yourdfpy==0.0.60' \
    'scikit-image==0.25.2' \
    'warp-lang==1.12.0' \
    pybind11 \
    setuptools-scm \
    wheel \
    importlib-resources \
    pyquaternion \
    ninja \
    addict \
    configargparse \
    plotly \
    lazy-loader \
    vcs-versioning \
    'scikit-learn==1.9.0' \
    joblib \
    threadpoolctl

# Open3D imports its Dash visualization module eagerly, even though RMBench
# only uses point-cloud containers.  Install Dash and its lightweight web
# dependencies in the overlay so importing Open3D remains deterministic.
uv pip install --python "${SIM_ENV}/bin/python" 'dash==4.4.1'

export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
uv pip install --python "${SIM_ENV}/bin/python" \
    --no-deps \
    --no-build-isolation \
    --editable "${CUROBO_ROOT}"

cd "${RMBENCH_ROOT}"
PYTHONDONTWRITEBYTECODE=1 "${SIM_ENV}/bin/python" - <<'PY'
import mplib
import numpy
import open3d
import sapien
import torch
import warp
from curobo.types.math import Pose

assert mplib.__version__ == "0.2.1"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)
print(
    "RMBench simulator imports complete:",
    f"python_torch={torch.__version__}",
    f"numpy={numpy.__version__}",
    f"sapien={sapien.__version__}",
    f"mplib={mplib.__version__}",
    f"open3d={open3d.__version__}",
    f"warp={warp.__version__}",
    f"curobo_pose={Pose.__name__}",
)
PY

du -sh "${SIM_ENV}" "${CUROBO_ROOT}"
df -h "${RMBENCH_ROOT}"

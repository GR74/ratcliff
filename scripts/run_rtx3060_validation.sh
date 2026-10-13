#!/usr/bin/env bash
set -euo pipefail
# Reuse the isolated WSL environment; never install a Linux NVIDIA driver.
task_base=${RATCLIFF_LAPTOP_BASE:-/mnt/d/ratcliff-3060}
task_python="$task_base/.venv/bin/python"
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$task_base/tmp" "$task_base/cuda-cache" "$task_base/mpl-cache"
export TMPDIR="$task_base/tmp"
export CUDA_CACHE_PATH="$task_base/cuda-cache"
export MPLCONFIGDIR="$task_base/mpl-cache"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
unset LD_LIBRARY_PATH
cd "$task_root"
exec "$task_python" scripts/validate_early_stopping.py "$@"

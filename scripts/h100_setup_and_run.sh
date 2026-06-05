#!/usr/bin/env bash
# Stage 5 — One-shot setup + benchmark for any rented GPU box.
#
# Usage (Vast.ai On-Start Script field, RunPod startup, Lambda Labs SSH):
#     export REPO_URL="https://github.com/YOUR_USERNAME/ratcliff.git"   # ← edit this
#     bash <(curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/ratcliff/main/scripts/h100_setup_and_run.sh)
#
# Or, if the repo is already cloned:
#     bash scripts/h100_setup_and_run.sh
#
# What it does:
#   1. Install git/python if missing
#   2. Clone repo (if not already)
#   3. Set up venv with JAX CUDA + project deps
#   4. Confirm GPU is visible
#   5. Run the Model B benchmark
#   6. Save results to /workspace/h100_results.txt

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/ratcliff.git}"
WORKDIR="${WORKDIR:-/workspace}"
RESULTS_OUT="${RESULTS_OUT:-$WORKDIR/h100_results.txt}"

echo "=============================================================="
echo "Stage 5 H100 Benchmark — Ratcliff Model B"
echo "=============================================================="
echo "Repo:     $REPO_URL"
echo "Workdir:  $WORKDIR"
echo "Results:  $RESULTS_OUT"
echo "=============================================================="

# 1. Install system deps (only if missing)
if ! command -v git >/dev/null 2>&1; then
  echo "[setup] Installing git..."
  apt-get update -qq && apt-get install -y -qq git
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] Installing python3..."
  apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip
fi

# 2. Clone repo if needed
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -d ratcliff ]; then
  echo "[setup] Cloning $REPO_URL..."
  git clone "$REPO_URL" ratcliff
else
  echo "[setup] ratcliff dir exists — pulling latest..."
  cd ratcliff && git pull --quiet && cd ..
fi
cd ratcliff

# 3. Set up venv + install
echo "[setup] Creating venv + installing deps (this is the slow part, ~3-5 min)..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -U "jax[cuda12]"
pip install --quiet -e ".[dev,fit]"

# 4. Confirm GPU
echo "[verify] JAX devices:"
python -c "import jax; print('  ', jax.devices())"
GPU_CHECK=$(python -c "import jax; print(jax.default_backend())")
if [ "$GPU_CHECK" != "gpu" ]; then
  echo "!!! WARNING: JAX default backend is '$GPU_CHECK', not 'gpu'."
  echo "!!! Benchmark numbers will be CPU-meaningless."
  echo "!!! Continuing anyway in case devices are visible but backend is mislabeled."
fi

# 5. Run benchmark
echo
echo "=============================================================="
echo "Running benchmark — first-call JIT compile may take 2-5 min."
echo "=============================================================="
python scripts/h100_model_b_benchmark.py 2>&1 | tee "$RESULTS_OUT"

echo
echo "=============================================================="
echo "DONE. Results saved to $RESULTS_OUT"
echo "Send the contents of that file back to Claude for the write-up."
echo "=============================================================="

#!/usr/bin/env bash
# Run on the existing user-provided pod. This does not create or stop a pod.
set -euo pipefail
total_budget=${TOTAL_BUDGET_SECONDS:-3300}
if ! [[ "$total_budget" =~ ^[0-9]+$ ]] || ((total_budget < 120 || total_budget > 3300)); then
    echo 'TOTAL_BUDGET_SECONDS must be between 120 and 3300.' >&2
    exit 2
fi
# Cover installation and testing too, with five minutes left for downloading
# results and stopping the pod within the authorized one-hour rental.
if [[ ${RATCLIFF_BUDGET_GUARDED:-0} != 1 ]]; then
    export RATCLIFF_BUDGET_GUARDED=1
    exec timeout --signal=TERM --kill-after=15s "$total_budget" bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
started=$(date +%s)
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'jax[cuda12]==0.10.2' -e '.[dev,fit]'
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
.venv/bin/python -c 'import jax; d=jax.devices(); print(d); assert jax.default_backend()=="gpu" and "H100" in d[0].device_kind'
.venv/bin/python -m pytest shared/tests/test_blocked.py shared/tests/test_staged_fit.py \
    shared/tests/test_fit_integration.py shared/tests/test_benchmark_stats.py \
    shared/tests/test_benchmark_runner.py \
    model_a/tests/test_blocked_simulate.py model_b/tests/test_blocked_simulate.py -q
remaining=$((total_budget - $(date +%s) + started))
if ((remaining < 120)); then
    echo 'Setup used the time budget; no benchmark started.'
    exit 2
fi
.venv/bin/python scripts/benchmark_speedups.py --budget-seconds "$remaining" "$@"

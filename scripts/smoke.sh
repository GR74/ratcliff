#!/usr/bin/env bash
# Smoke test runner. Stages 1 + 2 + 3 + 3.5 + 4 gate.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python -c "import jax" 2>/dev/null; then
  echo "Error: JAX not importable. Activate the venv: source .venv/bin/activate" >&2
  exit 1
fi
pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py model_a/tests/test_objective_smoke.py model_a/tests/test_objective_parity.py model_a/tests/test_simulate_smooth_smoke.py model_a/tests/test_objective_smooth_smoke.py model_a/tests/test_fit_smoke.py model_b/tests/test_grf_smoke.py model_b/tests/test_simulate_b_smoke.py model_b/tests/test_objective_b_smoke.py model_b/tests/test_fit_b_smoke.py -v

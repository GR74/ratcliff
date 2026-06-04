#!/usr/bin/env bash
# Smoke test runner. Stage 1 + 2 gate.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python -c "import jax" 2>/dev/null; then
  echo "Error: JAX not importable. Activate the venv: source .venv/bin/activate" >&2
  exit 1
fi
pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py -v

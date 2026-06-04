#!/usr/bin/env bash
# Smoke test runner. Stage 1 gate.
set -euo pipefail
cd "$(dirname "$0")/.."
pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py -v

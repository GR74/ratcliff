# Smoke test runner. Stage 1 gate.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
.\.venv\Scripts\python.exe -m pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py -v

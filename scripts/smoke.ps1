# Smoke test runner. Stage 1 + 2 gate.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    Write-Error "venv not found at .\.venv. Run: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e '.[dev]'"
    exit 1
}
.\.venv\Scripts\python.exe -m pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Ratcliff JAX Speedup

JAX port of two Ratcliff spatially-extended diffusion models:

- **Model A** — 1D, `N=72` accumulators, Cholesky-correlated Gaussian noise (`reference/gpgsq5deg3twod24.f`).
- **Model B** — 2D, `100×160` accumulator grid, FFT-based circulant-embedding GRF noise (`reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`).

Targets: laptop (CPU dev), 64-core workstation (CPU production), H100 supercomputer (GPU production). Same source, one JAX install per machine.

See `docs/plans/2026-06-03-ratcliff-speedup-design.md` for the full design and rationale.

## Quick start

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
.\scripts\smoke.ps1
```

If PowerShell blocks `.ps1` scripts, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke.ps1
```

First smoke run takes 30–90 seconds (JAX JIT compile); subsequent runs are 9–14 seconds on a warm cache. A successful run ends with a `N passed` summary line (currently `25 passed`).

On the Linux workstation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
./scripts/smoke.sh
```

## Stage status

- [x] Stage 0 — repo init, env setup
- [x] Stage 1 — validation infrastructure, smoke tests pass on laptop CPU
- [ ] Stage 2 — single-GEMM + cumsum rewrite of `model_a/simulate.py`
- [ ] Stage 3 — `vmap` `fofs` over conditions, L-BFGS optimizer, simplex fallback
- [ ] Stage 4 — Model B GRF + accumulator port
- [ ] Stage 5 — benchmark report across laptop / 64-core / H100
- [ ] Stage 6 (optional) — Triton kernel for per-step scan

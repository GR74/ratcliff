# Ratcliff JAX Speedup

JAX port of two Ratcliff spatially-extended diffusion models:

- **Model A** — 1D, `N=72` accumulators, Cholesky-correlated Gaussian noise (`reference/gpgsq5deg3twod24.f`).
- **Model B** — 2D, `100×160` accumulator grid, FFT-based circulant-embedding GRF noise (`reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`).

Targets: laptop (CPU dev), 64-core workstation (CPU production), H100 supercomputer (GPU production). Same source, one JAX install per machine.

See `docs/plans/2026-06-03-ratcliff-speedup-design.md` for the full design and rationale.

## Status

Stage 0 — repo init. See design doc §6 for the staged plan.

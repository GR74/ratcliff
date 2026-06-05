# Stage 4 — Completion Summary

**Date:** 2026-06-04 → 2026-06-05
**HEAD:** `d2b3a04` (smoke runner update)
**Status:** Implementation complete. End-to-end verified per-task. Slow synthetic-recovery test deferred.

## What shipped

| Component | File | Verified |
|---|---|---|
| Data parser (2-cond × 5-cat × 5-quantile) | `shared/data_io.py::load_twod3datanew` | 2 tests pass; subject 0 shape (2, 5) |
| Archive Fortran outputs | `reference/archive_outputs/benchtwod3mpi.{1..6}` + `twod3parallelmpi.out` | files committed |
| Spectral sqrt of circulant kernel | `model_b/grf.py::calc_LAM` | 4 tests pass; JIT-friendly after Task 4.C.x fix |
| Positive-definite gate (eager) | `model_b/grf.py::assert_pd_embedding` | 2 tests pass; raises ValueError for s>17.95 |
| 2-GRFs-per-FFT generator | `model_b/grf.py::circulant_grf` | 3 tests pass; empirical variance 0.997 (theoretical 1.0) |
| Constants + drift bumps + 5-zone array | `model_b/simulate.py` (top) | 4 tests pass; (50,80)→cat 1, (0,0)→cat 5 |
| Per-chunk 2D accumulator simulator | `model_b/simulate.py::_simulate_chunk_b` | 3 tests pass; cats in {1..5} |
| JIT'd full-nsim wrapper | `model_b/simulate.py::simulate_b` | 3 tests pass; deterministic for same key |
| 5-category G² + bounds | `model_b/objective.py` | 4 tests pass; clamp_b caps sig at 17.0 |
| Vectorized 2-condition objective | `model_b/objective.py::fofs_b_new` | 3 tests pass; G²=4162.12 on real twod3datanew[0] |
| Simplex fit driver | `model_b/fit.py::fit_simplex_b` | 1 fast structural test pass |
| Synthetic recovery (slow) | `model_b/tests/test_fit_b_smoke.py` (slow mark) | not run; deferred |

## Stage 4 success criteria check

1. **`model_b/grf.py`, `model_b/simulate.py`, `model_b/objective.py`, `model_b/fit.py` exist** ✅
2. **GRF empirical variance ~1.0** ✅ (0.997 measured at 200 samples)
3. **`simulate_b` runs end-to-end at small `nsim` on laptop CPU** ✅ (no memory blowup at nsim=16, chunk_size=4)
4. **`fofs_b_new` returns finite scalar on real `twod3datanew` data** ✅ (subject 0: 4162.12)
5. **`fit_simplex_b` recovers synthetic params** ⏸ deferred (slow test; structural test confirmed fit driver returns FitResult correctly)
6. **`model_a/` files UNTOUCHED** ✅ (no commits to model_a/ since Stage 3)
7. **`scripts/smoke.{ps1,sh}` updated to include Model B tests** ✅ (commit `d2b3a04`)

## Key engineering decisions

1. **Skipped the FFT-per-2-steps F1/F2 trick.** The Fortran uses it to halve FFT count. JAX implementation calls `circulant_grf` once per step, uses F1 only. Simpler code; the trick is documented as a Stage 5 GPU-perf optimization if needed.

2. **JIT-friendliness of `calc_LAM`.** The original implementation had a Python `raise ValueError` for positive-definite failures, which blocked JIT tracing. Task 4.C.x refactored: `calc_LAM` is now pure (no raise), and the explicit check lives in a separate `assert_pd_embedding` function. `clamp_b` upstream caps sig at 17.0, well within the empirical 17.95 ceiling, so production-fit workflows are safe without an explicit check.

3. **`simulate_b` static_argnums shrunk** from `(8, 9, 10, 11, 12)` to `(11, 12)` after the JIT fix. Now sis/sig/si trace as runtime args — the simplex fit can perturb them without triggering re-compile.

4. **Default `chunk_size=4`** on laptop CPU. Per-chunk peak working set ~3 MB. H100 will scale to chunk_size=64-128. Memory math documented in `simulate.py` docstring.

5. **Simplex only** (no L-BFGS). Gradient-zero issue from Stage 3 applies equally to Model B (same indicator-CDF in G² objective). Stage 3.5 may revisit with smooth surrogates or REINFORCE.

## What's deferred / out of scope

- **Synthetic recovery test pass.** The `@pytest.mark.slow` test was committed but not run to completion in this session. Estimated wall-clock: 5-15 minutes on laptop. Will exercise it when needed.
- **Parity against Fortran recorded outputs.** Archive files at `reference/archive_outputs/` are committed but no parser/comparison test against them yet. Future task.
- **FFT-per-2-steps trick.** Documented Stage 5 optimization.
- **GPU benchmarks.** Stage 5.
- **L-BFGS / gradient-based optimizer.** Awaits Stage 3.5 (smoothing or REINFORCE).
- **Fortran validation (Stage 2.5).** One-time gate, separate effort.

## Test totals

- Tests collected at completion: **81** (80 fast + 1 slow recovery)
- Stage 4 added (since `af8035d`): **~30 tests**
- Effective default smoke: 80 tests (slow test auto-skipped without `-m slow`)

## Stage 4 wall-clock observations

| Scale | Wall-clock per simulate call | Notes |
|---|---|---|
| nsim=16, chunk=4 on laptop CPU | ~11 s | warm cache; cold compile ~40-60 s |
| nsim=128, chunk=8 on laptop CPU | ~30-90 s (estimate) | per-call basis; fit needs many calls |
| nsim=9000, chunk=128 on H100 | ~1-3 s (estimate) | Stage 5 will measure |

Simplex fit on laptop = many calls × per-call cost = several hours per fit. H100 deployment via Stage 5 should bring fit time to minutes.

## Forward-looking items for Stage 5+

1. **GPU benchmarks** — measure simulate_b and fit_simplex_b on H100.
2. **Restore F1/F2 FFT-per-2-steps trick** if GPU FFT throughput is the bottleneck.
3. **Stage 2.5 Fortran validation** for jax_port (Model A side).
4. **Stage 3.5 smoothing for L-BFGS** (sigmoid CDFs / soft argmax) if simplex turns out to be too slow even on H100.
5. **Parity against archive's recorded `benchtwod3mpi.{1..6}`** — actually verify the JAX simulator's aggregates match those outputs.
6. **`fofs_b_new`'s `float(sis)/float(sig)/float(si)` conversions** are now redundant (sis/sig/si are no longer static_argnums of simulate_b). Cleanup task.
7. **Smoke runner PowerShell exit-code reliability** — the JAX backend access-violation noise on Windows fakes exit 1 even on green runs. Investigate `2>$null` redirect or move to bash-on-WSL for CI.
8. **Hierarchical / Bayesian fits.** Once the deterministic fit works, NumPyro / BlackJAX wraps cleanly around `fofs_b_new`.

# Model B — Stage 4 Design Sketch: 2D GRF Diffusion Port

**Date:** 2026-06-04
**Status:** SKETCH — directional, not yet a committed implementation plan
**Scope:** Port `benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS` (2D Gaussian random field diffusion model) to JAX. This is the "took weeks to run" model in the original Fortran archive.

---

## 1. Background

Model B is structurally different from Model A:

| Dimension | Model A | Model B |
|---|---|---|
| Spatial axis | 1D, `N=72` accumulators | 2D, `100 × 160` accumulator field |
| Correlated noise | GP kernel + Cholesky | Circulant embedding + FFT |
| Timesteps | 400 | 400 |
| Trials | 4000 | 9000 |
| Categories | 3 (position bands) | 5 (2D regions via `k(i,j)`) |
| Drift bumps | 1 Gaussian (`av * Normal(i; U, si)`) | 3 Gaussian bumps (`av1, av2, av3` at different `(uj, ui)`) |
| Parameters | 10 | 13 |
| Conditions | 4 | 2 (each with all 3 drifts) |
| Parallelism (Fortran) | OpenMP threads | MPI + OpenMP |
| Reference doc | gpgsq5deg3twod24.f + twod24_jax.py | benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS + Russ Childers' README_random_field |

The key algorithmic asset in Model B is the FFT-based 2D GRF generator using **circulant embedding** (Dietrich & Newsam method, Kroese §2.2 of MCSpatial.pdf). The Fortran's clever optimization: each FFT call produces **two** independent GRFs (real and imaginary parts of the complex output), so the time loop uses one GRF on odd steps and the cached second GRF on even steps — halves the FFT count.

## 2. Goals

1. `model_b/grf.py` — `calc_LAM` (one-time spectral sqrt) and `circulant_grf` (per-sim FFT giving two GRFs).
2. `model_b/simulate.py` — 2D accumulator simulator with the FFT-per-two-steps trick.
3. `model_b/objective.py` — vectorized 5-category G² objective.
4. `model_b/fit.py` — simplex driver (deferring L-BFGS pending Stage 3.5 resolution).
5. Validation: aggregate statistics match the archive's recorded Fortran outputs (`benchtwod3mpi.1` through `benchtwod3mpi.6`).
6. Wall-clock target: one full fit in **under 30 min on H100** (down from "weeks").

## 3. Non-goals

- No new optimizer infrastructure (use simplex until Stage 3.5 resolves gradients).
- No re-derivation of the circulant kernel — copy the formula from the Fortran exactly.
- No MPI/multi-node — single H100 + vmap should be enough.
- No GPU benchmarking infrastructure — Stage 5.
- No backwards compatibility with the original Fortran's binary file format.

## 4. Architecture

### 4.1 Files

```
model_b/
├── __init__.py           (exists, empty)
├── grf.py                # NEW — circulant embedding + FFT GRF generator
├── simulate.py           # NEW — 2D accumulator simulator
├── objective.py          # NEW — 5-category G² objective
├── fit.py                # NEW — simplex fit driver
└── tests/
    ├── __init__.py       (exists, empty)
    ├── test_grf_smoke.py            # NEW
    ├── test_simulate_b_smoke.py     # NEW
    ├── test_simulate_b_parity.py    # NEW (against archive's recorded outputs)
    ├── test_objective_b_smoke.py    # NEW
    └── test_fit_b_smoke.py          # NEW
shared/
├── data_io.py            # MODIFIED — add `load_twod3datanew` parser
└── tests/test_data_io.py # MODIFIED — add 5-cat / 2-cond parsing tests
```

### 4.2 GRF algorithm — `model_b/grf.py`

Two functions, mirroring the Fortran subroutines `calc_LAM` and `circulant_grf`.

**`calc_LAM(n=100, m=160, s1=15.0, s2=15.0)` — one-time per (sig) value:**

1. Build the autocovariance kernel at displacements `(dx, dy)`:
   ```
   ρ(x, y) = (1 - x²/s2² - xy/(s1·s2) - y²/s1²) · exp(-(x²/s2² + y²/s1²))
   ```
   (This is the Matern-class kernel from Kroese §2.2.)

2. Embed in a `(2n-1) × (2m-1)` block-circulant matrix.

3. One 2D FFT (`jnp.fft.fft2`).

4. Take the real part, clip negatives to zero (assert positive-definite embedding), take sqrt.

Returns `LAM` of shape `(2n-1, 2m-1)`.

**`circulant_grf(LAM, gauss_pairs)` — per simulation, returns two GRFs:**

1. Build complex array `X = LAM * (g1 + 1j*g2)` where `g1, g2` are iid normals of shape `(2n-1, 2m-1)`.

2. One 2D FFT.

3. Return `(F1, F2) = (X.real[:n, :m], X.imag[:n, :m])` — both shape `(n, m)`, both independent samples from the GRF.

The Fortran's `gauss_pairs` is an `(2*(2n-1)*(2m-1),)` array of iid normals; we'll pass it as a `(2, 2n-1, 2m-1)` JAX array.

**Critical constraint (from Russ's README_random_field):** At field size 100×160, the spectral sqrt fails when `s1, s2 > 17.95`. The kernel becomes non-positive-definite as the autocorrelation length approaches the field size. Add a runtime check in `calc_LAM` that raises `ValueError` if any `LAM[i,j] < -1e-10`, mirroring the Fortran's `Could not find positive definite embedding!` error.

### 4.3 Simulator — `model_b/simulate.py`

Mirrors `_simulate_chunk` from Model A but in 2D and with the FFT-per-two-steps trick:

```python
def _simulate_chunk_b(key, ter, st, cr, crsd, av1, av2, av3, sis, sig, chunk_size):
    # Build drift bumps v1, v2, v3 (each shape (n, m)) at fixed positions
    # Build LAM = calc_LAM(sig, sig)
    # Per chunk:
    #   - Generate iid normals for (NSTEP/2 + 1) FFT calls per trial.
    #   - On odd steps, do circulant_grf → use F1, cache F2.
    #   - On even steps, use the cached F2.
    #   - Accumulate: a = a + av1·v1 + av2·v2 + av3·v3 + agrf - mean(a)
    #   - Track first crossing as in Model A.
    # Return (rt, cat) where cat ∈ {1,2,3,4,5} from k[id(1), id(2)].
```

The `k[i,j]` 5-category zone array is precomputed from the drift bump positions — it's the categorical "this position is in zone X" classifier. Same logic as Fortran `accum` lines 432-450.

### 4.4 Memory budget

This is the dangerous part for Model B. At one trial × NSTEP=400 × 100×160 × fp64:
- `a`: 51 MB
- `agrf` per step: 51 MB
- LAM: ~0.5 MB (fixed)
- Per-FFT working space: ~1.5 MB

So per-trial working set is ~150 MB. For chunked vmap:
- chunk_size=4: 600 MB. Probably the max for laptop CPU.
- chunk_size=16: 2.4 GB. Workstation 64-core, OK if 64 GB RAM.
- chunk_size=128: 19 GB. H100 only.

Smaller default than Model A. The plan should expose `chunk_size` as a tunable.

### 4.5 Objective — `model_b/objective.py`

5 categories instead of 3. 2 conditions (each uses all three drifts; the conditions differ only by parameter values — Fortran lines 280-300 show two "if ij.eq.1" / "if ij.eq.2" blocks setting `av1, av2, av3` from different param indices).

The 2-condition vmap structure is simpler than Model A's 4-condition COND_MAP, but the per-condition G² has 5 categories not 3.

`condition_g2_vectorized_b` mirrors Model A's version but with `MC=5`. The `PQQ` array is still `[.1, .2, .2, .2, .2, .1]` (NQ=5 quantiles → 6 defective bins).

### 4.6 Validation strategy

**Tier 1: GRF aggregate match against analytic kernel.** For a fixed (s1, s2), generate 1000 GRFs via `circulant_grf`, compute empirical covariance at known displacements, compare to the analytic ρ(dx, dy). Tolerance ~5% relative.

**Tier 2: Simulator parity against archive's recorded Fortran outputs.** The archive at `reference/` (after copying from the unzipped benchtwod3) has `benchtwod3mpi.1` through `benchtwod3mpi.6` — recorded Fortran outputs at known parameter values. Parse those and compare proportions + quantiles per category to JAX simulator output.

**Tier 3: Synthetic parameter recovery.** Same pattern as Model A — generate data with known params, simplex-fit, recover within ±15%.

## 5. Risk surface

| Risk | Likelihood | Mitigation |
|---|---|---|
| Memory blowup on laptop at any nontrivial chunk size | high | Default chunk_size=2; document expected H100 vs laptop usage |
| LAM positive-definite check fires unexpectedly | medium | Document the s1, s2 < 17.95 ceiling; consider auto-shrinking sig in clamp |
| JAX `jnp.fft.fft2` numerical differences vs MKL DFTI cause Tier 2 failures | medium | Aggregate-statistical match (not bit-exact); should be robust |
| The "two GRFs per FFT" trick breaks vmap semantics (chunk-level caching of F2) | medium | Use `jax.lax.scan` with two-step strides, or pre-generate all FFTs upfront (more memory) |
| Recorded Fortran outputs use a slightly different parameter set than what's documented in the .f file | medium | First inspect the parinp file and Fortran source carefully before trying to match aggregates |
| 5-category indicator-CDF in objective kills gradients (same issue as Stage 3) | certain | Don't try L-BFGS for Model B. Simplex only. Document. |

## 6. Implementation stages (sketch)

1. **Stage 4.A** — copy archive's `twod3datanew`, `parinp`, `benchtwod3mpi.1`-`6` into `data/` and `reference/`. Add `shared/data_io.load_twod3datanew`.
2. **Stage 4.B** — implement `model_b/grf.py::calc_LAM` and `circulant_grf`; smoke tests + GRF aggregate match.
3. **Stage 4.C** — implement `model_b/simulate.py::_simulate_chunk_b` (one chunk, no vmap yet); smoke tests for shape and determinism.
4. **Stage 4.D** — wrap `simulate_b()` over chunks via `lax.map`. Smoke + parity against Model A-style smoke tests.
5. **Stage 4.E** — implement `model_b/objective.py::fofs_b_new`; parity against archive's recorded `benchtwod3mpi.N` outputs.
6. **Stage 4.F** — implement `model_b/fit.py::fit_simplex_b`; synthetic recovery test.
7. **Stage 4.G** — update smoke runner.
8. **Stage 4.H** — Stage 4 completion gate.

Each stage is multi-task in the bite-sized TDD sense. Plan into ~12-15 implementation tasks total.

## 7. Success criteria

Stage 4 is "done" when ALL of:

1. `model_b/grf.py` produces GRFs whose empirical covariance matches the analytic kernel within 5% relative.
2. `model_b/simulate.py::simulate_b` runs end-to-end on real `twod3datanew` data with `nsim=512` on laptop CPU (no memory blowup).
3. JAX simulator aggregates match Fortran recorded outputs within 2% absolute (proportions) and 3% relative (quantiles).
4. Synthetic parameter recovery test passes within ±15% per parameter via simplex.
5. `model_a/jax_port.py`, `model_a/simulate.py`, `model_a/objective.py` UNTOUCHED.
6. `scripts/smoke.{ps1,sh}` updated to include Model B tests.
7. Forward-looking items captured for Stage 5.

## 8. Stage 5 connection

Once Model B works on CPU, Stage 5 measures GPU speedup. The 2D GRF + FFT should benefit enormously from H100's cuFFT — likely 50-100× over CPU. Combined with simplex (gradient-free), a Model B fit could land at minutes-to-an-hour on H100, down from the original "weeks."

## 9. Decision log

- **Optimizer**: simplex only (gradient issue from Stage 3 applies here too). Decision-by-default until Stage 3.5 resolves smoothing.
- **Algorithm**: copy the Fortran's circulant embedding + FFT-per-two-steps verbatim. Same Kroese §2.2 reference.
- **Validation**: against archive's recorded outputs (no live Fortran needed). Stage 2.5's Fortran-validation work for Model A is independent.
- **`condition_g2` for 5 categories**: same indicator-CDF structure as Model A (will have zero gradient — fine, we're using simplex).
- **Memory chunking**: tiny default (2), documented; H100 can scale up.
- **GRF storage**: don't pre-allocate all FFTs (would be 16 GB at nsim=9000); use lax.scan with two-step caching.

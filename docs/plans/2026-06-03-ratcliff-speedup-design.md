# Ratcliff JAX Speedup — Design

**Date:** 2026-06-03
**Status:** Approved, ready for implementation planning
**Scope:** Port and accelerate two of Roger Ratcliff's spatially-extended diffusion / competing-accumulator models from Fortran+MKL+MPI to JAX, targeting laptop / 64-core workstation / H100 supercomputer.

---

## 1. Background

Two existing Fortran codebases implement related but distinct simulators:

- **Model A — 1D diffusion** (`gpgsq5deg3twod24.f`): `N=72` accumulators along one axis, correlated Gaussian noise via Cholesky factor of a GP kernel, `NSTEP=400`, `nsim=4000` trials, 3 response categories, 10-parameter fit via Nelder-Mead. A first-pass JAX port (`twod24_jax.py`) already exists and is marked "first faithful port — must be validated."

- **Model B — 2D Gaussian Random Field diffusion** (`benchtwod3mpi.f`): `100×160` accumulator field, correlated noise via 2D circulant embedding + MKL DFTI FFT, `NSTEP=400`, `nsim=9000`, 5 response categories, 13-parameter fit, hybrid MPI+OpenMP parallelization. Full fits previously took weeks. No JAX port exists yet.

Both models fit observed RT quantile data (`twod24data` for A, `twod3datanew` for B) by minimizing a G² objective on response proportions and quantiles.

## 2. Goals

Concrete deliverables ranked by importance:

1. JAX implementations of both models with the same source running on laptop (CPU), 64-core workstation (CPU), and H100 supercomputer (GPU).
2. Validation harness proving JAX output matches Fortran within Monte Carlo tolerance.
3. Wall-clock targets: Model A fit < 30 s on H100; Model B fit < 30 min on H100.
4. Drop-in gradient-based optimizer (L-BFGS) replacing the simplex, with simplex retained as fallback.
5. Benchmark report documenting speedups vs the original Fortran across all three hardware tiers.

## 3. Non-goals

Explicit scope boundaries (decisions logged from brainstorming):

- Not switching frameworks: not Rust, not Go, not Julia, not Mojo, not C+CUDA, not SpaceX's announced C stack.
- Not keeping MPI: one H100 replaces the 6-node cluster setup.
- Not re-implementing `SIMPLX` line-by-line: use `scipy.optimize.minimize(method='Nelder-Mead')` or a small JAX simplex for fallback parity.
- Not building a GUI or web service.
- Not pursuing Bayesian hierarchical fits in v1 (post-MVP follow-up).
- Not modifying the Fortran reference files — they remain in `reference/` as oracle.

## 4. Architecture

### 4.1 Repository layout

```
ratcliff/
├── README.md
├── pyproject.toml             # JAX, jaxopt, optax, scipy, pytest, matplotlib
├── data/
│   ├── twod24data
│   ├── twod3datanew
│   └── parinp
├── reference/                 # Untouched Fortran originals
│   ├── gpgsq5deg3twod24.f
│   └── benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS
├── model_a/                   # 1D diffusion
│   ├── simulate.py            # single-GEMM + cumsum + first-crossing
│   ├── objective.py           # fofs, condition_g2, vmap over conditions
│   ├── fit.py                 # L-BFGS primary, simplex fallback
│   └── tests/
├── model_b/                   # 2D GRF diffusion
│   ├── grf.py                 # calc_LAM + circulant_grf via jnp.fft.fft2
│   ├── simulate.py            # 2D accumulator with FFT-per-2-steps trick
│   ├── objective.py
│   ├── fit.py
│   └── tests/
├── shared/
│   ├── data_io.py             # parse twod24data / twod3datanew
│   ├── validation.py          # statistical aggregate match vs Fortran
│   ├── benchmark.py           # wall-clock + memory across CPU/GPU
│   └── prng.py                # PRNG helpers, deterministic key splits
├── notebooks/
│   ├── 01_validate_model_a.ipynb
│   ├── 02_speedup_ladder.ipynb
│   └── 03_bootstrap_ci.ipynb
└── docs/
    └── plans/
        └── 2026-06-03-ratcliff-speedup-design.md
```

### 4.2 Key algorithmic decisions

**Model A: collapse the time loop into one matmul + cumsum.**
The Fortran demeans the accumulator every timestep, so the accumulator path is exactly the cumulative sum of demeaned per-step increments. Pre-generate the full `(NSTEP, N)` Gaussian noise block, apply the Cholesky factor `L` as one big `Z @ L.T` GEMM, demean per step, cumsum along time, then find the first row where `a.max() > crr`. This removes the per-step state from `lax.scan` and reduces the simulator to three GPU-friendly ops: one GEMM, one cumsum, one argmax+threshold.

**Model B: port circulant-embedding FFT to `jnp.fft.fft2`.**
`calc_LAM` (one-time spectral square root via 2D FFT of the embedded autocovariance) and `circulant_grf` (per-sim FFT producing two independent GRFs from one FFT via real/imag parts) translate directly to JAX FFTs. Preserve the Fortran's clever trick of using one GRF on odd timesteps and the cached second GRF on even timesteps — halves the FFT count.

**Both models: `vmap` over trials, then over conditions.**
`one_trial` is written for one trial. `jax.vmap` stacks `(trials × conditions × subjects)` as batch dims, and `jax.jit` fuses the entire objective into one XLA program. Gradients of this single program come for free via `jax.grad(fofs)`.

### 4.3 Optimizer

L-BFGS via `jaxopt.LBFGS` (or `optax.scale_by_lbfgs` chain) is the default. Hyperparameters: `maxiter=200`, `tol=1e-6`. Simplex is retained behind an `--optimizer=simplex` flag for Fortran-compatible runs and as automatic fallback when L-BFGS gradient norm plateaus before convergence. Hybrid mode (`--optimizer=hybrid`: simplex coarse → L-BFGS refine) is offered for bumpy likelihoods.

## 5. Validation

**Truth source.** Fortran outputs at the same parameter point, run on the 64-core workstation. Recorded outputs from the `benchtwod3` archive (`twod3parallelmpi.out`, `benchtwod3mpi.1-6`) serve as backup oracle for Model B.

**Test.** For fixed parameters `x`, run `nsim=50_000` in Fortran and JAX. Compare:

| Quantity | Tolerance |
|---|---|
| Response proportions per category | ±0.5% absolute |
| 5 RT quantiles per category | ±1% relative |
| `fofs` total | ±2% |

**Gate.** All three pass → port is valid. Failure blocks all downstream speedups until resolved.

**Secondary check.** Synthetic parameter recovery: generate data with known params using the JAX simulator, fit, confirm recovery within Monte Carlo noise. Runs even if Fortran is unavailable.

**PRNG note.** MKL VSL and JAX PRNG cannot bit-match. Validation is aggregate-statistical, not per-trial deterministic. This is the standard correctness gate for ported Monte Carlo code.

## 6. Implementation stages

Each stage ends with a benchmark and a passing validation gate. No stage starts until the previous gate is green.

| Stage | Scope | Deliverable | Gate |
|---|---|---|---|
| 0 | Repo init, env setup, copy data and reference files | git repo + working JAX install on laptop | imports work |
| 1 | Wire `twod24_jax.py` into `model_a/`, run validation | `tests/test_validate_a.py` passes | aggregates match Fortran |
| 2 | Rewrite `model_a/simulate.py` as single-GEMM + cumsum + first-crossing | New simulator; validation still passes | ≥ 5× speed over Stage 1 |
| 3 | `vmap` `fofs` over 4 conditions; switch simplex → L-BFGS; simplex retained as flag | `model_a/fit.py` with both optimizers | L-BFGS converges within MC noise of simplex |
| 4 | Port Model B GRF + accumulator + objective to JAX | `model_b/*.py`; validation against archive outputs | aggregates match `twod3parallelmpi.out` |
| 5 | Benchmark all three tiers (laptop / 64-core / H100), write report | `shared/benchmark.py` + numbers in `docs/` | report committed |
| 6 (optional) | Triton kernel for per-step scan via `jax.experimental.pallas` | `model_a/simulate_triton.py` | ≥ 1.5× over Stage 5 to justify |

## 7. Expected outcomes

Conservative speedup ranges from the JAX port plus optimizer change:

### Model A (1D)

| Workload | Fortran (1 core) | JAX Stage 3 (H100) | Multiplier |
|---|---|---|---|
| One `simulate()` (4000 trials × 400 steps × 72 positions) | 1–3 s | 5–20 ms | 50–500× |
| One `fofs()` (4 conditions) | 5–15 s | 50–200 ms | 50–200× |
| Full simplex fit | tens of min | 30–90 s | 20–50× |
| Full L-BFGS fit | not possible in Fortran | 2–10 s | qualitative |

### Model B (2D GRF)

| Workload | Fortran (1 MPI node / 6 nodes) | JAX Stage 4 (H100) | Multiplier |
|---|---|---|---|
| One `simulate()` (9000 trials × 400 steps × 100×160 grid) | 36 s / 11 s | 0.5–2 s | 20–70× vs 1 node |
| One `fofs()` (2 condition blocks) | 1–3 min | 2–5 s | 30–60× |
| Full simplex fit (historical: weeks) | days to weeks | minutes to a few hours | 100–10000× |
| Full L-BFGS fit | not possible | tens of minutes | qualitative |

### New capabilities (qualitative)

- Free gradients → L-BFGS / Adam optimization instead of gradient-free simplex.
- Bootstrap confidence intervals via `vmap` over resampled datasets.
- Parameter recovery studies as routine afternoon experiments.
- Plug into `NumPyro` / `BlackJAX` for hierarchical Bayesian fits (post-MVP).
- GPU profile sweeps (e.g. 50×50 parameter grids) as a matter of course.
- Deterministic, reproducible runs via JAX PRNG keys.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| JAX aggregates don't match Fortran within tolerance | medium | Debug port (especially the `sv`/`as` dummy-arg flag and the data parser). Hold all speedups until fixed. |
| L-BFGS gets stuck on bumpy likelihood | medium | Hybrid mode (simplex coarse → L-BFGS refine) auto-engages if L-BFGS gradient norm plateaus |
| Fortran doesn't build on workstation | low-medium | Fall back to recorded archive outputs (Model B) + synthetic recovery (Model A) |
| H100 unavailable when needed | medium | Workstation tier still provides 10–50× over current Fortran. Not blocking. |
| Model B GRF memory blows up on H100 (batched FFTs over 9000 trials × 199 × 319 complex) | low | Chunked vmap: batch trials in groups of ~1000. Standard JAX pattern. |
| `pallas` API churn for Triton (Stage 6) | low | Only enter Stage 6 if Stage 5 benchmarks demand it. Defer the risk. |

## 9. Success criteria

The project is "done" when all of the following hold:

1. Model A fits one dataset in < 30 s on H100 (stretch: ≤ 10 s).
2. Model B fits one dataset in < 30 min on H100.
3. Validation tests for both models pass within tolerance.
4. The same source runs on laptop, 64-core workstation, and H100 with one config switch.
5. Benchmark report committed showing speedup numbers across all three tiers.

## 10. Decision log

- **Framework:** JAX. Considered and rejected: Rust, Go, Julia, Mojo, C+CUDA, SpaceX C stack. Reasoning: existing JAX port works; hot path is GEMM + FFT + cumsum which JAX already dispatches to cuBLAS / cuFFT; free autodiff and vmap; same source runs on all three target machines.
- **Triton:** Reserved as Stage 6 escape hatch (called from JAX via `pallas`), not a JAX replacement.
- **Validation:** Statistical aggregate match against Fortran, not per-trial deterministic (PRNG cannot bit-match between MKL VSL and JAX).
- **Optimizer:** L-BFGS primary, simplex retained as `--optimizer` flag and as auto-fallback.
- **Layout:** One repo with `model_a/`, `model_b/`, `shared/` subfolders.
- **MPI:** Dropped. One H100 replaces the 6-node cluster.

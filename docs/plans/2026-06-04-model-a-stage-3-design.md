# Model A — Stage 3 Design: vmap Conditions + L-BFGS Optimizer

**Date:** 2026-06-04
**Status:** Approved (autonomous mode, user-pre-approved scope)
**Scope:** Vectorize `fofs` over 4 conditions into a single fused JAX program; replace Nelder-Mead simplex with L-BFGS via `jaxopt`. Validate via parity against `jax_port.fofs` and via synthetic parameter recovery. This is where the headline speedup lives — 50–100× fewer fofs evaluations per fit.

---

## 1. Background

Stage 2 delivered a new fast simulator (`model_a/simulate.py`). What it didn't deliver was anything that uses `jax.grad`. The full objective `fofs` still lives in `jax_port.py` with two layers of Python loops:

- An outer loop over 4 conditions (`for ci, (di, bi) in enumerate(COND_MAP)`) calling `simulate` sequentially.
- An inner loop in `condition_g2` over MC=3 response categories and NQ=5 quantiles.

Both loops trace correctly under jit (they unroll at trace time), but they:
- Block `vmap`-over-conditions parallelism that would give 2–4× per-call.
- Prevent `jax.grad(fofs)` from being practical — the unrolled graph is large and Python-loop control flow makes scaling fragile.

The bigger story is what gradient-based optimization unlocks. Nelder-Mead simplex (the current Fortran-faithful optimizer) takes ~5,000–20,000 `fofs` evaluations to converge on a 10-parameter problem. L-BFGS with free gradients takes ~50–200. That's the 50–100× reduction that, combined with the Stage 2 per-call work and the future Stage 5 GPU deployment, lands the project at "fit one dataset in seconds."

## 2. Goals

1. New file `model_a/objective.py` with fully vectorized `fofs_new(params, data, key, nsim)`. Same numerical contract as `jax_port.fofs`.
2. New file `model_a/fit.py` with two optimizer drivers: L-BFGS (primary) via `jaxopt.LBFGS`, and Nelder-Mead simplex (fallback) via `scipy.optimize.minimize`.
3. Parity: `fofs_new` matches `jax_port.fofs` to MC tolerance at the same `(params, key, nsim)`.
4. Synthetic parameter recovery: generating data with known params and fitting recovers them within MC noise.
5. Wall-clock target: full L-BFGS fit completes in **under 60 seconds on laptop CPU** at `nsim=512`.

## 3. Non-goals

- Not touching `model_a/jax_port.py` (oracle).
- Not switching to `optimistix` (jaxopt is in `[fit]` extras, mature enough for our purposes; optimistix is the long-term successor but adds dependency churn here).
- Not adding Bayesian/MCMC fitting (NumPyro / BlackJAX — Stage 4+).
- Not handling Model B (Stage 4).
- Not adding hierarchical fits across subjects (post-MVP).
- Not adding GPU benchmark harness (Stage 5).
- Not fixing the `jax_port.one_trial` vs Fortran `pos`-at-NSTEP divergence (Stage 2.5).

## 4. Architecture

### 4.1 Files

```
model_a/
├── __init__.py
├── jax_port.py        # untouched (oracle)
├── simulate.py        # untouched (Stage 2 deliverable)
├── objective.py       # NEW — vectorized condition_g2 + fofs_new
├── fit.py             # NEW — L-BFGS + simplex drivers
└── tests/
    ├── (existing files untouched)
    ├── test_objective_smoke.py          # NEW — basic fofs_new tests
    ├── test_objective_parity.py         # NEW — fofs_new vs jax_port.fofs
    ├── test_fit_lbfgs_smoke.py          # NEW — L-BFGS recovery on synthetic data
    └── test_fit_simplex_smoke.py        # NEW — simplex recovery on synthetic data
```

### 4.2 `model_a/objective.py` — vectorized condition_g2 + fofs_new

**`condition_g2_vectorized(rt, cat, obs_prop, obs_count, obs_quant)`** — same scalar G² output as the existing `condition_g2`, but uses:
- `jnp.where`-based category masking instead of Python `for i in range(MC)`.
- Reduces over categories with `jnp.sum(... ,axis=0)` and a `jax.lax.cond`-style switch between full-quantile and lumped terms based on `obs_count >= NCUT`.
- No Python loops in the hot path.

**`fofs_new(params, data, key, nsim)`** — same signature, vmap'd over conditions:
- `clamp(params)` → bounded parameter vector (reusing `jax_port.clamp` semantics).
- Build `(4,)` arrays for the 4-condition `(drift, boundary)` per `COND_MAP`.
- `vmap(simulate, in_axes=(0, None, None, 0, None, None, None, 0, None, None))` over `(key, ter, st, cr, crsd, si, sig, av, nsim, chunk_size)` — keys are folded per condition, then `cr` and `av` vary per condition.
- Compute G² per condition with `vmap(condition_g2_vectorized)`.
- Sum to a scalar.

The objective is fully JIT'd. `jax.grad(fofs_new)` produces a scalar-valued gradient w.r.t. the 10-parameter vector.

### 4.3 `model_a/fit.py` — L-BFGS primary, simplex fallback

**`fit_lbfgs(data, key, x0, nsim=512, maxiter=200, tol=1e-6)`**
- Wraps `jaxopt.LBFGS(fun=lambda p: fofs_new(p, data, key, nsim), maxiter=maxiter, tol=tol)`.
- `clamp()` is applied INSIDE `fofs_new`, so the optimizer can see any unbounded `params` and the projection happens internally. This is robust against L-BFGS proposing out-of-bounds points (which it will, sometimes).
- Returns `(best_params, final_loss, n_iters, converged)`.

**`fit_simplex(data, key, x0, nsim=512, maxiter=2000, tol=1e-7)`**
- Uses `scipy.optimize.minimize(method='Nelder-Mead', ...)`.
- Wraps a NumPy adapter around `fofs_new` (JAX → NumPy at each call).
- Slower (sequential, many evals), but gradient-free — robust on noisy/bumpy likelihoods.
- Same return type.

**`fit(data, key, x0, nsim=512, optimizer="lbfgs", **kwargs)`**
- Dispatcher: `optimizer in {"lbfgs", "simplex", "hybrid"}`.
- `"hybrid"` = simplex coarse pass (maxiter=50) → L-BFGS refine. Most robust.

### 4.4 Parameter bounds policy

`clamp(params)` already exists in `jax_port.py:175` and applies:
- `ter ≥ 175.0`, `st ∈ [20, 1.5*ter]`, `a1 ≥ 1.0`, `sa ∈ [0.01, a1/2]`, `sig ≥ 1.0`, `sv ≥ 0.3`, `d1,d2,a2 ≥ 0.01`.

Stage 3 reuses `jax_port.clamp` directly (importable; we're not modifying jax_port, just calling its function).

**Why not `LBFGSB` with explicit bounds?** Some bounds are dependent (`sa` depends on `a1`, `st` depends on `ter`). Box bounds can't express this. The `clamp` approach handles dependencies correctly.

## 5. Parity validation

### 5.1 `fofs_new` vs `jax_port.fofs` at same params

Two test tiers:

**Tier 1 — Scalar match within MC tolerance**

For each of 3 parameter regimes (realistic, high-drift, low-drift), compute `fofs_new` and `jax_port.fofs` at the same `(params, key, nsim=2048)`. Require `|new - old| / old < 0.05` (5% relative).

5% is appropriate because:
- The two implementations use different PRNG threading (jax_port splits keys per condition + per trial; objective.py vmaps a single key over conditions).
- At nsim=2048, the per-condition G² standard error is several percent.
- The categorical-bound and continuous-quantile mixed structure of G² makes it more sensitive to PRNG choice than raw proportions.

**Tier 2 — Gradient sanity**

`jax.grad(fofs_new)(params)` must produce a finite vector of length 10, not all zeros, with reasonable scale (no NaN, no Inf, magnitude similar to a numerical finite-difference estimate). This catches "gradients are broken" without requiring exact correctness.

### 5.2 Synthetic parameter recovery

Generate observed-data-like input from `jax_port.simulate` at known parameters `x_true`. Run `fit_lbfgs(data, x0=perturbed(x_true))` and verify recovered parameters are within ±10% of `x_true` per parameter. This is the end-to-end correctness gate for the fitter.

## 6. Implementation stages

| Stage | Scope | Gate |
|---|---|---|
| 3.A | Write `model_a/objective.py` skeleton (constants, signature) + skeleton smoke | smoke passes (import + signature) |
| 3.B | Implement vectorized `condition_g2_vectorized`; parity vs `jax_port.condition_g2` at one set of inputs | scalar G² matches within 1e-6 |
| 3.C | Implement `fofs_new` via vmap over conditions; parity vs `jax_port.fofs` at 3 param sets | within 5% relative |
| 3.D | Smoke test `jax.grad(fofs_new)` produces finite non-zero gradient | gradient test passes |
| 3.E | Write `model_a/fit.py` with `fit_lbfgs` driver | fit converges on synthetic data |
| 3.F | Add `fit_simplex` (scipy NM) as fallback | simplex converges (slower, same recovery) |
| 3.G | Add `fit_hybrid` (simplex coarse → L-BFGS refine) | hybrid converges |
| 3.H | Update smoke runner to include new tests | full smoke green |

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `fofs_new` doesn't match `jax_port.fofs` within 5% due to PRNG threading | medium | Diagnose if it's MC noise (raise to 10%) or algorithmic (debug); document tolerance choice |
| L-BFGS gets stuck on bumpy MC likelihood (especially at low nsim) | medium | Use hybrid mode as default fallback; raise `nsim` for the fit |
| `jax.grad(fofs_new)` produces NaN at parameter boundaries (clamp's max/clip create subgradient kinks) | medium | Add small smoothing in clamp if needed, or use `LBFGSB` with relaxed bounds |
| Synthetic recovery fails (params don't recover) | low–medium | If recovery error > 10%, increase `nsim` or `maxiter`; if still failing, investigate `condition_g2` mismatch |
| `jaxopt.LBFGS` API churn vs jax 0.10.1 | low | Already pinned `jax<0.12`; `jaxopt>=0.8` was Stage-2-test-installed |
| Synthetic recovery too slow (>5 min per fit) on laptop | medium | Lower `nsim` to 256 for tests; document expected wall-clock |
| Hybrid mode brittle (simplex doesn't help) | low | Hybrid is opt-in; default to L-BFGS pure |

## 8. Success criteria

Stage 3 is "done" when ALL of:

1. `model_a/objective.py` exists with `fofs_new` that takes 10-param vector, returns scalar.
2. `jax.grad(fofs_new)(params)` produces finite, non-zero gradient.
3. Parity tests against `jax_port.fofs` pass within 5% at 3 param regimes.
4. `fit_lbfgs` recovers known parameters in synthetic data within ±10%, in under 60 seconds on laptop.
5. `fit_simplex` recovers known parameters (slower acceptable; ~5–10 minutes ok).
6. `scripts/smoke.ps1` runs all tests green (target ~55–60 total).
7. New simulator (Stage 2) is the one being called inside `fofs_new` — close the loop on the Stage 2 deliverable.

## 9. Decision log

- **Optimizer**: jaxopt.LBFGS (primary), scipy Nelder-Mead (fallback), hybrid (opt-in). Optimistix deferred.
- **Bounds**: reuse `jax_port.clamp()` via projection inside `fofs_new`. Not `LBFGSB` because some bounds are dependent.
- **Vectorization**: vmap over conditions in `fofs_new`. `condition_g2_vectorized` removes the inner Python loops via masking + `lax.cond`.
- **PRNG threading**: single key passed to `fofs_new`; vmap folds-in the condition index internally. Different from jax_port's "fold_in per condition, split per trial" — accept 5% parity tolerance.
- **`fofs_new` uses Stage 2 simulator** (`model_a.simulate.simulate`), NOT `jax_port.simulate`. This is the load-bearing change that connects Stage 2 to the rest of the project.
- **`clamp` is shared with jax_port** — we import it; we don't duplicate it.

## 10. Expected outcomes

### Speedup chain (laptop CPU, conservative)

| Stage | Per-fofs cost | Evals/fit | Wall-clock/fit |
|---|---|---|---|
| Fortran simplex (baseline) | ~5 s | 5,000 | ~7 hours |
| jax_port + simplex | ~3 s | 5,000 | ~4 hours |
| Stage 2 + simplex | ~2.7 s | 5,000 | ~3.7 hours |
| **Stage 3 + L-BFGS (target)** | **~1.0 s** | **100** | **~100 s** |

Per-fofs cost drops because vmap-over-conditions fuses 4 sequential simulate calls into one program. Eval count drops because L-BFGS reads gradients. Combined: ~250× faster wall-clock fit on laptop CPU.

### On H100 (later, Stage 5)

Per-fofs cost: ~50 ms (estimate; depends on GPU benchmarking).
Eval count: ~50–100 (L-BFGS).
Wall-clock/fit: **~2.5–5 seconds**.

That's the "fit one dataset in under 30 seconds on H100" success criterion from the master design doc §9.

## 11. Stage 2.5 acknowledgement

The `pos`-at-NSTEP divergence between `jax_port.one_trial` and the Fortran is still pending Stage 2.5 resolution. Stage 3 is unaffected because parity is against `jax_port.fofs`, not against Fortran. Stage 2.5 can land at any time (one-time effort) before final production fits.

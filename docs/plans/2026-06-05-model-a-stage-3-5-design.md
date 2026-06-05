# Stage 3.5 Design — Smooth-Objective Surrogate for Gradient-Based Fitting (Model A)

**Date:** 2026-06-05
**Status:** Approved (autonomous mode)
**Scope:** Build a smooth, differentiable variant of `fofs_new` (`fofs_smooth`) so `jax.grad` returns finite, useful gradients. Wire it into `fit_lbfgs_smooth` and a `fit_hybrid` (smooth-LBFGS coarse → discrete-simplex polish) driver. Validate that the hybrid pipeline recovers known parameters faster than simplex-only.

---

## 1. Background

Stage 3 discovered that `jax.grad(fofs_new)` is structurally zero because every path from `params` to the scalar G² passes through one of three non-differentiable ops:

1. `jnp.argmax(crossed, axis=1)` — discrete first-crossing index
2. `cat = where(pos band)` — discrete category assignment
3. `(rt_i <= obs_quant[j,i]).sum()` — indicator-based empirical CDF (step function in rt)

Documented in `docs/notes/2026-06-04-stage-3-gradient-zero-finding.md`. Without gradients, L-BFGS is dead and the headline 100-1000× fit-wall-clock speedup is gone. Stage 3.5 restores the gradient via smooth surrogates.

## 2. Goals

1. New `model_a/simulate_smooth.py` returning soft `jstop`, soft `cat` probabilities, and the smooth `rt` derived from soft jstop.
2. New `model_a/objective_smooth.py::fofs_smooth(params, data, key, nsim, tau, sigma)` returning a smooth scalar G² loss.
3. `jax.grad(fofs_smooth)(params)` is finite and non-zero on at least 8 of 10 active parameters.
4. `fit_lbfgs_smooth(data, key, x0, ...)` converges in ≤ 100 iterations on synthetic data.
5. `fit_hybrid(data, key, x0, ...)` = smooth L-BFGS coarse → discrete simplex polish, recovers known params within ±15% in **under 60 seconds on laptop CPU** at `nsim=512`.

## 3. Non-goals

- Not modifying `model_a/jax_port.py` (oracle, immutable).
- Not modifying `model_a/simulate.py` (Stage 2 deliverable, parity-tested).
- Not modifying `model_a/objective.py` (Stage 3 deliverable, bit-exact parity).
- Not running on Model B in this stage (replicate in Stage 3.5.B if Model A succeeds).
- Not implementing REINFORCE / score-function (different approach, deferred).
- Not annealing temperatures (keep static; rely on hybrid polish to close the bias).
- Not adding GPU benchmarks (Stage 5).

## 4. Architecture

### 4.1 Files

```
model_a/
├── jax_port.py             # untouched
├── simulate.py             # untouched (Stage 2)
├── objective.py            # untouched (Stage 3)
├── simulate_smooth.py      # NEW — smooth-jstop, soft-cat simulator
├── objective_smooth.py     # NEW — fofs_smooth with sigmoid CDF, soft-cat aggregation
├── fit.py                  # NEW (Stage 3.5 lands fit drivers here, since none existed in Stage 3)
└── tests/
    ├── (existing files untouched)
    ├── test_simulate_smooth_smoke.py    # NEW
    ├── test_objective_smooth_smoke.py   # NEW — gradient finite + non-zero
    └── test_fit_smooth_smoke.py         # NEW — recovery test
```

### 4.2 `simulate_smooth.py` — smooth simulator

Returns per-trial:
- `soft_jstop` (chunk_size,) — expected first-crossing time as a smooth function of params
- `cat_probs` (chunk_size, 3) — soft membership in each response category (sums to 1 per trial)
- `rt` (chunk_size,) = (soft_jstop + ndt) * E — smooth in params

Algorithm: pre-generate all noise, run cumsum exactly like Stage 2's `simulate`. Then:

```python
# At each step t, "crossed_score_t" measures how much a.max() exceeds crr
crossed_score = a.max(axis=-1) - crr[:, None]    # (chunk, NSTEP)

# Soft first-crossing weights (softmax over time, peaked at first big positive)
# Apply a one-sided sigmoid first so post-cross times get high weight
post_cross = jax.nn.sigmoid(crossed_score / tau_step)   # (chunk, NSTEP)
# "Has not crossed by time t" = product(1 - post_cross[s] for s < t)
# Soft jstop weights: prob that first crossing is AT time t
not_yet = jnp.cumprod(1.0 - post_cross + 1e-9, axis=1)
not_yet = jnp.concatenate([jnp.ones((chunk, 1)), not_yet[:, :-1]], axis=1)
weights = post_cross * not_yet                          # (chunk, NSTEP)
# Renormalize to sum to 1 (in case crossing never happens, push to last step)
weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-9)
timesteps = jnp.arange(1, NSTEP + 1)
soft_jstop = (weights * timesteps).sum(axis=1)          # (chunk,)

# Soft category membership: weighted average of per-step cat probs
# For each step t, pos_t = argmax of a[trial, t, :] (still discrete)
# But we can compute soft cat probs from the FULL accumulator value at each step
# weighted softmax over positions, then mapped to category bands
pos_logits = a / tau_pos                                # (chunk, NSTEP, N)
pos_probs = jax.nn.softmax(pos_logits, axis=-1)         # soft position probs per step
# Build band-membership masks (3 categories from IPA..IPD bounds)
positions = jnp.arange(1, N + 1, dtype=jnp.float64)
mask_1 = (positions > IPA) & (positions < IPB)
mask_3 = (positions <= IPC) | (positions >= IPD)
mask_2 = ~(mask_1 | mask_3)
# Per-step cat probs
cat_prob_at_step = jnp.stack([
    (pos_probs * mask_1).sum(axis=-1),
    (pos_probs * mask_2).sum(axis=-1),
    (pos_probs * mask_3).sum(axis=-1),
], axis=-1)                                              # (chunk, NSTEP, 3)
# Weighted average over time (weighted by the same first-crossing weights)
cat_probs = (cat_prob_at_step * weights[:, :, None]).sum(axis=1)  # (chunk, 3)

rt = (soft_jstop + ndt) * E
```

Two temperatures: `tau_step` (for cross-time softening) and `tau_pos` (for position softening). Defaults: `tau_step = 0.5`, `tau_pos = 2.0`.

### 4.3 `objective_smooth.py` — smooth G²

Mirrors `condition_g2_vectorized` but:
- `cat == c` indicator → use `cat_probs[:, c-1]` (the soft membership directly)
- `pxy = mean(cat == c)` → `pxy = mean(cat_probs[:, c-1])`
- `qc[j] = (rt_i <= obs_quant[j, i]).sum() / denom` → `(sigmoid((obs_quant[j, i] - rt) / sigma_cdf) * cat_probs[:, i]).sum() / cat_probs[:, i].sum()`

```python
def condition_g2_smooth(rt, cat_probs, obs_prop, obs_count, obs_quant, sigma_cdf):
    mmn = obs_count.sum()
    def per_cat(i):
        weight = cat_probs[:, i]
        pxy = jnp.mean(weight)
        denom = jnp.maximum(weight.sum(), 1e-6)
        # Soft empirical CDF at each obs quantile, weighted by cat membership
        qc = jnp.array([
            (jax.nn.sigmoid((obs_quant[j, i] - rt) / sigma_cdf) * weight).sum() / denom
            for j in range(NQ)
        ])
        # ... rest of G² identical to condition_g2_vectorized
        ...
    return jnp.array([per_cat(i) for i in range(MC)]).sum()
```

Single new temperature: `sigma_cdf`. Default: `10.0` (in ms; aggressive smoothing).

### 4.4 `fofs_smooth` wrapper

```python
def fofs_smooth(params, data, key, nsim=512, chunk_size=256,
                tau_step=0.5, tau_pos=2.0, sigma_cdf=10.0):
    p = clamp(params)
    ter, st, sa, si, sig = p[0], p[1], p[3], p[4], p[5]
    drifts = jnp.stack([p[di] for (di, _) in COND_MAP])
    boundaries = jnp.stack([p[bi] for (_, bi) in COND_MAP])
    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(4)])

    sim_vmap = jax.vmap(
        sim_smooth.simulate_smooth,
        in_axes=(0, None, None, 0, None, None, None, 0, None, None, None, None, None),
    )
    rts, cat_probs = sim_vmap(
        cond_keys, ter, st, boundaries, sa, si, sig, drifts,
        nsim, chunk_size, tau_step, tau_pos, sigma_cdf,
    )
    # rts: (4, nsim); cat_probs: (4, nsim, 3)
    g2_vmap = jax.vmap(condition_g2_smooth, in_axes=(0, 0, 0, 0, 0, None))
    g2_per_cond = g2_vmap(rts, cat_probs, data["prop"], data["count"], data["quant"], sigma_cdf)
    return g2_per_cond.sum()
```

### 4.5 `fit.py` — fit drivers (new file)

```python
@dataclass
class FitResult:
    params: jnp.ndarray
    loss: float
    n_iters: int
    converged: bool
    optimizer: str

def fit_lbfgs_smooth(data, key, x0, nsim=512, maxiter=100, tol=1e-5,
                     tau_step=0.5, tau_pos=2.0, sigma_cdf=10.0):
    import jaxopt
    def loss_fn(p):
        return objective_smooth.fofs_smooth(p, data, key, nsim, 256,
                                             tau_step, tau_pos, sigma_cdf)
    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=maxiter, tol=tol)
    result = solver.run(x0)
    return FitResult(
        params=result.params, loss=float(result.state.value),
        n_iters=int(result.state.iter_num), converged=True, optimizer="lbfgs_smooth",
    )

def fit_simplex(data, key, x0, nsim=256, maxiter=500, tol=1e-7):
    from scipy.optimize import minimize
    from model_a import objective
    def loss_np(p_np):
        return float(objective.fofs_new(jnp.asarray(p_np), data, key, nsim))
    res = minimize(loss_np, np.asarray(x0), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": tol, "fatol": tol})
    return FitResult(params=jnp.asarray(res.x), loss=float(res.fun),
                     n_iters=int(res.nit), converged=bool(res.success), optimizer="simplex")

def fit_hybrid(data, key, x0, nsim=512, lbfgs_maxiter=50, polish_maxiter=200, **kwargs):
    coarse = fit_lbfgs_smooth(data, key, x0, nsim, maxiter=lbfgs_maxiter, **kwargs)
    polish = fit_simplex(data, key, coarse.params, nsim=nsim, maxiter=polish_maxiter)
    return FitResult(
        params=polish.params, loss=polish.loss,
        n_iters=coarse.n_iters + polish.n_iters,
        converged=polish.converged, optimizer="hybrid",
    )
```

## 5. Validation

### 5.1 Gradient finite + non-zero

`jax.grad(fofs_smooth)(params)` must:
- Be of shape (10,)
- All entries finite
- At least 8/10 entries with magnitude > 1e-4

The two parameters that might still have small gradient: `sv` (which is inert in the simulator) and potentially `si` (the drift bump width — only enters through scaling the deterministic drift). Acceptable.

### 5.2 Synthetic recovery

Generate synthetic data via `jax_port.simulate` at known `x_true`, perturb 15% to get `x0`, run `fit_hybrid`. Recovery target: ±15% per active parameter. Wall-clock target: under 60s on laptop CPU at nsim=512.

### 5.3 Comparison vs simplex-only

Optional benchmark: same recovery via `fit_simplex` alone vs `fit_hybrid`. Expect hybrid to be 5-20× faster wall-clock at the same accuracy.

## 6. Implementation stages

| Task | Scope | Gate |
|---|---|---|
| 3.5.A | `simulate_smooth.py` skeleton + soft-jstop + soft-cat | shape tests pass |
| 3.5.B | `objective_smooth.py::fofs_smooth` end-to-end | returns finite scalar |
| 3.5.C | Gradient test | jax.grad finite + non-zero on ≥ 8 params |
| 3.5.D | `fit.py::fit_lbfgs_smooth` driver | converges in ≤ 100 iter |
| 3.5.E | `fit.py::fit_hybrid` (smooth → polish) | hybrid runs to completion |
| 3.5.F | Synthetic recovery test (slow mark) | recovers within ±15% in < 60s |
| 3.5.G | Stage 3.5 gate review | all of above |

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Smooth optimum biased away from discrete optimum | medium | Hybrid polish closes the gap |
| L-BFGS unstable on MC-noisy gradients | medium | jaxopt's line search; raise nsim if needed |
| Soft-cat normalization fails at extreme params (all weights → 0) | low-medium | Add 1e-9 floors throughout |
| `cumprod` over NSTEP=400 underflows fp64 | low | Replace with logsumexp form if needed |
| Soft-jstop biased toward NSTEP (no crossings ever) | medium | Tune tau_step; sanity-check on high-drift case where crossing is certain |
| Recovery fails: hybrid doesn't beat simplex | medium-high | If smooth-L-BFGS converges but recovery is poor, that's smooth-bias; if it doesn't converge, that's optimizer instability — diagnose either |

## 8. Success criteria

Stage 3.5 is "done" when:
1. `jax.grad(fofs_smooth)` is finite + non-zero on ≥ 8 active params.
2. `fit_lbfgs_smooth` converges in ≤ 100 iter on synthetic data.
3. `fit_hybrid` recovers known params within ±15% in **under 60s on laptop CPU**.
4. Synthetic recovery test passes (slow mark).
5. `jax_port.py`, `simulate.py`, `objective.py` UNTOUCHED.
6. Smoke runner includes new tests, all green.

## 9. Decision log

- **Smoothing only, no REINFORCE** — simpler infrastructure; well-known technique.
- **Static temperatures, no annealing** — keep code simple; hybrid polish handles bias.
- **Defaults**: `tau_step=0.5`, `tau_pos=2.0`, `sigma_cdf=10.0` ms.
- **Hybrid = smooth-coarse → discrete-polish** — best of both worlds.
- **Model A only this stage** — if successful, port to Model B in Stage 3.5.B.
- **Reuse Stage 2's `simulate.py`** for the cumsum/noise generation — don't duplicate; smooth wrapper does post-processing.

Actually correction on the last point: the smooth simulator needs access to the full `(chunk, NSTEP, N)` accumulator array, which the current `simulate` doesn't expose. Either (a) refactor `simulate.py` to optionally return the accumulator, or (b) duplicate the simulation logic in `simulate_smooth.py`. **Decision: duplicate (b).** Keeps `simulate.py` clean; Stage 5 GPU work won't be affected by smooth-simulator changes.

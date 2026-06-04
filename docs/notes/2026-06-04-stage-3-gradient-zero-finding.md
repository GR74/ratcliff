# Stage 3 — `jax.grad(fofs_new)` is structurally zero

**Date:** 2026-06-04
**Discovered in:** Task 3.D.1 (gradient smoke test)
**HEAD when discovered:** `815bf87`

## The finding

`jax.grad(fofs_new)(params)` returns `[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]` for any params. Not NaN. Not boundary-clamp artifacts. **Exactly zero in every component.**

## Why

Every smooth path from `params` to the scalar G² is severed by a non-differentiable op:

| Location | Op | Gradient impact |
|---|---|---|
| `simulate._simulate_chunk` line 80 | `jstop = jnp.argmax(crossed, axis=1) + 1` | Discrete first-crossing index. Zero gradient w.r.t. params. |
| `simulate._simulate_chunk` line 88 | `cat = jnp.where((pos > IPA) & (pos < IPB), 1, ...)` | Discrete category. Zero gradient w.r.t. params. |
| `objective.condition_g2_vectorized` | `(rt_i <= obs_quant[j, i]).sum() / denom` | Indicator-based empirical CDF. Step function in `rt_i`, zero gradient. |

Even the one smooth path that survives in the simulator (`(ter, st) → rt` via `rt = (jstop + ndt) * E`) gets killed downstream by the indicator CDF inside the objective.

## Verified component-by-component

```
grad of mean(rt) w.r.t. ter:  1.0    ← only surviving smooth path inside simulate
grad of mean(rt) w.r.t. av:   0.0    ← killed by argmax(jstop)
grad of P(cat=1) w.r.t. av:   0.0    ← killed by discrete cat
```

Once `rt` is passed through `(rt <= q).sum()` in `condition_g2_vectorized`, even the `ter` path dies.

## What this means for Stage 3

The current Stage 3 design's headline win (L-BFGS with `jax.grad` → 50-100× fewer fofs evals) **does not work as designed**. The simulator + objective formulation is mathematically not differentiable from `params` to the scalar G².

This is not a bug in the implementation. It's a property of the model formulation (RT quantile binning + discrete category assignment). Even a perfect re-implementation would have the same problem.

## Three real paths forward (any future Stage 3 rewrite must pick one)

### Option A: Smooth surrogates
Replace each discrete op with a smooth, differentiable approximation:
- `argmax(crossed)` → soft-argmax: `sum(t * softmax(crossed_logit, dim=t))`. Anneal temperature toward zero during training.
- `cat = where(pos in band1, 1, 2, 3)` → soft-membership scores: `softmax([d_band1(pos), d_band2(pos), d_band3(pos)] / τ)`.
- `(rt <= q).sum()` → `sigmoid((q - rt) / σ).sum()`. σ small (~10 ms).

Pros: deterministic, JAX-native, gradients flow.
Cons: introduces bias (need to control τ, σ); two simulators to maintain (smooth and discrete); equivalence-against-discrete is hard to prove.

### Option B: Score-function (REINFORCE) estimator
Treat the simulator as a sampler `x ~ p(x | params)` and use the likelihood-ratio gradient:

  `∇_params E[loss(x)] ≈ E[loss(x) · ∇_params log p(x | params)]`

Pros: unbiased gradient of the original (discrete) objective. Well-studied technique in cognitive modeling (e.g., variational MCMC).
Cons: very high variance unless paired with control variates; many existing Ratcliff fits don't use this; significant infrastructure.

### Option C: Don't use `jax.grad` — stick with simplex / gradient-free
Stage 3's L-BFGS leg is abandoned. The simplex driver (Task 3.F.1) still works. Wall-clock fit on laptop CPU is now ~3-7 hours per dataset (basically the same as Fortran), not the 100s target.

Pros: zero new code; gradient-free; matches existing Ratcliff-fit literature.
Cons: kills the speedup story. Stage 5 GPU speedup on per-simulate-call is still real, but combined wins are 10-50× max, not 100-1000×.

## What's in the tree right now

Already committed and tested:
- `model_a/objective.py` with `condition_g2_vectorized` (bit-exact match to `jax_port.condition_g2`) and `fofs_new` (parity within 5%).
- `model_a/tests/test_objective_smoke.py` (3 tests, all passing — gradient test NOT committed).
- `model_a/tests/test_objective_parity.py` (3 tests, all passing).

NOT committed (because the gate failed):
- The gradient smoke test from Task 3.D.1.
- `fit_lbfgs`, `fit_simplex`, `fit_hybrid` (Tasks 3.E.1, 3.F.1, 3.G.1).

## Recommendation

Park Stage 3. Implement the `fit_simplex` driver (Task 3.F.1) when it's needed for a production fit — it's gradient-free and small (~50 LOC). Defer the L-BFGS decision to a Stage 3.5 where we evaluate Options A vs B properly. Move on to Stage 4 (Model B port) since the headline-speedup story there is per-simulate-call on GPU, not optimizer convergence.

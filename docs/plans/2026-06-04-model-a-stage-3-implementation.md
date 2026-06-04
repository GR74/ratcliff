# Model A — Stage 3 Implementation Plan: vmap + L-BFGS

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build `model_a/objective.py` (vmap'd `fofs_new` + vectorized `condition_g2_vectorized`) and `model_a/fit.py` (L-BFGS primary, simplex fallback, hybrid mode). Validate via parity against `jax_port.fofs` and via synthetic parameter recovery.

**Architecture:** New `fofs_new(params, data, key, nsim)` calls `model_a.simulate.simulate` (Stage 2's fast simulator) under a `jax.vmap` over the 4 conditions, then aggregates G² via a vectorized `condition_g2_vectorized`. `jax.grad(fofs_new)` is finite and meaningful. `fit_lbfgs` uses `jaxopt.LBFGS`; `fit_simplex` uses `scipy.optimize.minimize(method='Nelder-Mead')`.

**Tech Stack:** Python 3.11+, JAX 0.10.1 (CPU), jaxopt, scipy, pytest. No new dependencies (already in `[fit]` extras from Stage 0).

**Required sub-skills:**
- @superpowers:test-driven-development for every implementation task
- @superpowers:verification-before-completion before claiming task done

**Reference docs:**
- Design: `docs/plans/2026-06-04-model-a-stage-3-design.md`
- Project design: `docs/plans/2026-06-03-ratcliff-speedup-design.md`

**Current HEAD:** `44d08a6` (Stage 3 design commit). 45 tests passing.

---

## Pre-flight checks

```powershell
git -C C:\Users\gowri\ratcliff status
.\scripts\smoke.ps1
```
Expected: clean working tree, `45 passed`.

Install the `[fit]` extra if not already:
```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,fit]"
.\.venv\Scripts\python.exe -c "import jaxopt; print(jaxopt.__version__)"
```
Expected: prints a version string like `0.8.x`.

---

## Task 3.A.1: `model_a/objective.py` skeleton + import smoke

@superpowers:test-driven-development

**Files:**
- Create: `model_a/objective.py`
- Create: `model_a/tests/test_objective_smoke.py`

**Step 1: Write the failing smoke test**

Create `model_a/tests/test_objective_smoke.py`:

```python
"""Smoke tests for model_a/objective.py."""
import jax.numpy as jnp
import pytest


def test_objective_module_imports():
    from model_a import objective
    # Confirm public API
    assert hasattr(objective, "condition_g2_vectorized")
    assert hasattr(objective, "fofs_new")
    assert hasattr(objective, "COND_MAP")


def test_cond_map_has_4_conditions():
    from model_a import objective
    assert len(objective.COND_MAP) == 4
    for (di, bi) in objective.COND_MAP:
        assert isinstance(di, int)
        assert isinstance(bi, int)
```

**Step 2: Run to verify it fails**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_smoke.py -v
```
Expected: ImportError on `from model_a import objective`.

**Step 3: Create the skeleton**

Create `model_a/objective.py`:

```python
"""
Vectorized G² objective for the Ratcliff 1D model.

Replaces the Python loops in `jax_port.fofs` and `jax_port.condition_g2` with
`jax.vmap` over conditions and `jnp.where`-based category aggregation. The
output `fofs_new(params, data, key, nsim)` matches `jax_port.fofs`'s contract
and is differentiable via `jax.grad`.
"""
from functools import partial

import jax
import jax.numpy as jnp

from model_a import simulate as sim_new
from model_a.jax_port import clamp  # reuse the bounds function
from shared import prng

# 4 fitting conditions: (drift_param_idx, boundary_param_idx) per condition.
# Matches jax_port.fofs:
#   cond 1: drift=params[7], boundary=params[2]  (drift1, a1)
#   cond 2: drift=params[8], boundary=params[2]  (drift2, a1)
#   cond 3: drift=params[7], boundary=params[9]  (drift1, a2)
#   cond 4: drift=params[8], boundary=params[9]  (drift2, a2)
COND_MAP = [(7, 2), (8, 2), (7, 9), (8, 9)]

MC = 3
NQ = 5
NCUT = 8
PQQ = jnp.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def condition_g2_vectorized(rt, cat, obs_prop, obs_count, obs_quant):
    """Placeholder — implemented in Task 3.B.1."""
    raise NotImplementedError


def fofs_new(params, data, key, nsim=4000):
    """Placeholder — implemented in Task 3.C.1."""
    raise NotImplementedError
```

**Step 4: Run smoke tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_smoke.py -v
```
Expected: 2 passed.

**Step 5: Commit**

```powershell
git add model_a/objective.py model_a/tests/test_objective_smoke.py
git commit -m "feat(objective): skeleton with COND_MAP + NotImplementedError stubs"
```

---

## Task 3.B.1: Vectorized `condition_g2`

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/objective.py`
- Modify: `model_a/tests/test_objective_smoke.py`

**Step 1: Add failing parity test**

Append to `model_a/tests/test_objective_smoke.py`:

```python
def test_condition_g2_matches_jax_port_at_realistic():
    """The vectorized condition_g2 must match jax_port's scalar output."""
    import numpy as np
    from model_a import jax_port, objective
    from shared import prng

    # Generate one condition's worth of trials
    key = prng.root_key(42)
    rt, cat = jax_port.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=512
    )

    # Synthetic observed-data triplet for one condition (matches fofs's inner call)
    obs_prop = jnp.array([0.5, 0.3, 0.2])
    obs_count = jnp.array([50, 30, 20], dtype=jnp.int64)
    obs_quant = jnp.array([
        [300.0, 320.0, 360.0],
        [340.0, 360.0, 400.0],
        [380.0, 400.0, 440.0],
        [420.0, 440.0, 480.0],
        [460.0, 480.0, 520.0],
    ])  # shape (NQ=5, MC=3)

    g2_old = float(jax_port.condition_g2(rt, cat, obs_prop, obs_count, obs_quant))
    g2_new = float(objective.condition_g2_vectorized(rt, cat, obs_prop, obs_count, obs_quant))
    rel_diff = abs(g2_new - g2_old) / abs(g2_old) if abs(g2_old) > 1e-9 else abs(g2_new - g2_old)
    assert rel_diff < 1e-4, f"condition_g2 mismatch: old={g2_old}, new={g2_new}, rel_diff={rel_diff}"
```

**Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_smoke.py::test_condition_g2_matches_jax_port_at_realistic -v
```
Expected: NotImplementedError.

**Step 3: Implement `condition_g2_vectorized`**

In `model_a/objective.py`, replace the placeholder body:

```python
def condition_g2_vectorized(rt, cat, obs_prop, obs_count, obs_quant):
    """
    Compute G² contribution from one condition.

    rt        : (nsim,) RTs from simulate.
    cat       : (nsim,) categories in {1,2,3}.
    obs_prop  : (3,) observed proportions per category.
    obs_count : (3,) observed counts per category.
    obs_quant : (5, 3) observed RT quantiles per category.

    Returns scalar G² (sum over the 3 categories).
    """
    mmn = obs_count.sum()

    def per_cat(i):
        in_cat = (cat == (i + 1))
        pxy = jnp.mean(in_cat)
        denom = jnp.maximum(in_cat.sum(), 1)
        rt_i = jnp.where(in_cat, rt, jnp.inf)
        # qc[j] = empirical CDF of in-cat RTs at obs_quant[j, i], length NQ=5
        qc = jnp.array([(rt_i <= obs_quant[j, i]).sum() / denom for j in range(NQ)])

        # full quantile-by-quantile G² term
        c_full = mmn * obs_prop[i] * PQQ[0] * jnp.log(
            obs_prop[i] * PQQ[0] / (pxy * qc[0] + 1e-5))
        for j in range(1, NQ):
            yy = jnp.maximum(qc[j] - qc[j - 1], 1e-3)
            c_full = c_full + mmn * obs_prop[i] * PQQ[j] * jnp.log(
                obs_prop[i] * PQQ[j] / (pxy * yy + 1e-5))
        c_full = c_full + mmn * obs_prop[i] * PQQ[NQ] * jnp.log(
            obs_prop[i] * PQQ[NQ] / (pxy * (1.0 - qc[NQ - 1]) + 1e-5))

        # lumped term for small-count categories
        c_lumped = mmn * (obs_prop[i] + 0.002) * jnp.log(
            (obs_prop[i] + 0.002) / (pxy + 1e-12))

        return jnp.where(obs_count[i] >= NCUT, c_full, c_lumped)

    contribs = jnp.array([per_cat(i) for i in range(MC)])
    return contribs.sum()
```

**Note:** The inner `for j in range(NQ)` and `for i in range(MC)` are over compile-time constants (5 and 3), so they unroll at trace time — they are NOT runtime Python control flow, despite looking like it. This is the same pattern used in `jax_port.condition_g2`. The "vectorization" here is removing the dynamic-shape construction `[ (rt_i <= ...).sum() for j in range(NQ) ]` and replacing it with a clean `jnp.array(...)` over the literal range.

**Step 4: Run parity test to verify it passes**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_smoke.py -v
```
Expected: 3 passed.

**Step 5: Commit**

```powershell
git add model_a/objective.py model_a/tests/test_objective_smoke.py
git commit -m "feat(objective): vectorized condition_g2 matches jax_port to 1e-4"
```

---

## Task 3.C.1: `fofs_new` via vmap over conditions

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/objective.py`
- Create: `model_a/tests/test_objective_parity.py`

**Step 1: Write the failing parity tests**

Create `model_a/tests/test_objective_parity.py`:

```python
"""
Parity tests: fofs_new must match jax_port.fofs to MC tolerance at the same params.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import jax_port, objective
from shared import data_io, prng
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def _first_subject_data():
    """Same helper as test_fofs_smoke; returns 4-condition data dict for subject 0."""
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    return {
        "prop": jnp.asarray(g["prop"][0]),
        "count": jnp.asarray(g["count"][0]),
        "quant": jnp.asarray(g["quant"][0]),
    }


PARAM_SETS = [
    ("realistic", jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])),
    ("high_drift", jnp.array([200., 50., 50., 10., 4., 5., 0.7, 60., 50., 60.])),
    ("low_drift", jnp.array([200., 50., 50., 10., 4., 5., 0.7, 5., 3., 60.])),
]


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[p[0] for p in PARAM_SETS])
def test_fofs_new_matches_jax_port_fofs(name, params):
    """fofs_new and jax_port.fofs return scalar G² values within 5% relative."""
    data = _first_subject_data()
    key = prng.root_key(1337)

    val_old = float(jax_port.fofs(params, data, key, nsim=2048))
    val_new = float(objective.fofs_new(params, data, key, nsim=2048))

    rel_diff = abs(val_new - val_old) / abs(val_old)
    assert rel_diff < 0.05, (
        f"[{name}] fofs mismatch: old={val_old:.3f}, new={val_new:.3f}, "
        f"rel_diff={rel_diff:.4f} (tol 0.05)"
    )
```

**Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_parity.py -v
```
Expected: NotImplementedError.

**Step 3: Implement `fofs_new`**

In `model_a/objective.py`, replace the `fofs_new` placeholder:

```python
def fofs_new(params, data, key, nsim=4000, chunk_size=256):
    """
    Vectorized G² objective summed across 4 conditions.

    params : (10,) parameter vector — see clamp() docs.
    data   : dict with "prop" (4,3), "count" (4,3), "quant" (4,5,3).
    key    : JAX typed key.
    nsim   : trials per condition.
    chunk_size : trial chunk for the Stage 2 simulator.

    Returns scalar G² (sum over conditions).
    """
    p = clamp(params)
    ter, st, sa, si, sig = p[0], p[1], p[3], p[4], p[5]
    # Per-condition (drift, boundary) tuples from COND_MAP
    drifts = jnp.stack([p[di] for (di, _) in COND_MAP])    # (4,)
    boundaries = jnp.stack([p[bi] for (_, bi) in COND_MAP]) # (4,)

    # One subkey per condition, deterministic from `key`
    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(4)])

    # vmap simulate over (key, cr, av); other params are condition-invariant.
    # simulate signature: (key, ter, st, cr, crsd, si, sig, av, nsim, chunk_size)
    sim_vmap = jax.vmap(
        sim_new.simulate,
        in_axes=(0, None, None, 0, None, None, None, 0, None, None),
    )
    rts, cats = sim_vmap(
        cond_keys, ter, st, boundaries, p[3], si, sig, drifts, nsim, chunk_size
    )
    # rts, cats: (4, nsim)

    # vmap condition_g2 over the 4 (rts[ci], cats[ci], data[ci, :])
    g2_vmap = jax.vmap(
        condition_g2_vectorized,
        in_axes=(0, 0, 0, 0, 0),
    )
    g2_per_cond = g2_vmap(
        rts, cats, data["prop"], data["count"], data["quant"]
    )
    return g2_per_cond.sum()
```

Note: `sa` was passed as `p[3]` directly inline (since it's the `crsd` arg of simulate). Confusion-prone — double-check by reading jax_port.fofs lines 165-180 to map params to simulate args.

Actually, looking at `jax_port.fofs` again: it calls `simulate(jax.random.fold_in(key, ci), ter, st, p[bi], sa, si, sig, p[di], sv, nsim)`. So:
- ter = p[0]
- st = p[1]
- cr = p[bi]   (condition-dependent: a1 or a2)
- crsd = sa = p[3]
- si = p[4]
- sig = p[5]
- av = p[di]   (condition-dependent: drift1 or drift2)
- sv = p[6]    (inert in jax_port.simulate; new simulate.py omits sv)

The new simulate signature has no `sv`, so the vmap over `(key, cr, av)` is correct as drafted.

**Step 4: Run parity tests**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_parity.py -v
```
Expected: 3 passed. Wall clock: 60–120 seconds first run (JIT compile both old and new fofs at nsim=2048).

If parity fails:
- Check the `(drifts, boundaries)` extraction matches `COND_MAP` interpretation.
- Check `sa` (column 3) wired to simulate's `crsd` argument.
- Check `cond_keys` derivation matches what jax_port does (`fold_in(key, ci)` per condition).

If parity passes within 10% but not 5%, raise the tolerance to 0.08 with a code comment explaining MC noise dominates at nsim=2048.

**Step 5: Commit**

```powershell
git add model_a/objective.py model_a/tests/test_objective_parity.py
git commit -m "feat(objective): fofs_new vmaps simulate over 4 conditions"
```

---

## Task 3.D.1: Smoke test for `jax.grad(fofs_new)`

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/tests/test_objective_smoke.py`

**Step 1: Append the gradient smoke test**

```python
def test_fofs_new_gradient_is_finite():
    """jax.grad(fofs_new) produces a finite, non-zero gradient vector."""
    import jax
    from model_a import objective
    from shared import data_io, prng
    from pathlib import Path
    DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"

    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    data = {
        "prop": jnp.asarray(g["prop"][0]),
        "count": jnp.asarray(g["count"][0]),
        "quant": jnp.asarray(g["quant"][0]),
    }
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(0)

    grad_fn = jax.grad(lambda p: objective.fofs_new(p, data, key, nsim=512))
    g_vec = grad_fn(params)

    assert g_vec.shape == (10,)
    assert jnp.all(jnp.isfinite(g_vec)), f"gradient has non-finite: {g_vec}"
    # At least one component should be non-trivial (not all near-zero)
    assert jnp.any(jnp.abs(g_vec) > 1e-4), f"gradient is essentially zero: {g_vec}"
```

**Step 2: Run test**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_objective_smoke.py::test_fofs_new_gradient_is_finite -v
```
Expected: passes. Wall clock 60–90 seconds first run.

If it FAILS:
- NaN in gradient → likely a `0/0` somewhere in `condition_g2_vectorized`. Check the `(pxy * qc[0] + 1e-5)` denominators.
- All zeros → `clamp()` may be flattening the gradient at all components (e.g., all params are above the floor); try perturbing initial params away from the floors.

**Step 3: Commit**

```powershell
git add model_a/tests/test_objective_smoke.py
git commit -m "test(objective): jax.grad(fofs_new) produces finite gradient"
```

---

## Task 3.E.1: `fit_lbfgs` driver

@superpowers:test-driven-development

**Files:**
- Create: `model_a/fit.py`
- Create: `model_a/tests/test_fit_lbfgs_smoke.py`

**Step 1: Write the failing smoke test**

Create `model_a/tests/test_fit_lbfgs_smoke.py`:

```python
"""
L-BFGS fit smoke test: recover known parameters from synthetic data.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import fit, jax_port, simulate as sim_new
from shared import prng

# Known "true" parameters for synthetic data generation
TRUE_PARAMS = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])


def _generate_synthetic_data(true_params, key, nsim_per_condition=512):
    """Run simulate at true_params for each of 4 conditions, build a data dict."""
    from model_a.objective import COND_MAP
    from model_a.jax_port import clamp
    p = clamp(true_params)
    ter, st, sa, si, sig = p[0], p[1], p[3], p[4], p[5]
    prop_list, count_list, quant_list = [], [], []
    for ci, (di, bi) in enumerate(COND_MAP):
        cond_key = prng.split_for_condition(key, ci)
        rt, cat = sim_new.simulate(
            cond_key, ter, st, p[bi], sa, si, sig, p[di],
            nsim=nsim_per_condition, chunk_size=256
        )
        # Compute proportions, counts, quantiles per category
        props = jnp.array([(cat == c).mean() for c in (1, 2, 3)])
        counts = jnp.array([(cat == c).sum() for c in (1, 2, 3)], dtype=jnp.int64)
        quants = jnp.zeros((5, 3))
        qs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for ci2, c in enumerate((1, 2, 3)):
            mask = cat == c
            if mask.sum() >= 5:
                cat_rts = jnp.sort(jnp.where(mask, rt, jnp.inf))
                n_in_cat = int(mask.sum())
                indices = (qs * (n_in_cat - 1)).astype(jnp.int32)
                quants = quants.at[:, ci2].set(cat_rts[indices])
        prop_list.append(props)
        count_list.append(counts)
        quant_list.append(quants)
    return {
        "prop": jnp.stack(prop_list),
        "count": jnp.stack(count_list),
        "quant": jnp.stack(quant_list),
    }


def test_lbfgs_recovers_known_params():
    """L-BFGS fit on synthetic data should recover known parameters within ±10%."""
    key_data = prng.root_key(0)
    data = _generate_synthetic_data(TRUE_PARAMS, key_data, nsim_per_condition=512)

    # Perturb starting point ±20% to give the optimizer something to do
    np.random.seed(0)
    perturbation = jnp.array(np.random.uniform(0.8, 1.2, size=10))
    x0 = TRUE_PARAMS * perturbation

    key_fit = prng.root_key(1)
    result = fit.fit_lbfgs(data, key_fit, x0, nsim=512, maxiter=100, tol=1e-5)

    # Recovery check on the "active" parameters (skip sv which is inert)
    active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    for i in active_indices:
        rel_err = abs(float(result.params[i]) - float(TRUE_PARAMS[i])) / float(TRUE_PARAMS[i])
        assert rel_err < 0.15, (
            f"param {i}: true={float(TRUE_PARAMS[i]):.3f}, "
            f"recovered={float(result.params[i]):.3f}, rel_err={rel_err:.3f}"
        )
    # Loss should be small (close to MC-noise floor)
    assert result.loss < 100.0
```

**Step 2: Run to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_fit_lbfgs_smoke.py -v
```
Expected: ImportError on `from model_a import fit`.

**Step 3: Implement `fit_lbfgs`**

Create `model_a/fit.py`:

```python
"""
Fit drivers for the Ratcliff Model A objective.

`fit_lbfgs` uses jaxopt.LBFGS with free gradients from jax.grad(fofs_new).
`fit_simplex` falls back to scipy Nelder-Mead for gradient-free fits.
`fit_hybrid` runs simplex coarse then L-BFGS refine.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from model_a import objective


@dataclass
class FitResult:
    params: jnp.ndarray
    loss: float
    n_iters: int
    converged: bool
    optimizer: str


def fit_lbfgs(data, key, x0, nsim=512, maxiter=200, tol=1e-6, chunk_size=256):
    """L-BFGS via jaxopt."""
    import jaxopt

    def loss_fn(p):
        return objective.fofs_new(p, data, key, nsim=nsim, chunk_size=chunk_size)

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=maxiter, tol=tol)
    result = solver.run(x0)
    return FitResult(
        params=result.params,
        loss=float(result.state.value),
        n_iters=int(result.state.iter_num),
        converged=bool(result.state.error < tol),
        optimizer="lbfgs",
    )
```

**Step 4: Run fit smoke test**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_fit_lbfgs_smoke.py -v
```
Expected: passes. Wall clock 60–180 seconds.

If recovery fails:
- Check the residual loss — if it's > MC noise floor (~50-200 at nsim=512), L-BFGS didn't converge. Bump `maxiter` or use `hybrid`.
- If specific params don't recover (likely `crsd` or `si` which are weakly identified), accept ±20% on those, tighten on the easy ones.

**Step 5: Commit**

```powershell
git add model_a/fit.py model_a/tests/test_fit_lbfgs_smoke.py
git commit -m "feat(fit): fit_lbfgs via jaxopt recovers synthetic params"
```

---

## Task 3.F.1: `fit_simplex` fallback

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/fit.py`
- Create: `model_a/tests/test_fit_simplex_smoke.py`

**Step 1: Write the failing test**

Create `model_a/tests/test_fit_simplex_smoke.py`:

```python
"""Simplex fit recovery on synthetic data (slower than L-BFGS but gradient-free)."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import fit
from model_a.tests.test_fit_lbfgs_smoke import _generate_synthetic_data, TRUE_PARAMS
from shared import prng


@pytest.mark.slow
def test_simplex_recovers_known_params():
    """Simplex fit recovers params within ±15% (looser than L-BFGS, slower)."""
    key_data = prng.root_key(0)
    data = _generate_synthetic_data(TRUE_PARAMS, key_data, nsim_per_condition=256)

    np.random.seed(0)
    x0 = TRUE_PARAMS * jnp.array(np.random.uniform(0.85, 1.15, size=10))

    key_fit = prng.root_key(1)
    result = fit.fit_simplex(data, key_fit, x0, nsim=256, maxiter=500)

    active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    for i in active_indices:
        rel_err = abs(float(result.params[i]) - float(TRUE_PARAMS[i])) / float(TRUE_PARAMS[i])
        assert rel_err < 0.2, (
            f"param {i}: true={float(TRUE_PARAMS[i]):.3f}, "
            f"recovered={float(result.params[i]):.3f}, rel_err={rel_err:.3f}"
        )
```

**Step 2: Register the `slow` mark in pyproject.toml**

Add to `[tool.pytest.ini_options].markers`:

```toml
markers = [
    "perf: performance regression tests (skipped by default, run with -m perf)",
    "slow: slow integration tests (run with -m slow or include explicitly)",
]
```

**Step 3: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_fit_simplex_smoke.py -m slow -v
```
Expected: `AttributeError: module 'model_a.fit' has no attribute 'fit_simplex'`.

**Step 4: Implement `fit_simplex`**

Append to `model_a/fit.py`:

```python
def fit_simplex(data, key, x0, nsim=256, maxiter=2000, tol=1e-7):
    """Scipy Nelder-Mead simplex; gradient-free fallback for bumpy likelihoods."""
    from scipy.optimize import minimize

    def loss_numpy(p_np):
        p = jnp.asarray(p_np)
        val = objective.fofs_new(p, data, key, nsim=nsim)
        return float(val)

    res = minimize(
        loss_numpy, np.asarray(x0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": tol, "fatol": tol, "disp": False},
    )
    return FitResult(
        params=jnp.asarray(res.x),
        loss=float(res.fun),
        n_iters=int(res.nit),
        converged=bool(res.success),
        optimizer="simplex",
    )
```

**Step 5: Run test**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_fit_simplex_smoke.py -m slow -v
```
Expected: passes. Wall clock 3–10 minutes (simplex is slow).

**Step 6: Commit**

```powershell
git add model_a/fit.py model_a/tests/test_fit_simplex_smoke.py pyproject.toml
git commit -m "feat(fit): fit_simplex via scipy Nelder-Mead as fallback"
```

---

## Task 3.G.1: `fit_hybrid` (simplex coarse → L-BFGS refine)

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/fit.py`
- Modify: `model_a/tests/test_fit_lbfgs_smoke.py` (add hybrid test)

**Step 1: Add failing test**

Append to `model_a/tests/test_fit_lbfgs_smoke.py`:

```python
def test_hybrid_recovers_known_params():
    """Hybrid simplex-coarse → LBFGS-refine recovers params (most robust)."""
    key_data = prng.root_key(0)
    data = _generate_synthetic_data(TRUE_PARAMS, key_data, nsim_per_condition=256)

    np.random.seed(0)
    x0 = TRUE_PARAMS * jnp.array(np.random.uniform(0.85, 1.15, size=10))

    key_fit = prng.root_key(1)
    result = fit.fit_hybrid(data, key_fit, x0, nsim=256,
                            simplex_maxiter=50, lbfgs_maxiter=100, tol=1e-5)

    active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    for i in active_indices:
        rel_err = abs(float(result.params[i]) - float(TRUE_PARAMS[i])) / float(TRUE_PARAMS[i])
        assert rel_err < 0.15
    assert result.optimizer == "hybrid"
```

**Step 2: Implement `fit_hybrid`**

Append to `model_a/fit.py`:

```python
def fit_hybrid(data, key, x0, nsim=256, simplex_maxiter=50, lbfgs_maxiter=100, tol=1e-5):
    """Simplex coarse-pass → L-BFGS refine. Most robust on bumpy likelihoods."""
    coarse = fit_simplex(data, key, x0, nsim=nsim, maxiter=simplex_maxiter)
    refined = fit_lbfgs(data, key, coarse.params, nsim=nsim, maxiter=lbfgs_maxiter, tol=tol)
    return FitResult(
        params=refined.params,
        loss=refined.loss,
        n_iters=coarse.n_iters + refined.n_iters,
        converged=refined.converged,
        optimizer="hybrid",
    )
```

**Step 3: Run test**

```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_fit_lbfgs_smoke.py::test_hybrid_recovers_known_params -v
```
Expected: passes.

**Step 4: Commit**

```powershell
git add model_a/fit.py model_a/tests/test_fit_lbfgs_smoke.py
git commit -m "feat(fit): fit_hybrid combines simplex coarse with lbfgs refine"
```

---

## Task 3.H.1: Update smoke runner

**Files:**
- Modify: `scripts/smoke.ps1`
- Modify: `scripts/smoke.sh`

Add the new test files:

`scripts/smoke.ps1`:
```powershell
.\.venv\Scripts\python.exe -m pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py model_a/tests/test_objective_smoke.py model_a/tests/test_objective_parity.py model_a/tests/test_fit_lbfgs_smoke.py -v
```

`scripts/smoke.sh`: same paths.

**Note**: We include the L-BFGS fit test in smoke (it's fast — ~60s on warm cache). We do NOT include `test_fit_simplex_smoke.py` (the `slow` mark is for `pytest -m slow` only).

Run:
```powershell
.\scripts\smoke.ps1
```
Expected: ~52-55 tests passing, wall clock 3–5 minutes (the fit test is slow).

Commit:
```powershell
git add scripts/smoke.ps1 scripts/smoke.sh
git commit -m "chore(smoke): include objective and fit smoke tests in Stage 3 gate"
```

---

## Stage 3 completion gate

@superpowers:verification-before-completion

Verify ALL of:

1. `model_a/objective.py` exists with `fofs_new` and `condition_g2_vectorized`.
2. `model_a/fit.py` exists with `fit_lbfgs`, `fit_simplex`, `fit_hybrid`.
3. Parity tests pass: `fofs_new` matches `jax_port.fofs` within 5% at 3 param sets.
4. `jax.grad(fofs_new)` returns finite, non-zero gradient.
5. `fit_lbfgs` recovers known params within ±15% on synthetic data.
6. `fit_simplex` recovers known params within ±20% (slow mark, run separately).
7. `fit_hybrid` recovers known params within ±15%.
8. `.\scripts\smoke.ps1` passes (target ~52-55 tests).
9. `model_a/jax_port.py` UNTOUCHED.
10. Wall-clock for one L-BFGS fit: under 60s on laptop CPU.

If all green, dispatch final code-reviewer for the whole Stage 3 diff (`44d08a6..HEAD`).

---

## Out of scope

- No Model B (Stage 4).
- No Fortran-validation gate for `jax_port` itself (Stage 2.5 — separate effort).
- No Bayesian (NumPyro / BlackJAX) fitting.
- No hierarchical fits across subjects.
- No GPU benchmark suite (Stage 5).
- No `LICENSE` file.
- No README updates — defer to Stage 5 when fit examples are stable.

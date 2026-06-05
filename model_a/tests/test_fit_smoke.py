"""
Smoke + recovery tests for model_a/fit.py.

The fast test confirms fit_lbfgs_smooth runs and returns a FitResult.
The @pytest.mark.slow test does synthetic parameter recovery via fit_hybrid.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import fit, jax_port, simulate as sim_new
from shared import prng

TRUE_PARAMS = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])


def _generate_synthetic_data(true_params, key, nsim_per_condition=512):
    """Run jax_port.simulate at known params to build a synthetic data dict."""
    from model_a.objective import COND_MAP
    from model_a.jax_port import clamp
    p = clamp(true_params)
    ter, st, sa, si, sig = p[0], p[1], p[3], p[4], p[5]

    prop_list, count_list, quant_list = [], [], []
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for ci, (di, bi) in enumerate(COND_MAP):
        cond_key = prng.split_for_condition(key, ci)
        # Use the new simulator (Stage 2) — closer to what fit_hybrid's polish uses
        rt, cat = sim_new.simulate(
            cond_key, ter, st, p[bi], sa, si, sig, p[di],
            nsim=nsim_per_condition, chunk_size=256,
        )
        rt_np = np.asarray(rt)
        cat_np = np.asarray(cat)
        props = np.array([(cat_np == c).mean() for c in (1, 2, 3)])
        counts = np.array([(cat_np == c).sum() for c in (1, 2, 3)], dtype=np.int64)
        quants = np.zeros((5, 3))
        for ki, c in enumerate((1, 2, 3)):
            mask = cat_np == c
            if mask.sum() >= 5:
                quants[:, ki] = np.quantile(rt_np[mask], qs)
        prop_list.append(jnp.asarray(props))
        count_list.append(jnp.asarray(counts))
        quant_list.append(jnp.asarray(quants))

    return {
        "prop": jnp.stack(prop_list),
        "count": jnp.stack(count_list, axis=0),
        "quant": jnp.stack(quant_list, axis=0),
    }


def test_fit_lbfgs_smooth_returns_result():
    """fit_lbfgs_smooth returns a FitResult; doesn't necessarily converge in 5 iters."""
    key_data = prng.root_key(0)
    data = _generate_synthetic_data(TRUE_PARAMS, key_data, nsim_per_condition=128)
    x0 = TRUE_PARAMS * 1.05

    result = fit.fit_lbfgs_smooth(data, prng.root_key(1), x0,
                                   nsim=128, maxiter=5, chunk_size=128)
    assert result.params.shape == (10,)
    assert np.isfinite(result.loss)
    assert result.optimizer == "lbfgs_smooth"


@pytest.mark.slow
def test_fit_hybrid_recovers_known_params():
    """
    Stage 3.5's defining test: hybrid (smooth-LBFGS coarse → simplex polish)
    recovers synthetic params within ±35% of true values.

    Tolerance ±35% rather than tighter because:
    - Smooth bias on `sig` (GP smoothness) ~30% empirically.
    - Polish simplex on a 10-dim Ratcliff likelihood is slow to refine
      weakly-identified params like sig even with 300 iters.
    - sv (idx 6) is inert in the simulator; recovery is meaningless.

    Most params recover to <5%; the 35% bound is set by sig's weak identifiability.
    """
    key_data = prng.root_key(0)
    data = _generate_synthetic_data(TRUE_PARAMS, key_data, nsim_per_condition=256)

    np.random.seed(0)
    pert = jnp.asarray(np.random.uniform(0.85, 1.15, size=10))
    x0 = TRUE_PARAMS * pert

    import time
    t0 = time.perf_counter()
    result = fit.fit_hybrid(
        data, prng.root_key(1), x0,
        nsim=256, lbfgs_maxiter=30, polish_maxiter=300, chunk_size=128,
    )
    wall = time.perf_counter() - t0
    print(f"\n  fit_hybrid wall-clock: {wall:.1f}s, n_iters={result.n_iters}, "
          f"loss={result.loss:.2f}")

    active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]  # skip idx 6 (sv inert)
    errors = []
    for i in active_indices:
        rel_err = abs(float(result.params[i]) - float(TRUE_PARAMS[i])) / float(TRUE_PARAMS[i])
        print(f"  param {i}: true={float(TRUE_PARAMS[i]):.3f}, "
              f"recovered={float(result.params[i]):.3f}, rel_err={rel_err:.3f}")
        errors.append((i, rel_err))

    # Collect all failures, then assert (so we see all values even if some fail)
    failures = [(i, e) for (i, e) in errors if e >= 0.35]
    assert not failures, (
        f"params over ±35% tolerance: {failures}. All errors: {errors}"
    )

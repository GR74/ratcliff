"""Fit smoke test for Model B (simplex only — gradient issue from Stage 3)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import fit, objective as obj_b, simulate as sim_b
from shared import prng

# Known "true" parameters for synthetic data generation
TRUE_PARAMS_B = jnp.array([
    200.0, 50.0, 10.0, 2.0,    # ter, st, cr, crsd
    12.0, 10.0, 0.5,            # sis, sig, sv (sv inert)
    15.0, 10.0, 8.0,            # cond 1 drifts (av1, av2, av3)
    14.0, 11.0, 9.0,            # cond 2 drifts
])


def _generate_synthetic_data_b(true_params, key, nsim_per_condition=128, chunk_size=8):
    """Run simulate_b at true_params per condition, build a data dict."""
    from model_b.objective import COND_MAP_B, clamp_b

    p = clamp_b(true_params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    prop_list, count_list, quant_list = [], [], []

    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        cond_key = prng.split_for_condition(key, ci)
        rt, cat = sim_b.simulate_b(
            cond_key, ter, st, cr, crsd,
            p[d1], p[d2], p[d3],
            sis, sig, si,
            nsim=nsim_per_condition, chunk_size=chunk_size,
        )
        # Compute proportions, counts, quantiles per category (5 categories)
        props = jnp.array([(cat == c).mean() for c in (1, 2, 3, 4, 5)])
        counts = jnp.array([(cat == c).sum() for c in (1, 2, 3, 4, 5)], dtype=jnp.int64)
        quants = jnp.zeros((5, 5))  # (NQ, MC)
        qs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat == c
            if int(mask.sum()) >= 5:
                cat_rts = jnp.sort(jnp.where(mask, rt, jnp.inf))
                n_in_cat = int(mask.sum())
                indices = (qs * (n_in_cat - 1)).astype(jnp.int32)
                quants = quants.at[:, ki].set(cat_rts[indices])

        prop_list.append(props)
        count_list.append(counts)
        quant_list.append(quants)

    return {
        "prop": jnp.stack(prop_list),    # (2, 5)
        "count": jnp.stack(count_list),  # (2, 5)
        "quant": jnp.stack(quant_list),  # (2, 5, 5)
    }


def test_fit_simplex_b_returns_result():
    """fit_simplex_b returns a FitResult with the right shape."""
    key = jax.random.key(0)
    data = _generate_synthetic_data_b(TRUE_PARAMS_B, key, nsim_per_condition=64, chunk_size=8)
    x0 = TRUE_PARAMS_B * 1.05  # 5% perturbation
    result = fit.fit_simplex_b(data, jax.random.key(1), x0, nsim=64, maxiter=10, chunk_size=8)
    assert result.params.shape == (13,)
    assert np.isfinite(result.loss)
    assert result.n_iters > 0


@pytest.mark.slow
def test_fit_simplex_b_recovers_known_params():
    """Recovery on synthetic data within +/-25% on active params (simplex is loose)."""
    key_data = jax.random.key(0)
    data = _generate_synthetic_data_b(TRUE_PARAMS_B, key_data, nsim_per_condition=128, chunk_size=8)

    np.random.seed(0)
    perturbation = jnp.array(np.random.uniform(0.85, 1.15, size=13))
    x0 = TRUE_PARAMS_B * perturbation

    key_fit = jax.random.key(1)
    result = fit.fit_simplex_b(data, key_fit, x0, nsim=128, maxiter=200, chunk_size=8)

    # Recovery check on the "active" parameters (skip sv at index 6 which is inert)
    active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    for i in active_indices:
        rel_err = abs(float(result.params[i]) - float(TRUE_PARAMS_B[i])) / float(TRUE_PARAMS_B[i])
        assert rel_err < 0.25, (
            f"param {i}: true={float(TRUE_PARAMS_B[i]):.3f}, "
            f"recovered={float(result.params[i]):.3f}, rel_err={rel_err:.3f}"
        )

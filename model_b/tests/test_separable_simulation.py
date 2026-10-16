"""Separable K-L noise must reproduce the dense K-L simulator trial by trial."""
import jax
import numpy as np
import pytest

from model_b import simulate

TRUE = dict(ter=200., st=50., cr=10., crsd=2., av1=15., av2=10., av3=8., sis=12., si=6.)


@pytest.mark.parametrize("sig,stable", [(10.0, True), (10.0, False), (15.0, True)])
def test_separable_matches_dense_trials(sig, stable):
    p = TRUE
    runs = {}
    with jax.default_matmul_precision("highest"):
        for sep in (False, True):
            rt, cat, work = simulate.simulate_b_blocked_with_work(
                jax.random.key(3), p["ter"], p["st"], p["cr"], p["crsd"],
                p["av1"], p["av2"], p["av3"], p["sis"], sig, p["si"],
                nsim=512, chunk_size=64, use_kl=True, time_block_size=32,
                stable_kl_noise=stable, separable_kl=sep)
            runs[sep] = (np.asarray(rt), np.asarray(cat))
    (rt_d, cat_d), (rt_s, cat_s) = runs[False], runs[True]
    # float32 rounding differs between the two contractions, so a trial whose
    # path grazes the boundary can flip; anything beyond a handful is a bug.
    same = (rt_d == rt_s) & (cat_d == cat_s)
    assert same.mean() >= 0.99, same.mean()

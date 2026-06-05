"""
Smoke tests for Stage 6 K-L path through fofs_b_new and fit_simplex_b.

Lightweight checks that fofs_b_new + fit_simplex_b accept use_kl=True and
produce finite values. NOT a full benchmark — that lives on H100 (Phase 4).
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate as sim_b
from model_b import objective as obj_b
from model_b import fit as fit_b
from model_b.objective import COND_MAP_B, clamp_b
from shared import prng


TRUE_PARAMS = jnp.array([
    200.0, 50.0, 10.0, 2.0,
    12.0, 10.0, 0.5,
    15.0, 10.0, 8.0,
    14.0, 11.0, 9.0,
])


def _make_synthetic(true_params, key, nsim, chunk_size, use_kl):
    """Generate synthetic data via simulate_b at true params."""
    p = clamp_b(true_params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    props_l, counts_l, quants_l = [], [], []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(key, ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
            sis, sig, si, nsim=nsim, chunk_size=chunk_size, use_kl=use_kl,
        )
        cat_np = np.asarray(cat); rt_np = np.asarray(rt)
        props = np.array([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)])
        counts = np.array([(cat_np == c).sum() for c in (1, 2, 3, 4, 5)], dtype=np.int64)
        quants = np.zeros((5, 5))
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat_np == c
            if mask.sum() >= 5:
                quants[:, ki] = np.quantile(rt_np[mask], qs)
        props_l.append(jnp.asarray(props))
        counts_l.append(jnp.asarray(counts))
        quants_l.append(jnp.asarray(quants))
    return {"prop": jnp.stack(props_l), "count": jnp.stack(counts_l),
            "quant": jnp.stack(quants_l)}


def test_fofs_b_new_accepts_use_kl_default():
    """fofs_b_new should accept use_kl=False kwarg (fast, FFT path only)."""
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=32,
                          chunk_size=8, use_kl=False)
    np.random.seed(1)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.95, 1.05, size=13))
    val_fft = float(obj_b.fofs_b_new(x0, syn, jax.random.key(1),
                                      nsim=32, chunk_size=8, use_kl=False))
    assert np.isfinite(val_fft), f"FFT G^2 must be finite, got {val_fft}"


@pytest.mark.slow
def test_fofs_b_new_kl_path_returns_finite():
    """fofs_b_new with use_kl=True returns finite (slow because of K-L basis cost)."""
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=32,
                          chunk_size=8, use_kl=False)
    np.random.seed(1)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.95, 1.05, size=13))
    val_kl = float(obj_b.fofs_b_new(x0, syn, jax.random.key(1),
                                     nsim=32, chunk_size=8, use_kl=True))
    assert np.isfinite(val_kl), f"K-L G^2 must be finite, got {val_kl}"


def test_fit_simplex_b_accepts_use_kl():
    """fit_simplex_b should accept use_kl kwarg and produce a finite-loss result."""
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=32,
                          chunk_size=8, use_kl=False)
    np.random.seed(2)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.98, 1.02, size=13))
    res = fit_b.fit_simplex_b(
        syn, jax.random.key(1), x0,
        nsim=32, maxiter=3, chunk_size=8, use_kl=False,
    )
    assert np.isfinite(res.loss), f"loss must be finite, got {res.loss}"
    assert res.n_iters >= 0


@pytest.mark.slow
def test_kl_recovery_small_scale_completes():
    """
    Run a small simplex fit with use_kl=True and confirm it terminates.
    NOT a parameter-quality test (too small nsim for that) — just smoke.
    Slow because K=1325 GEMM is heavy on laptop CPU. Fast on H100.
    """
    nsim = 256
    chunk = 16
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=nsim,
                          chunk_size=chunk, use_kl=True)
    np.random.seed(2)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.95, 1.05, size=13))

    t0 = time.perf_counter()
    res = fit_b.fit_simplex_b(
        syn, jax.random.key(1), x0,
        nsim=nsim, maxiter=10, chunk_size=chunk, use_kl=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nK-L recovery (nsim={nsim}, maxiter=10) took {elapsed:.1f}s, "
          f"loss {res.loss:.2f}")
    assert np.isfinite(res.loss)
    assert res.n_iters > 0

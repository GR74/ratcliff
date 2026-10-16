"""Fortran variants: bench/fit zones, sv, binned CDF (qant1)."""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import objective as obj
from model_b import simulate

jax.config.update("jax_enable_x64", True)


def _radius(k, zone, row=50, col=80):
    cols = np.where(np.asarray(k)[row] == zone)[0]
    return (cols.max() - cols.min()) / 2.0


def test_fit_zones_are_smaller_than_bench_zones():
    fit, bench = simulate.zone_array_fit(6.0), simulate.zone_array(6.0)
    # twod3fitmpi.f: zone 1 radius ~4.5, zone 2 outer ~14.5; bench ~7.6 / ~19.7
    assert _radius(fit, 1) == 4.0
    assert _radius(bench, 1) == 7.0
    zone12_fit = np.isin(np.asarray(fit)[50], [1, 2]).sum()
    zone12_bench = np.isin(np.asarray(bench)[50], [1, 2]).sum()
    assert zone12_fit == 29 and zone12_bench == 39


def _qant1_reference(rt_ms, q_ms, e=10.0, n=400):
    """Line-by-line Python of Fortran accum binning + qant1 (1-based)."""
    rc = np.zeros(10001)
    for x in rt_ms / e:
        rc[int(x)] += 1.0
    r = np.zeros(n + 1)
    r[1:] = rc[1:n + 1] / len(rt_ms)
    rr = np.zeros(n + 1)
    for i in range(2, n + 1):
        rr[i] = rr[i - 1] + r[i]
    rr[1:] /= rr[n]
    art = q_ms / e
    irt = int(art)
    irt = max(irt, 1)
    irt = min(irt, n - 1)
    return max(rr[irt] + (art - irt) * (rr[irt + 1] - rr[irt]), 0.01)


def test_binned_cdf_matches_fortran_transliteration():
    rng = np.random.default_rng(0)
    rt = 250 + rng.gamma(3, 60, size=700)
    for q in [300.0, 347.5, 401.2, 512.9, 900.0]:
        ours = float(obj._cdf_fortran_binned(jnp.asarray(rt), jnp.ones(700, bool),
                                             jnp.asarray(q)))
        assert abs(ours - _qant1_reference(rt, q)) < 1e-12, q


def test_sv_matters_only_in_fortran_variants():
    args = (jax.random.key(1), 200., 45., 17., 7., 15., 10., 8., 11., 9.)
    kw = dict(nsim=256, chunk_size=64, use_kl=False, time_block_size=32)
    def run(sv, variant):
        rt, cat = simulate.simulate_b_blocked(*args, 6.0, **kw, sv=sv,
                                              fortran_variant=variant)
        return np.asarray(rt), np.asarray(cat)
    a, b = run(0.0, None), run(0.9, None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    for variant in ("bench", "fit"):
        c, d = run(0.0, variant), run(0.9, variant)
        assert not np.array_equal(c[0], d[0])

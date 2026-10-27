"""Even FFT embedding preserves the requested covariance, not individual draws."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import grf, simulate, objective, fit


@pytest.mark.parametrize("sig", [.2, 1., 3., 5., 8., 10., 12., 15., 17.])
def test_even_covariance_all_required_lags_and_spectral_symmetry(sig):
    covariance = np.asarray(grf.even_embedding(s1=sig, s2=sig))
    assert covariance.shape == (200, 320)
    np.testing.assert_array_equal(covariance, covariance[(-np.arange(200)) % 200][:, (-np.arange(320)) % 320])
    dy = np.arange(-99, 100)[:, None]
    dx = np.arange(-159, 160)[None, :]
    expected = (1-(dx*dx+dx*dy+dy*dy)/sig**2)*np.exp(-(dx*dx+dy*dy)/sig**2)
    np.testing.assert_allclose(covariance[dy % 200, dx % 320], expected, atol=1e-14, rtol=1e-12)
    spectral = np.fft.fft2(covariance)/covariance.size
    assert abs(spectral.imag).max() < 1e-15
    assert spectral.real.min() > -1e-14
    lam = np.asarray(grf.calc_LAM_even(s1=sig, s2=sig))
    effective = np.fft.fft2(lam**2).real
    np.testing.assert_allclose(effective[dy % 200, dx % 320], expected, atol=1e-10, rtol=0)


def test_nyquist_cross_term_is_averaged():
    c = np.asarray(grf.even_embedding(n=3, m=4, s1=2., s2=2.))
    expected = .5*(grf._kernel_value(jnp.array(1), jnp.array(3), 2., 2.)+
                   grf._kernel_value(jnp.array(-1), jnp.array(3), 2., 2.))
    np.testing.assert_allclose(c[3, 1], expected)


def test_even_generator_returns_original_grid():
    lam = grf.calc_LAM_even(n=4, m=6, s1=.5, s2=.5)
    a, b = grf.circulant_grf(lam, jnp.ones((8, 12)), jnp.zeros((8, 12)))
    assert a.shape == b.shape == (4, 6)


@pytest.mark.parametrize("sig", [0., .1, 17.1, float("nan")])
def test_fft200_rejects_unaudited_sig(sig):
    with pytest.raises(ValueError, match="audited"):
        simulate.simulate_b(jax.random.key(0), *([1.] * 8), sig, 6., 2, fft200=True)


def test_fft200_rejects_kl_combination():
    with pytest.raises(ValueError, match="isolated"):
        simulate.simulate_b(jax.random.key(0), *([1.] * 8), 10., 6., 2,
                            fft200=True, use_kl=True)


@pytest.mark.parametrize("block", [0, 16])
def test_objective_forwards_fft200(monkeypatch, block):
    seen = []
    def fake(*args):
        seen.append(args[-1])
        return jnp.full((2,), 300.), jnp.ones(2, dtype=jnp.int32)
    monkeypatch.setattr(simulate, "simulate_b_blocked" if block else "simulate_b", fake)
    data = {"prop": jnp.full((2, 5), .2), "count": jnp.full((2, 5), 100), "quant": jnp.full((2, 5, 5), 300.)}
    p = jnp.array([200., 50., 10., 2., 12., 10., .5, 15., 10., 8., 14., 11., 9.])
    assert np.isfinite(objective.fofs_b_new(p, data, jax.random.key(0), nsim=2,
                                           fft200=True, time_block_size=block))
    assert seen and all(seen)


def test_fit_forwards_fft200(monkeypatch):
    seen = []
    def loss(p, *args, **kwargs):
        seen.append(kwargs["fft200"])
        return float(jnp.sum(p*p))
    monkeypatch.setattr(fit.obj_b, "fofs_b_new", loss)
    fit.fit_simplex_b({}, jax.random.key(0), np.ones(13), maxiter=1, fft200=True)
    assert seen and all(seen)

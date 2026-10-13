import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import fit, grf_kl, simulate
from shared import blocked


def test_bucket_padding_retains_identical_modes_and_covariance(monkeypatch):
    monkeypatch.setattr(grf_kl.grf_circulant, "calc_LAM",
                        lambda **kwargs: jnp.sqrt(jnp.arange(20., 0., -1.).reshape(4, 5)))
    grf_kl._basis_cache.clear()
    grf_kl._basis_cache_order.clear()
    options = dict(sig=51.125, n=3, m=4, k_max=20,
                   variance_threshold=.7, pad_to_k_max=True)
    reference, k, variance = grf_kl.calc_kl_basis(**options)
    candidate, k_new, variance_new = grf_kl.calc_kl_basis(
        **options, mode_buckets=(8, 12, 20))
    assert k_new == k and variance_new == variance
    assert candidate.shape[1] < reference.shape[1]
    np.testing.assert_array_equal(candidate[:, :2 * k], reference[:, :2 * k])
    np.testing.assert_array_equal(candidate[:, 2 * k:], 0)
    np.testing.assert_allclose(candidate @ candidate.T, reference @ reference.T,
                               rtol=2e-6, atol=2e-5)
    grf_kl._basis_cache.clear()
    grf_kl._basis_cache_order.clear()


@pytest.mark.parametrize("buckets", [(8, 8, 20), (12, 8, 20), (8, 12), ()])
def test_invalid_mode_buckets_are_rejected(buckets):
    with pytest.raises(ValueError, match="mode buckets"):
        grf_kl.calc_kl_basis(10., k_max=20, mode_buckets=buckets)


def test_cumulative_weights_reconstruct_every_intermediate_field():
    rng = np.random.default_rng(2)
    weights = jnp.asarray(rng.normal(size=(3, 7, 5)), dtype=jnp.float64)
    basis = jnp.asarray(rng.normal(size=(9, 5)), dtype=jnp.float64)
    drift = jnp.asarray(rng.normal(size=9), dtype=jnp.float64)
    increments = jnp.einsum("bsk,Nk->bsN", weights, basis) + drift
    reference = jnp.cumsum(increments - increments.mean(axis=-1, keepdims=True), axis=1)
    candidate = jnp.einsum("bsk,Nk->bsN", jnp.cumsum(weights, axis=1), basis)
    candidate += jnp.arange(1, 8)[None, :, None] * drift
    candidate -= candidate.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(candidate, reference, rtol=2e-12, atol=2e-12)


def test_cumulative_paths_do_not_skip_a_crossing_followed_by_a_fall():
    keys = blocked.indexed_keys(jax.random.key(0), jnp.arange(1))
    increments = jnp.asarray([[[3., -3.], [-4., 4.], [0., 0.], [0., 0.]]])
    options = dict(n_steps=4, block_size=4)
    arguments = (keys, jnp.asarray([2.]), jnp.zeros(2), jnp.asarray([True]))
    reference = blocked.first_passage(*arguments, lambda keys: increments, **options)
    candidate = blocked.first_passage(*arguments, lambda keys: jnp.cumsum(increments, axis=1),
                                     noise_is_path=True, **options)
    np.testing.assert_array_equal(candidate.steps, reference.steps)
    np.testing.assert_array_equal(candidate.positions, reference.positions)
    assert candidate.steps.tolist() == [1]


def test_active_fit_preserves_full_vector_and_fixed_sv(monkeypatch):
    active = [i for i in range(13) if i != 6]
    monkeypatch.setattr(fit.obj_b, "fofs_b_new",
                        lambda p, *args, **kwargs: float(np.sum(np.asarray(p)[active] ** 2)))
    x0 = np.ones(13)
    x0[6] = .73
    callbacks = []
    result = fit.fit_simplex_b({}, jax.random.key(0), x0, nsim=8, maxiter=1200,
                              optimize_active_only=True,
                              on_update=lambda n, loss, p: callbacks.append(p))
    assert result.params.shape == (13,)
    assert float(result.params[6]) == x0[6]
    assert np.linalg.norm(np.asarray(result.params)[active]) < .01
    assert callbacks and all(p.shape == (13,) and p[6] == x0[6] for p in callbacks)
    np.testing.assert_array_equal(x0, np.where(np.arange(13) == 6, .73, 1.))


def test_kl_options_reject_fft_before_simulation():
    with pytest.raises(ValueError, match="K-L extensions"):
        simulate.simulate_b_blocked_with_work(
            jax.random.key(0), *([1.] * 10), nsim=1, use_kl=False,
            cumulative_kl=True)

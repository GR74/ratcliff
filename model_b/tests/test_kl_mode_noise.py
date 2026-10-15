"""Regressions for assigning random coefficients by physical Fourier mode."""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf_kl, simulate
from shared import blocked


def test_mode_coefficients_survive_permutation_insertion_and_padding():
    keys = blocked.indexed_keys(jax.random.key(17), jnp.arange(2))
    ids = jnp.asarray([8, 3, 27], dtype=jnp.uint32)
    changed = jnp.asarray([27, 19, 8, 3, 0xffffffff], dtype=jnp.uint32)
    a = grf_kl.mode_normal_blocks(keys, ids, 7, jnp.float32).reshape(2, 7, 3, 2)
    b = grf_kl.mode_normal_blocks(keys, changed, 7, jnp.float32).reshape(2, 7, 5, 2)
    np.testing.assert_array_equal(a, b[:, :, [2, 3, 0], :])
    assert not np.array_equal(a[:, :, 0], a[:, :, 1])


def test_real_basis_noise_is_continuous_at_unchanged_retained_set():
    options = dict(n=20, m=32, k_max=200, variance_threshold=.9,
                   pad_to_k_max=True, return_mode_ids=True)
    a, ka, _, ids_a = grf_kl.calc_kl_basis(2.2, **options)
    b, kb, _, ids_b = grf_kl.calc_kl_basis(2.200001, **options)
    assert ka == kb and set(np.asarray(ids_a[:ka])) == set(np.asarray(ids_b[:kb]))
    keys = blocked.indexed_keys(jax.random.key(4), jnp.arange(2))
    ranked = blocked.normal_blocks(keys, 8, a.shape[1], jnp.float32)
    old_a, old_b = ranked @ a.T, ranked @ b.T
    assert np.linalg.norm(old_a-old_b) / np.linalg.norm(old_a) > .5
    z_a = grf_kl.mode_normal_blocks(keys, ids_a, 8, jnp.float32)
    z_b = grf_kl.mode_normal_blocks(keys, ids_b, 8, jnp.float32)
    fields_a, fields_b = z_a @ a.T, z_b @ b.T
    relative_change = np.linalg.norm(fields_a-fields_b) / np.linalg.norm(fields_a)
    assert relative_change < 1e-4


def test_mode_metadata_preserves_historical_basis_and_covariance():
    options = dict(n=20, m=32, k_max=200, variance_threshold=.9, pad_to_k_max=True)
    old, k, variance = grf_kl.calc_kl_basis(2.2, **options)
    new, k_new, variance_new, ids = grf_kl.calc_kl_basis(2.2, **options, return_mode_ids=True)
    np.testing.assert_array_equal(new, old)
    assert k_new == k and variance_new == variance
    assert len(np.unique(np.asarray(ids[:k]))) == k
    assert np.all(np.asarray(ids[k:]) == np.uint32(0xffffffff))


def test_mode_coefficients_keep_independent_unit_normal_covariance():
    keys = blocked.indexed_keys(jax.random.key(21), jnp.arange(1))
    z = np.asarray(grf_kl.mode_normal_blocks(keys, jnp.asarray([3, 7, 11, 19]),
                                            32768, jnp.float32))[0]
    np.testing.assert_allclose(z.mean(axis=0), 0., atol=.03)
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(8), atol=.04)


def test_conjugate_cutoff_representative_flip_preserves_field(monkeypatch):
    def lam(**kwargs):
        values = np.zeros((3, 5))
        values.flat[1] = 2. + (1e-7 if kwargs["s1"] == 10 else -1e-7)
        values.flat[4] = 4. - values.flat[1]
        return jnp.sqrt(jnp.asarray(values))

    monkeypatch.setattr(grf_kl.grf_circulant, "calc_LAM", lam)
    grf_kl._basis_cache.clear()
    grf_kl._basis_cache_order.clear()
    options = dict(n=3, m=3, k_max=4, variance_threshold=.4,
                   pad_to_k_max=True, return_mode_ids=True)
    a, ka, _, ids_a = grf_kl.calc_kl_basis(10., **options)
    b, kb, _, ids_b = grf_kl.calc_kl_basis(10.000001, **options)
    assert ka == kb == 1
    assert ids_a[0] != ids_b[0]
    assert ids_a[0] & grf_kl.MODE_KEY_MASK == ids_b[0] & grf_kl.MODE_KEY_MASK
    keys = blocked.indexed_keys(jax.random.key(12), jnp.arange(2))
    fields_a = grf_kl.mode_normal_blocks(keys, ids_a, 8) @ a.T
    fields_b = grf_kl.mode_normal_blocks(keys, ids_b, 8) @ b.T
    np.testing.assert_allclose(fields_a, fields_b, atol=1e-6, rtol=1e-6)
    both, k, _, codes = grf_kl.calc_kl_basis(10., **{**options, "variance_threshold": 1.})
    assert k == 2 and len(np.unique(np.asarray(codes[:k]) & grf_kl.MODE_KEY_MASK)) == 2
    z = np.asarray(grf_kl.mode_normal_blocks(keys[:1], codes[:k], 32768))[0]
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(4), atol=.04)
    grf_kl._basis_cache.clear()
    grf_kl._basis_cache_order.clear()


def test_repaired_wrappers_use_ids_and_preserve_paired_stopping(monkeypatch):
    rng = np.random.default_rng(2)
    basis = jnp.asarray(rng.normal(size=(16000, 6)), dtype=jnp.float32)
    ids = jnp.asarray([8, 3, 27], dtype=jnp.uint32)
    permutation = np.asarray([2, 0, 1])
    calls = []

    def provide_basis(**kwargs):
        assert kwargs["return_mode_ids"] is True
        order = permutation if kwargs["sig"] > 10 else np.arange(3)
        columns = np.c_[2 * order, 2 * order + 1].ravel()
        calls.append(kwargs["sig"])
        return basis[:, columns], 3, .99, ids[order]

    monkeypatch.setattr(grf_kl, "calc_kl_basis", provide_basis)
    params = dict(ter=200., st=50., cr=10., crsd=2., av1=15., av2=10., av3=8.,
                  sis=12., si=6., nsim=3, chunk_size=3, use_kl=True, stable_kl_noise=True)
    key = jax.random.key(9)
    for function in (simulate.simulate_b, simulate.simulate_b_blocked):
        a = function(key, sig=10., **params)
        b = function(key, sig=10.000001, **params)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    reference = simulate.simulate_b_blocked(key, sig=10., stop_early=False, **params)
    candidate = simulate.simulate_b_blocked(key, sig=10., stop_early=True, **params)
    np.testing.assert_array_equal(reference[0], candidate[0])
    np.testing.assert_array_equal(reference[1], candidate[1])
    assert len(calls) == 6

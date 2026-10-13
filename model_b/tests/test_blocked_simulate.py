import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate


PARAMS = (200., 50., 10., 2., 15., 10., 8., 12., 10., 6.)


@pytest.fixture
def small_basis(monkeypatch):
    # Exercise the real wrapper/core without a production-size CPU GEMM.
    rng = np.random.default_rng(0)
    basis = jnp.asarray(rng.normal(size=(100 * 160, 8)), dtype=jnp.float32)
    monkeypatch.setattr(simulate.grf_kl, "calc_kl_basis",
                        lambda **kwargs: (basis, 4, 0.99))


def test_blocked_model_b_matches_full_horizon_control(small_basis):
    key = jax.random.key(7)
    control = simulate.simulate_b_blocked_with_work(
        key, *PARAMS, 11, 8, True, 32, False, False,
    )
    early = simulate.simulate_b_blocked_with_work(
        key, *PARAMS, 11, 8, True, 32, True, True,
    )
    np.testing.assert_array_equal(early[0], control[0])
    np.testing.assert_array_equal(early[1], control[1])
    assert int(early[2]) < int(control[2])


def test_model_b_trials_are_nested_across_budget(small_basis):
    key = jax.random.key(3)
    a = simulate.simulate_b_blocked(key, *PARAMS, 11, 8, True, 32)
    b = simulate.simulate_b_blocked(key, *PARAMS, 19, 8, True, 32)
    np.testing.assert_array_equal(a[0], b[0][:11])
    np.testing.assert_array_equal(a[1], b[1][:11])


def test_blocked_preserves_model_b_no_crossing_fallback(small_basis):
    params = (200., 0., 1e8, 0., 0., 0., 0., 12., 10., 6.)
    rt, cat = simulate.simulate_b_blocked(jax.random.key(0), *params, 3, 8, True, 32)
    np.testing.assert_array_equal(rt, [4200.] * 3)
    np.testing.assert_array_equal(cat, [5] * 3)


def test_blocked_fft_path_checks_actual_grid():
    # Low threshold finishes the first block; this runs the real FFT sampler.
    params = (200., 0., 0.1, 0., 15., 10., 8., 12., 10., 6.)
    rt, cat, work = simulate.simulate_b_blocked_with_work(
        jax.random.key(0), *params, 2, 2, False, 4,
    )
    assert np.all(np.asarray(rt) == 210.)
    assert set(np.asarray(cat).tolist()) <= {1, 2, 3, 4, 5}
    assert int(work) == 8

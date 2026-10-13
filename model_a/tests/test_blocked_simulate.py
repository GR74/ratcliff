import jax
import numpy as np

from model_a import simulate


PARAMS = (200., 50., 50., 10., 4., 5., 20.)


def test_blocked_matches_full_horizon_control_with_padding():
    key = jax.random.key(7)
    control = simulate.simulate_blocked_with_work(
        key, *PARAMS, 11, 8, 32, False, False,
    )
    early = simulate.simulate_blocked_with_work(
        key, *PARAMS, 11, 8, 32, True, True,
    )
    np.testing.assert_array_equal(early[0], control[0])
    np.testing.assert_array_equal(early[1], control[1])
    assert int(early[2]) < int(control[2])


def test_blocked_trials_are_nested_across_budget_and_chunk_size():
    key = jax.random.key(3)
    a = simulate.simulate_blocked(key, *PARAMS, 11, 8, 32)
    b = simulate.simulate_blocked(key, *PARAMS, 23, 16, 32)
    np.testing.assert_allclose(a[0], b[0][:11], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(a[1], b[1][:11])


def test_blocked_preserves_model_a_no_crossing_fallback():
    rt, cat = simulate.simulate_blocked(
        jax.random.key(0), 200., 0., 1e8, 0., 4., 5., 0., 3, 8, 32,
    )
    np.testing.assert_array_equal(rt, [4200.] * 3)
    np.testing.assert_array_equal(cat, [3] * 3)

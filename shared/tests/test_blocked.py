"""Deterministic checks for crossings, tails, compaction and indexed RNG."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shared import blocked


def _run(thresholds, drift, *, n_steps=12, block_size=4, stop_early=True,
         compact=True, random_noise=False, valid=None):
    thresholds = jnp.asarray(thresholds, dtype=jnp.float64)
    drift = jnp.asarray(drift, dtype=jnp.float64)
    keys = blocked.indexed_keys(jax.random.key(0), jnp.arange(len(thresholds)))
    valid = jnp.ones(thresholds.shape, dtype=bool) if valid is None else jnp.asarray(valid)

    def sample(keys):
        if random_noise:
            return blocked.normal_blocks(keys, block_size, len(drift), drift.dtype)
        return jnp.zeros((len(keys), block_size, len(drift)), dtype=drift.dtype)

    return jax.jit(lambda: blocked.first_passage(
        keys, thresholds, drift, valid, sample, n_steps=n_steps,
        block_size=block_size, stop_early=stop_early, compact=compact,
    ))()


def test_checks_crossing_inside_block_not_only_at_end():
    out = _run([2.5], [1., -1.])
    assert out.steps.tolist() == [3]
    assert out.positions.tolist() == [0]
    assert out.crossed.tolist() == [True]
    assert int(out.computed_trial_steps) == 4


def test_threshold_is_strict_and_no_crossing_keeps_cap():
    out = _run([1., 100.], [1., -1.])
    assert out.steps.tolist() == [2, 12]
    assert out.crossed.tolist() == [True, False]


def test_padded_final_steps_cannot_create_false_crossing():
    out = _run([10.5], [1., -1.], n_steps=10, block_size=4)
    assert out.steps.tolist() == [10]
    assert out.crossed.tolist() == [False]


def test_does_not_skip_a_crossing_followed_by_a_fall():
    keys = blocked.indexed_keys(jax.random.key(0), jnp.arange(1))

    def sample(keys):
        return jnp.asarray([[[3., -3.], [-4., 4.], [0., 0.], [0., 0.]]])

    out = blocked.first_passage(keys, jnp.asarray([2.]), jnp.zeros(2),
                                 jnp.asarray([True]), sample, n_steps=4,
                                 block_size=4)
    assert out.steps.tolist() == [1]
    assert out.positions.tolist() == [0]


def test_early_stop_and_compaction_preserve_trial_noise_and_outputs():
    args = ([1.] * 15 + [40.], [1., -.2, -.8])
    control = _run(*args, n_steps=30, block_size=4, stop_early=False,
                   random_noise=True)
    early = _run(*args, n_steps=30, block_size=4, random_noise=True)
    np.testing.assert_array_equal(early.steps, control.steps)
    np.testing.assert_array_equal(early.positions, control.positions)
    np.testing.assert_array_equal(early.crossed, control.crossed)
    assert int(early.computed_trial_steps) < int(control.computed_trial_steps)


def test_no_valid_trials_generate_no_noise():
    out = _run([2., 2.], [1., -1.], valid=[False, False])
    assert int(out.computed_trial_steps) == 0
    assert not np.asarray(out.crossed).any()


def test_indexed_randomness_is_nested_across_trial_counts():
    a, ka = blocked.trial_inputs(jax.random.key(4), jnp.arange(3), jnp.float64)
    b, kb = blocked.trial_inputs(jax.random.key(4), jnp.arange(9), jnp.float64)
    np.testing.assert_array_equal(a, b[:3])
    np.testing.assert_array_equal(jax.random.key_data(ka), jax.random.key_data(kb[:3]))


def test_invalid_block_size_rejected():
    with pytest.raises(ValueError, match="block_size"):
        _run([2.], [1., -1.], block_size=0)

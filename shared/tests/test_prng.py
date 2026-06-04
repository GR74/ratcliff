import jax
import jax.numpy as jnp
import pytest

from shared import prng


def test_root_key_is_deterministic_for_same_seed():
    k1 = prng.root_key(42)
    k2 = prng.root_key(42)
    assert jnp.array_equal(k1, k2)


def test_root_key_differs_for_different_seeds():
    k1 = prng.root_key(42)
    k2 = prng.root_key(43)
    assert not jnp.array_equal(k1, k2)


def test_split_for_condition_is_deterministic():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=2)
    b = prng.split_for_condition(k, condition_idx=2)
    assert jnp.array_equal(a, b)


def test_split_for_condition_differs_across_conditions():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=0)
    b = prng.split_for_condition(k, condition_idx=1)
    assert not jnp.array_equal(a, b)


def test_trial_keys_returns_n_distinct_keys():
    k = prng.root_key(7)
    keys = prng.trial_keys(k, n=100)
    assert keys.shape == (100, 2)  # JAX keys are (2,) uint32 pairs
    # All keys should be distinct
    flat = keys.reshape(100, -1)
    assert len({tuple(row.tolist()) for row in flat}) == 100

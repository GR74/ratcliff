import jax
import jax.numpy as jnp
import pytest

from shared import prng


def test_root_key_is_deterministic_for_same_seed():
    k1 = prng.root_key(42)
    k2 = prng.root_key(42)
    assert jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))


def test_root_key_differs_for_different_seeds():
    k1 = prng.root_key(42)
    k2 = prng.root_key(43)
    assert not jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))


def test_split_for_condition_is_deterministic():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=2)
    b = prng.split_for_condition(k, condition_idx=2)
    assert jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b))


def test_split_for_condition_differs_across_conditions():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=0)
    b = prng.split_for_condition(k, condition_idx=1)
    assert not jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b))


def test_trial_keys_returns_n_distinct_keys():
    k = prng.root_key(7)
    keys = prng.trial_keys(k, n=100)
    # Typed keys: outer shape is (n,), underlying buffer is (n, 2) uint32
    assert keys.shape == (100,)
    raw = jax.random.key_data(keys)
    assert raw.shape == (100, 2)
    # All keys should be distinct
    assert len({tuple(row.tolist()) for row in raw}) == 100


def test_root_key_returns_typed_key():
    """The migrated API returns a typed key, not a raw uint32 array."""
    k = prng.root_key(0)
    # Typed keys have dtype kind that's not 'u' or 'i' (it's a special key dtype)
    assert jnp.issubdtype(k.dtype, jax.dtypes.prng_key)

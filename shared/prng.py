"""Deterministic PRNG helpers for reproducible Monte Carlo runs."""
import jax


def root_key(seed: int):
    """Top-level JAX PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def split_for_condition(key, condition_idx: int):
    """Derive a condition-specific subkey deterministically from a root key."""
    return jax.random.fold_in(key, condition_idx)


def trial_keys(key, n: int):
    """Return an (n, 2) array of distinct trial-level subkeys."""
    return jax.random.split(key, n)

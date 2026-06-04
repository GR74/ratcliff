"""Deterministic PRNG helpers for reproducible Monte Carlo runs.

Uses JAX's typed-key API (jax.random.key) introduced in 0.4.16+. The typed
key is opaque; access the raw (n, 2) uint32 buffer via jax.random.key_data
when needed for distinctness checks.
"""
import jax


def root_key(seed: int):
    """Top-level JAX typed PRNG key from an integer seed."""
    return jax.random.key(seed)


def split_for_condition(key, condition_idx: int):
    """Derive a condition-specific subkey deterministically from a root key."""
    return jax.random.fold_in(key, condition_idx)


def trial_keys(key, n: int):
    """Return n distinct trial-level subkeys as a (n,) shape typed-key array."""
    return jax.random.split(key, n)

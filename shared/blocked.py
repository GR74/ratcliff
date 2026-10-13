"""Optional time-blocked first-passage simulation with stable trial randomness.

The legacy simulators remain the default. This implementation supplies both a
full-horizon control and an early-stop variant using identical per-trial,
per-block noise keys. Compaction uses a small set of static batch sizes rather
than multiplying an active mask against an otherwise full-size calculation.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class PassageResult(NamedTuple):
    steps: jax.Array
    positions: jax.Array
    crossed: jax.Array
    computed_trial_steps: jax.Array


class _State(NamedTuple):
    block: jax.Array
    accumulator: jax.Array
    active: jax.Array
    steps: jax.Array
    positions: jax.Array
    crossed: jax.Array
    computed_trial_steps: jax.Array


def indexed_keys(key, indices):
    """Keys depend on absolute trial ID, not requested nsim or chunk size."""
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(indices)


def trial_inputs(key, indices, dtype):
    """Independent uniforms/noise streams, stable across staged trial counts."""
    keys = indexed_keys(key, indices)
    uniform_keys = jax.vmap(lambda k: jax.random.fold_in(k, 0))(keys)
    noise_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(keys)
    uniforms = jax.vmap(lambda k: jax.random.uniform(k, (10,), dtype=dtype))(
        uniform_keys,
    )
    return uniforms, noise_keys


def normal_blocks(keys, block_size, width, dtype):
    """One normal array per trial; compaction does not change its random draws."""
    return jax.vmap(
        lambda k: jax.random.normal(k, (block_size, width), dtype=dtype),
    )(keys)


def _batch_sizes(size):
    sizes = []
    current = min(8, size)
    while current < size:
        sizes.append(current)
        current *= 2
    return tuple(sizes + [size])


def first_passage(noise_keys, thresholds, drift, valid_trials,
                  sample_noise: Callable, *, n_steps=400, block_size=32,
                  stop_early=True, compact=True):
    """Return first crossings, preserving the shared upper-boundary rule.

    sample_noise(keys) returns (batch, block_size, locations). It receives
    trial/block keys so an unfinished trial sees the same noise when others
    finish. All intermediate steps are checked, including a short final block.
    Full-horizon control: stop_early=False (also disables compaction).

    Keep the condition loop outside vmap: batched conditional predicates can
    execute all lax.switch branches and defeat the purpose of compaction.
    """
    if not isinstance(block_size, int) or not 1 <= block_size <= n_steps:
        raise ValueError(f"block_size must be an integer in [1, {n_steps}]")
    size = thresholds.shape[0]
    if size < 1:
        raise ValueError("at least one trial is required")
    n_blocks = (n_steps + block_size - 1) // block_size
    state = _State(
        jnp.int32(0),
        jnp.zeros((size, drift.shape[0]), dtype=drift.dtype),
        valid_trials,
        jnp.full((size,), n_steps, dtype=jnp.int32),
        jnp.zeros((size,), dtype=jnp.int32),
        jnp.zeros((size,), dtype=jnp.bool_),
        jnp.int32(0),
    )
    sizes = _batch_sizes(size) if stop_early and compact else (size,)

    def make_branch(batch_size):
        def advance(s):
            if stop_early and compact:
                indices = jnp.argsort(~s.active, stable=True)[:batch_size]
            else:
                indices = jnp.arange(size)
            active = s.active[indices]
            block_keys = jax.vmap(lambda k: jax.random.fold_in(k, s.block))(
                noise_keys[indices],
            )
            increments = sample_noise(block_keys) + drift[None, None, :]
            increments -= increments.mean(axis=-1, keepdims=True)
            paths = jnp.cumsum(increments, axis=1)
            paths += s.accumulator[indices, None, :]
            start = s.block * block_size
            within_horizon = start + jnp.arange(block_size) < n_steps
            hits = (paths.max(axis=-1) > thresholds[indices, None])
            hits &= within_horizon[None, :] & active[:, None]
            hit = hits.any(axis=1)
            first = jnp.argmax(hits, axis=1).astype(jnp.int32)
            position = jnp.argmax(
                paths[jnp.arange(batch_size), first], axis=-1,
            ).astype(jnp.int32)
            last = jnp.minimum(block_size, n_steps - start) - 1
            accumulator = jnp.where(
                active[:, None], paths[:, last, :], s.accumulator[indices],
            )
            steps = jnp.where(hit, start + first + 1, s.steps[indices])
            positions = jnp.where(hit, position, s.positions[indices])
            return _State(
                s.block + 1,
                s.accumulator.at[indices].set(accumulator),
                s.active.at[indices].set(active & ~hit),
                s.steps.at[indices].set(steps),
                s.positions.at[indices].set(positions),
                s.crossed.at[indices].set(s.crossed[indices] | hit),
                s.computed_trial_steps + batch_size * block_size,
            )
        return advance

    branches = tuple(make_branch(batch_size) for batch_size in sizes)

    def cond(s):
        unfinished = s.active.any() if stop_early else jnp.bool_(True)
        return (s.block < n_blocks) & unfinished

    def body(s):
        count = s.active.sum()
        branch = jnp.sum(count > jnp.asarray(sizes[:-1]))
        return jax.lax.switch(branch, branches, s)

    result = jax.lax.while_loop(cond, body, state)
    return PassageResult(result.steps, result.positions, result.crossed,
                         result.computed_trial_steps)

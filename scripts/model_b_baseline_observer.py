"""Expose existing first-passage results in an isolated measurement module.

The frozen simulator is never edited or monkey-patched. The two substitutions
only return values its first-passage kernel already computes. Callers must check
RT/category parity before relying on this instrumented module. It is excluded
from all performance measurements and optimizer evaluations.
"""
from pathlib import Path
import types


def load_observer(source):
    path = Path(source) / 'model_b/simulate.py'
    text = path.read_text()
    old = '''        return rt, cat, result.computed_trial_steps

    rts, cats, work = jax.lax.map(run_chunk, jnp.arange(n_chunks))
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim], work.sum()'''
    new = '''        return rt, cat, result.computed_trial_steps, result.steps, result.positions, result.crossed

    rts, cats, work, steps, positions, crossed = jax.lax.map(run_chunk, jnp.arange(n_chunks))
    return (rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim], work.sum(),
            steps.reshape(-1)[:nsim], positions.reshape(-1)[:nsim], crossed.reshape(-1)[:nsim])'''
    if text.count(old) != 1:
        raise ValueError('Frozen-source observer pattern changed; inspect before adapting')
    module = types.ModuleType('model_b_baseline_observed')
    module.__file__ = str(path)
    exec(compile(text.replace(old, new), str(path) + ':observed', 'exec'), module.__dict__)
    return module


def sample_observed(module, params, seed, nsim=9000, stop_early=True, kl_bucketed=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from model_b import objective
    from shared import prng
    p = objective.clamp_b(jnp.asarray(params))
    outputs = []
    for ci, (d1, d2, d3) in enumerate(objective.COND_MAP_B):
        outputs.append(module.simulate_b_blocked_with_work(
            prng.split_for_condition(jax.random.key(seed), ci),
            p[0], p[1], p[2], p[3], p[d1], p[d2], p[d3], p[4], float(p[5]), 6.,
            nsim, 8, True, 32, stop_early, True, stable_kl_noise=True,
            kl_bucketed=kl_bucketed))
    jax.block_until_ready(outputs)
    rt, cat, work, steps, pos, crossed = [np.stack([np.asarray(o[i]) for o in outputs]) for i in range(6)]
    # Manuscript Experiment 9, p.38: fitted spatial parameters are pixels / 2.
    xy_pixels = 2. * np.stack([pos % module.M, pos // module.M], axis=-1)
    # Uncrossed positions are kernel placeholders, not observed response locations.
    xy_pixels = np.where(crossed[..., None], xy_pixels, np.nan)
    return dict(rt=rt, cat=cat, work=work, steps=steps, positions=pos,
                crossed=crossed, xy_pixels=xy_pixels)


def sliced_wasserstein(a, b, directions):
    import numpy as np
    from scipy.stats import wasserstein_distance
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a).all(axis=-1)]
    b = b[np.isfinite(b).all(axis=-1)]
    if not len(a) or not len(b):
        return None
    return float(np.mean([wasserstein_distance(a @ d, b @ d) for d in directions]))

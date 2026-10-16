"""Soft-count G2: tau=0 is the hard (Fortran) count; tau>0 is smooth in RT."""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import objective as obj

jax.config.update("jax_enable_x64", True)


def _synthetic(seed=0, n=600):
    rng = np.random.default_rng(seed)
    cat = rng.choice([1, 2, 3, 4, 5], size=n, p=[.3, .4, .2, .07, .03])
    rt = 300 + rng.gamma(3, 120, size=n)
    obs_rt = 300 + rng.gamma(3, 115, size=n)
    obs_cat = rng.choice([1, 2, 3, 4, 5], size=n, p=[.29, .42, .2, .07, .02])
    prop = np.array([(obs_cat == k).mean() for k in range(1, 6)])
    count = np.array([(obs_cat == k).sum() for k in range(1, 6)])
    quant = np.stack([np.quantile(obs_rt[obs_cat == k], [.1, .3, .5, .7, .9])
                      for k in range(1, 6)], axis=1)
    return jnp.asarray(rt), jnp.asarray(cat), prop, count, quant


def test_tau_zero_is_hard_count():
    rt, cat, prop, count, quant = _synthetic()
    hard = obj.condition_g2_b(rt, cat, prop, count, quant)
    zero = obj.condition_g2_b(rt, cat, prop, count, quant, soft_tau_ms=0.0)
    assert float(hard) == float(zero)


def test_small_tau_converges_to_hard():
    rt, cat, prop, count, quant = _synthetic()
    hard = float(obj.condition_g2_b(rt, cat, prop, count, quant))
    soft = float(obj.condition_g2_b(rt, cat, prop, count, quant, soft_tau_ms=1e-4))
    assert abs(soft - hard) < 1e-6 * max(1.0, abs(hard))


def test_soft_is_smooth_in_rt_shift_hard_is_not():
    rt, cat, prop, count, quant = _synthetic()
    shifts = np.linspace(-20, 20, 401)          # 0.1 ms grid, like moving ter
    def curve(tau):
        return np.array([float(obj.condition_g2_b(rt + d, cat, prop, count, quant,
                                                  soft_tau_ms=tau)) for d in shifts])
    hard, soft = curve(0.0), curve(5.0)
    # Hard count: piecewise constant -> many exactly-flat steps then jumps.
    assert (np.diff(hard) == 0).mean() > 0.3
    # Soft count: no flat steps, and no jump much larger than its typical step.
    d = np.abs(np.diff(soft))
    assert (d == 0).mean() == 0.0
    assert d.max() < 5 * np.median(d)

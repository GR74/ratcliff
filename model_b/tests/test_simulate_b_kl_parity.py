"""
End-to-end parity: simulate_b with use_kl=True vs use_kl=False.

We don't expect bit-exact agreement (different random sampling paths).
We test that aggregate statistics (RT mean, RT std, category proportions)
match within Monte Carlo tolerance at nsim=2000.

Marked slow because K=1325 GEMM is heavy on laptop CPU (~36 min). This
test is the right shape for an H100 run where it completes in seconds.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate as sim_b


def _summary_stats(rt, cat):
    """RT mean, RT std, category proportions."""
    rt_np = np.asarray(rt)
    cat_np = np.asarray(cat)
    props = np.array([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)])
    return rt_np.mean(), rt_np.std(), props


@pytest.mark.slow
def test_simulate_b_kl_matches_fft_in_aggregate():
    """K-L and FFT paths should give matching aggregate statistics."""
    nsim = 2000
    params = dict(
        ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
    )

    key_fft = jax.random.key(42)
    key_kl = jax.random.key(43)

    rt_fft, cat_fft = sim_b.simulate_b(
        key_fft, **params, nsim=nsim, chunk_size=16, use_kl=False,
    )
    rt_kl, cat_kl = sim_b.simulate_b(
        key_kl, **params, nsim=nsim, chunk_size=16, use_kl=True,
    )

    rt_mean_fft, rt_std_fft, props_fft = _summary_stats(rt_fft, cat_fft)
    rt_mean_kl, rt_std_kl, props_kl = _summary_stats(rt_kl, cat_kl)

    rel_err_mean = abs(rt_mean_kl - rt_mean_fft) / rt_mean_fft
    assert rel_err_mean < 0.03, (
        f"RT mean mismatch: FFT={rt_mean_fft:.1f}, K-L={rt_mean_kl:.1f}, "
        f"rel_err={rel_err_mean:.4f}"
    )

    rel_err_std = abs(rt_std_kl - rt_std_fft) / rt_std_fft
    assert rel_err_std < 0.10, (
        f"RT std mismatch: FFT={rt_std_fft:.1f}, K-L={rt_std_kl:.1f}, "
        f"rel_err={rel_err_std:.4f}"
    )

    prop_diff = np.abs(props_kl - props_fft).max()
    assert prop_diff < 0.05, (
        f"Category proportion mismatch: FFT={props_fft}, K-L={props_kl}, "
        f"max_diff={prop_diff:.4f}"
    )

"""
Performance test: new simulator should beat jax_port.simulate on CPU.

Marked `@pytest.mark.perf` so it doesn't run in the smoke gate by default.
Run explicitly with: pytest -m perf model_a/tests/test_simulate_perf.py
"""
import time

import jax
import pytest

from model_a import jax_port, simulate as sim_new
from shared import prng


@pytest.mark.perf
def test_simulate_cpu_speedup():
    """Wall-clock comparison after warmup. Target: >= 5x speedup on CPU."""
    params = dict(ter=200.0, st=50.0, cr=50.0, crsd=10.0,
                  si=4.0, sig=5.0, av=20.0)
    nsim = 4000
    key = prng.root_key(0)

    # Warmup both
    rt_old, _ = jax_port.simulate(key, **params, sv=0.7, nsim=nsim)
    rt_old.block_until_ready()
    rt_new, _ = sim_new.simulate(key, **params, nsim=nsim, chunk_size=256)
    rt_new.block_until_ready()

    # Time old
    n_iter = 3
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rt, _ = jax_port.simulate(key, **params, sv=0.7, nsim=nsim)
        rt.block_until_ready()
    t_old = (time.perf_counter() - t0) / n_iter

    # Time new
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rt, _ = sim_new.simulate(key, **params, nsim=nsim, chunk_size=256)
        rt.block_until_ready()
    t_new = (time.perf_counter() - t0) / n_iter

    speedup = t_old / t_new
    print(f"\n  jax_port.simulate:  {t_old*1000:.1f} ms / call")
    print(f"  sim_new.simulate:   {t_new*1000:.1f} ms / call")
    print(f"  speedup: {speedup:.2f}x")

    # Soft target: 5x. Hard target: at least not slower than the oracle.
    assert t_new <= t_old, f"new simulator is SLOWER: {t_new*1000:.1f}ms vs {t_old*1000:.1f}ms"
    # Print warning if we miss the soft target but don't fail
    if speedup < 5.0:
        print(f"  WARNING: speedup {speedup:.2f}x is below the 5x soft target")

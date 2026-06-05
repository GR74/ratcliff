"""Quick smoke + timing test for the optimized simulate_b. ~30 seconds total."""
import jax
import time
from model_b import simulate as sim_b

p = dict(ter=200.0, st=50.0, cr=10.0, crsd=2.0,
         av1=15.0, av2=10.0, av3=8.0,
         sis=12.0, sig=10.0, si=6.0)
key = jax.random.key(0)

print(f"JAX device: {jax.devices()[0]}")
print("Warming up (JIT compile may take 30-90s)...")
rt, _ = sim_b.simulate_b(key, **p, nsim=512, chunk_size=64)
rt.block_until_ready()
print("Compiled. Timing 3 iterations...")

t0 = time.perf_counter()
for _ in range(3):
    rt, _ = sim_b.simulate_b(key, **p, nsim=512, chunk_size=64)
    rt.block_until_ready()
elapsed = (time.perf_counter() - t0) / 3
extrapolated = elapsed * (9000.0 / 512.0)

print("")
print(f"NEW nsim=512, chunk=64:    {elapsed*1000:.0f} ms/call")
print(f"Extrapolated to nsim=9000: {extrapolated:.2f}s")
print(f"OLD Stage 4 chunk=64:      25.24s for nsim=9000")
print(f"SPEEDUP: {25.24 / extrapolated:.2f}x")

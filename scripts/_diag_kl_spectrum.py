"""One-off diagnostic: distribution of eigenvalues in K-L spectrum."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model_b import grf as grf_circulant

for sig in [5.0, 10.0, 15.0]:
    LAM = grf_circulant.calc_LAM(s1=sig, s2=sig)
    eigvals = np.asarray(LAM ** 2).flatten()
    print(f"\n=== sig={sig} ===")
    print(f"LAM shape: {LAM.shape}, total cells: {len(eigvals)}")
    print(f"sum eigvals: {eigvals.sum():.6f}")
    print(f"max: {eigvals.max():.6e}, min: {eigvals.min():.6e}")
    print(f"nonzero (>1e-12): {(eigvals > 1e-12).sum()}")

    sorted_ev = np.sort(eigvals)[::-1]
    cumvar = np.cumsum(sorted_ev) / sorted_ev.sum()
    for K in [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if K <= len(cumvar):
            print(f"  K={K:5d}: var_captured={cumvar[K-1]:.6f}")
    K_999 = int(np.searchsorted(cumvar, 0.999) + 1)
    K_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    K_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"  K_for_95%   = {K_95}")
    print(f"  K_for_99%   = {K_99}")
    print(f"  K_for_99.9% = {K_999}")

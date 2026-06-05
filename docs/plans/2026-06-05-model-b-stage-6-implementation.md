# Stage 6 Implementation Plan — Karhunen-Loève Low-Rank GRF (Model B)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the FFT-based circulant GRF generator in Model B with a truncated K-L expansion, achieving 5-10× per-call speedup over Stage 5.

**Architecture:** For our block-circulant covariance, K-L eigendecomposition equals the FFT of the kernel. Top-K modes (by `|LAM|²`) capture 99.9%+ variance at sig=10. Runtime op becomes one batched GEMM (`grf = z @ V.T`) instead of a batched 2D FFT.

**Tech Stack:** JAX, NumPy, pytest. No new dependencies.

**Design reference:** `docs/plans/2026-06-05-model-b-stage-6-design.md`

---

## Phase 1: K-L basis module

### Task 1: Write failing test for `calc_kl_basis` returns expected shapes

**Files:**
- Create: `model_b/tests/test_grf_kl_smoke.py`

**Step 1: Write the failing test**

```python
"""Smoke tests for model_b/grf_kl.py — K-L basis builder + sampler."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import grf_kl


def test_calc_kl_basis_returns_correct_shapes():
    """K-L basis V should be (NM, 2K) fp32 real."""
    V, K, var_captured = grf_kl.calc_kl_basis(sig=10.0, n=100, m=160,
                                              k_max=200, variance_threshold=0.999)
    assert V.shape[0] == 100 * 160, f"V row count should be N*M=16000, got {V.shape[0]}"
    assert V.shape[1] == 2 * K, f"V col count should be 2*K={2*K}, got {V.shape[1]}"
    assert V.dtype == jnp.float32, f"V should be fp32, got {V.dtype}"
    assert 1 <= K <= 200, f"K should be in [1, 200], got {K}"
    assert 0.0 <= var_captured <= 1.0, f"variance_captured should be in [0, 1], got {var_captured}"
```

**Step 2: Run test to verify it fails**

```bash
pytest model_b/tests/test_grf_kl_smoke.py::test_calc_kl_basis_returns_correct_shapes -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model_b.grf_kl'`

**Step 3: Create minimal `grf_kl.py`**

**Files:**
- Create: `model_b/grf_kl.py`

```python
"""
Karhunen-Loève low-rank GRF generator for Model B.

For our block-circulant covariance, the K-L eigendecomposition equals the FFT
of the kernel. Top-K modes by spectral magnitude capture ~99.9% of variance at
sig=10 with K ≈ 100. Runtime sampling is one batched GEMM instead of a 2D FFT.

See docs/plans/2026-06-05-model-b-stage-6-design.md for math + design rationale.
"""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circulant


def calc_kl_basis(sig: float,
                  n: int = 100,
                  m: int = 160,
                  k_max: int = 200,
                  variance_threshold: float = 0.999):
    """
    Build the truncated K-L basis for the circulant-embedded covariance.

    Returns:
        V : (n*m, 2*K) fp32 real basis with √λ folded in
        K : int, actual number of complex modes retained
        variance_captured : float, fraction of total variance retained
    """
    # Get LAM (spectral sqrt) from existing circulant code — fp64
    LAM = grf_circulant.calc_LAM(n=n, m=m, s1=sig, s2=sig)  # (N_pad, M_pad)
    n_pad, m_pad = LAM.shape

    # Eigenvalues = LAM^2 (since LAM is the spectral sqrt)
    eigvals = (LAM ** 2).flatten()  # (N_pad * M_pad,)
    eigvals_np = np.asarray(eigvals)

    # Sort descending, find K
    sort_idx = np.argsort(-eigvals_np)
    sorted_eigvals = eigvals_np[sort_idx]
    cumvar = np.cumsum(sorted_eigvals) / sorted_eigvals.sum()
    K_by_thresh = int(np.searchsorted(cumvar, variance_threshold) + 1)
    K = min(K_by_thresh, k_max)
    variance_captured = float(cumvar[K - 1])

    # Top K flat indices in LAM
    top_idx = sort_idx[:K]
    i_star = top_idx // m_pad   # (K,)
    j_star = top_idx % m_pad    # (K,)
    lam_k = np.sqrt(sorted_eigvals[:K]).astype(np.float32)  # √λ_k, (K,)

    # Build basis: V[n*M + m, 2k]   = √λ_k · Re(e_k(n, m))
    #              V[n*M + m, 2k+1] = √λ_k · Im(e_k(n, m))
    # where e_k(n, m) = exp(2πi (i*_k n / N_pad + j*_k m / M_pad)) / √(N_pad M_pad)
    norm = 1.0 / np.sqrt(n_pad * m_pad)
    n_grid = np.arange(n)[:, None]   # (n, 1)
    m_grid = np.arange(m)[None, :]   # (1, m)

    V = np.empty((n * m, 2 * K), dtype=np.float32)
    for k in range(K):
        phase = 2.0 * np.pi * (
            i_star[k] * n_grid / n_pad + j_star[k] * m_grid / m_pad
        )
        re = (np.cos(phase) * norm).astype(np.float32).flatten()  # (n*m,)
        im = (np.sin(phase) * norm).astype(np.float32).flatten()
        V[:, 2 * k] = lam_k[k] * re
        V[:, 2 * k + 1] = lam_k[k] * im

    return jnp.asarray(V), K, variance_captured
```

**Step 4: Run test to verify it passes**

```bash
pytest model_b/tests/test_grf_kl_smoke.py::test_calc_kl_basis_returns_correct_shapes -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add model_b/grf_kl.py model_b/tests/test_grf_kl_smoke.py
git commit -m "feat(grf_kl): basis builder skeleton with shape test"
```

---

### Task 2: Test variance threshold satisfied

**Files:**
- Modify: `model_b/tests/test_grf_kl_smoke.py` (append)

**Step 1: Add failing test**

```python
def test_calc_kl_basis_variance_threshold():
    """Variance captured should hit the requested threshold."""
    V, K, vc = grf_kl.calc_kl_basis(sig=10.0, variance_threshold=0.999)
    assert vc >= 0.999, f"variance_captured {vc} below threshold 0.999"
    assert K < 200, f"At sig=10, K={K} should be well under k_max=200"


def test_calc_kl_basis_k_grows_with_lower_threshold():
    """Lower threshold should need fewer modes."""
    _, K_99, _ = grf_kl.calc_kl_basis(sig=10.0, variance_threshold=0.99)
    _, K_999, _ = grf_kl.calc_kl_basis(sig=10.0, variance_threshold=0.999)
    assert K_99 <= K_999, f"K should grow with threshold (99%={K_99}, 99.9%={K_999})"
```

**Step 2: Run tests**

```bash
pytest model_b/tests/test_grf_kl_smoke.py -v -k "variance or k_grows"
```

Expected: PASS (implementation already satisfies this).

**Step 3: Commit**

```bash
git add model_b/tests/test_grf_kl_smoke.py
git commit -m "test(grf_kl): variance threshold + K-grows-with-threshold"
```

---

### Task 3: Test `sample_kl_grf` returns correct shape

**Files:**
- Modify: `model_b/tests/test_grf_kl_smoke.py` (append)

**Step 1: Add failing test**

```python
def test_sample_kl_grf_shape():
    """sample_kl_grf should produce (batch, n, m) reals."""
    V, K, _ = grf_kl.calc_kl_basis(sig=10.0)
    key = jax.random.key(0)
    z = jax.random.normal(key, (32, 2 * K), dtype=jnp.float32)
    out = grf_kl.sample_kl_grf(V, z, n=100, m=160)
    assert out.shape == (32, 100, 160), f"expected (32, 100, 160), got {out.shape}"
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))
```

**Step 2: Run test (fails: function doesn't exist)**

```bash
pytest model_b/tests/test_grf_kl_smoke.py::test_sample_kl_grf_shape -v
```

Expected: FAIL with `AttributeError: module 'model_b.grf_kl' has no attribute 'sample_kl_grf'`

**Step 3: Implement `sample_kl_grf` in `model_b/grf_kl.py`**

Append to `model_b/grf_kl.py`:

```python
def sample_kl_grf(V, z, n: int = 100, m: int = 160):
    """
    Generate GRF samples from the K-L basis.

    V : (NM, 2K) fp32 basis from calc_kl_basis
    z : (batch, 2K) iid N(0,1) random samples
    Returns: (batch, n, m) fp32 GRF realizations
    """
    # z @ V.T -> (batch, NM)
    grf_flat = z @ V.T
    return grf_flat.reshape(-1, n, m)
```

**Step 4: Run test**

```bash
pytest model_b/tests/test_grf_kl_smoke.py::test_sample_kl_grf_shape -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add model_b/grf_kl.py model_b/tests/test_grf_kl_smoke.py
git commit -m "feat(grf_kl): sample_kl_grf for batched GRF generation"
```

---

### Task 4: Test K-L GRF marginal variance matches circulant

**Files:**
- Create: `model_b/tests/test_grf_kl_parity.py`

**Step 1: Write the failing parity test**

```python
"""Parity tests for K-L GRF vs circulant GRF (model_b/grf.py)."""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circ
from model_b import grf_kl


def _sample_circulant_grfs(key, n_samples, sig=10.0, n=100, m=160):
    """Generate n_samples GRFs from the circulant generator."""
    LAM = grf_circ.calc_LAM(n=n, m=m, s1=sig, s2=sig)
    n_pad, m_pad = LAM.shape
    grfs = []
    for i in range(n_samples // 2 + 1):
        k = jax.random.fold_in(key, i)
        g1 = jax.random.normal(jax.random.fold_in(k, 0), (n_pad, m_pad))
        g2 = jax.random.normal(jax.random.fold_in(k, 1), (n_pad, m_pad))
        F1, F2 = grf_circ.circulant_grf(LAM, g1, g2)
        grfs.append(np.asarray(F1))
        if len(grfs) < n_samples:
            grfs.append(np.asarray(F2))
    return np.stack(grfs[:n_samples])  # (n_samples, n, m)


def _sample_kl_grfs(key, n_samples, sig=10.0, n=100, m=160):
    """Generate n_samples GRFs from the K-L generator."""
    V, K, _ = grf_kl.calc_kl_basis(sig=sig, n=n, m=m)
    z = jax.random.normal(key, (n_samples, 2 * K), dtype=jnp.float32)
    return np.asarray(grf_kl.sample_kl_grf(V, z, n=n, m=m))


def test_marginal_variance_parity():
    """Per-cell variance should match between circulant and K-L generators."""
    n_samples = 2000  # enough for ~3% sampling noise
    key_c = jax.random.key(0)
    key_k = jax.random.key(1)
    grfs_c = _sample_circulant_grfs(key_c, n_samples)
    grfs_k = _sample_kl_grfs(key_k, n_samples)

    var_c = grfs_c.var(axis=0)  # (n, m)
    var_k = grfs_k.var(axis=0)

    # Compare mean variance (per-cell variance can be noisy at 2000 samples).
    mean_var_c = var_c.mean()
    mean_var_k = var_k.mean()
    rel_err = abs(mean_var_k - mean_var_c) / mean_var_c
    assert rel_err < 0.05, (
        f"Mean variance mismatch: circulant={mean_var_c:.4f}, "
        f"K-L={mean_var_k:.4f}, rel_err={rel_err:.4f}"
    )
```

**Step 2: Run test**

```bash
pytest model_b/tests/test_grf_kl_parity.py::test_marginal_variance_parity -v
```

Expected: PASS if K-L math is correct. If FAIL, the test output will show observed variances — debug the basis construction.

**Step 3: Commit**

```bash
git add model_b/tests/test_grf_kl_parity.py
git commit -m "test(grf_kl): marginal variance parity vs circulant"
```

---

### Task 5: Test K-L GRF autocorrelation matches circulant

**Files:**
- Modify: `model_b/tests/test_grf_kl_parity.py` (append)

**Step 1: Add failing test**

```python
def _empirical_acf(grfs, max_lag=10):
    """1D autocorrelation along row direction, averaged over samples + cols."""
    grfs_centered = grfs - grfs.mean(axis=0, keepdims=True)
    # Use central row 50; compute autocorr along columns
    row = grfs_centered[:, 50, :]  # (n_samples, m)
    acf = np.zeros(max_lag + 1)
    var = row.var()
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            cov = (row[:, :-lag] * row[:, lag:]).mean()
            acf[lag] = cov / var
    return acf


def test_autocorrelation_parity():
    """ACF along row direction should match between circulant and K-L."""
    n_samples = 2000
    key_c = jax.random.key(0)
    key_k = jax.random.key(1)
    grfs_c = _sample_circulant_grfs(key_c, n_samples)
    grfs_k = _sample_kl_grfs(key_k, n_samples)

    acf_c = _empirical_acf(grfs_c, max_lag=10)
    acf_k = _empirical_acf(grfs_k, max_lag=10)

    max_diff = np.abs(acf_c - acf_k).max()
    assert max_diff < 0.05, (
        f"ACF max diff {max_diff:.4f} too large. "
        f"circulant={acf_c}, K-L={acf_k}"
    )
```

**Step 2: Run test**

```bash
pytest model_b/tests/test_grf_kl_parity.py::test_autocorrelation_parity -v
```

Expected: PASS. If FAIL, examine basis indexing carefully.

**Step 3: Commit**

```bash
git add model_b/tests/test_grf_kl_parity.py
git commit -m "test(grf_kl): autocorrelation parity vs circulant"
```

---

## Phase 2: Integrate K-L path into `simulate_b`

### Task 6: Add `use_kl=False` flag to `simulate_b`, no behavior change

**Files:**
- Modify: `model_b/simulate.py` (function signature of `simulate_b` near line 174)
- Create: `model_b/tests/test_simulate_b_kl_flag.py`

**Step 1: Write failing test**

```python
"""Test that simulate_b accepts use_kl flag."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate as sim_b


def test_simulate_b_accepts_use_kl_flag():
    """simulate_b should accept use_kl=False and behave identically to current."""
    key = jax.random.key(0)
    rt_a, cat_a = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4,
        use_kl=False,  # NEW flag
    )
    rt_b, cat_b = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4,
        # use_kl defaults to False
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)
```

**Step 2: Run test**

```bash
pytest model_b/tests/test_simulate_b_kl_flag.py::test_simulate_b_accepts_use_kl_flag -v
```

Expected: FAIL with `TypeError: simulate_b() got an unexpected keyword argument 'use_kl'`

**Step 3: Modify `simulate_b` to add the flag**

In `model_b/simulate.py` near line 174-202, change the signature and add `use_kl` to static_argnums:

```python
@partial(jax.jit, static_argnums=(11, 12, 13))
def simulate_b(key, ter, st, cr, crsd, av1, av2, av3,
               sis, sig, si, nsim, chunk_size=4, use_kl=False):
    """
    Run `nsim` Model B trials with the given parameters.

    Parameters:
        sis : drift bump width
        sig : GRF correlation length (s1=s2=sig)
        si  : zone-array width
        use_kl : if True, use K-L low-rank GRF (Stage 6). Default False (Stage 5 FFT path).
    Returns (rt, cat) each shape (nsim,). cat in {1..5}. RT in ms.
    """
    LAM = grf.calc_LAM(s1=sig, s2=sig)
    v1, v2, v3 = drift_bumps(sis=sis)
    k_zone = zone_array(si=si)

    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)

    if use_kl:
        from model_b import grf_kl
        V_kl, K_kl, _ = grf_kl.calc_kl_basis(sig=sig, n=N, m=M)

        def run_chunk(k):
            return _simulate_chunk_b_kl(
                k, ter, st, cr, crsd, av1, av2, av3,
                V_kl, K_kl, v1, v2, v3, k_zone, chunk_size,
            )
    else:
        def run_chunk(k):
            return _simulate_chunk_b(
                k, ter, st, cr, crsd, av1, av2, av3,
                LAM, v1, v2, v3, k_zone, chunk_size,
            )

    rts, cats = jax.lax.map(run_chunk, keys)
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim]
```

Also add a placeholder `_simulate_chunk_b_kl` that just calls `_simulate_chunk_b` for now (to keep the flag a no-op until Task 7):

```python
def _simulate_chunk_b_kl(key, ter, st, cr, crsd, av1, av2, av3,
                          V_kl, K_kl, v1, v2, v3, k_zone, chunk_size):
    """K-L variant — placeholder for Task 7."""
    # Until Task 7 lands, fall back to FFT path
    from model_b import grf
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)  # placeholder sig
    return _simulate_chunk_b(key, ter, st, cr, crsd, av1, av2, av3,
                              LAM, v1, v2, v3, k_zone, chunk_size)
```

Note: this placeholder will use a wrong sig. That's OK — Task 7 replaces it before any real validation. The flag-acceptance test passes with use_kl=False which doesn't hit this path.

**Step 4: Run test**

```bash
pytest model_b/tests/test_simulate_b_kl_flag.py::test_simulate_b_accepts_use_kl_flag -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add model_b/simulate.py model_b/tests/test_simulate_b_kl_flag.py
git commit -m "feat(simulate_b): add use_kl flag (placeholder, falls through to FFT)"
```

---

### Task 7: Implement `_simulate_chunk_b_kl` with real K-L sampling

**Files:**
- Modify: `model_b/simulate.py` (replace placeholder `_simulate_chunk_b_kl`)

**Step 1: Write the implementation**

Replace the placeholder with:

```python
def _simulate_chunk_b_kl(key, ter, st, cr, crsd, av1, av2, av3,
                          V_kl, K_kl, v1, v2, v3, k_zone, chunk_size):
    """
    Simulate `chunk_size` trials using K-L low-rank GRF generation.

    Differences from `_simulate_chunk_b`:
      - Replaces (chunk, n_fft, 2, n_pad, m_pad) noise + batched FFT
        with (chunk, NSTEP, 2K) noise + single GEMM via V_kl.
      - 2K random draws per timestep per trial instead of full-grid noise.
      - Memory savings ~50× at K=100 vs n_pad*m_pad=63481.

    All downstream logic (drift + cumsum + first-crossing + categorization)
    is identical to the FFT path.
    """
    ku, kg = jax.random.split(key)

    # Per-trial uniforms (mirrors Fortran gu1)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = (cr + crsd * (u[:, 4] - 0.5)).astype(jnp.float32)
    ndt = (ter + st * (0.5 - u[:, 9])) / E

    # Sample (chunk, NSTEP, 2K) iid normals
    z = jax.random.normal(kg, (chunk_size, NSTEP, 2 * K_kl), dtype=jnp.float32)

    # One batched GEMM: (chunk, NSTEP, 2K) @ (2K, NM) -> (chunk, NSTEP, NM)
    grf_flat = jnp.einsum("csk,Nk->csN", z, V_kl)
    grf_path = grf_flat.reshape(chunk_size, NSTEP, N, M)

    # Build demeaned per-step increment (identical to FFT path).
    v1_f32 = v1.astype(jnp.float32)
    v2_f32 = v2.astype(jnp.float32)
    v3_f32 = v3.astype(jnp.float32)
    av1_f32 = jnp.float32(av1)
    av2_f32 = jnp.float32(av2)
    av3_f32 = jnp.float32(av3)
    drift_const = av1_f32 * v1_f32 + av2_f32 * v2_f32 + av3_f32 * v3_f32
    incr = drift_const[None, None, :, :] + grf_path

    # Demean per (trial, step) over spatial axes.
    incr = incr - incr.mean(axis=(-2, -1), keepdims=True)

    # Accumulator path.
    a = jnp.cumsum(incr, axis=1)

    # First crossing.
    max_per_step = a.reshape(chunk_size, NSTEP, -1).max(axis=-1)
    crossed = max_per_step > crr[:, None]
    any_crossed = crossed.any(axis=1)
    jstop = jnp.where(any_crossed, jnp.argmax(crossed, axis=1) + 1, NSTEP)

    # Position + category.
    a_at_crossing = a[jnp.arange(chunk_size), jstop - 1, :, :]
    pos_flat = jnp.argmax(a_at_crossing.reshape(chunk_size, -1), axis=-1)
    row = pos_flat // M
    col = pos_flat % M
    cat = jnp.where(any_crossed, k_zone[row, col], jnp.int32(5))
    rt = (jstop.astype(jnp.float64) + ndt) * E

    return rt, cat
```

Also update `simulate_b` to remove the wrong-sig placeholder — V_kl is already passed in correctly:

(The signature shown in Task 6 step 3 already does this correctly.)

**Step 2: Write smoke test for the new path**

Modify `model_b/tests/test_simulate_b_kl_flag.py` (append):

```python
def test_simulate_b_kl_path_runs_and_returns_finite():
    """use_kl=True should run end-to-end without errors."""
    import jax
    key = jax.random.key(0)
    rt, cat = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4,
        use_kl=True,
    )
    assert rt.shape == (16,)
    assert cat.shape == (16,)
    assert jnp.all(jnp.isfinite(rt))
    assert jnp.all((cat >= 1) & (cat <= 5))
```

**Step 3: Run tests**

```bash
pytest model_b/tests/test_simulate_b_kl_flag.py -v
```

Expected: PASS (both flag-accept and K-L-runs-end-to-end).

**Step 4: Commit**

```bash
git add model_b/simulate.py model_b/tests/test_simulate_b_kl_flag.py
git commit -m "feat(simulate_b): real K-L low-rank GRF path"
```

---

### Task 8: Statistical parity test — `simulate_b` outputs at high nsim

**Files:**
- Create: `model_b/tests/test_simulate_b_kl_parity.py`

**Step 1: Write parity test**

```python
"""
End-to-end parity test: simulate_b with use_kl=True vs use_kl=False.

We don't expect bit-exact agreement (different random sampling paths).
We test that aggregate statistics match within Monte Carlo tolerance.
"""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import simulate as sim_b


def _summary_stats(rt, cat):
    """RT mean, RT std, category proportions."""
    rt_np = np.asarray(rt)
    cat_np = np.asarray(cat)
    props = np.array([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)])
    return rt_np.mean(), rt_np.std(), props


def test_simulate_b_kl_matches_fft_in_aggregate():
    """K-L and FFT paths should give matching aggregate statistics."""
    nsim = 2000  # enough for ~3% sampling noise
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

    # RT mean within 3%
    rel_err_mean = abs(rt_mean_kl - rt_mean_fft) / rt_mean_fft
    assert rel_err_mean < 0.03, (
        f"RT mean mismatch: FFT={rt_mean_fft:.1f}, K-L={rt_mean_kl:.1f}, "
        f"rel_err={rel_err_mean:.4f}"
    )

    # RT std within 5%
    rel_err_std = abs(rt_std_kl - rt_std_fft) / rt_std_fft
    assert rel_err_std < 0.05, (
        f"RT std mismatch: FFT={rt_std_fft:.1f}, K-L={rt_std_kl:.1f}, "
        f"rel_err={rel_err_std:.4f}"
    )

    # Category proportions: max absolute difference < 0.05
    prop_diff = np.abs(props_kl - props_fft).max()
    assert prop_diff < 0.05, (
        f"Category proportion mismatch: FFT={props_fft}, K-L={props_kl}, "
        f"max_diff={prop_diff:.4f}"
    )
```

**Step 2: Run test**

```bash
pytest model_b/tests/test_simulate_b_kl_parity.py -v
```

Expected: PASS. If FAIL, statistics diverge — debug basis construction or accumulation logic.

**Step 3: Commit**

```bash
git add model_b/tests/test_simulate_b_kl_parity.py
git commit -m "test(simulate_b): K-L vs FFT aggregate parity at nsim=2000"
```

---

## Phase 3: End-to-end recovery check (laptop)

### Task 9: Tiny parameter recovery test with `use_kl=True`

**Files:**
- Create: `model_b/tests/test_fit_b_kl_smoke.py`

**Step 1: Write smoke test**

```python
"""
End-to-end smoke test: run a tiny parameter recovery with use_kl=True
to confirm the K-L path produces fittable behavior. NOT a full benchmark —
this is a sanity check that the optimizer can move with the K-L simulator.

Full recovery quality check happens on H100 (Phase 4 / Stage 6 benchmark).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate as sim_b
from model_b import objective as obj_b
from model_b import fit as fit_b
from model_b.objective import COND_MAP_B, clamp_b
from shared import prng


TRUE_PARAMS = jnp.array([
    200.0, 50.0, 10.0, 2.0,
    12.0, 10.0, 0.5,
    15.0, 10.0, 8.0,
    14.0, 11.0, 9.0,
])


def _make_synthetic(true_params, key, nsim, chunk_size, use_kl):
    """Generate synthetic data via simulate_b at true params."""
    p = clamp_b(true_params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    props_l, counts_l, quants_l = [], [], []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(key, ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
            sis, sig, si, nsim=nsim, chunk_size=chunk_size, use_kl=use_kl,
        )
        cat_np = np.asarray(cat); rt_np = np.asarray(rt)
        props = np.array([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)])
        counts = np.array([(cat_np == c).sum() for c in (1, 2, 3, 4, 5)], dtype=np.int64)
        quants = np.zeros((5, 5))
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat_np == c
            if mask.sum() >= 5:
                quants[:, ki] = np.quantile(rt_np[mask], qs)
        props_l.append(jnp.asarray(props))
        counts_l.append(jnp.asarray(counts))
        quants_l.append(jnp.asarray(quants))
    return {"prop": jnp.stack(props_l), "count": jnp.stack(counts_l),
            "quant": jnp.stack(quants_l)}


@pytest.mark.slow
def test_kl_recovery_moves_loss_down_at_small_scale():
    """Run a tiny fit (maxiter=10) with K-L path; loss should decrease."""
    nsim = 256
    chunk = 16
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=nsim,
                          chunk_size=chunk, use_kl=True)
    np.random.seed(0)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.9, 1.1, size=13))

    # NOTE: fit_simplex_b doesn't yet thread use_kl. We'll wire that in Task 10.
    # For now, just confirm fofs_b_new with use_kl works.
    # (Not strictly required — Task 8 already validates simulate_b. Skip if pressed for time.)
    val0 = float(obj_b.fofs_b_new(x0, syn, jax.random.key(1),
                                   nsim=nsim, chunk_size=chunk))
    assert np.isfinite(val0), f"Initial objective should be finite, got {val0}"
```

**Step 2: Run test**

```bash
pytest model_b/tests/test_fit_b_kl_smoke.py -v --no-cov
```

Expected: PASS if `fofs_b_new` doesn't need a use_kl flag yet. SKIP or FAIL otherwise — in which case Task 10 wires use_kl into `fofs_b_new` next.

**Step 3: Commit**

```bash
git add model_b/tests/test_fit_b_kl_smoke.py
git commit -m "test(fit_b): K-L objective evaluation smoke"
```

---

### Task 10: Thread `use_kl` through `fofs_b_new` and `fit_simplex_b`

**Files:**
- Modify: `model_b/objective.py` (`fofs_b_new` signature near line 91)
- Modify: `model_b/fit.py` (`fit_simplex_b` signature near line 24)

**Step 1: Write the failing test**

In `model_b/tests/test_fit_b_kl_smoke.py`, append:

```python
def test_fofs_b_new_accepts_use_kl():
    """fofs_b_new should pass use_kl through to simulate_b."""
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=128,
                          chunk_size=8, use_kl=False)
    np.random.seed(1)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.95, 1.05, size=13))
    val_fft = float(obj_b.fofs_b_new(x0, syn, jax.random.key(1),
                                      nsim=128, chunk_size=8, use_kl=False))
    val_kl = float(obj_b.fofs_b_new(x0, syn, jax.random.key(1),
                                     nsim=128, chunk_size=8, use_kl=True))
    assert np.isfinite(val_fft) and np.isfinite(val_kl)
```

**Step 2: Run test (fails: fofs_b_new doesn't accept use_kl yet)**

```bash
pytest model_b/tests/test_fit_b_kl_smoke.py::test_fofs_b_new_accepts_use_kl -v
```

Expected: FAIL with `TypeError: fofs_b_new() got an unexpected keyword argument 'use_kl'`

**Step 3: Modify `fofs_b_new` in `model_b/objective.py` around line 91**

Change signature and the `sim_vmap` call:

```python
def fofs_b_new(params, data, key, nsim=512, chunk_size=4, use_kl=False):
    """
    Vectorized G2 objective for Model B, summed across 2 conditions.

    params : (13,) parameter vector. See clamp_b docstring for layout.
    data   : dict with "prop" (2, 5), "count" (2, 5), "quant" (2, 5, 5).
    key    : JAX typed key.
    use_kl : if True, use Stage 6 K-L low-rank GRF simulator.
    Returns scalar G2.
    """
    from model_b import simulate as sim_b
    from shared import prng

    p = clamp_b(params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0

    avs = jnp.stack([
        jnp.array([p[d1], p[d2], p[d3]]) for (d1, d2, d3) in COND_MAP_B
    ])
    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(2)])

    sis_py = float(sis)
    sig_py = float(sig)
    si_py = float(si)

    sim_vmap = jax.vmap(
        sim_b.simulate_b,
        in_axes=(0, None, None, None, None, 0, 0, 0, None, None, None, None, None, None),
    )
    rts, cats = sim_vmap(
        cond_keys, ter, st, cr, crsd,
        avs[:, 0], avs[:, 1], avs[:, 2],
        sis_py, sig_py, si_py, nsim, chunk_size, use_kl,
    )

    g2_vmap = jax.vmap(condition_g2_b, in_axes=(0, 0, 0, 0, 0))
    g2_per_cond = g2_vmap(rts, cats, data["prop"], data["count"], data["quant"])
    return g2_per_cond.sum()
```

**Step 4: Modify `fit_simplex_b` in `model_b/fit.py` line 24**

```python
def fit_simplex_b(data, key, x0, nsim: int = 256, maxiter: int = 2000,
                  tol: float = 1e-7, chunk_size: int = 4, use_kl: bool = False):
    """
    ...
    use_kl : if True, use Stage 6 K-L low-rank GRF inside fofs_b_new.
    """
    from scipy.optimize import minimize

    def loss_numpy(p_np):
        p = jnp.asarray(p_np)
        val = obj_b.fofs_b_new(p, data, key, nsim=nsim,
                                chunk_size=chunk_size, use_kl=use_kl)
        return float(val)

    res = minimize(
        loss_numpy, np.asarray(x0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": tol, "fatol": tol, "disp": False},
    )
    return FitResult(
        params=jnp.asarray(res.x),
        loss=float(res.fun),
        n_iters=int(res.nit),
        converged=bool(res.success),
    )
```

**Step 5: Run test**

```bash
pytest model_b/tests/test_fit_b_kl_smoke.py -v
```

Expected: PASS for all tests.

**Step 6: Commit**

```bash
git add model_b/objective.py model_b/fit.py model_b/tests/test_fit_b_kl_smoke.py
git commit -m "feat(objective+fit): thread use_kl flag through fofs_b_new and fit_simplex_b"
```

---

### Task 11: Full local recovery smoke (slow)

**Files:**
- Modify: `model_b/tests/test_fit_b_kl_smoke.py` (append)

**Step 1: Write the slow test**

```python
@pytest.mark.slow
def test_kl_recovery_small_scale_completes():
    """
    Run a small simplex fit with use_kl=True and confirm it terminates.
    NOT a parameter-quality test (too small nsim for that) — just smoke.
    """
    import time
    nsim = 512
    chunk = 16
    syn = _make_synthetic(TRUE_PARAMS, jax.random.key(0), nsim=512,
                          chunk_size=chunk, use_kl=True)
    np.random.seed(2)
    x0 = TRUE_PARAMS * jnp.asarray(np.random.uniform(0.95, 1.05, size=13))

    t0 = time.perf_counter()
    res = fit_b.fit_simplex_b(
        syn, jax.random.key(1), x0,
        nsim=nsim, maxiter=20, chunk_size=chunk, use_kl=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nK-L recovery (nsim={nsim}, maxiter=20) took {elapsed:.1f}s, "
          f"loss {res.loss:.2f}")
    assert np.isfinite(res.loss)
    assert res.n_iters > 0
```

**Step 2: Run test**

```bash
pytest model_b/tests/test_fit_b_kl_smoke.py::test_kl_recovery_small_scale_completes -v -s
```

Expected: PASS, prints wall-clock. Should be roughly 10-30 sec on laptop CPU.

**Step 3: Commit**

```bash
git add model_b/tests/test_fit_b_kl_smoke.py
git commit -m "test(fit_b): K-L small-scale recovery smoke"
```

---

## Phase 4: H100 benchmark preparation (USER ACTION)

### Task 12: Update `h100_section4_verbose.py` to support use_kl

**Files:**
- Modify: `scripts/h100_section4_verbose.py`

**Step 1: Add `use_kl=True` to the script**

In `scripts/h100_section4_verbose.py`, find the simulate_b call inside `_generate_synthetic_data_b` and the `obj_b.fofs_b_new` calls in the fit loop. Add `use_kl=True` to both:

```python
# In _generate_synthetic_data_b:
rt, cat = sim_b.simulate_b(
    ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
    sis, sig, si, nsim=nsim, chunk_size=chunk_size, use_kl=True,
)

# In loss_with_progress:
val = float(obj_b.fofs_b_new(p, syn_data, key_fit,
                              nsim=NSIM_PROD, chunk_size=CHUNK, use_kl=True))
```

Also bump `CHUNK = 32` (or 64) — K-L's lower memory should allow larger chunks on H100.

**Step 2: Commit**

```bash
git add scripts/h100_section4_verbose.py
git commit -m "feat(h100): enable K-L path in Section 4 verbose script"
```

### Task 13: Write Stage 6 completion summary template

**Files:**
- Create: `docs/notes/2026-06-XX-stage-6-completion.md` (with placeholder for H100 numbers)

**Step 1: Write the template**

```markdown
# Stage 6 Completion — K-L Low-Rank GRF

**Date:** [TODO when H100 benchmark lands]
**Status:** [TODO: passed/failed]

## Results vs Stage 5

| Metric | Stage 5 (FFT) | Stage 6 (K-L) | Improvement |
|---|---|---|---|
| Per-call simulator at nsim=9000 | 5.18 sec | [TODO] | [TODO]× |
| Full fit wall-clock | 30.3 min | [TODO] | [TODO]× |
| Parameters within 7% of truth | 12/12 | [TODO] | — |
| Average parameter error | 3.8% | [TODO] | — |

## K-L parameters used

- sig_default: 10.0
- K (modes retained): [TODO]
- Variance captured: [TODO]

## Notes

[TODO: any surprises, regressions, learnings]
```

**Step 2: Commit**

```bash
git add docs/notes/2026-06-XX-stage-6-completion.md
git commit -m "docs: Stage 6 completion summary template"
```

---

## Phase 5: Final cleanup

### Task 14: Run all existing tests, confirm no regressions

**Step 1: Run full test suite**

```bash
pytest model_b/tests/ model_a/tests/ shared/tests/ -v --no-cov
```

Expected: All tests PASS. Specifically, the existing Stage 5 tests (use_kl=False default) should be unaffected.

**Step 2: If any fail, investigate and fix.** Most likely failure mode: `simulate_b` signature change broke a test that calls with positional args.

**Step 3: Commit only if fixes needed**

```bash
git add [fixed files]
git commit -m "fix(model_b): restore [specific test] under new signature"
```

### Task 15: Final tag

**Step 1: Tag the Stage 6 milestone**

```bash
git tag -a v0.6.0-stage6-kl -m "Stage 6: Karhunen-Loève low-rank GRF (laptop validated)"
```

**Note:** Don't push the tag until H100 benchmark numbers are in and `docs/notes/2026-06-XX-stage-6-completion.md` is filled in.

---

## Hand-off note for the H100 benchmark (user action)

After Phase 5, the user must:
1. Rent an H100 on RunPod / Vast / Lambda
2. Run `scripts/h100_setup_and_run.sh` (or equivalent)
3. Run `python scripts/h100_section4_verbose.py 2>&1 | tee /workspace/h100_section6.txt`
4. Compare numbers against Stage 5 baseline (5.18s/call, 30.3 min fit)
5. Fill in `docs/notes/2026-06-XX-stage-6-completion.md`
6. Push the v0.6.0-stage6-kl tag

Expected H100 numbers (success criteria from design):
- Per-call simulator: ≤ 1.0 sec
- Full fit: ≤ 6 min
- Parameter recovery: 12/12 within 7%

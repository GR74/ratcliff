# Stage 6 Design — Karhunen-Loève Low-Rank GRF (Model B)

**Date:** 2026-06-05
**Status:** AWAITING APPROVAL
**Scope:** Replace the FFT-based circulant GRF generator in `model_b/simulate.py` with a truncated Karhunen-Loève expansion. Target: 5-10× per-call speedup on top of Stage 5, taking the production fit from ~30 min to ~3-5 min on H100.

---

## 1. Background

Stage 5 H100 benchmarking (2026-06-05) confirmed `simulate_b` runs at 5.18s per call for nsim=9000 — 2× faster than the Fortran 6-node MPI cluster, 7× faster than 1-node. A chunk-size sweep showed flat 5.2s across chunk ∈ {16, 32, 64, 128}, plus a hard cuFFT batched-plan failure at chunk=256. The diagnosis: **we have saturated the FFT memory-bandwidth path.** Further per-call speedup requires changing the GRF generation algorithm, not the chunking strategy.

The current generator uses Dietrich-Newsam circulant embedding via batched 2D FFT (`model_b/grf.py::circulant_grf`). Per chunk: generate (chunk, n_fft, 2, n_pad, m_pad) fp32 normals (~13 GB at chunk=64), do one batched 2D FFT (~12,800 transforms at chunk=64), cumsum the resulting drift field over time. The FFT itself is bandwidth-bound on H100 and contributes the dominant cost.

The Karhunen-Loève (K-L) expansion observation: for a block-circulant covariance like ours, the eigendecomposition **is** the FFT of the kernel. The eigenvalues are `|LAM|²` at each frequency; the eigenvectors are columns of the discrete Fourier basis. Truncating to the top K modes by eigenvalue gives a rank-K approximation.

**Empirical spectrum measurement (2026-06-05, Task 2 diagnostic).** The Kroese §2.2 kernel has a polynomial prefactor (`(1 - x²/s²² - xy/(s₁s₂) - y²/s₁²)`) that broadens the spectrum substantially relative to a pure Gaussian. Actual K values needed at our grid:

| sig | K for 95% | K for 99% | K for 99.9% |
|---|---|---|---|
| 5  | 3744 | 5295 | 7399 |
| 10 | 936  | 1325 | 1850 |
| 15 | 417  | 588  | 821  |

Original design assumed K ≈ 100 for sig=10 — that was off by ~13×. The speedup story still works because the K-L speedup is **memory-bandwidth-dominated** (smaller noise tensor, fewer random draws) not compute-dominated (GEMM cost scales with K). At K=1325 the noise tensor is ~410 MB per chunk vs 6.5 GB for the FFT path — still an ~16× memory reduction. Defaults are now `variance_threshold=0.99`, `k_max=2000`.

Replacing the full FFT with a (NM × K) basis matrix × (K, batch) random sample matrix gives a single batched GEMM instead of a batched FFT. H100 fp32 GEMM throughput is ~67 TFLOPS sustained; H100 batched FFT at our sizes is bandwidth-bound at ~20% of that. The runtime should be dominated by GEMM cost, which scales O(NM × K × batch) instead of O(N_pad × M_pad × log(N_pad × M_pad) × batch).

## 2. Goals

1. New `model_b/grf_kl.py` exposing `calc_kl_basis(sig, k_max=200, variance_threshold=0.999)` returning the (NM, 2K) real basis matrix and (2K,) eigenvalue vector.
2. Modified `model_b/simulate.py::_simulate_chunk_b` to use the K-L basis when a flag `use_kl=True` is set. Existing FFT path stays as the default for one release cycle for safety.
3. Parity validation:
   - Marginal variance per cell matches circulant GRF within 1% relative error.
   - Autocorrelation function at lags 0..20 (in both row and column directions) matches within 1%.
   - End-to-end Model B parameter recovery on synthetic data: 12/12 active params within 7% of truth (same threshold as Stage 5).
4. Performance: at nsim=9000, chunk=64, target per-call simulator time ≤ 1.0s on H100 (~5× speedup over Stage 5's 5.18s).
5. Full fit wall-clock target: ≤ 6 min on H100 at production scale.

## 3. Non-goals

- Not modifying `model_b/grf.py::circulant_grf` (keep as oracle).
- Not modifying `model_b/objective.py` or `model_b/fit.py` (works as-is).
- Not changing the model spec (same kernel, same drift bumps, same zone array).
- Not extending to n-D > 2 (separate work, requires different math).
- Not running on CPU benchmarks (Stage 5 already established the GPU path is the relevant one).
- Not implementing bf16 mixed precision (separate optimization, harder validation).

## 4. Math

### 4.1 K-L decomposition of the circulant covariance

The current circulant embedding builds a `(2N-1) × (2M-1)` block-circulant covariance matrix `C`. Its eigendecomposition is:

```
C = F* Λ F
```

where `F` is the 2D DFT matrix and `Λ = diag(LAM²)` (where `LAM` is what `calc_LAM` already returns — the elementwise sqrt of the spectral density).

To sample a GRF realization on the `(N, M)` grid:
1. Sample `z ~ N(0, I)` complex Gaussian, shape `(N_pad, M_pad)`.
2. Compute `X = LAM ⊙ z`.
3. Compute `F_inv(X)` (or `F(X)` — convention differs).
4. Take real part on the top-left `(N, M)` slab.

This is what `circulant_grf` does. The "F1/F2 trick" extracts both real and imaginary parts as two independent GRFs.

### 4.2 Truncated K-L

Let `(i*, j*)` be the indices of the top K largest `|LAM[i, j]|²` values, sorted by magnitude. Define:

- `λ_k = LAM[i*_k, j*_k]² ` for k=1..K
- `e_k(n, m) = (1/√(N_pad M_pad)) · exp(2πi (i*_k n / N_pad + j*_k m / M_pad))` — the corresponding Fourier basis function restricted to the `(N, M)` grid.

A K-mode K-L approximation to one GRF sample:

```
GRF(n, m) ≈ Σ_{k=1}^{K} √λ_k · ξ_k · e_k(n, m)
```

where `ξ_k ~ N(0, 1)` complex.

Since the GRF is real-valued, we work with real and imaginary parts:

```
GRF(n, m) = Σ_{k=1}^{K} √λ_k [a_k Re(e_k(n, m)) + b_k Im(e_k(n, m))]
```

with `a_k, b_k ~ N(0, 1)` real iid.

### 4.3 Materialized basis matrix

Build once at trace time:

```
V[n*M + m, 2k]     = √λ_k · Re(e_k(n, m))
V[n*M + m, 2k + 1] = √λ_k · Im(e_k(n, m))
```

So `V` is shape `(NM, 2K)`. Then one GRF is `V @ z` where `z ~ N(0, 1)` shape `(2K,)`.

Batched across `(chunk, NSTEP)` samples:

```
z ~ N(0, 1)   shape (chunk, NSTEP, 2K)
grf_path = z @ V.T   shape (chunk, NSTEP, NM)
grf_path = grf_path.reshape(chunk, NSTEP, N, M)
```

That's a single batched GEMM with output shape `(chunk × NSTEP, NM)`. At chunk=64, NSTEP=400, NM=16000, K=100: GEMM size is (64·400, 200) × (200, 16000) = (25600, 200) × (200, 16000). Modest by H100 standards; should run in tens of milliseconds.

### 4.4 Variance check

The total variance of the full circulant GRF is `Σ LAM²[i, j] / (N_pad · M_pad)`. The truncated K-L variance is `Σ_{k=1}^K λ_k / (N_pad · M_pad)`. We want:

```
Σ_{k=1}^K λ_k / Σ all λ_i,j ≥ 0.999
```

If for some sig value the threshold can't be hit at K=k_max, we either (a) raise k_max or (b) emit a warning and proceed.

## 5. Architecture

### 5.1 Files

```
model_b/
├── grf.py             # untouched (oracle)
├── grf_kl.py          # NEW — K-L basis builder + sampler
├── simulate.py        # MODIFIED — _simulate_chunk_b takes use_kl flag
└── tests/
    ├── (existing files untouched)
    ├── test_grf_kl_smoke.py         # NEW — basis shape, eigenvalue sum, variance threshold
    ├── test_grf_kl_parity.py        # NEW — marginal variance + ACF parity vs circulant
    └── test_simulate_b_kl_parity.py # NEW — end-to-end recovery parity
```

### 5.2 `grf_kl.py` API

```python
def calc_kl_basis(sig: float,
                  n: int = 100,
                  m: int = 160,
                  k_max: int = 200,
                  variance_threshold: float = 0.999):
    """
    Build the truncated K-L basis for the circulant-embedded covariance.

    Returns:
        V : (n*m, 2*K) fp32 real basis matrix (with √λ folded in)
        K : int, actual number of complex modes retained
        variance_captured : float, fraction of total variance in the retained modes

    K is determined by either k_max OR variance_threshold, whichever triggers first.
    """


def sample_kl_grf(V, z):
    """
    Generate GRF samples from the K-L basis.

    V : (NM, 2K) basis (from calc_kl_basis)
    z : (batch, 2K) iid N(0,1) samples
    Returns: (batch, n, m) GRF realizations
    """
```

### 5.3 `simulate.py` changes

Add a module-level constant or argument `USE_KL` (or equivalent). When True, `_simulate_chunk_b` calls `calc_kl_basis` once (cached on first JIT trace) and `sample_kl_grf` per chunk instead of `circulant_grf`.

```python
@partial(jax.jit, static_argnums=(11, 12, 13))
def simulate_b(key, ter, st, cr, crsd, av1, av2, av3,
               sis, sig, si, nsim, chunk_size=4, use_kl=False):
    ...
```

`use_kl` is static (added to `static_argnums`). Default is False — opt-in for the first release cycle.

## 6. Algorithm

### 6.1 Setup (once per `(sig, k_max, variance_threshold)`)

1. Compute `LAM` via existing `grf.calc_LAM(s1=sig, s2=sig)` (already exists, fp64).
2. Flatten `LAM²` into a (N_pad · M_pad,) vector of eigenvalues.
3. Argsort descending. Pick top K such that cumulative sum / total sum ≥ variance_threshold, capped at k_max.
4. For each kept index `flat_k` → unflatten to `(i*_k, j*_k)`.
5. Build `V[NM, 2K]` by evaluating `Re(e_k(n, m))` and `Im(e_k(n, m))` × √λ_k at all (n, m). fp32.
6. Return `V, K, variance_captured`.

This is done in NumPy / JAX once per JIT trace. ~10 ms cost. Cached implicitly via JIT.

### 6.2 Per-chunk simulation

Replaces the current FFT path in `_simulate_chunk_b`:

```python
# OLD (FFT path):
z = jax.random.normal(kg, (chunk, n_fft, 2, n_pad, m_pad), fp32)
X = LAM * (z[..., 0, :, :] + 1j * z[..., 1, :, :])
F = jnp.fft.fft2(X)
F1 = F[..., :N, :M].real
F2 = F[..., :N, :M].imag
grf_path = jnp.stack([F1, F2], axis=2).reshape(chunk, NSTEP, N, M)

# NEW (K-L path):
z = jax.random.normal(kg, (chunk, NSTEP, 2 * K), fp32)
grf_path = jnp.einsum("csk,Nk->csN", z, V)   # (chunk, NSTEP, NM)
grf_path = grf_path.reshape(chunk, NSTEP, N, M)
```

Everything downstream (drift + cumsum + first-crossing + categorization) is unchanged.

## 7. Validation strategy

Three tests, each must pass:

### 7.1 Marginal variance parity (`test_grf_kl_parity.py`)

Sample 10,000 GRFs from both the circulant generator and the K-L generator at sig=10. Compute the empirical variance per cell on the (N, M) grid. Both should match the analytic variance to within 1% relative error.

### 7.2 Autocorrelation parity (`test_grf_kl_parity.py`)

Compute the empirical autocorrelation function at lags (0..20) in row direction and column direction from 10,000 samples of each generator. The two ACFs should agree within 1% absolute at all lags.

### 7.3 End-to-end recovery (`test_simulate_b_kl_parity.py`)

Run the same parameter-recovery experiment as Stage 5 (synthetic data at known params, ±10% perturbed start, simplex Nelder-Mead). With K-L active, recovery should still hit 12/12 params within 7% of truth and average error ≤ 5%.

### 7.4 Speed check

Smoke benchmark at nsim=512 on CPU: K-L path should be at least 2× faster than FFT path. On H100 at nsim=9000 (separate run): target ≤ 1.0s per call.

## 8. Open decisions for user approval

These are the assumptions I'd default to if you don't push back; flagged here for explicit confirmation when you're back:

1. **K selection**: auto-tune to `variance_threshold=0.999` with `k_max=200`. Alternative: hardcode K=100 for simplicity. **Recommend auto-tune** — handles varying sig during fits without manual retuning.

2. **API compatibility**: opt-in `use_kl=False` flag in `simulate_b` for one release cycle, then flip default to True after parity is established. Alternative: just replace the FFT path entirely. **Recommend opt-in flag** — safer, lets fit recovery tests confirm parity before switching default.

3. **Variance threshold**: 0.999. Alternative: 0.99 (faster, less rigorous) or 0.9999 (slower, tighter). **Recommend 0.999** — matches the typical analytic tolerance in GRF simulation papers, gives K ≈ 100 at sig=10.

4. **fp32 precision**: keep as Stage 5. Alternative: bf16 for the GEMM (faster, riskier for variance bias). **Recommend fp32** — defer bf16 to a separate stage.

5. **Implementation pattern**: same as Stage 4/5 — subagent writes the implementation, fix-agent loop on test failures, single review pass at the end. Alternative: I write it directly. **Recommend subagent pattern** — matches what worked for Stage 4 and Stage 3.5.

6. **Validation data scale**: 10,000 GRF samples per parity test. This takes ~30 sec on laptop CPU. Acceptable, or run on H100 for tighter empirical bounds? **Recommend laptop CPU** — sufficient signal for parity.

## 9. Implementation plan summary

Phase 1 (Day 1): Implement `grf_kl.py` with the basis builder, write smoke tests, verify on laptop.
Phase 2 (Day 2): Modify `simulate.py` to support `use_kl=True` path. Run parity tests on laptop.
Phase 3 (Day 3): If parity passes, run end-to-end recovery on laptop at small nsim. If recovery matches Stage 5 quality, prepare H100 benchmark script.
Phase 4 (Day 4): Rent H100, run K-L benchmark + parity, write completion summary doc.
Phase 5 (Day 5): If H100 numbers hit target (≤ 1s/call, ≤ 6 min fit), commit + tag. Update methods paper draft.

Total estimated effort: ~5 working days. Could be faster if subagent execution goes cleanly.

## 10. Success criteria (committed)

- Parity tests pass (variance, ACF, end-to-end recovery)
- H100 benchmark: per-call simulator ≤ 1.0s at nsim=9000
- Full fit at production scale ≤ 6 min wall clock
- No regression in 1D (Model A) or any existing test
- Completion doc with measured numbers + comparison to Stage 5 baseline

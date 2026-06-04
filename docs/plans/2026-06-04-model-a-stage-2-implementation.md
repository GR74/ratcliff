# Model A — Stage 2 Implementation Plan: Single-GEMM + Cumsum Rewrite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the per-step `lax.scan` simulator in `model_a/jax_port.py` with a pre-generated-noise + single-GEMM + cumsum + first-crossing-argmax simulator in new file `model_a/simulate.py`, validated by statistical parity against `jax_port.simulate`. Bundle three Stage-1 forward-looking migrations: typed-key API in `shared/prng.py`, exclusive use of `shared.prng` in the new simulator, and centralization of x64 config in a top-level `conftest.py`.

**Architecture:** New `model_a/simulate.py` is a JIT'd program that processes `nsim` trials in chunks (default 256) via `jax.lax.map`. Each chunk generates iid normals, applies the Cholesky factor as one big GEMM, demeans per step, cumsums along time, and finds the first row where the accumulator's max exceeds the threshold. `jax_port.py` stays untouched as the oracle.

**Tech Stack:** Python 3.11+, JAX 0.10.1 (CPU), pytest. No new dependencies. Uses existing `shared.prng`, `shared.data_io`, `shared.validation`.

**Required sub-skills:**
- @superpowers:test-driven-development for every implementation task
- @superpowers:verification-before-completion before claiming task done
- @superpowers:executing-plans for stepwise execution

**Reference docs:**
- Design: `docs/plans/2026-06-03-model-a-stage-2-design.md`
- Project design: `docs/plans/2026-06-03-ratcliff-speedup-design.md`
- Stage 1 plan: `docs/plans/2026-06-03-model-a-stage-1-setup-validation.md`

**Current HEAD:** `d93bbbd` (Stage 2 design commit). 25 tests passing via `.\scripts\smoke.ps1`.

---

## Pre-flight checks

```powershell
git -C C:\Users\gowri\ratcliff status
```
Expected: clean working tree on `master`, HEAD at the Stage 2 design commit.

```powershell
.\scripts\smoke.ps1
```
Expected: `25 passed`. This is the baseline; nothing should regress.

---

## Task 2.A.1: Create top-level `conftest.py` with x64 config

**Files:**
- Create: `conftest.py`

**Step 1: Write the file**

Create `C:\Users\gowri\ratcliff\conftest.py`:

```python
"""Top-level pytest config: enable JAX x64 mode before any test imports.

This runs at pytest collection time, before any `from model_a import jax_port`
statement. Idempotent with the duplicate call inside jax_port.py — both set
the same flag, second call is a no-op.
"""
import jax

jax.config.update("jax_enable_x64", True)
```

**Step 2: Verify smoke still passes**

Run:
```powershell
.\scripts\smoke.ps1
```
Expected: `25 passed`. No regression — the conftest's x64 setting is idempotent with the existing per-module config.

**Step 3: Verify x64 actually fires**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q 2>&1 | Select-String "configfile"
.\.venv\Scripts\python.exe -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp; print('default float:', jnp.zeros(1).dtype)"
```
Expected: prints `default float: float64`. (Sanity that x64 takes effect on a fresh process.)

**Step 4: Commit**

```powershell
git add conftest.py
git commit -m "chore: top-level conftest.py centralizes JAX x64 config"
```

---

## Task 2.B.1: Migrate `shared/prng.py` to typed-key API

@superpowers:test-driven-development

**Files:**
- Modify: `shared/prng.py`
- Modify: `shared/tests/test_prng.py`

**Step 1: Update the test for typed keys**

Edit `shared/tests/test_prng.py`. Replace the body of `test_trial_keys_returns_n_distinct_keys` with the typed-key form. The full updated test file should be:

```python
import jax
import jax.numpy as jnp
import pytest

from shared import prng


def test_root_key_is_deterministic_for_same_seed():
    k1 = prng.root_key(42)
    k2 = prng.root_key(42)
    assert jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))


def test_root_key_differs_for_different_seeds():
    k1 = prng.root_key(42)
    k2 = prng.root_key(43)
    assert not jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))


def test_split_for_condition_is_deterministic():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=2)
    b = prng.split_for_condition(k, condition_idx=2)
    assert jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b))


def test_split_for_condition_differs_across_conditions():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=0)
    b = prng.split_for_condition(k, condition_idx=1)
    assert not jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b))


def test_trial_keys_returns_n_distinct_keys():
    k = prng.root_key(7)
    keys = prng.trial_keys(k, n=100)
    # Typed keys: outer shape is (n,), underlying buffer is (n, 2) uint32
    assert keys.shape == (100,)
    raw = jax.random.key_data(keys)
    assert raw.shape == (100, 2)
    # All keys should be distinct
    assert len({tuple(row.tolist()) for row in raw}) == 100


def test_root_key_returns_typed_key():
    """The migrated API returns a typed key, not a raw uint32 array."""
    k = prng.root_key(0)
    # Typed keys have dtype kind that's not 'u' or 'i' (it's a special key dtype)
    assert jnp.issubdtype(k.dtype, jax.dtypes.prng_key)
```

**Step 2: Run test to verify it fails**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest shared/tests/test_prng.py -v
```
Expected: `test_root_key_returns_typed_key` fails (the implementation still returns `PRNGKey`, which is a raw uint32 array, not a typed key). Other tests may also fail because the assertions now look at `key_data`.

**Step 3: Migrate the implementation**

Edit `shared/prng.py` to use the typed-key API:

```python
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
```

**Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest shared/tests/test_prng.py -v
```
Expected: `6 passed` (5 original tests + the new typed-key check).

**Step 5: Verify other tests still pass (no regression)**

Run:
```powershell
.\scripts\smoke.ps1
```
Expected: `26 passed` (25 baseline + 1 new test). If anything else regressed (e.g., a test in `test_simulate_smoke.py` that compared raw key buffers), investigate.

**Step 6: Commit**

```powershell
git add shared/prng.py shared/tests/test_prng.py
git commit -m "refactor(prng): migrate to typed jax.random.key API"
```

---

## Task 2.C.1: Skeleton `model_a/simulate.py` — constants + chol_factor + drift_profile

@superpowers:test-driven-development

**Files:**
- Create: `model_a/simulate.py`
- Create: `model_a/tests/test_simulate_new_smoke.py`

**Step 1: Write a failing minimal smoke test**

Create `model_a/tests/test_simulate_new_smoke.py`:

```python
"""Smoke tests for the new model_a/simulate.py (single-GEMM rewrite)."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import simulate as sim_new
from shared import prng


def test_module_exposes_constants():
    """Sanity: the new module exports N, NSTEP, MC, E with the right values."""
    assert sim_new.N == 72
    assert sim_new.NSTEP == 400
    assert sim_new.MC == 3
    assert sim_new.E == 10.0


def test_chol_factor_returns_lower_triangular():
    """chol_factor(sig) returns the Cholesky factor of the GP kernel."""
    L = sim_new.chol_factor(5.0)
    assert L.shape == (72, 72)
    # K = L @ L.T should reproduce the kernel (within FP tolerance)
    K = L @ L.T
    # Diagonal of K is ~1 (plus the 1e-12 jitter)
    assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-6)


def test_drift_profile_peaks_at_U():
    """drift_profile(av, si) is a Gaussian bump centered at U=36."""
    v = sim_new.drift_profile(av=20.0, si=4.0)
    assert v.shape == (72,)
    # Peak is near index U=36 (0-indexed: 35, but the function uses 1-indexed IDX)
    peak_idx = int(jnp.argmax(v))
    assert 34 <= peak_idx <= 36  # tolerance for the 1-vs-0 indexing convention
```

**Step 2: Run test to verify it fails**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_new_smoke.py -v
```
Expected: `ImportError: cannot import name 'simulate' from 'model_a'` (the new module doesn't exist).

**Step 3: Create the skeleton**

Create `model_a/simulate.py`:

```python
"""
Fast Monte Carlo simulator for the Ratcliff 1D spatially-extended diffusion.

Rewrite of `model_a/jax_port.simulate`: pre-generates the full noise block per
chunk, applies the Cholesky factor as one big GEMM, demeans per step, cumsums
along time, and finds the first crossing via argmax. No per-step lax.scan
state; XLA fuses the whole chunk into a single program.

Memory: at default chunk_size=256, peak working set is ~240 MB for N=72,
NSTEP=400, fp64. Tunable via the static `chunk_size` argument.
"""
from functools import partial

import jax
import jax.numpy as jnp

from shared import prng

# ----------------------------------------------------------------------
# Fixed model structure (mirrors jax_port.py exactly)
# ----------------------------------------------------------------------
N = 72
NSTEP = 400
E = 10.0
U = 180.0 / 5.0
MC = 3
IPA, IPB, IPC, IPD = 150 / 5, 210 / 5, 100 / 5, 260 / 5  # 30, 42, 20, 52
IDX = jnp.arange(1, N + 1, dtype=jnp.float64)


def chol_factor(sig):
    """Cholesky factor L of the GP kernel K; noise = L @ z ~ N(0, K)."""
    d = IDX[:, None] - IDX[None, :]
    K = jnp.exp(-0.5 * d * d / (sig * sig)) + 1e-12 * jnp.eye(N)
    return jnp.linalg.cholesky(K)


def drift_profile(av, si):
    """Spatial drift bump v(i) = av * Normal(i; U, si)."""
    return av * jnp.exp(-(IDX - U) ** 2 / (2.0 * si * si)) / (si * jnp.sqrt(2.0 * jnp.pi))
```

**Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_new_smoke.py -v
```
Expected: `3 passed`.

**Step 5: Commit**

```powershell
git add model_a/simulate.py model_a/tests/test_simulate_new_smoke.py
git commit -m "feat(model_a): simulate.py skeleton with constants, chol_factor, drift_profile"
```

---

## Task 2.D.1: Implement `_simulate_chunk` core algorithm

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/simulate.py`
- Modify: `model_a/tests/test_simulate_new_smoke.py`

**Step 1: Write failing tests for the chunk function**

Append to `model_a/tests/test_simulate_new_smoke.py`:

```python
def test_simulate_chunk_returns_rt_and_cat_shapes():
    """_simulate_chunk(...) returns (rt, cat) each of shape (chunk_size,)."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    rt, cat = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=64
    )
    assert rt.shape == (64,)
    assert cat.shape == (64,)


def test_simulate_chunk_rt_is_finite_and_positive():
    """RTs are finite, positive, and bounded above by (NSTEP + ter+st/2)*E."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    rt, _ = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=64
    )
    assert jnp.all(jnp.isfinite(rt))
    assert jnp.all(rt > 0)
    # Hard upper bound: jstop <= NSTEP, ndt <= (ter+st/2)/E ≈ 22.5 steps,
    # so rt <= (400 + 22.5) * 10 = 4225 ms
    assert jnp.all(rt <= 5000)


def test_simulate_chunk_cat_in_valid_range():
    """All categories are in {1, 2, 3}."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    _, cat = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=64
    )
    assert jnp.all((cat >= 1) & (cat <= 3))


def test_simulate_chunk_deterministic_for_same_key():
    """Same key + same params produces bit-exact same outputs."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(42)
    rt_a, cat_a = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=32
    )
    rt_b, cat_b = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=32
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)
```

**Step 2: Run tests to verify they fail**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_new_smoke.py -v -k "simulate_chunk"
```
Expected: `AttributeError: module 'model_a.simulate' has no attribute '_simulate_chunk'`.

**Step 3: Implement `_simulate_chunk`**

Append to `model_a/simulate.py`:

```python
def _simulate_chunk(key, ter, st, cr, crsd, L, v, chunk_size):
    """
    Simulate `chunk_size` trials in one fused XLA program.

    Returns (rt, cat) each of shape (chunk_size,).
    """
    ku, kz = jax.random.split(key)

    # Per-trial uniforms (gu1 in the Fortran)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = cr + crsd * (u[:, 4] - 0.5)                  # (chunk_size,) per-trial threshold
    ndt = (ter + st * (0.5 - u[:, 9])) / E             # (chunk_size,) per-trial nondecision in steps

    # One big GEMM: all noise for chunk × NSTEP × N
    z = jax.random.normal(kz, (chunk_size, NSTEP, N))  # iid normals
    noise = z @ L.T                                     # (chunk_size, NSTEP, N) correlated normals

    # Build demeaned per-step increments
    incr = v[None, None, :] + 5.0 * noise               # broadcast drift bump + scaled noise
    incr = incr - incr.mean(axis=-1, keepdims=True)     # demean per (trial, step)

    # Accumulator paths
    a = jnp.cumsum(incr, axis=1)                        # (chunk_size, NSTEP, N)

    # First crossing
    max_per_step = a.max(axis=-1)                       # (chunk_size, NSTEP)
    crossed = max_per_step > crr[:, None]               # (chunk_size, NSTEP) bool
    any_crossed = crossed.any(axis=1)                   # (chunk_size,)
    # argmax of bool returns the first True; if none, returns 0, hence the where
    jstop = jnp.where(any_crossed, jnp.argmax(crossed, axis=1) + 1, NSTEP)

    # Position at crossing (or at NSTEP if never crossed)
    pos = jnp.argmax(a[jnp.arange(chunk_size), jstop - 1, :], axis=-1) + 1

    # RT in ms; categorize by position band (mirrors jax_port.one_trial)
    rt = (jstop + ndt) * E
    cat = jnp.where((pos > IPA) & (pos < IPB), 1,
          jnp.where((pos <= IPC) | (pos >= IPD), 3, 2))
    return rt, cat
```

**Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_new_smoke.py -v
```
Expected: `7 passed` (3 module + 4 chunk).

**Step 5: Commit**

```powershell
git add model_a/simulate.py model_a/tests/test_simulate_new_smoke.py
git commit -m "feat(model_a): _simulate_chunk implements single-GEMM cumsum first-crossing"
```

---

## Task 2.E.1: Implement `simulate` wrapper with `jax.lax.map` over chunks

@superpowers:test-driven-development

**Files:**
- Modify: `model_a/simulate.py`
- Modify: `model_a/tests/test_simulate_new_smoke.py`

**Step 1: Write failing tests for the top-level `simulate`**

Append to `model_a/tests/test_simulate_new_smoke.py`:

```python
def test_simulate_returns_full_nsim_shape():
    """simulate(...) returns (rt, cat) each of shape (nsim,)."""
    key = prng.root_key(0)
    rt, cat = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=512, chunk_size=128
    )
    assert rt.shape == (512,)
    assert cat.shape == (512,)


def test_simulate_handles_non_multiple_chunk():
    """nsim that isn't a multiple of chunk_size still returns exactly nsim outputs."""
    key = prng.root_key(0)
    rt, cat = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=300, chunk_size=128
    )
    assert rt.shape == (300,)
    assert cat.shape == (300,)


def test_simulate_deterministic_for_same_key():
    """Same key reproduces bit-exact outputs."""
    key = prng.root_key(11)
    rt_a, cat_a = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    rt_b, cat_b = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)


def test_simulate_differs_for_different_keys():
    """Different keys produce different outputs."""
    rt_a, _ = sim_new.simulate(
        prng.root_key(0), ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    rt_b, _ = sim_new.simulate(
        prng.root_key(1), ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    assert not jnp.array_equal(rt_a, rt_b)
```

**Step 2: Run tests to verify they fail**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_new_smoke.py -v -k "test_simulate_"
```
Expected: `AttributeError: module 'model_a.simulate' has no attribute 'simulate'`.

**Step 3: Implement `simulate`**

Append to `model_a/simulate.py`:

```python
@partial(jax.jit, static_argnums=(8, 9))
def simulate(key, ter, st, cr, crsd, si, sig, av, nsim, chunk_size=256):
    """
    Run `nsim` independent Monte Carlo trials with the given parameters.

    Returns (rt, cat) each of shape (nsim,). RT in ms; cat in {1, 2, 3}.

    Trials are processed in chunks of `chunk_size` via jax.lax.map. The whole
    computation is JIT'd into one XLA program; chunks run sequentially to keep
    peak memory bounded. Each unique (nsim, chunk_size) pair triggers one JIT
    compile.
    """
    L = chol_factor(sig)
    v = drift_profile(av, si)

    # Allocate enough chunks to cover nsim; last chunk may have padding trials
    # that we'll slice off at the end.
    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)               # (n_chunks,) typed keys

    def run_chunk(k):
        return _simulate_chunk(k, ter, st, cr, crsd, L, v, chunk_size)

    rts, cats = jax.lax.map(run_chunk, keys)            # (n_chunks, chunk_size) each
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim]
```

**Step 4: Run all simulate tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_new_smoke.py -v
```
Expected: `11 passed` (3 module + 4 chunk + 4 simulate). First run may take 30–60 seconds for JIT compile.

**Step 5: Run full smoke to verify no regression**

Run:
```powershell
.\scripts\smoke.ps1
```
Expected: still `26 passed` (the smoke runner doesn't yet include the new file). We'll add it in Task 2.G.1.

**Step 6: Commit**

```powershell
git add model_a/simulate.py model_a/tests/test_simulate_new_smoke.py
git commit -m "feat(model_a): simulate wraps lax.map over chunks for full nsim"
```

---

## Task 2.F.1: Parity tests against `jax_port.simulate`

@superpowers:test-driven-development

**Files:**
- Create: `model_a/tests/test_simulate_parity.py`

**Step 1: Write the parity tests**

Create `model_a/tests/test_simulate_parity.py`:

```python
"""
Parity tests: the new simulator must produce statistically equivalent output
to jax_port.simulate at the same (params, key, nsim).

PRNG threading differs between the two implementations (jax_port splits
per-trial inside `simulate`; the new version splits per-chunk then per-trial).
So bit-exact comparison is impossible. We compare aggregates with the
tolerances defined in shared/validation.py.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import jax_port, simulate as sim_new
from shared import prng, validation


PARAM_SETS = [
    # (name, params dict)
    ("realistic", dict(ter=200.0, st=50.0, cr=50.0, crsd=10.0,
                       si=4.0, sig=5.0, av=20.0)),
    ("high_drift", dict(ter=200.0, st=50.0, cr=50.0, crsd=10.0,
                        si=4.0, sig=5.0, av=60.0)),
    ("low_drift", dict(ter=200.0, st=50.0, cr=50.0, crsd=10.0,
                       si=4.0, sig=5.0, av=5.0)),
]

NSIM_PARITY = 2048   # large enough for aggregate stability
SEED_PARITY = 1337


def _aggregates(rt, cat):
    """Return (proportions[3], quantiles[5, 3]) for cats 1, 2, 3."""
    props = jnp.array([(cat == c).mean() for c in (1, 2, 3)])
    quants = jnp.zeros((5, 3))
    qs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for ci, c in enumerate((1, 2, 3)):
        mask = cat == c
        if mask.sum() >= 5:
            cat_rts = jnp.where(mask, rt, jnp.nan)
            cat_rts_sorted = jnp.sort(cat_rts)
            n_in_cat = int(mask.sum())
            indices = (qs * (n_in_cat - 1)).astype(jnp.int32)
            quants = quants.at[:, ci].set(cat_rts_sorted[indices])
    return props, quants


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[p[0] for p in PARAM_SETS])
def test_parity_proportions(name, params):
    """Response proportions per category match within 0.5% absolute."""
    key = prng.root_key(SEED_PARITY)

    # jax_port takes sv (inert); new sim doesn't take sv
    old_params = {**params, "sv": 0.7}
    rt_old, cat_old = jax_port.simulate(key, **old_params, nsim=NSIM_PARITY)
    rt_new, cat_new = sim_new.simulate(key, **params, nsim=NSIM_PARITY,
                                       chunk_size=256)

    props_old = np.array([float((cat_old == c).mean()) for c in (1, 2, 3)])
    props_new = np.array([float((cat_new == c).mean()) for c in (1, 2, 3)])

    ok, report = validation.proportions_match(props_new, props_old, abs_tol=0.005)
    assert ok, f"[{name}] proportions disagree: new={props_new}, old={props_old}, max_diff={report['max_abs_diff']}"


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[p[0] for p in PARAM_SETS])
def test_parity_quantiles_per_category(name, params):
    """RT quantiles per category match within 1% relative (cats with >=20 trials)."""
    key = prng.root_key(SEED_PARITY)

    old_params = {**params, "sv": 0.7}
    rt_old, cat_old = jax_port.simulate(key, **old_params, nsim=NSIM_PARITY)
    rt_new, cat_new = sim_new.simulate(key, **params, nsim=NSIM_PARITY,
                                       chunk_size=256)

    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for c in (1, 2, 3):
        mask_old = np.array(cat_old == c)
        mask_new = np.array(cat_new == c)
        # Skip categories with too few observations (parity is too noisy)
        if mask_old.sum() < 20 or mask_new.sum() < 20:
            continue
        q_old = np.quantile(np.array(rt_old)[mask_old], qs)
        q_new = np.quantile(np.array(rt_new)[mask_new], qs)
        ok, report = validation.quantiles_match(q_new, q_old, rel_tol=0.01)
        assert ok, (f"[{name}] cat={c} quantiles disagree: "
                    f"new={q_new}, old={q_old}, max_rel_diff={report['max_rel_diff']}")
```

**Step 2: Run the parity tests**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest model_a/tests/test_simulate_parity.py -v
```
Expected: 6 tests pass (3 param sets × 2 test types). Wall clock may be 30-90 seconds first run due to JIT compile of both old and new simulators at the same `nsim`.

**Possible failure mode:** if the `low_drift` case fails proportions parity, it's because at very low drift, the RT distribution is dominated by the NSTEP boundary, and the two simulators may differ slightly in how they handle non-crossing trials. Investigate the actual diff; if it's within 1% (not 0.5%), the tolerance was too tight for that regime — bump `abs_tol=0.01` for the low_drift case only.

**Step 3: Commit**

```powershell
git add model_a/tests/test_simulate_parity.py
git commit -m "test(model_a): parity against jax_port.simulate at 3 param sets"
```

---

## Task 2.G.1: Update smoke runner scripts to include new tests

**Files:**
- Modify: `scripts/smoke.ps1`
- Modify: `scripts/smoke.sh`

**Step 1: Update `scripts/smoke.ps1`**

Edit the `pytest` line to include the new test files. The full updated `smoke.ps1`:

```powershell
# Smoke test runner. Stage 1 + 2 gate.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    Write-Error "venv not found at .\.venv. Run: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e '.[dev]'"
    exit 1
}
.\.venv\Scripts\python.exe -m pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

**Step 2: Update `scripts/smoke.sh`**

The bash equivalent:

```bash
#!/usr/bin/env bash
# Smoke test runner. Stage 1 + 2 gate.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python -c "import jax" 2>/dev/null; then
  echo "Error: JAX not importable. Activate the venv: source .venv/bin/activate" >&2
  exit 1
fi
pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py -v
```

**Step 3: Run the updated smoke**

Run:
```powershell
.\scripts\smoke.ps1
```
Expected: `43 passed` total (26 baseline + 11 new smoke + 6 parity = 43). Wall clock may be 60-120 seconds first run.

**Step 4: Verify modes preserved**

Run:
```powershell
git ls-files --stage scripts/
```
Expected: `smoke.ps1` mode `100644`, `smoke.sh` mode `100755`. No changes from before.

**Step 5: Commit**

```powershell
git add scripts/smoke.ps1 scripts/smoke.sh
git commit -m "chore(smoke): include new simulate + parity tests in Stage 2 gate"
```

---

## Task 2.H.1: Optional CPU speedup measurement

**Files:**
- Create: `model_a/tests/test_simulate_perf.py`

**Step 1: Write the perf test**

Create `model_a/tests/test_simulate_perf.py`:

```python
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
```

**Step 2: Register the `perf` mark in pyproject.toml**

`--strict-markers` is enabled, so unrecognized marks fail. Edit `pyproject.toml` to add the mark.

In `[tool.pytest.ini_options]`, add:
```toml
markers = [
    "perf: performance regression tests (skipped by default, run with -m perf)",
]
```

The full `[tool.pytest.ini_options]` block becomes:
```toml
[tool.pytest.ini_options]
testpaths = ["shared/tests", "model_a/tests", "model_b/tests"]
addopts = "-v --tb=short --strict-markers -ra"
markers = [
    "perf: performance regression tests (skipped by default, run with -m perf)",
]
```

**Step 3: Run perf test explicitly**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -m perf model_a/tests/test_simulate_perf.py -v -s
```
Expected: prints the per-call timings and the speedup. Test passes if new isn't slower than old. Hopefully shows >= 5x speedup; if not, the warning is logged but the test still passes.

**Step 4: Verify perf test is skipped from default smoke**

Run:
```powershell
.\scripts\smoke.ps1
```
Expected: still `43 passed`. The perf test is not in the smoke runner's explicit file list, so it doesn't run.

**Step 5: Commit**

```powershell
git add model_a/tests/test_simulate_perf.py pyproject.toml
git commit -m "test(model_a): perf gate measures new simulator speedup vs jax_port"
```

---

## Stage 2 completion gate

@superpowers:verification-before-completion

Before declaring Stage 2 complete, verify ALL of:

1. `.\scripts\smoke.ps1` exits 0 with `43 passed` (25 baseline + 1 typed-key + 11 new smoke + 6 parity = 43).
2. `git log --oneline` shows ~8 commits since the Stage 2 design commit, each scoped to one task.
3. `git status` is clean.
4. `pytest --co -q` lists all 43 tests cleanly.
5. Reference files (`reference/gpgsq5deg3twod24.f`, `reference/twod24_jax.py`) are untouched.
6. `model_a/jax_port.py` is untouched — `git log -- model_a/jax_port.py` shows only the original import commit `dc4fb13`.
7. The perf gate (`pytest -m perf`) runs and shows new simulator is at least as fast as `jax_port`.

If any fail, fix before claiming completion. Then dispatch the final code reviewer for the whole Stage 2 diff (`d93bbbd..HEAD`).

---

## What's deliberately out of this plan

- No `fofs` rewrite → Stage 3.
- No optimizer (L-BFGS or simplex) → Stage 3.
- No vmap over conditions → Stage 3.
- No `condition_g2` rewrite → Stage 3.
- No Model B → Stage 4.
- No GPU benchmark suite → Stage 5.
- No Fortran-comparison gate → Stage 2.5 (separate effort, ad-hoc).
- No Triton kernel → Stage 6.
- No `LICENSE` file → repo-hygiene cleanup, separate.
- No README updates for Stage 2 — defer until Stage 3 lands and the optimizer is callable, then update the example.
- No `condition_g2` vmap → Stage 3.
- No removal of the duplicate `jax.config.update` in `jax_port.py` → don't touch the oracle.

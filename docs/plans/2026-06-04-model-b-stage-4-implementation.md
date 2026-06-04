# Model B — Stage 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Port the 2D Gaussian-random-field diffusion model from `benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS` (the "took weeks" Fortran). Deliver `model_b/grf.py`, `model_b/simulate.py`, `model_b/objective.py`, `model_b/fit.py`, plus data parsing and validation against the archive's recorded Fortran outputs. Simplex-only optimizer (gradient issue from Stage 3 applies).

**Architecture:** GRF noise via FFT-based circulant embedding (Kroese §2.2). Per-timestep one FFT (skipping the F1/F2 caching trick for simpler code — documented as Stage 5 optimization target). 2D `100×160` accumulator. 5 response categories from `k(i,j)` zones. 2 conditions. `lax.scan` over timesteps with small chunked vmap over trials. Scipy NM for fits.

**Tech Stack:** JAX 0.10.1, scipy, pytest. No new deps.

**Current HEAD:** `af8035d`. 52 tests passing. `model_a/objective.py::fofs_new` ships with bit-exact `condition_g2_vectorized`.

**Reference docs:**
- Sketch: `docs/plans/2026-06-04-model-b-stage-4-design-sketch.md`
- Fortran source: `reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`
- GRF history: `reference/README_random_field` (Russ's 2016 journal)

**Memory budget on laptop:** keep `chunk_size` small (default 4). Per-chunk peak working set ~3 MB with skip-trick FFT. Fast on H100, tolerable on laptop CPU.

---

## Pre-flight checks

```powershell
git -C C:\Users\gowri\ratcliff status
.\scripts\smoke.ps1
```
Expected: clean tree at `af8035d`, `52 passed`.

Copy archive's recorded Fortran outputs into `reference/` and `data/`:

```powershell
$src = 'C:\Users\gowri\Downloads\benchtwod3\benchtwod3'
$dst_data = 'C:\Users\gowri\ratcliff\data'
$dst_ref  = 'C:\Users\gowri\ratcliff\reference\archive_outputs'
New-Item -ItemType Directory -Path $dst_ref -Force | Out-Null
Copy-Item "$src\benchtwod3mpi.1","$src\benchtwod3mpi.2","$src\benchtwod3mpi.3","$src\benchtwod3mpi.4","$src\benchtwod3mpi.5","$src\benchtwod3mpi.6" -Destination $dst_ref
Copy-Item "$src\twod3parallelmpi.out" -Destination $dst_ref
# parinp + twod3datanew already in data/ from Stage 0
Get-ChildItem $dst_ref | Format-Table Name, Length
```

---

## Task 4.A.1: `shared/data_io.load_twod3datanew` parser

@superpowers:test-driven-development

**Files:**
- Modify: `shared/data_io.py`
- Modify: `shared/tests/test_data_io.py`

The Model B data file `data/twod3datanew` (already in repo) has a different format than `twod24data`. The Fortran reads it as (from `benchtwod3mpi.f` lines 38-44):
```fortran
do 10 i1=1,nc          ! nc=2 conditions
   read(1,*) acc(1),mn(1),(rry(j,1),j=1,nq),x1,x2,
             acc(2),mn(2),(rry(j,2),j=1,nq),x1,x2,
             ... up to acc(5),mn(5)        ! mc=5 categories
```
So one record per condition: 5 categories × 9 fields (prop, count, q1..q5, x1, x2) = 45 fields per condition. With `nc=2` conditions, 90 fields per "subject". Total file has multiple subjects.

**Step 1: Write failing test**

Add to `shared/tests/test_data_io.py`:

```python
def test_load_twod3datanew_returns_expected_shape():
    from shared import data_io
    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    d = data_io.load_twod3datanew(path)
    # 2 conditions per subject, 5 categories, 5 quantiles
    # Shape: (n_subjects, 2, 5) prop, (n_subjects, 2, 5) count, (n_subjects, 2, 5, 5) quant
    assert d["prop"].ndim == 3 and d["prop"].shape[1:] == (2, 5)
    assert d["count"].ndim == 3 and d["count"].shape[1:] == (2, 5)
    assert d["quant"].ndim == 4 and d["quant"].shape[1:] == (2, 5, 5)
    # Proportions per condition sum close to 1
    prop_sums = d["prop"].sum(axis=2)
    assert np.all(np.abs(prop_sums - 1.0) < 0.05)


def test_load_twod3datanew_first_record_values():
    from shared import data_io
    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    d = data_io.load_twod3datanew(path)
    # First subject, first condition, first category — sanity check the parser
    # (Actual values depend on the data file; this test confirms it's not all zeros)
    assert d["prop"][0, 0, 0] > 0.0
    assert d["count"][0, 0, 0] > 0
```

Run, confirm failure (AttributeError on `load_twod3datanew`).

**Step 2: Implement**

Append to `shared/data_io.py`:

```python
def load_twod3datanew(path, n_conditions: int = 2, n_categories: int = 5, n_quantiles: int = 5):
    """
    Parse a Model-B-style RT/proportion data file.

    Format per "subject record": n_conditions lines, each containing
    n_categories blocks of (prop, count, q1..q5, x1, x2) = 9 fields.
    Total fields per condition-line = n_categories * 9.

    Returns dict with arrays shaped (n_subjects, n_conditions, ...):
      prop  : (S, C, K) float
      count : (S, C, K) int
      quant : (S, C, Q, K) float
    """
    fields_per_cat = 2 + n_quantiles + 2          # prop, count, q1..qN, x1, x2 = 9
    fields_per_line = n_categories * fields_per_cat   # 45

    raw_lines = Path(path).read_text().splitlines()
    indexed = [(i, ln) for i, ln in enumerate(raw_lines, start=1) if ln.strip()]
    if len(indexed) % n_conditions != 0:
        raise ValueError(
            f"{path}: {len(indexed)} non-empty lines is not divisible by "
            f"n_conditions={n_conditions}"
        )
    n_subjects = len(indexed) // n_conditions

    prop  = np.zeros((n_subjects, n_conditions, n_categories), dtype=np.float64)
    count = np.zeros((n_subjects, n_conditions, n_categories), dtype=np.int64)
    quant = np.zeros((n_subjects, n_conditions, n_quantiles, n_categories), dtype=np.float64)

    for line_idx, (file_lineno, ln) in enumerate(indexed):
        fields = ln.split()
        if len(fields) != fields_per_line:
            raise ValueError(
                f"{path}:{file_lineno}: expected {fields_per_line} fields, "
                f"got {len(fields)}"
            )
        s, c = divmod(line_idx, n_conditions)
        for k in range(n_categories):
            base = k * fields_per_cat
            prop[s, c, k] = float(fields[base])
            count[s, c, k] = int(float(fields[base + 1]))
            for q in range(n_quantiles):
                quant[s, c, q, k] = float(fields[base + 2 + q])

    return {"prop": prop, "count": count, "quant": quant}
```

Run tests, verify they pass.

**Step 3: Commit**

```powershell
git add shared/data_io.py shared/tests/test_data_io.py
git commit -m "feat(data_io): add load_twod3datanew for Model B data"
```

---

## Task 4.A.2: Copy archive Fortran outputs into reference/

**Files:**
- Create: `reference/archive_outputs/benchtwod3mpi.1` (and 2–6)
- Create: `reference/archive_outputs/twod3parallelmpi.out`

Run the pre-flight copy command shown above. Then:

```powershell
git -C C:\Users\gowri\ratcliff add reference/archive_outputs/
git -C C:\Users\gowri\ratcliff status
git -C C:\Users\gowri\ratcliff commit -m "chore: import Fortran recorded outputs from benchtwod3 archive"
```

These files are the oracle for Model B parity tests (Task 4.E.1).

---

## Task 4.B.1: `model_b/grf.py::calc_LAM`

@superpowers:test-driven-development

**Files:**
- Create: `model_b/grf.py`
- Create: `model_b/tests/test_grf_smoke.py`

**Step 1: Failing test**

```python
"""Smoke tests for model_b/grf.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import grf


def test_calc_LAM_returns_correct_shape():
    LAM = grf.calc_LAM(n=100, m=160, s1=15.0, s2=15.0)
    assert LAM.shape == (199, 319)  # (2n-1, 2m-1)


def test_calc_LAM_is_nonneg():
    """LAM should be non-negative everywhere (it's the sqrt of a spectral density)."""
    LAM = grf.calc_LAM(n=100, m=160, s1=15.0, s2=15.0)
    assert jnp.all(LAM >= 0)


def test_calc_LAM_fails_on_large_s():
    """Per Russ's notes, s > 17.95 breaks the positive-definite embedding."""
    with pytest.raises(ValueError, match="positive definite"):
        grf.calc_LAM(n=100, m=160, s1=20.0, s2=20.0)
```

Run, confirm ImportError.

**Step 2: Implement**

Create `model_b/grf.py`:

```python
"""
Gaussian Random Field generator via circulant embedding (Kroese §2.2).

Mirrors `calc_LAM` and `circulant_grf` from the Fortran reference
`benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`. The kernel formula is the
Matern-class variogram documented in `reference/README_random_field`.
"""
import jax
import jax.numpy as jnp

# Default field size matches the Fortran (n=100 rows, m=160 cols).
DEFAULT_N = 100
DEFAULT_M = 160


def _kernel_row(i, j, s1, s2):
    """ρ(x=i, y=j) = (1 - x²/s2² - xy/(s1·s2) - y²/s1²) · exp(-(x²/s2² + y²/s1²))"""
    x = i.astype(jnp.float64)
    y = j.astype(jnp.float64)
    return (1.0 - x * x / (s2 * s2) - x * y / (s1 * s2) - y * y / (s1 * s1)) \
        * jnp.exp(-(x * x / (s2 * s2) + y * y / (s1 * s1)))


def calc_LAM(n: int = DEFAULT_N, m: int = DEFAULT_M, s1: float = 15.0, s2: float = 15.0):
    """
    Compute the spectral square root LAM of the block-circulant embedded kernel.

    Returns LAM of shape (2n-1, 2m-1). Used by circulant_grf to generate
    correlated GRF samples via FFT.

    Raises ValueError if the embedding is not positive-definite (i.e., the
    smallest spectral value is below -1e-10). Per Russ's notes, this happens
    at field 100x160 when s1, s2 > 17.95.
    """
    # Build the autocovariance over displacements 0..n-1 (rows) and 0..m-1 (cols).
    # rows shape (n, m): ρ(dx=i, dy=j) for i in 0..n-1, j in 0..m-1.
    i_grid = jnp.arange(n)[:, None]
    j_grid = jnp.arange(m)[None, :]
    rows = _kernel_row(j_grid, i_grid, s1, s2)
    # cols shape (n, m): ρ(dx=-i, dy=j) = ρ(dx=i, dy=j) for this even kernel.
    cols = _kernel_row(-j_grid, i_grid, s1, s2)

    # Embed in a (2n-1) x (2m-1) block-circulant matrix.
    blkcirc = jnp.zeros((2 * n - 1, 2 * m - 1), dtype=jnp.complex128)
    # Top-left quadrant: rows[j, i]
    blkcirc = blkcirc.at[:n, :m].set(rows)
    # Top-right quadrant: cols[j, n_m - i] (reverse along m axis, shifted)
    for i in range(1, m):
        blkcirc = blkcirc.at[:n, i + m - 1].set(cols[:, m - i])
    # Bottom-left: cols reverse along n axis
    for j in range(1, n):
        blkcirc = blkcirc.at[j + n - 1, :m].set(cols[n - j, :])
    # Bottom-right: rows reverse along both axes
    for i in range(1, m):
        for j in range(1, n):
            blkcirc = blkcirc.at[j + n - 1, i + m - 1].set(rows[n - j, m - i])

    # 2D FFT
    spectral = jnp.fft.fft2(blkcirc).real / ((2 * n - 1) * (2 * m - 1))

    # Positive-definite check (allow small numerical negatives).
    min_val = jnp.min(spectral)
    if min_val < -1e-10:
        raise ValueError(
            f"Could not find positive definite embedding (min spectral = {float(min_val)}). "
            f"For 100x160, s1, s2 must be < ~17.95."
        )

    # Clamp negatives to zero, then sqrt.
    return jnp.sqrt(jnp.maximum(spectral, 0.0))
```

Note: the explicit Python `for` loops in the embedding step are over compile-time constants (`m`, `n`) and trace-unrolled. For Stage 5 optimization, they can be replaced with `jnp.flip` + `jnp.roll` operations. Keep the explicit form here for clarity.

Run tests, verify they pass.

**Step 3: Commit**

```powershell
git add model_b/grf.py model_b/tests/test_grf_smoke.py
git commit -m "feat(model_b): calc_LAM for circulant-embedding GRF"
```

---

## Task 4.B.2: `model_b/grf.py::circulant_grf`

@superpowers:test-driven-development

**Files:**
- Modify: `model_b/grf.py`
- Modify: `model_b/tests/test_grf_smoke.py`

**Step 1: Append failing tests**

```python
def test_circulant_grf_returns_two_grfs():
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    key = jax.random.key(0)
    g1, g2 = jax.random.split(jax.random.normal(key, (2, 199, 319)).reshape(2, -1).at[0].get(),
                              None)
    # cleaner:
    g_all = jax.random.normal(jax.random.key(0), (2, 199, 319))
    F1, F2 = grf.circulant_grf(LAM, g_all[0], g_all[1])
    assert F1.shape == (100, 160)
    assert F2.shape == (100, 160)


def test_circulant_grf_outputs_are_real_finite():
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    g_all = jax.random.normal(jax.random.key(0), (2, 199, 319))
    F1, F2 = grf.circulant_grf(LAM, g_all[0], g_all[1])
    assert jnp.all(jnp.isfinite(F1))
    assert jnp.all(jnp.isfinite(F2))


def test_circulant_grf_aggregate_variance_matches_kernel():
    """At the zero displacement, empirical variance should match the kernel's ρ(0,0)=1."""
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    key = jax.random.key(42)
    n_samples = 500
    variances = []
    for s in range(n_samples):
        k = jax.random.fold_in(key, s)
        g = jax.random.normal(k, (2, 199, 319))
        F1, F2 = grf.circulant_grf(LAM, g[0], g[1])
        variances.append(float(F1[50, 80] ** 2))
        variances.append(float(F2[50, 80] ** 2))
    emp_var = np.mean(variances)
    # ρ(0, 0) = 1 in the kernel. Empirical variance should be ~1 within MC noise.
    assert 0.9 < emp_var < 1.1, f"empirical variance {emp_var:.3f}, expected ~1.0"
```

Run, expect AttributeError on `circulant_grf`.

**Step 2: Implement**

Append to `model_b/grf.py`:

```python
def circulant_grf(LAM, g1, g2):
    """
    Generate two independent GRF samples via one 2D FFT.

    LAM : (2n-1, 2m-1) — spectral sqrt from calc_LAM.
    g1, g2 : (2n-1, 2m-1) — iid N(0,1) arrays.

    Returns (F1, F2), each of shape (n, m), both samples from the GRF.
    """
    n_pad, m_pad = LAM.shape
    n = (n_pad + 1) // 2
    m = (m_pad + 1) // 2
    X = LAM * (g1 + 1j * g2)
    F = jnp.fft.fft2(X)
    return F[:n, :m].real, F[:n, :m].imag
```

Run tests, verify all 6 pass.

**Step 3: Commit**

```powershell
git add model_b/grf.py model_b/tests/test_grf_smoke.py
git commit -m "feat(model_b): circulant_grf yields two GRFs per FFT call"
```

---

## Task 4.C.1: `model_b/simulate.py` constants + bump/zone constructors

@superpowers:test-driven-development

**Files:**
- Create: `model_b/simulate.py`
- Create: `model_b/tests/test_simulate_b_smoke.py`

**Step 1: Failing test**

```python
"""Smoke tests for model_b/simulate.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate as sim_b


def test_simulate_b_exposes_constants():
    assert sim_b.N == 100
    assert sim_b.M == 160
    assert sim_b.NSTEP == 400
    assert sim_b.E == 10.0


def test_drift_bumps_shape_and_peak():
    """Three drift Gaussians at (uj1=80, uj2=30, uj3=130), all with ui=50."""
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    assert v1.shape == (100, 160)
    assert v2.shape == (100, 160)
    assert v3.shape == (100, 160)
    # Peaks at the documented positions
    assert int(jnp.argmax(v1)) // 160 == 50  # row
    assert int(jnp.argmax(v1)) % 160 == 80   # col
    assert int(jnp.argmax(v2)) % 160 == 30
    assert int(jnp.argmax(v3)) % 160 == 130


def test_zone_array_has_5_categories():
    """k(i,j) classifies positions into {1, 2, 3, 4, 5}."""
    k = sim_b.zone_array(si=6.0)
    assert k.shape == (100, 160)
    unique = set(int(x) for x in jnp.unique(k))
    assert unique == {1, 2, 3, 4, 5}
```

**Step 2: Implement**

Create `model_b/simulate.py`:

```python
"""
2D Gaussian-random-field diffusion simulator (Model B).

Mirrors `accum` from benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS. The simulator:
1. Builds 3 drift Gaussian bumps at fixed positions (uj1=80, uj2=30, uj3=130).
2. Builds 5-category zone array k(i,j) from bump positions.
3. Per timestep: generate one GRF via circulant_grf, accumulate, demean, check for crossing.

Note: the Fortran uses an F1/F2 caching trick to halve FFT count. We skip
this trick for simpler code (Stage 5 GPU benchmark will revisit).
"""
from functools import partial

import jax
import jax.numpy as jnp

from model_b import grf
from shared import prng

N = 100
M = 160
NSTEP = 400
E = 10.0

# Drift bump positions (mirrors Fortran accum lines 432-450)
UI = 50.0       # row center, common to all 3 bumps
UJ1 = 80.0      # cat 1 bump col
UJ2 = 30.0      # cat 3 bump col
UJ3 = 130.0     # cat 4 bump col

_I = jnp.arange(N, dtype=jnp.float64)
_J = jnp.arange(M, dtype=jnp.float64)
_I_GRID, _J_GRID = jnp.meshgrid(_I, _J, indexing="ij")  # both shape (N, M)


def drift_bumps(sis: float):
    """
    Return (v1, v2, v3), each (N, M) Gaussian bumps centered at
    (UI=50, UJ=80/30/130) with stddev `sis`.
    """
    s3 = 2.0 * sis * sis
    s4 = sis * jnp.sqrt(2.0 * jnp.pi)
    def bump(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s3) / s4
    return bump(UJ1), bump(UJ2), bump(UJ3)


def zone_array(si: float = 6.0):
    """
    Build the 5-category zone array k(i,j). Mirrors Fortran accum lines 432-450:
    - k initialized to 5.
    - cat 1: position near (UI, UJ1), above the b(i,j) > .03 threshold.
    - cat 2: position near (UI, UJ1), above b(i,j) > .0003 (broader ring).
    - cat 3: position near (UI, UJ2), above b(i,j) > .0003.
    - cat 4: position near (UI, UJ3), above b(i,j) > .0003.
    - cat 5: everything else.
    """
    s1 = 2.0 * si * si
    s2 = si * jnp.sqrt(2.0 * jnp.pi)
    def b_field(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s1) / s2

    k = jnp.full((N, M), 5, dtype=jnp.int32)
    # Cat 1 (innermost): around UJ1, b > 0.03
    b1 = b_field(UJ1)
    k = jnp.where(b1 > 0.0003, jnp.int32(2), k)
    k = jnp.where(b1 > 0.03,   jnp.int32(1), k)
    # Cat 3: around UJ2
    b2 = b_field(UJ2)
    k = jnp.where(b2 > 0.0003, jnp.int32(3), k)
    # Cat 4: around UJ3
    b3 = b_field(UJ3)
    k = jnp.where(b3 > 0.0003, jnp.int32(4), k)
    return k
```

Run tests, verify they pass.

**Step 3: Commit**

```powershell
git add model_b/simulate.py model_b/tests/test_simulate_b_smoke.py
git commit -m "feat(model_b): simulate.py constants, drift_bumps, zone_array"
```

---

## Task 4.C.2: `model_b/simulate.py::_simulate_chunk_b`

@superpowers:test-driven-development

**Files:**
- Modify: `model_b/simulate.py`
- Modify: `model_b/tests/test_simulate_b_smoke.py`

**Step 1: Failing tests**

```python
def test_simulate_chunk_b_returns_shapes():
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    k_zone = sim_b.zone_array(si=6.0)
    key = jax.random.key(0)
    rt, cat = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=2,
    )
    assert rt.shape == (2,)
    assert cat.shape == (2,)


def test_simulate_chunk_b_cat_in_valid_range():
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    k_zone = sim_b.zone_array(si=6.0)
    key = jax.random.key(0)
    _, cat = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=4,
    )
    assert jnp.all((cat >= 1) & (cat <= 5))


def test_simulate_chunk_b_deterministic():
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    k_zone = sim_b.zone_array(si=6.0)
    key = jax.random.key(42)
    rt_a, _ = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=2,
    )
    rt_b, _ = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=2,
    )
    np.testing.assert_array_equal(rt_a, rt_b)
```

**Step 2: Implement**

Append to `model_b/simulate.py`:

```python
def _simulate_chunk_b(key, ter, st, cr, crsd, av1, av2, av3,
                     LAM, v1, v2, v3, k_zone, chunk_size):
    """
    Simulate `chunk_size` trials of the 2D GRF accumulator model.

    Returns (rt, cat) each of shape (chunk_size,). cat in {1,2,3,4,5}.
    """
    ku, kt = jax.random.split(key)
    # Per-trial uniforms (mirrors Fortran gu1)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = cr + crsd * (u[:, 4] - 0.5)        # (chunk_size,) per-trial threshold
    ndt = (ter + st * (0.5 - u[:, 9])) / E   # (chunk_size,) nondecision in steps

    # Per-step keys per trial
    step_keys = jax.random.split(kt, chunk_size * NSTEP).reshape(chunk_size, NSTEP, -1)
    # Actually we want one key per (trial, step) pair; flatten and split:
    flat_keys = jax.random.split(kt, chunk_size * NSTEP)
    step_keys = flat_keys.reshape(chunk_size, NSTEP)

    n_pad, m_pad = LAM.shape

    def per_trial(trial_idx, ck_keys, ck_crr, ck_ndt):
        # ck_keys: (NSTEP,) keys for this trial
        # Scan over time
        def step(carry, sk_step_pair):
            a, jstop, pos_flat, done = carry
            sk, t = sk_step_pair
            # Generate GRF for this step
            g_pair = jax.random.normal(sk, (2, n_pad, m_pad))
            F1, _ = grf.circulant_grf(LAM, g_pair[0], g_pair[1])
            # Accumulate
            a = a + av1 * v1 + av2 * v2 + av3 * v3 + F1
            # Demean
            a = a - a.mean()
            # Check crossing
            am = a.max()
            crossed = am > ck_crr
            newly = crossed & (~done)
            jstop = jnp.where(newly, t + 1, jstop)
            pos_flat = jnp.where(newly, jnp.argmax(a), pos_flat)
            return (a, jstop, pos_flat, done | crossed), None

        init = (jnp.zeros((N, M)), NSTEP, 0, False)
        ts = jnp.arange(NSTEP)
        (a_final, jstop, pos_flat, done), _ = jax.lax.scan(
            step, init, (ck_keys, ts),
        )
        # If never crossed, pos_flat stays 0 (matches Model A's jax_port behavior).
        # Compute (row, col) from flat index
        row = pos_flat // M
        col = pos_flat % M
        cat = jnp.where(done, k_zone[row, col], jnp.int32(5))
        rt = (jstop + ck_ndt) * E
        return rt, cat

    rt_chunk, cat_chunk = jax.vmap(per_trial)(
        jnp.arange(chunk_size), step_keys, crr, ndt
    )
    return rt_chunk, cat_chunk
```

This is complex. Be patient with the JIT compile time on first run (expect 30-60s).

**If shape errors occur during test:** the most common issue is `jax.random.split(kt, N*M)` returning shape `(N*M,)` but `reshape(N, M)` failing because typed keys have outer shape `(N*M,)` already. Use `flat_keys[i*NSTEP + j]` indexing instead of reshape if needed.

**Step 3: Commit**

```powershell
git add model_b/simulate.py model_b/tests/test_simulate_b_smoke.py
git commit -m "feat(model_b): _simulate_chunk_b core algorithm with lax.scan"
```

---

## Task 4.C.3: `model_b/simulate.py::simulate_b` wrapper

@superpowers:test-driven-development

**Files:**
- Modify: `model_b/simulate.py`
- Modify: `model_b/tests/test_simulate_b_smoke.py`

**Step 1: Failing test**

```python
def test_simulate_b_full_nsim_shape():
    key = jax.random.key(0)
    rt, cat = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4,
    )
    assert rt.shape == (16,)
    assert cat.shape == (16,)


def test_simulate_b_deterministic_for_same_key():
    key = jax.random.key(11)
    rt_a, _ = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=8, chunk_size=4,
    )
    rt_b, _ = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=8, chunk_size=4,
    )
    np.testing.assert_array_equal(rt_a, rt_b)
```

**Step 2: Implement**

Append to `model_b/simulate.py`:

```python
@partial(jax.jit, static_argnums=(10, 11))
def simulate_b(key, ter, st, cr, crsd, av1, av2, av3,
               sis, sig, si, nsim, chunk_size=4):
    """
    Run `nsim` Model B trials with the given parameters.

    Parameters:
        sis : drift bump width
        sig : GRF correlation length (s1=s2=sig)
        si  : zone-array width
    Returns (rt, cat) each shape (nsim,). cat in {1,...,5}. RT in ms.

    Memory note: chunk_size=4 is the default for laptop CPU. H100 can use
    much larger (32-128).
    """
    LAM = grf.calc_LAM(s1=sig, s2=sig)
    v1, v2, v3 = drift_bumps(sis=sis)
    k_zone = zone_array(si=si)

    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)

    def run_chunk(k):
        return _simulate_chunk_b(
            k, ter, st, cr, crsd, av1, av2, av3,
            LAM, v1, v2, v3, k_zone, chunk_size,
        )

    rts, cats = jax.lax.map(run_chunk, keys)
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim]
```

Run tests. Expect first call to take 60-120s on laptop CPU (JIT compile of complex scan).

**Step 3: Commit**

```powershell
git add model_b/simulate.py model_b/tests/test_simulate_b_smoke.py
git commit -m "feat(model_b): simulate_b wraps lax.map over chunks"
```

---

## Task 4.D.1: `model_b/objective.py::condition_g2_b`

@superpowers:test-driven-development

5-category version of `condition_g2_vectorized`. Same indicator-CDF pattern, different `MC`.

**Files:**
- Create: `model_b/objective.py`
- Create: `model_b/tests/test_objective_b_smoke.py`

The structure mirrors `model_a/objective.py::condition_g2_vectorized` exactly but with `MC=5`. Reuse the same `PQQ = [.1, .2, .2, .2, .2, .1]` (still NQ=5 quantiles), `NCUT = 10` (note: Fortran uses ncut=10 for Model B vs ncut=8 for Model A).

```python
"""
G² objective for Model B (2D GRF model).

5-category version, summed across 2 conditions. Each condition uses all
three drift bumps (av1, av2, av3); conditions differ only by parameter values.
"""
import jax
import jax.numpy as jnp

from model_b import simulate as sim_b
from shared import prng

# Two conditions; each condition reads its own (av1, av2, av3) from the param vector.
# Param layout (13 total): [ter, st, cr, crsd, sis, sig, sv, av1_c1, av2_c1, av3_c1,
#                           av1_c2, av2_c2, av3_c2]
# Note: si (zone width) is fixed at 6.0; not a fit parameter.
MC = 5
NQ = 5
NCUT = 10
PQQ = jnp.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def clamp_b(params):
    """Bounds for Model B parameters. Mirrors Fortran fofs lines 158-184."""
    ter = jnp.maximum(params[0], 175.0)
    st = jnp.clip(params[1], 10.0, ter * 1.5)
    cr = jnp.maximum(params[2], 1.0)
    crsd = jnp.clip(params[3], 0.01, cr / 2.0)
    sis = params[4]
    sig = jnp.clip(params[5], 0.2, 17.0)
    sv = jnp.maximum(params[6], 0.2)
    rest = jnp.maximum(params[7:], 0.01)
    return jnp.concatenate([
        jnp.array([ter, st, cr, crsd, sis, sig, sv]),
        rest,
    ])


def condition_g2_b(rt, cat, obs_prop, obs_count, obs_quant):
    """
    G² contribution from one Model B condition.
    Same shape as model_a but with MC=5.

    rt        : (nsim,) RTs from simulate_b.
    cat       : (nsim,) categories in {1..5}.
    obs_prop  : (5,) observed proportions per category.
    obs_count : (5,) observed counts per category.
    obs_quant : (5, 5) observed RT quantiles per category. (NQ, MC)
    """
    mmn = obs_count.sum()

    def per_cat(i):
        in_cat = (cat == (i + 1))
        pxy = jnp.mean(in_cat)
        denom = jnp.maximum(in_cat.sum(), 1)
        rt_i = jnp.where(in_cat, rt, jnp.inf)
        qc = jnp.array([(rt_i <= obs_quant[j, i]).sum() / denom for j in range(NQ)])

        c_full = mmn * obs_prop[i] * PQQ[0] * jnp.log(
            obs_prop[i] * PQQ[0] / (pxy * qc[0] + 1e-5))
        for j in range(1, NQ):
            yy = jnp.maximum(qc[j] - qc[j - 1], 1e-3)
            c_full = c_full + mmn * obs_prop[i] * PQQ[j] * jnp.log(
                obs_prop[i] * PQQ[j] / (pxy * yy + 1e-5))
        c_full = c_full + mmn * obs_prop[i] * PQQ[NQ] * jnp.log(
            obs_prop[i] * PQQ[NQ] / (pxy * (1.0 - qc[NQ - 1]) + 1e-5))

        c_lumped = mmn * (obs_prop[i] + 0.002) * jnp.log(
            (obs_prop[i] + 0.002) / (pxy + 1e-12))
        return jnp.where(obs_count[i] >= NCUT, c_full, c_lumped)

    return jnp.array([per_cat(i) for i in range(MC)]).sum()
```

Write a smoke test that confirms `condition_g2_b` returns a finite positive value.

Commit:
```powershell
git add model_b/objective.py model_b/tests/test_objective_b_smoke.py
git commit -m "feat(model_b): condition_g2_b (5-category G²) + clamp_b"
```

---

## Task 4.D.2: `model_b/objective.py::fofs_b_new` with vmap over 2 conditions

Similar structure to Model A's fofs_new but with 2 conditions and Model-B-specific param layout.

```python
COND_MAP_B = [
    (7, 8, 9),    # cond 1: av1, av2, av3 from p[7], p[8], p[9]
    (10, 11, 12), # cond 2: av1, av2, av3 from p[10], p[11], p[12]
]


def fofs_b_new(params, data, key, nsim=512, chunk_size=4):
    """
    Vectorized G² objective for Model B, summed across 2 conditions.

    params : (13,) parameter vector.
    data   : dict with "prop" (2, 5), "count" (2, 5), "quant" (2, 5, 5).
    key    : JAX typed key.
    Returns scalar G².
    """
    p = clamp_b(params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0  # zone width, fixed

    # Per-condition drift vectors
    avs = jnp.stack([
        jnp.array([p[d1], p[d2], p[d3]]) for (d1, d2, d3) in COND_MAP_B
    ])  # (2, 3)

    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(2)])

    sim_vmap = jax.vmap(
        sim_b.simulate_b,
        in_axes=(0, None, None, None, None, 0, 0, 0, None, None, None, None, None),
    )
    rts, cats = sim_vmap(
        cond_keys, ter, st, cr, crsd,
        avs[:, 0], avs[:, 1], avs[:, 2],
        sis, sig, si, nsim, chunk_size,
    )

    g2_vmap = jax.vmap(condition_g2_b, in_axes=(0, 0, 0, 0, 0))
    g2_per_cond = g2_vmap(rts, cats, data["prop"], data["count"], data["quant"])
    return g2_per_cond.sum()
```

Smoke test: confirms `fofs_b_new` returns finite positive scalar.

Commit:
```powershell
git add model_b/objective.py model_b/tests/test_objective_b_smoke.py
git commit -m "feat(model_b): fofs_b_new vmaps simulate_b over 2 conditions"
```

---

## Task 4.D.3: `model_b/fit.py::fit_simplex_b`

Simplex driver for Model B. Same structure as Model A's `fit_simplex` would be — scipy NM wrapper.

```python
"""Fit drivers for Model B. Simplex only (gradient issue from Stage 3)."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from model_b import objective as obj_b


@dataclass
class FitResult:
    params: jnp.ndarray
    loss: float
    n_iters: int
    converged: bool


def fit_simplex_b(data, key, x0, nsim=256, maxiter=2000, tol=1e-7, chunk_size=4):
    """Scipy Nelder-Mead simplex for Model B."""
    from scipy.optimize import minimize

    def loss_numpy(p_np):
        p = jnp.asarray(p_np)
        val = obj_b.fofs_b_new(p, data, key, nsim=nsim, chunk_size=chunk_size)
        return float(val)

    res = minimize(
        loss_numpy, np.asarray(x0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": tol, "fatol": tol},
    )
    return FitResult(
        params=jnp.asarray(res.x),
        loss=float(res.fun),
        n_iters=int(res.nit),
        converged=bool(res.success),
    )
```

Smoke test:
- Generate synthetic data with `simulate_b` at known params.
- Fit with `fit_simplex_b` from a perturbed starting point.
- Assert recovery within ±20% on active params.
- Mark `@pytest.mark.slow` (will take minutes).

Commit:
```powershell
git add model_b/fit.py model_b/tests/test_fit_b_smoke.py
git commit -m "feat(model_b): fit_simplex_b via scipy Nelder-Mead"
```

---

## Task 4.E.1: Update smoke runner

Add Model B test files to smoke runner. Use only the FAST tests (skip the @slow simplex recovery).

```powershell
.\.venv\Scripts\python.exe -m pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py model_a/tests/test_simulate_new_smoke.py model_a/tests/test_simulate_parity.py model_a/tests/test_objective_smoke.py model_a/tests/test_objective_parity.py model_b/tests/test_grf_smoke.py model_b/tests/test_simulate_b_smoke.py model_b/tests/test_objective_b_smoke.py -v
```

Run smoke. Target: ~65 tests passing. Wall clock 5-10 min first run.

Commit:
```powershell
git add scripts/smoke.ps1 scripts/smoke.sh
git commit -m "chore(smoke): include Model B tests in Stage 4 gate"
```

---

## Task 4.F.1: Stage 4 completion gate

Verify ALL:
1. `model_b/grf.py`, `model_b/simulate.py`, `model_b/objective.py`, `model_b/fit.py` exist.
2. GRF empirical variance test passes (~1.0).
3. `simulate_b` runs end-to-end at small `nsim` on laptop CPU.
4. `fofs_b_new` returns finite scalar on real `twod3datanew` data (subject 0).
5. `fit_simplex_b` recovers synthetic params within ±20% (slow mark, run separately).
6. `model_a/` files UNTOUCHED.
7. `scripts/smoke.ps1` shows ~65 passed.

Dispatch final code reviewer for the whole Stage 4 diff (`af8035d..HEAD`).

---

## Out of scope

- F1/F2 FFT caching trick (Stage 5 optimization).
- L-BFGS for Model B (gradient issue — Stage 3.5 first).
- Parity against archive's recorded Fortran outputs as a hard gate. This is genuinely hard (matching exact Fortran params + seed + nsim). Defer to a later validation pass; sketch documented as a Stage 4.G follow-up.
- GPU benchmarks (Stage 5).
- Multi-subject hierarchical fits.
- Bayesian / NumPyro inference.

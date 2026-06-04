# Model A — Stage 0+1 Implementation Plan: Setup and Validation Infrastructure

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the existing JAX port (`reference/twod24_jax.py`) running inside the repo as `model_a/jax_port.py`, with data parsing, deterministic PRNG helpers, and a validation infrastructure ready to compare JAX aggregate statistics against the Fortran reference. End state: smoke tests pass on laptop CPU and we can run `simulate()` and `fofs()` end-to-end against real `twod24data`.

**Architecture:** Layered Python: `shared/` provides data IO, PRNG helpers, and aggregate-comparison utilities; `model_a/` houses the JAX simulator and objective. Tests are pytest, test-first. No GPU code in this plan — Stage 2 onward handles speedups. No fit driver in this plan — Stage 3 handles optimizers.

**Tech Stack:** Python 3.11+, JAX (CPU), pytest, numpy, scipy. JAX runs on Windows CPU here; same source runs on workstation/H100 later. No CUDA install needed for Stage 0+1.

**Required sub-skills referenced:**
- @superpowers:test-driven-development for every implementation task
- @superpowers:verification-before-completion before marking any task done
- @superpowers:executing-plans for stepwise execution

**Reference files (do not modify):**
- `reference/gpgsq5deg3twod24.f` — Fortran source of truth for Model A
- `reference/twod24_jax.py` — first-pass JAX port, will be copied to `model_a/jax_port.py`
- `data/twod24data` — observed RT/proportion data (64 non-empty lines, 27 fields each)

---

## Pre-flight checks

Run these before Task 0.1:

```bash
git -C C:\Users\gowri\ratcliff status
```
Expected: clean working tree on `master`, no untracked files.

```bash
python --version
```
Expected: Python 3.11 or newer. If not, install before continuing.

---

## Task 0.1: Add `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

**Step 1: Write the file**

```toml
[project]
name = "ratcliff"
version = "0.1.0"
description = "JAX port of Ratcliff spatially-extended diffusion models"
requires-python = ">=3.11"
dependencies = [
    "jax>=0.4.28",
    "jaxlib>=0.4.28",
    "numpy>=1.26",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
]
fit = [
    "jaxopt>=0.8",
    "optax>=0.2",
]
viz = [
    "matplotlib>=3.8",
    "jupyter>=1.0",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["shared*", "model_a*", "model_b*"]

[tool.pytest.ini_options]
testpaths = ["shared/tests", "model_a/tests", "model_b/tests"]
addopts = "-v --tb=short"
```

**Step 2: Install in editable mode**

Run:
```powershell
cd C:\Users\gowri\ratcliff
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Expected: clean install, no errors. JAX CPU build pulls in.

**Step 3: Verify JAX imports**

Run:
```powershell
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Expected: prints a version number ≥ 0.4.28 and `[CpuDevice(id=0)]`.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: pyproject.toml with JAX + pytest deps"
```

---

## Task 0.2: Create package skeleton

**Files:**
- Create: `shared/__init__.py` (empty)
- Create: `shared/tests/__init__.py` (empty)
- Create: `model_a/__init__.py` (empty)
- Create: `model_a/tests/__init__.py` (empty)
- Create: `model_b/__init__.py` (empty)
- Create: `model_b/tests/__init__.py` (empty)

**Step 1: Create all six empty files**

Each should be a zero-byte file. Use `Write` tool with empty content.

**Step 2: Verify directory tree**

Run:
```powershell
Get-ChildItem 'C:\Users\gowri\ratcliff' -Recurse -Directory | Select-Object FullName
```

Expected: includes `shared/`, `shared/tests/`, `model_a/`, `model_a/tests/`, `model_b/`, `model_b/tests/`.

**Step 3: Verify pytest discovers nothing yet**

Run:
```powershell
pytest
```

Expected: exits 0 (or 5 "no tests collected"). No errors.

**Step 4: Commit**

```bash
git add shared/__init__.py shared/tests/__init__.py model_a/__init__.py model_a/tests/__init__.py model_b/__init__.py model_b/tests/__init__.py
git commit -m "chore: package skeleton for shared, model_a, model_b"
```

---

## Task 0.3: Copy `twod24_jax.py` to `model_a/jax_port.py`

**Files:**
- Create: `model_a/jax_port.py` (verbatim copy of `reference/twod24_jax.py` with a one-line provenance header)

**Step 1: Copy with header**

Open `reference/twod24_jax.py`, prepend a single-line comment to the top, write to `model_a/jax_port.py`:

```python
# Imported verbatim from reference/twod24_jax.py. Reference copy is the immutable original.
"""
twod24_jax.py
=============
... (rest of existing docstring) ...
"""
```

The rest of the file is byte-for-byte identical to the reference. No logic changes in this task.

**Step 2: Verify it imports**

Run:
```powershell
python -c "from model_a import jax_port; print(jax_port.N, jax_port.NSTEP, jax_port.MC)"
```

Expected: `72 400 3`.

**Step 3: Commit**

```bash
git add model_a/jax_port.py
git commit -m "feat(model_a): import twod24_jax.py as model_a/jax_port.py"
```

---

## Task 1.1: `shared/prng.py` — deterministic key helpers

@superpowers:test-driven-development

**Files:**
- Test: `shared/tests/test_prng.py`
- Create: `shared/prng.py`

**Step 1: Write the failing tests**

Create `shared/tests/test_prng.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from shared import prng


def test_root_key_is_deterministic_for_same_seed():
    k1 = prng.root_key(42)
    k2 = prng.root_key(42)
    assert jnp.array_equal(k1, k2)


def test_root_key_differs_for_different_seeds():
    k1 = prng.root_key(42)
    k2 = prng.root_key(43)
    assert not jnp.array_equal(k1, k2)


def test_split_for_condition_is_deterministic():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=2)
    b = prng.split_for_condition(k, condition_idx=2)
    assert jnp.array_equal(a, b)


def test_split_for_condition_differs_across_conditions():
    k = prng.root_key(0)
    a = prng.split_for_condition(k, condition_idx=0)
    b = prng.split_for_condition(k, condition_idx=1)
    assert not jnp.array_equal(a, b)


def test_trial_keys_returns_n_distinct_keys():
    k = prng.root_key(7)
    keys = prng.trial_keys(k, n=100)
    assert keys.shape == (100, 2)  # JAX keys are (2,) uint32 pairs
    # All keys should be distinct
    flat = keys.reshape(100, -1)
    assert len({tuple(row.tolist()) for row in flat}) == 100
```

**Step 2: Run test to verify it fails**

Run:
```powershell
pytest shared/tests/test_prng.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'shared.prng'`.

**Step 3: Write minimal implementation**

Create `shared/prng.py`:

```python
"""Deterministic PRNG helpers for reproducible Monte Carlo runs."""
import jax


def root_key(seed: int):
    """Top-level JAX PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def split_for_condition(key, condition_idx: int):
    """Derive a condition-specific subkey deterministically from a root key."""
    return jax.random.fold_in(key, condition_idx)


def trial_keys(key, n: int):
    """Return an (n, 2) array of distinct trial-level subkeys."""
    return jax.random.split(key, n)
```

**Step 4: Run test to verify it passes**

Run:
```powershell
pytest shared/tests/test_prng.py -v
```

Expected: all 5 tests pass.

**Step 5: Commit**

```bash
git add shared/prng.py shared/tests/test_prng.py
git commit -m "feat(shared): deterministic PRNG key helpers"
```

---

## Task 1.2: `shared/data_io.py` — parse `twod24data`

@superpowers:test-driven-development

The Fortran reads `twod24data` as: per line, 3 categories × (acc, count, 5 quantiles, x1, x2) = 27 fields. `nsim=16` outer fitting loops × `nc=4` conditions = 64 data lines.

**Files:**
- Test: `shared/tests/test_data_io.py`
- Create: `shared/data_io.py`

**Step 1: Write the failing tests**

Create `shared/tests/test_data_io.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from shared import data_io

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def test_data_file_exists():
    assert DATA_PATH.exists(), f"missing {DATA_PATH}"


def test_load_twod24data_returns_expected_shape():
    d = data_io.load_twod24data(DATA_PATH)
    # 16 subjects × 4 conditions = 64 condition-records
    assert d["prop"].shape == (64, 3)
    assert d["count"].shape == (64, 3)
    assert d["quant"].shape == (64, 5, 3)


def test_proportions_sum_close_to_one_per_line():
    d = data_io.load_twod24data(DATA_PATH)
    sums = d["prop"].sum(axis=1)
    # Some lines have empty categories — tolerate ±0.05
    assert np.all(np.abs(sums - 1.0) < 0.05), f"max deviation {np.abs(sums - 1.0).max()}"


def test_counts_are_nonneg_integers():
    d = data_io.load_twod24data(DATA_PATH)
    assert d["count"].dtype.kind in ("i", "u")
    assert np.all(d["count"] >= 0)


def test_quantiles_are_monotone_per_category():
    d = data_io.load_twod24data(DATA_PATH)
    # For categories with nonzero count, quantiles must be non-decreasing along axis 1
    for ci in range(64):
        for cat in range(3):
            if d["count"][ci, cat] > 0:
                q = d["quant"][ci, :, cat]
                assert np.all(np.diff(q) >= 0), f"line {ci} cat {cat} not monotone: {q}"


def test_grouped_by_subject_returns_4_conditions_per_subject():
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    assert g["prop"].shape == (16, 4, 3)
    assert g["count"].shape == (16, 4, 3)
    assert g["quant"].shape == (16, 4, 5, 3)
```

**Step 2: Run test to verify it fails**

Run:
```powershell
pytest shared/tests/test_data_io.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'shared.data_io'`.

**Step 3: Write implementation**

Create `shared/data_io.py`:

```python
"""Parsers for Ratcliff observed RT/proportion data files."""
from pathlib import Path

import numpy as np


def load_twod24data(path):
    """
    Parse `twod24data` as 64 lines × 27 fields = 3 categories × (prop, count, q1..q5, x1, x2).

    Returns dict with:
      prop  : (64, 3) float — observed response proportion per category
      count : (64, 3) int   — observed trial count per category
      quant : (64, 5, 3) float — 5 RT quantiles per category
    """
    lines = [
        ln for ln in Path(path).read_text().splitlines() if ln.strip()
    ]
    n_lines = len(lines)

    prop = np.zeros((n_lines, 3), dtype=np.float64)
    count = np.zeros((n_lines, 3), dtype=np.int64)
    quant = np.zeros((n_lines, 5, 3), dtype=np.float64)

    for i, ln in enumerate(lines):
        fields = ln.split()
        if len(fields) != 27:
            raise ValueError(
                f"line {i}: expected 27 fields, got {len(fields)} — line: {ln!r}"
            )
        for cat in range(3):
            base = cat * 9
            prop[i, cat] = float(fields[base])
            count[i, cat] = int(float(fields[base + 1]))
            for q in range(5):
                quant[i, q, cat] = float(fields[base + 2 + q])
            # fields base+7 (x1) and base+8 (x2) ignored — RT extremes, not used by fofs

    return {"prop": prop, "count": count, "quant": quant}


def group_by_subject(data, conditions_per_subject: int = 4):
    """Reshape (64, ...) flat arrays into (n_subjects, conditions_per_subject, ...)."""
    n_records = data["prop"].shape[0]
    if n_records % conditions_per_subject != 0:
        raise ValueError(
            f"{n_records} records is not divisible by {conditions_per_subject}"
        )
    n_subjects = n_records // conditions_per_subject
    return {
        "prop": data["prop"].reshape(n_subjects, conditions_per_subject, 3),
        "count": data["count"].reshape(n_subjects, conditions_per_subject, 3),
        "quant": data["quant"].reshape(n_subjects, conditions_per_subject, 5, 3),
    }
```

**Step 4: Run test to verify it passes**

Run:
```powershell
pytest shared/tests/test_data_io.py -v
```

Expected: all 6 tests pass.

**Note:** if `test_proportions_sum_close_to_one_per_line` fails for a few lines because the data has zero-count "empty" third categories where the proportions sum to less than 1.0, the test already allows 0.05 slack. If that's insufficient, raise the tolerance to 0.10 — but **do not** silently mutate the data. Inspect the failing lines first.

**Step 5: Commit**

```bash
git add shared/data_io.py shared/tests/test_data_io.py
git commit -m "feat(shared): parse twod24data into prop/count/quant arrays"
```

---

## Task 1.3: `shared/validation.py` — aggregate comparison utilities

@superpowers:test-driven-development

We need helpers to compare two sets of Monte Carlo aggregates within tolerance — this is what every validation gate will use later.

**Files:**
- Test: `shared/tests/test_validation.py`
- Create: `shared/validation.py`

**Step 1: Write the failing tests**

Create `shared/tests/test_validation.py`:

```python
import numpy as np
import pytest

from shared import validation


def test_proportions_match_when_identical():
    a = np.array([0.3, 0.5, 0.2])
    ok, report = validation.proportions_match(a, a, abs_tol=0.005)
    assert ok
    assert report["max_abs_diff"] == 0.0


def test_proportions_match_within_tolerance():
    a = np.array([0.3, 0.5, 0.2])
    b = np.array([0.302, 0.499, 0.199])
    ok, _ = validation.proportions_match(a, b, abs_tol=0.005)
    assert ok


def test_proportions_match_fails_outside_tolerance():
    a = np.array([0.3, 0.5, 0.2])
    b = np.array([0.31, 0.49, 0.20])
    ok, report = validation.proportions_match(a, b, abs_tol=0.005)
    assert not ok
    assert report["max_abs_diff"] > 0.005


def test_quantiles_match_within_relative_tolerance():
    a = np.array([300.0, 400.0, 500.0])
    b = np.array([301.0, 401.0, 501.0])
    ok, _ = validation.quantiles_match(a, b, rel_tol=0.01)
    assert ok


def test_quantiles_match_fails_outside_relative_tolerance():
    a = np.array([300.0, 400.0, 500.0])
    b = np.array([330.0, 400.0, 500.0])  # 10% off on first
    ok, _ = validation.quantiles_match(a, b, rel_tol=0.01)
    assert not ok


def test_aggregate_match_combines_both():
    obs_prop = np.array([0.3, 0.5, 0.2])
    obs_quant = np.array([[300.0, 400.0], [310.0, 410.0]])  # (n_quantiles=2, n_cat=2)... ignored shape, just illustrative
    sim_prop = np.array([0.302, 0.499, 0.199])
    sim_quant = obs_quant.copy()
    result = validation.aggregate_match(
        obs_prop, sim_prop, obs_quant, sim_quant, prop_abs_tol=0.005, quant_rel_tol=0.01
    )
    assert result["passed"]
    assert result["prop_passed"]
    assert result["quant_passed"]
```

**Step 2: Run test to verify it fails**

Run:
```powershell
pytest shared/tests/test_validation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'shared.validation'`.

**Step 3: Write implementation**

Create `shared/validation.py`:

```python
"""Aggregate-statistical comparison between Fortran reference and JAX port."""
import numpy as np


def proportions_match(a, b, abs_tol: float = 0.005):
    """
    Compare response proportions elementwise within an absolute tolerance.
    Returns (ok, report_dict).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = np.abs(a - b)
    return bool(np.all(diffs <= abs_tol)), {
        "max_abs_diff": float(diffs.max()),
        "tol": abs_tol,
    }


def quantiles_match(a, b, rel_tol: float = 0.01):
    """
    Compare RT quantiles within a relative tolerance |a-b| / max(|a|, |b|, eps).
    Returns (ok, report_dict).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-9)
    rels = np.abs(a - b) / denom
    return bool(np.all(rels <= rel_tol)), {
        "max_rel_diff": float(rels.max()),
        "tol": rel_tol,
    }


def aggregate_match(
    obs_prop, sim_prop, obs_quant, sim_quant,
    prop_abs_tol: float = 0.005, quant_rel_tol: float = 0.01,
):
    """Combined gate: proportions AND quantiles within their respective tolerances."""
    prop_ok, prop_report = proportions_match(obs_prop, sim_prop, prop_abs_tol)
    quant_ok, quant_report = quantiles_match(obs_quant, sim_quant, quant_rel_tol)
    return {
        "passed": prop_ok and quant_ok,
        "prop_passed": prop_ok,
        "quant_passed": quant_ok,
        "prop": prop_report,
        "quant": quant_report,
    }
```

**Step 4: Run test to verify it passes**

Run:
```powershell
pytest shared/tests/test_validation.py -v
```

Expected: all 6 tests pass.

**Step 5: Commit**

```bash
git add shared/validation.py shared/tests/test_validation.py
git commit -m "feat(shared): aggregate proportion/quantile comparison helpers"
```

---

## Task 1.4: Smoke test — `model_a.jax_port.simulate` runs end-to-end

@superpowers:test-driven-development

Goal: prove the existing JAX port at least *runs* on real data with `nsim=64` (small enough to be fast on laptop CPU). Not validating Fortran-parity yet — that needs Fortran running, which is a later plan once the user confirms workstation Fortran builds.

**Files:**
- Test: `model_a/tests/test_simulate_smoke.py`

**Step 1: Write the failing test**

Create `model_a/tests/test_simulate_smoke.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from model_a import jax_port
from shared import prng


def test_simulate_returns_finite_rt_and_categories():
    key = prng.root_key(0)
    rt, cat = jax_port.simulate(
        key,
        ter=200.0,
        st=50.0,
        cr=50.0,
        crsd=10.0,
        si=4.0,
        sig=5.0,
        av=20.0,
        sv=0.7,
        nsim=64,
    )
    assert rt.shape == (64,)
    assert cat.shape == (64,)
    assert jnp.all(jnp.isfinite(rt))
    # Categories are {1, 2, 3} per the Fortran convention
    assert jnp.all((cat >= 1) & (cat <= 3))


def test_simulate_is_deterministic_for_same_key():
    key = prng.root_key(42)
    a_rt, a_cat = jax_port.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32
    )
    b_rt, b_cat = jax_port.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32
    )
    assert jnp.array_equal(a_rt, b_rt)
    assert jnp.array_equal(a_cat, b_cat)


def test_simulate_differs_for_different_keys():
    rt_a, _ = jax_port.simulate(
        prng.root_key(0),
        ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32,
    )
    rt_b, _ = jax_port.simulate(
        prng.root_key(1),
        ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32,
    )
    assert not jnp.array_equal(rt_a, rt_b)
```

**Step 2: Run tests**

Run:
```powershell
pytest model_a/tests/test_simulate_smoke.py -v
```

Expected: PASS on all 3. If `simulate` raises (e.g. dtype mismatch, JAX install issue), debug **without modifying `jax_port.py`** — the file is the user's authored work and we touch it only in Stage 2.

Common debugging tips if a test fails here:
- `ImportError: No module named jax`: re-run `pip install -e ".[dev]"` from Task 0.1.
- `XlaRuntimeError`: try `jax.config.update("jax_platform_name", "cpu")` at the top of the test (laptop CPU only for Stage 1).
- Shape errors: print `rt.shape` and `cat.shape` — `simulate`'s output may need a small adapter; if so, write it in `model_a/__init__.py`, not in `jax_port.py`.

**Step 3: Commit**

```bash
git add model_a/tests/test_simulate_smoke.py
git commit -m "test(model_a): smoke tests for jax_port.simulate determinism and shape"
```

---

## Task 1.5: Smoke test — `fofs` returns finite scalar on real `twod24data`

@superpowers:test-driven-development

**Files:**
- Test: `model_a/tests/test_fofs_smoke.py`

**Step 1: Write the failing test**

Create `model_a/tests/test_fofs_smoke.py`:

```python
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from model_a import jax_port
from shared import data_io, prng

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def _first_subject_4_conditions():
    """fofs in jax_port expects {"prop", "count", "quant"} for 4 conditions."""
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    return {
        "prop": jnp.asarray(g["prop"][0]),    # (4, 3)
        "count": jnp.asarray(g["count"][0]),  # (4, 3)
        "quant": jnp.asarray(g["quant"][0]),  # (4, 5, 3)
    }


def test_fofs_returns_finite_scalar():
    data = _first_subject_4_conditions()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(0)
    val = jax_port.fofs(params, data, key, nsim=512)
    val = float(val)
    assert np.isfinite(val), f"fofs returned non-finite: {val}"
    assert val > 0, f"fofs is a G^2 statistic and should be positive: {val}"


def test_fofs_is_deterministic_for_same_key():
    data = _first_subject_4_conditions()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(7)
    a = float(jax_port.fofs(params, data, key, nsim=256))
    b = float(jax_port.fofs(params, data, key, nsim=256))
    assert a == b


def test_fofs_changes_when_params_change():
    data = _first_subject_4_conditions()
    key = prng.root_key(0)
    params_a = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    params_b = jnp.array([250., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])  # ter different
    a = float(jax_port.fofs(params_a, data, key, nsim=512))
    b = float(jax_port.fofs(params_b, data, key, nsim=512))
    assert a != b
```

**Step 2: Run tests**

Run:
```powershell
pytest model_a/tests/test_fofs_smoke.py -v
```

Expected: PASS. **If anything fails**, the root cause is almost certainly the data parser (Task 1.2) or a JAX-version-specific issue, not `jax_port.py`. Debug there.

Performance note: each test calls `fofs` on a fresh JIT trace; first call may take 5–20 s on CPU. Subsequent calls reuse the cache.

**Step 3: Commit**

```bash
git add model_a/tests/test_fofs_smoke.py
git commit -m "test(model_a): smoke tests for fofs on real twod24data"
```

---

## Task 1.6: Add `make smoke` equivalent — one command runs all Stage-1 tests

**Files:**
- Create: `scripts/smoke.ps1`
- Create: `scripts/smoke.sh` (for the workstation later)

**Step 1: PowerShell script**

Create `scripts/smoke.ps1`:

```powershell
# Smoke test runner. Stage 1 gate.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py -v
```

Create `scripts/smoke.sh`:

```bash
#!/usr/bin/env bash
# Smoke test runner. Stage 1 gate.
set -euo pipefail
cd "$(dirname "$0")/.."
pytest shared/tests model_a/tests/test_simulate_smoke.py model_a/tests/test_fofs_smoke.py -v
```

**Step 2: Run the PowerShell version**

Run:
```powershell
.\scripts\smoke.ps1
```

Expected: All tests from Tasks 1.1, 1.2, 1.3, 1.4, 1.5 pass. Final summary: `XX passed`.

**Step 3: Commit**

```bash
git add scripts/smoke.ps1 scripts/smoke.sh
git commit -m "chore: smoke.ps1 / smoke.sh as the Stage-1 gate runner"
```

---

## Task 1.7: Update `README.md` with Stage 1 status

**Files:**
- Modify: `README.md`

**Step 1: Append to README**

Add after the existing "## Status" section:

```markdown
## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
.\scripts\smoke.ps1
```

## Stage status

- [x] Stage 0 — repo init, env setup
- [x] Stage 1 — validation infrastructure, smoke tests pass on laptop CPU
- [ ] Stage 2 — single-GEMM + cumsum rewrite of `model_a/simulate.py`
- [ ] Stage 3 — `vmap` `fofs` over conditions, L-BFGS optimizer, simplex fallback
- [ ] Stage 4 — Model B GRF + accumulator port
- [ ] Stage 5 — benchmark report across laptop / 64-core / H100
- [ ] Stage 6 (optional) — Triton kernel for per-step scan
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README quick start + stage checklist"
```

---

## Stage 1 completion gate

@superpowers:verification-before-completion

Before declaring Stage 0+1 complete, verify ALL of:

1. `.\scripts\smoke.ps1` exits 0 with all tests passing.
2. `git log --oneline` shows ~8 commits since the design-doc commit, each scoped to one task.
3. `git status` is clean.
4. `pytest --co -q` lists all the tests we wrote (no skipped imports).
5. Reference files (`reference/gpgsq5deg3twod24.f`, `reference/twod24_jax.py`) are unmodified — `git log -- reference/` shows only the initial commit.

If any of those fail, fix before claiming completion.

---

## What's deliberately out of this plan

These belong to later plans, do not creep them in:

- The `simulate.py` rewrite (single-GEMM + cumsum + first-crossing). → Plan 2 / Stage 2.
- Any optimizer (`fit.py`, L-BFGS, simplex). → Plan 3 / Stage 3.
- Strict Fortran-output comparison test (requires running Fortran somewhere). → Plan 2 once user confirms workstation Fortran is available, OR earlier if they share `twod24data` Fortran outputs.
- Anything Model B. → Plan 4 / Stage 4.
- Benchmarking. → Plan 5 / Stage 5.
- GPU testing. CPU only here.

If you find yourself wanting to add any of the above, stop and ask whether to expand the plan or hold for the next one.

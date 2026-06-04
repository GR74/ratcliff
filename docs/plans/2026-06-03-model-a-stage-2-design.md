# Model A — Stage 2 Design: Single-GEMM + Cumsum Rewrite

**Date:** 2026-06-03
**Status:** Approved (autonomous mode, user-pre-approved scope)
**Scope:** Rewrite `model_a/simulate.py` as a single-GEMM + cumsum + first-crossing-argmax program. Bundle three forward-looking migrations from Stage 1 reviews. Validate via parity against the existing `jax_port.simulate`. Fortran-comparison gate is deferred to Stage 2.5 (separate effort, not blocking).

---

## 1. Background

Stage 1 delivered the validation infrastructure and proved the user's existing JAX port (`model_a/jax_port.py`) works end-to-end on real `twod24data` (G² = 722.14, bit-deterministic). That port uses `jax.lax.scan` over timesteps with early termination, which is correct but leaves significant performance on the table:

- The scan carries state through 400 timesteps per trial → 400 serial dependency chain → poor GPU utilization.
- Each step does a `L @ z` matrix-vector product (small GEMM) inside the scan body.
- Early termination saves a few percent of FLOPs but the scan overhead dominates.

The design doc for the overall project (§4.2) identified that the accumulator path is exactly the cumulative sum of demeaned per-step increments. So the whole simulator collapses into three GPU-friendly ops: (1) one correlated-noise matmul, (2) a cumulative sum over time, (3) argmax + threshold reduction. This is what Stage 2 implements.

## 2. Goals

1. New file `model_a/simulate.py` with a single-GEMM + cumsum + first-crossing-argmax implementation.
2. Statistical parity against `jax_port.simulate` on real `twod24data` parameter sets.
3. Migrate `shared/prng.py` to JAX's typed-key API (`jax.random.key` instead of `PRNGKey`).
4. New simulator wires through `shared.prng` exclusively (no direct `jax.random.*` calls).
5. Centralize `jax.config.update("jax_enable_x64", True)` into a top-level `conftest.py`.
6. CPU speedup of ≥ 5× over `jax_port.simulate` at `nsim=4000` after warmup.
7. All Stage 1 tests still pass (zero regressions).

## 3. Non-goals

- Not rewriting `fofs` (Stage 3 handles vmap-over-conditions + L-BFGS).
- Not rewriting `condition_g2` (Stage 3).
- Not touching `model_a/jax_port.py` — it remains the JAX-side oracle.
- Not running Fortran (Stage 2.5).
- Not adding Model B (Stage 4).
- Not adding GPU benchmark harness (Stage 5).
- Not adding a `LICENSE` (out-of-band repo hygiene).

## 4. Architecture

### 4.1 Files

```
ratcliff/
├── conftest.py              # NEW — top-level x64 config + JAX cache dir
├── model_a/
│   ├── jax_port.py          # untouched (oracle)
│   ├── simulate.py          # NEW — the fast rewrite
│   └── tests/
│       ├── test_simulate_smoke.py       # existing, untouched
│       ├── test_fofs_smoke.py           # existing, untouched
│       ├── test_simulate_new_smoke.py   # NEW — basic smoke for new simulator
│       └── test_simulate_parity.py      # NEW — parity vs jax_port.simulate
├── shared/
│   ├── prng.py              # MODIFIED — typed-key API
│   └── tests/test_prng.py   # MODIFIED — assertion updated for typed keys
└── scripts/
    ├── smoke.ps1            # MODIFIED — include new test files
    └── smoke.sh             # MODIFIED — same
```

### 4.2 Algorithm: `model_a/simulate.py`

```python
@partial(jax.jit, static_argnums=(8, 9))
def simulate(key, ter, st, cr, crsd, si, sig, av, nsim, chunk_size=256):
    L = chol_factor(sig)              # (N, N)
    v = drift_profile(av, si)          # (N,)
    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)

    def run_chunk(k):
        return _simulate_chunk(k, ter, st, cr, crsd, L, v, chunk_size)

    rts, cats = jax.lax.map(run_chunk, keys)  # (n_chunks, chunk_size) each
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim]
```

The inner `_simulate_chunk`:

```python
def _simulate_chunk(key, ter, st, cr, crsd, L, v, chunk_size):
    ku, kz = jax.random.split(key)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = cr + crsd * (u[:, 4] - 0.5)               # (chunk_size,)
    ndt = (ter + st * (0.5 - u[:, 9])) / E          # (chunk_size,)

    # Single big GEMM: all noise for chunk × NSTEP × N
    z = jax.random.normal(kz, (chunk_size, NSTEP, N))   # iid normals
    noise = z @ L.T                                      # (chunk_size, NSTEP, N)

    # Build demeaned increments
    incr = v[None, None, :] + 5.0 * noise               # (chunk_size, NSTEP, N)
    incr = incr - incr.mean(axis=-1, keepdims=True)     # demean per step

    # Accumulator paths
    a = jnp.cumsum(incr, axis=1)                        # (chunk_size, NSTEP, N)

    # First crossing
    max_per_step = a.max(axis=-1)                       # (chunk_size, NSTEP)
    crossed = max_per_step > crr[:, None]               # (chunk_size, NSTEP)
    any_crossed = crossed.any(axis=1)                   # (chunk_size,)
    jstop = jnp.where(any_crossed, jnp.argmax(crossed, axis=1) + 1, NSTEP)

    # Position at crossing
    pos = jnp.argmax(a[jnp.arange(chunk_size), jstop - 1, :], axis=-1) + 1

    # RT and category
    rt = (jstop + ndt) * E
    cat = jnp.where((pos > IPA) & (pos < IPB), 1,
          jnp.where((pos <= IPC) | (pos >= IPD), 3, 2))
    return rt, cat
```

### 4.3 Memory budget

Per chunk at `chunk_size=256`: `256 × 400 × 72 × 8 bytes = 59 MB` (one for `z`, one for `noise`, one for `incr`, one for `a`). Peak working set ≈ 240 MB. Fits any laptop.

For `nsim=4000`, runs as 16 sequential chunks via `jax.lax.map`. No memory blowup.

For GPU: chunk size can scale up to e.g. 4096 (~950 MB per buffer, ~4 GB peak), well within H100 80 GB. Tunable via the static argument.

### 4.4 Chunking via `jax.lax.map` vs Python loop

`jax.lax.map` is preferred because:
- Stays inside the JIT region (no host-device sync per chunk).
- Single XLA program; simpler perf model.

Python loop over chunks is the fallback if `lax.map` has compilation issues with large chunk counts.

### 4.5 Why `sv` is dropped from the signature

`jax_port.py` keeps `sv` in the simulator signature for backward compatibility with the Fortran `accum` argument list, but `SV_ACTIVE = False` makes it inert. The new `simulate.py` omits the parameter entirely — clean signature, no dead arg. Callers that need across-trial drift variability will explicitly opt in via a future `SV_ACTIVE = True` rewrite (Stage 3 if Roger confirms; otherwise never).

## 5. Migration A — typed-key API in `shared/prng.py`

Replace `jax.random.PRNGKey(seed)` with `jax.random.key(seed)`. The returned object is opaque, not a `(2,) uint32` array.

```python
# Old
def root_key(seed: int):
    return jax.random.PRNGKey(seed)

# New
def root_key(seed: int):
    return jax.random.key(seed)
```

`split_for_condition` and `trial_keys` work identically with typed keys — they delegate to `fold_in` and `split`, both of which accept both APIs.

**Test update needed:** `test_trial_keys_returns_n_distinct_keys` currently asserts `keys.shape == (n, 2)`. With typed keys, `keys.shape == (n,)` (the underlying buffer is hidden). Use `jax.random.key_data(keys).shape == (n, 2)` to access the raw representation when needed for the distinctness check.

## 6. Migration B — `simulate.py` uses `shared.prng` exclusively

The new simulator:
- Takes `key` as input (a typed JAX key).
- Uses `prng.trial_keys(key, n_chunks)` to derive chunk-level keys.
- Inside the chunk, uses `jax.random.split` directly (one level deep is fine; this is the leaf of the key tree).

This closes the dead-code loop the Stage 1 reviewer flagged: `split_for_condition` and `trial_keys` now have one production caller.

## 7. Migration C — centralize x64 in `conftest.py`

New file at repo root:

```python
# conftest.py
"""Top-level pytest config: enable JAX x64 mode before any test imports."""
import jax

jax.config.update("jax_enable_x64", True)
```

This runs at pytest collection time, before any `from model_a import jax_port` statement. The duplicate `jax.config.update(...)` inside `jax_port.py` line 35 stays as a no-op (we're not allowed to touch the oracle). It's idempotent and safe — the second call is a no-op when the flag is already set.

For non-pytest direct imports (e.g., a script or notebook does `from model_a import simulate`), `simulate.py` does NOT call `jax.config.update`. Instead, the module-level docstring documents that callers must enable x64 before importing if they need it. This is acceptable because:
- All current callers (tests, fofs in jax_port) already handle x64 themselves.
- A future Stage 3+ benchmark script will set the flag explicitly.

## 8. Validation strategy — parity against `jax_port.simulate`

### 8.1 Why not Fortran here

Stage 1 deferred the Fortran-comparison test because (a) it requires Fortran running somewhere, and (b) it conflates "is the JAX port correct" with "is the rewrite correct." Stage 2 is about the rewrite; the port's correctness is a separate question handled in Stage 2.5.

### 8.2 Test tiers

**Tier 1: aggregate parity** — `model_a/tests/test_simulate_parity.py`

For a fixed parameter set and key, both simulators are run at `nsim=2048`. Comparison is via `shared.validation`:
- Response proportions per category: absolute tolerance 0.005.
- RT quantiles {0.1, 0.3, 0.5, 0.7, 0.9} per category: relative tolerance 0.01.

Three parameter sets:
- Realistic (matches `test_fofs_smoke.py`): `ter=200, st=50, cr=50, crsd=10, si=4, sig=5, av=20`.
- High-drift: `av=60` (most trials cross quickly, tests fast-RT regime).
- Low-drift: `av=5` (most trials saturate at NSTEP, tests slow-RT regime).

**Tier 2: smoke** — `model_a/tests/test_simulate_new_smoke.py`

Mirrors the existing `test_simulate_smoke.py` structure (shape, deterministic, key-differs), applied to the new simulator.

**Tier 3: performance** — `model_a/tests/test_simulate_perf.py` (optional, marked `@pytest.mark.perf`)

Measures wall-clock of `simulate_new(nsim=4000)` vs `jax_port.simulate(nsim=4000)` after warmup. Asserts the new simulator is ≥ 5× faster on CPU. Skipped by default in smoke gate; runs on `pytest -m perf`.

## 9. Implementation stages

| Stage | Scope | Gate |
|---|---|---|
| 2.A | Centralize x64 in `conftest.py`, update existing tests still pass | `smoke.ps1` shows 25 passed |
| 2.B | Migrate `shared/prng.py` to typed keys, update test | 25 still pass |
| 2.C | Write `model_a/simulate.py` skeleton (signature + chol_factor + drift_profile + return zero arrays) | new smoke test passes shape check |
| 2.D | Implement `_simulate_chunk` with single-GEMM + cumsum + first-crossing | smoke tests pass |
| 2.E | Implement `simulate` wrapping `lax.map` over chunks | smoke tests pass |
| 2.F | Add parity tests against `jax_port.simulate` | parity passes within tolerance |
| 2.G | Update `scripts/smoke.{ps1,sh}` to include new tests | smoke shows ~33 passed |
| 2.H | Optional perf gate: ≥ 5× CPU speedup | perf test passes when `pytest -m perf` |

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Memory blowup at large chunk_size on CPU | medium | Default `chunk_size=256`; user can override. Document in docstring. |
| `lax.map` over many chunks has slow compile | low | Fall back to Python `for` over `_simulate_chunk` if compile is > 30s. |
| Parity test fails at low-drift parameters (very long RTs / NSTEP saturation) | medium | Specifically test low-drift; if fails, investigate whether jax_port's scan has different boundary behavior. |
| Typed-key API changes break test on next JAX upgrade | low | Pin `jax<0.12` already in pyproject.toml. |
| `conftest.py` at repo root conflicts with future per-directory conftests | low | Document that this is the project-wide config; subdirectory conftests can override fields if needed. |
| Speedup is < 5× on laptop CPU | medium | Acceptable — laptop CPU is dev not production. Stage 5 will measure on H100. Soften assertion to "no slower than jax_port" if needed. |

## 11. Success criteria

Stage 2 is "done" when ALL of:

1. `model_a/simulate.py` exists, runs under JIT.
2. Parity tests against `jax_port.simulate` pass at three parameter sets.
3. `shared/prng.py` uses `jax.random.key`; `test_prng.py` updated and passing.
4. New simulator uses `shared.prng` for top-level key derivation (no direct `PRNGKey` calls).
5. `conftest.py` at repo root enables x64; `jax_port.py` is unchanged.
6. `scripts/smoke.ps1` runs all tests (~33) green.
7. New simulator is at least as fast as `jax_port.simulate` on CPU (perf gate, target ≥ 5× but soft).
8. Forward-looking items from Stage 2 reviews are captured for Stage 3 plan.

## 12. Decision log

- **Algorithm**: single-GEMM + cumsum + first-crossing argmax (locked from project design doc §4.2).
- **Memory**: chunked via `jax.lax.map`, default 256 trials per chunk.
- **Validation**: parity against `jax_port.simulate`, not Fortran (deferred to 2.5).
- **`sv` parameter**: removed from new signature (it's inert in oracle anyway).
- **x64 config**: centralized in `conftest.py`, oracle unchanged.
- **`shared.prng`**: migrated to typed-key API; new simulator uses prng helpers.
- **Chunk-size**: static_argnum, default 256 (CPU), exposed for tuning.
- **Out of scope**: no `fofs` rewrite, no optimizer, no Model B, no Fortran in this stage.

## 13. Stage 2.5 sketch (deferred)

After Stage 2 ships, run the Fortran code at a fixed parameter set (workstation, WSL2, or original cluster) and compare aggregate statistics against `jax_port.simulate(...)` at the same parameters. If they agree within MC noise, `jax_port` is enshrined as the JAX-side oracle and we never need Fortran in the loop again. This is a one-time effort, not part of any later stage's per-task gate.

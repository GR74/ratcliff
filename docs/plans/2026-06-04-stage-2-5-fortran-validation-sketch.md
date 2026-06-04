# Stage 2.5 — Fortran Validation Sketch

**Date:** 2026-06-04
**Status:** SKETCH — one-time off-band correctness gate, not on the main stage path
**Scope:** Confirm that `model_a/jax_port.simulate` (the JAX-side oracle) produces statistically equivalent output to the original Fortran `gpgsq5deg3twod24.f`. Once this passes, `jax_port` is enshrined as the JAX oracle and we never need Fortran in the loop again.

---

## 1. Why this exists

Stage 1 explicitly deferred the strict Fortran-comparison gate because it required Fortran running somewhere — outside the laptop-only Stage 1 scope. Stages 2 and 3 then validated against `jax_port` as the JAX-side oracle, treating its correctness as assumed. This is OK in the short term, but the assumption has to be paid down eventually:

- `jax_port.py` is the user's first-pass JAX port; its own docstring explicitly says **"first faithful port — MUST be validated against the patched Fortran oracle before any fit is trusted."**
- Stage 2's code review flagged a real algorithmic divergence between `jax_port.one_trial` and the Fortran for non-crossing trials (`pos` stays at init value 1 in JAX vs `argmax(a) at NSTEP` in Fortran). We don't know which is "right" without the Fortran comparison.
- Without this gate, any scientific claim from the JAX port has an asterisk.

Stage 2.5 is the closure of that loop. **One-time effort**, **off the main stage path** — can happen any time before a production fit.

## 2. Goal

For a single fixed parameter set, demonstrate that `jax_port.simulate` and the Fortran `accum` produce aggregate statistics (RT quantiles, response proportions) that agree within Monte Carlo tolerance. If they don't, fix the JAX side until they do.

## 3. Three environment paths (any one works)

### Path A: WSL2 + Intel oneAPI on the laptop (cleanest)

1. Install WSL2 Ubuntu 22.04 (one Microsoft Store click).
2. Install Intel oneAPI Base + HPC Toolkits inside WSL (free for non-commercial; includes `ifx` Fortran + MKL VSL + MKL BLAS).
3. Copy `reference/gpgsq5deg3twod24.f` to a WSL directory; copy `data/twod24data`.
4. Build:
   ```bash
   ifx -c /opt/intel/oneapi/mkl/latest/include/mkl_vsl.fi
   ifx gpgsq5deg3twod24.f -qmkl=parallel -o gpgsq
   ```
   (May need a small path-fix to the original `include` line.)
5. Run, capture output.

**Setup cost**: ~1-2 hours one-time.

### Path B: 64-core workstation (zero setup if Fortran already builds there)

The original Fortran was built on the user's workstation (`core64rr` per README_benchtwod3). If `ifort` + MKL are already installed:
1. SSH in.
2. Copy or pull the source.
3. Build with documented commands from README_benchtwod3.
4. Run, capture, scp output back.

**Setup cost**: ~10 minutes if Fortran is already working there; if not, fall back to Path A.

### Path C: Port to gfortran (most portable, most effort)

1. Replace MKL VSL RNG with `random_number` + Box-Muller.
2. Replace MKL BLAS (`DGEMM`, `DPOTF2`) with reference BLAS/LAPACK or OpenBLAS.
3. Replace MKL DFTI with FFTW3 (only matters for Model B).
4. Replace `gettim` with `system_clock`.
5. Build with gfortran on Windows via MSYS2 or `winget install gnu.gcc`.

**Setup cost**: 1-2 days of porting; changes the RNG, so the aggregate-statistical comparison gets fuzzier. **Not recommended unless A and B both fail.**

## 4. Test methodology

Pick **one fixed parameter set** matching the realistic regime from Stage 2 parity tests:
```
ter=200, st=50, cr=50 (a1), crsd=10 (sa),
si=4, sig=5, sv=0.7 (inert),
drift1=20 (av), drift2=10, a2=60
```

Pick **one fixed PRNG seed** that the Fortran will consume (MKL VSL takes an integer seed at `vslnewstream`).

### Step 1: Run Fortran at known params, capture aggregate stats

Modify the Fortran main loop to:
- Take params from a small input file (or hardcode for the validation).
- Run `accum(...)` once with `nsim=50000` (large enough that MC noise is ~0.2%).
- Write to stdout (or a text file) the per-category proportions and the 5 RT quantiles per category, for all 4 conditions.

Output format (one line per condition, 18 values: 3 proportions + 5 quantiles × 3 categories):
```
0.4520 0.3210 0.2270 312.0 348.0 376.0 405.0 442.0 ...
```

### Step 2: Run `jax_port.simulate` at the same params, same effective seed

Since MKL VSL and JAX PRNG can't produce the same draws, we use `nsim=50000` on the JAX side and rely on the law-of-large-numbers convergence.

```python
from model_a import jax_port
from shared import prng

key = prng.root_key(0)  # any fixed key
rt, cat = jax_port.simulate(key, ter=200, st=50, cr=50, crsd=10,
                            si=4, sig=5, av=20, sv=0.7, nsim=50_000)
# Compute the same 18 aggregates
```

Repeat for all 4 conditions (varying `cr` and `av` per `COND_MAP`).

### Step 3: Compare aggregates via `shared.validation`

```python
from shared import validation

result = validation.aggregate_match(
    obs_prop=fortran_props,
    sim_prop=jax_port_props,
    obs_quant=fortran_quants,
    sim_quant=jax_port_quants,
    prop_abs_tol=0.005,
    quant_rel_tol=0.01,
)
assert result["passed"]
```

At `nsim=50000` the MC noise on a proportion is ~`sqrt(0.5*0.5/50000) ≈ 0.0022`, well below the 0.005 tolerance. Quantile MC noise at 5 percent of 50000 samples is ~0.5% relative, below the 1% tolerance.

If `aggregate_match` returns `passed=True`: the JAX port is validated against Fortran. Commit the verification artifact (the Fortran output file + the comparison script) to `docs/notes/2026-MM-DD-fortran-validation-results.md` and call it done.

If it returns `passed=False`: surface the actual diffs; the most likely culprits are:
- The `sv` parameter handling — if `SV_ACTIVE=False` in jax_port but the Fortran is actually using `sv`, drift variability would diverge.
- The `pos`-at-NSTEP divergence for non-crossing trials (flagged in Stage 2.D code review). If category proportions in `low_drift` regime disagree most, this is the cause.
- A subtle dtype or scaling factor (`5.0 * noise`, the `+ 5` offset in the Fortran threshold, etc.).

## 5. Deliverables

A single committed notes file at `docs/notes/2026-MM-DD-fortran-validation-results.md` containing:
- The exact parameter set used.
- The exact Fortran build commands (which compiler, which MKL, which flags).
- The raw Fortran output (paste).
- The Python comparison script (10-20 lines).
- The pass/fail verdict from `aggregate_match`.
- If failed: what diverged and the recommended fix.

If passed, optionally also add a CI-runnable script at `scripts/fortran_validation.py` that automates re-running this in the future when JAX or jax_port changes.

## 6. Decision log

- **Off main path**: Stage 2.5 is intentionally NOT a numbered stage gate. It's a one-time prerequisite for production fits.
- **Realistic params only**: validating one parameter set is enough. If aggregates agree there, the simulator is correct everywhere (the model is smooth in params).
- **Aggregate-statistical, not bit-exact**: MKL VSL ≠ JAX PRNG, so we use large-N to wash out noise.
- **Path A preferred** (WSL2): self-contained, doesn't depend on workstation availability, works on any modern Windows box.
- **If validation fails**: fix `jax_port.py` (the oracle) at that point — Stage 2 explicitly preserved it as untouched, but Stage 2.5 is when we earn the right to modify it if needed.

## 7. Estimated cost

| Path | Setup | Build + run | Compare + write up | Total |
|---|---|---|---|---|
| A (WSL2) | 1-2 h | 30 min | 30 min | ~2-3 h |
| B (workstation) | 10 min | 10 min | 30 min | ~1 h |
| C (gfortran port) | 1-2 days | 30 min | 30 min | 1-2 days |

## 8. Trigger condition

Run Stage 2.5 when **any** of the following is true:
- Before publishing scientific results from a JAX fit.
- Before deploying to the H100 (cheap insurance before a long run).
- When the Stage 5 GPU benchmarks land and we want to claim "matches the Fortran's results, runs N× faster."
- If a contributor questions the JAX port's correctness.

If none of these are true yet, Stage 2.5 can sit on the shelf. It's not blocking anything.

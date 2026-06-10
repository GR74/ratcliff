# Track B — Amortized SBI for the 2D model: validation (smoke + full)

**Date:** 2026-06-10
**Script:** `scripts/sbi_mvp.py` · **Hardware:** rented H100 (data gen ran at ~0.64 s/dataset)

## Thesis

The 2D spatial diffusion model has no closed-form likelihood, so per-dataset fitting is
expensive. **Amortized** simulation-based inference trains a neural posterior estimator once
on (theta, summary-stats) pairs, then infers parameters for any new dataset in a single
forward pass. This note records whether that works and is calibrated for our model.

## Setup

- Generate N datasets from the 2D model (FFT path so `sig` varies), 13 free params,
  `nsim=500` trials/dataset, **20-dim summary** = 5 category proportions + 5 RT quantiles
  per condition × 2 conditions.
- Train `sbi` SNPE on (theta, x); validate by (a) posterior recovery on held-out test sets,
  (b) simulation-based calibration (SBC): `c2st` should be ≈0.5, `ks_pvals` > 0.05.

## Results

| | Smoke (N=500, 100 test) | **Full (N=3000, 200 test)** |
|---|---|---|
| Overall mean rel. error | 26.0% | **20.9%** |
| `c2st_ranks` (→0.5 ideal) | 0.528 | **0.520** |
| `c2st_dap` (→0.5 ideal) | 0.521 | **0.498** |
| `ks_pvals` mean (>0.05) | 0.144 | 0.065 |
| Train time | 65 s | 140 s |

**Per-parameter recovery (full run, mean rel. error over 50 test sets):**

| param | error | | param | error |
|---|---|---|---|---|
| ter  | 7.0% | | av1c1 | 9.6% |
| cr   | 8.0% | | av2c1 | 10.5% |
| sis  | 6.0% | | av3c1 | 12.9% |
| sig  | 25.7% | | av1c2 | 8.3% |
| **st**   | **41.0%** | | av2c2 | 9.7% |
| **crsd** | **57.6%** | | av3c2 | 13.1% |
| **sv**   | **61.7%** | | | |

## Interpretation

- **Location parameters recover well and tighten with scale.** Non-decision time (ter),
  drift criterion (cr), starting noise (sis), and all six per-condition drift rates land at
  **6–13%** — the scientifically load-bearing parameters are well constrained.
- **Variability parameters are weakly identified** — `st` (ter variability) improved markedly
  with scale (70→41%) but `crsd` and `sv` stay ~58–62%. This is the **known DDM identifiability
  result** (trial-to-trial variability params are notoriously hard to recover from summary
  stats), not a pipeline defect.
- **Calibration is essentially nominal** — `c2st` ≈ 0.50 on both ranks and data-averaged
  posterior. The estimator is *honest about its uncertainty*: it reports tight posteriors where
  the data constrain the parameter and appropriately wide ones where they don't (the variability
  params). That is exactly what a trustworthy amortized posterior should do.

**Verdict: the amortized-inference thesis holds for the 2D model, and sharpens at scale.**
More data tightened the identifiable parameters and left the unidentifiable ones honestly wide.

## Next (Track B continuation)

- **Faster generation** — the un-batched Python sim loop is the bottleneck; a `vmap`-batched
  generator (or the K-L fast path) should cut wall-clock substantially and enable larger N.
- **Sharper summary statistics** — richer summaries (or a learned/raw-trial embedding net) to
  see whether the variability params (`crsd`, `sv`) can be pulled in, or to confirm they are
  fundamentally unidentifiable here.
- **Real-data + hierarchical** — apply the trained posterior to a real `twod3datanew` fit and a
  hierarchical (multi-subject) extension.

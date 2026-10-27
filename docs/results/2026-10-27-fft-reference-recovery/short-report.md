# FFT199, FFT200 and KL: completed comparison

**October 27, 2026 — RTX 4060 Laptop GPU**

## Purpose and setup

Test whether FFT200 or KL reduces fitting time while preserving recovery and predictions relative to FFT199. All three methods fitted the **same newly generated FFT199 synthetic dataset**, with 9,000 trials per condition, the same initial parameters, training seed, SciPy Nelder–Mead optimizer and 135-minute cap. All used improvement 1: blocked simulation with early stopping and compaction.

The reference was a **corrected JAX port**, not Ratcliff's original Fortran executable or fitting routine. Shared corrections covered drift variability (`sv`), one-based spatial coordinates, exact-tie ordering, timeout handling and Fortran-style binned RT scoring. All 13 parameters were active. Independent preflight and observer-parity checks passed.

## Fitting results

| Method | Training time | Evaluations | Mean parameter error | Worst parameter error |
| --- | ---: | ---: | ---: | ---: |
| FFT199 | 74.01 min | 591 | 9.20% | 20.61% |
| **FFT200** | **66.40 min** | 630 | **7.68%** | **17.51%** |
| Repaired KL | 87.21 min | 550 | 9.05% | 18.78% |

All three optimizers converged before their caps. Errors are absolute relative differences from the known generating parameters. **FFT200 used 10.3% less training time than FFT199 and 23.9% less than KL**, with the lowest mean and worst recovery errors in this case. None met the proposed 5% mean / 10% worst recovery screens; those screens are not calibrated against original Fortran recovery.

## Independent prediction checks

Final fits and saved 30- and 60-minute checkpoints were scored with 20 fresh simulation seeds per prediction view. Views used each method's own simulator and a common FFT199 simulator. A separate FFT199 truth reference contained 180,000 trials per condition. These seeds are repeated predictions, **not repeated fits**.

The following summarizes each final fit's **native predictions versus truth-reference predictions**:

| Method | Largest region-probability difference | Largest conditional RT-quantile difference | Largest low-support probability difference |
| --- | ---: | ---: | ---: |
| FFT199 | 2.81 percentage points | 11.49 ms | 3.29 percentage points |
| FFT200 | **1.46 percentage points** | **11.34 ms** | **2.02 percentage points** |
| Repaired KL | 1.92 percentage points | 11.89 ms | 2.62 percentage points |

Each entry is the largest absolute **mean signed difference across seeds**, over both conditions and the relevant endpoints. Low-support results cover the fixed 5%, 10% and 20% drift masks. All listed endpoints had estimates for all 20 seeds. These are fitted-prediction errors, not same-parameter simulator-equivalence tests. Pointwise Monte Carlo intervals are saved separately; they do not include uncertainty across datasets or optimizer starts and do not certify equivalence.

## Conclusion and limits

**FFT200 performed best on training time, parameter recovery and these prediction summaries in this single case.** This supports advancing it to broader testing, but does not establish a universal advantage, a validated time-to-quality speedup, or equivalence to Ratcliff's original Fortran implementation.

The complete study took approximately **4 hours 20 minutes**, including setup, diagnostics, all three fits and independent scoring. Next, repeat across datasets, starting points and correlation scales, and establish direct original-Fortran agreement before making original-model fidelity claims.

## Evidence

- [Run report and status](README.md)
- [Protocol](protocol.json) and [preflight checks](preflight.json)
- [Detailed fits and prediction samples](results.json)
- [Prediction differences and pointwise intervals](prediction-differences.json)
- [Parameter and cost–recovery plots](figures.md)

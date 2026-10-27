# Ratcliff's concern, KL fidelity, and the move to FFT200

Research history and evidence summary, October 26, 2026.

This document explains the motivation, experiments, changes, and remaining questions. It distinguishes implemented changes from proposals. Numerical findings below are specific to their recorded workloads; they are not universal performance or recovery guarantees.

## 1. What we are trying to accomplish

Our goal is to make Ratcliff's spatial decision model computationally faster while preserving its scientific behavior and parameter interpretation. Making parameter recovery intrinsically better is welcome, but is not required for a successful optimization. Making the program faster by changing the model's predictions unintentionally is not an acceptable substitute.

The model accumulates evidence on a **100-by-160 decision grid: 16,000 locations**. Drift describes the average evidence supporting each location. Spatially correlated Gaussian noise creates trial-to-trial fluctuations. The location that first crosses the decision threshold determines the response and its decision time.

Several parameters are easy to confuse:

| Parameter | Meaning |
| --- | --- |
| `sig` | Spatial correlation scale of the noise. Larger values produce broader shared fluctuations; smaller values require finer spatial structure. It is not the drift hill width or simply the noise amplitude. |
| `cr` | Center of the decision threshold distribution. |
| `crsd` | Full width of the uniform across-trial threshold distribution, despite the name. A threshold is drawn as `cr + crsd * (u - 0.5)`. Thus `cr=10, crsd=2` gives thresholds from 9 to 11. |
| `sv` | Across-trial drift variability. This is implemented in the original Fortran but inactive in the frozen JAX benchmark version discussed below. |

## 2. What Ratcliff was concerned about

**According to the user's account of their discussion with Dr. Ratcliff**, the concern was that replacing FFT noise generation with a reduced PCA/KL representation could cause spurious wins in low-drift regions. This is a paraphrase of that account, not a verified quotation from Ratcliff.

A low-drift region receives little average evidence. It can nevertheless win because of a sufficiently large noise fluctuation. Those wins are part of the stochastic model; they are not automatically implementation errors or incorrect human responses.

The concern is whether the approximation changes their probability. Preserving most of the total noise variance does not necessarily preserve the spatial peaks and threshold-crossing events that determine winners. Changes can also affect high-drift regions and reaction times. Low-support wins are a sensitive diagnostic, not the only relevant outcome.

## 3. How FFT and KL generate the noise

### Original-size FFT: FFT199

The original construction uses a **199-by-319 covariance embedding**, then retains the required 100-by-160 decision grid. The dimensions follow `2N - 1`: 100 locations have separations from -99 through +99, giving 199 distinct lags. Similarly, 160 locations give 319 lags.

The FFT efficiently works with the full spectrum of that embedded covariance. Calling a JAX implementation “FFT199” identifies its noise construction; it does not establish that every other aspect of the program matches the original Fortran.

### Truncated KL

In this project, KL uses the covariance's Fourier eigenmodes. It is **not PCA trained on participants' behavioral data**. A noise field is constructed from retained spatial patterns with fresh Gaussian coefficients. Complex modes are represented through real sine/cosine basis columns.

KL keeps fewer modes, but still produces noise at every location on the same decision grid. It reduces the representation's rank, not the number of decision locations.

The possible speed benefit comes from fewer random coefficients and intermediates, reusable basis calculations, and GPU matrix multiplication. Whether this beats an FFT depends on the number of retained modes, padding, basis construction, and hardware.

The fidelity cost comes from **discarding modes**. Those modes carry covariance and fine spatial structure even when their individual variance contributions are small. A good mean-square reconstruction of the noise does not automatically imply accurate winning probabilities or reaction-time distributions.

## 4. What the first experiment found

The [initial winner-fidelity study](../results/2026-10-25-kl-winner-fidelity/README.md) used 1,024 trajectories per cell on the RTX 4060. It compared three correlation scales, three drift settings, and four noise methods.

The old policy requested 99% variance but imposed a **2,000-complex-mode cap**. At `sig=5`, the cap retained only **75.29%**. The requested percentage was therefore not the achieved percentage.

For ordinary drift at `sig=5`:

| Measurement | FFT199 | Capped KL | KL minus FFT199 |
| --- | --- | --- | --- |
| Low-support winning probability | 23.05% | 13.28% | -9.77 percentage points |
| Mean reaction time difference | Reference | — | +21.23 ms |

The reported paired 95% interval for the probability difference was [-14.40, -5.05] percentage points; the RT interval was [20.25, 22.22] ms.

**The observed problem was fewer low-support wins, not more.** This did not demonstrate the specific feared direction of excess spurious wins. It did demonstrate the broader concern: truncation changed the competition and its outcomes.

In an uncapped 99.9%-retention diagnostic at the same setting, 7,399 modes reduced the differences to +0.49 percentage points and +0.20 ms. This supported truncation as an important source of the mismatch. It did not establish that a dense implementation with that many modes would be fast.

At `sig=10`, the legacy method retained about 99% and differences were smaller: +0.78 percentage points in low-support wins and +1.10 ms in mean RT. Smaller differences were encouraging, but were not an equivalence certificate.

### What “low support” means in these tests

We defined a fixed spatial mask where deterministic drift was at most **10% of the maximum drift**, with 5% and 20% sensitivity checks. The same mask was used for all methods. The probability is the number of crossings in that mask divided by all valid attempted trials; non-crossings remain in the denominator.

This measures agreement between simulators, not behavioral “accuracy.” The initial study's intervals were diagnostic, not the later simultaneous qualification procedure.

## 5. What we tried to fix in KL

### Enforce the requested variance and fall back when necessary

The [guarded KL implementation](../results/2026-10-25-guarded-kl-screen/README.md) is opt-in. It requests **99.9% actual retention**, reports the achieved retention and mode counts, and uses full FFT when the required rank exceeds the 2,000-mode budget.

At `sig=5`, approximately 7,399 modes were needed, so it fell back to FFT. At `sig=10`, approximately 1,850 modes fit within the budget; at `sig=15`, approximately 821 did.

This fixed the **silent violation of the variance target**. It did not prove that 99.9% variance alone guarantees behavioral fidelity. Exact agreement in the `sig=5` fallback test was expected because that branch used the original FFT implementation.

The small-workload speed screen used 128 trials per condition, three warmups, and 30 alternating timing rounds:

| `sig` | FFT objective time | Legacy KL | Guarded KL |
| --- | --- | --- | --- |
| 5 | 0.413 s | 1.565 s | 0.426 s |
| 10 | 0.387 s | 1.156 s | 1.522 s |
| 15 | 0.383 s | 0.915 s | 0.916 s |

Guarded KL was about 73% faster than legacy KL at `sig=5`, about 32% slower at `sig=10`, and similar at `sig=15`. **The 73% figure was not a speedup over FFT. FFT was cheaper in all three screen settings.** These are small-workload measurements, not full fitting results.

### Repairs and bucketing address different issues

The later comparator named **repaired KL** includes stable mode-indexed random-coefficient assignment. That repair should not be confused with repairing truncation fidelity.

**Bucketed KL** reduces unnecessary padded computation. It can improve efficiency, but does not restore omitted modes. Both capped variants still showed a substantial behavioral mismatch at `sig=5` in the larger study.

## 6. Why we tried FFT200

FFT200 changes the embedding from **199-by-319 to 200-by-320**, while leaving the decision grid at **100-by-160**.

The performance hypothesis is simple: FFT algorithms can use efficient small-factor decompositions. Here, 199 is prime and 319 is 11 times 29, whereas 200 and 320 factor entirely into twos and fives. A slightly larger transform can therefore be cheaper.

This was not arbitrary zero-padding. We rebuilt the covariance embedding, including the new Nyquist row and column, with periodic-inversion symmetry. The implementation retains all required displacement covariances and does not intentionally discard positive spectral modes.

The [initial FFT200 audit](../results/2026-10-25-fft200-screen/README.md) tested 178 correlation widths. Maximum reconstructed covariance differences between FFT200 and FFT199 were:

- **5.054e-15** in float64 over the required spatial lags.
- **7.460e-8** in the float32 spectrum-factor check.

That provides strong evidence about the intended Gaussian noise law on the decision grid. Different embeddings consume different random draws, so equal seeds need not produce equal trajectories. These covariance checks also do not validate every downstream implementation detail or the full Fortran port. Other grid sizes require their own audit.

### Larger RTX 4060 speed comparison

The [optimizer-trace replay study](../results/2026-10-26-fft200-comparison/README.md) completed 24 paired timing rounds before the user stopped it. Each round replayed 30 archived parameter vectors; it was not a new optimization fit.

| Method | Median replay time | FFT200 time reduction relative to method |
| --- | --- | --- |
| FFT199 | 207.280 s | 19.97% |
| FFT200 | 165.891 s | Reference |
| Repaired KL | 262.355 s | 36.77% |
| Bucketed KL | 216.619 s | 23.42% |

The timing evidence strongly supported a speed benefit on this workload. Because results were inspected and the run stopped early, its ordinary uncertainty intervals should be treated descriptively rather than as an untouched confirmatory protocol.

## 7. Did the larger behavioral study confirm fidelity?

The [last formal assessment, at round 10](../results/2026-10-26-behavioral-assessment-20-runs/look-10/README.md), included **90,000 trials per condition, method, and case**. All four methods were evaluated each round across 17 cases. The study later stopped after 12 complete rounds and a partial thirteenth; the last formal analysis remained look 10.

For ordinary drift at `sig=5`, the differences from FFT199 were:

| Method | Condition 1 low-support difference | Condition 2 difference |
| --- | --- | --- |
| FFT200 | -0.338 percentage points | +0.090 percentage points |
| Repaired KL | -9.919 percentage points | -8.954 percentage points |
| Bucketed KL | -9.906 percentage points | -9.048 percentage points |

For example, the adjusted condition-1 interval was [-1.661, +0.986] for FFT200 and [-11.127, -8.707] for repaired KL. The capped KL mismatch was clear. FFT200's observed difference was small, but its interval was too wide for the tight qualification margin.

FFT200's low-support point differences were also small at `sig=10` (+0.119 and +0.043 percentage points) and `sig=15` (+0.308 and -0.500). These do not by themselves certify the complete behavior distribution.

**The appropriate conclusion is: capped KL showed a material mismatch; FFT200 had not shown a material mismatch, but full behavioral qualification remained inconclusive.** Rare conditional RT samples and conservative spatial uncertainty bounds contributed to that uncertainty.

## 8. What the fitting comparison actually measured

The [archived fitting comparison](../results/2026-10-26-fft200-archived-fit-comparison/README.md) reused two previous RTX 4060 KL fits and fitted FFT200 to their exact same archived dataset. This avoided rerunning the controls, as requested.

Crucially, the observations were **synthetic data generated by the older JAX/KL pipeline**, with 9,000 trials per condition. They were not human participant data or newly generated original-FFT data.

The matched setup used the same dataset, known truth, initial point, training key 321, Nelder-Mead settings, 1,000-iteration limit, and 135-minute soft cap. FFT200 and improved KL both used blocked early stopping.

| Method | Fit status | Training time | Mean active parameter error | Worst active error |
| --- | --- | --- | --- | --- |
| Legacy full-horizon JAX KL | Time limit; 84 evaluations | 135.92 min | 8.10% | 19.55% |
| Improved KL | Converged; 563 evaluations | 94.63 min | 4.59% | 16.64% |
| FFT200 | Converged; 586 evaluations | 56.97 min | 6.10% | 18.57% |

FFT200 took about **39.8% less training time than improved KL**, but had larger recovery errors in this case. Both methods' worst recovered parameter was `crsd`: truth 2, improved KL approximately 2.333, FFT200 approximately 2.371.

Recovery error here is `abs(estimate - truth) / abs(truth)`, summarized over active parameters. The inactive `sv` coordinate is excluded. Independent prediction scoring used 20 fresh seeds; that does **not** mean 20 independent recovery fits were performed.

This is useful evidence about one KL-generated dataset. It does not answer which method best preserves original-FFT recovery. A KL generator could favor KL, but that has not been established as the sole cause of the difference. Finite data, parameter tradeoffs, optimization paths, and simulation noise can all contribute. Optimizer convergence does not guarantee recovery of generating truth.

## 9. Important corrections about the reference model

Several uses of “original” in earlier discussion were ambiguous and needed correction:

- The archived RTX 4060 “original versus improved” fitting comparison was **full-horizon JAX KL versus improved JAX KL**, not original Fortran versus KL.
- Original Fortran outputs do exist in [the reference archive](../../reference/archive_outputs/), with provenance in [the benchmark notes](../../reference/README_benchtwod3). Those historical outputs are not a matched RTX 4060 recovery study.
- Selecting the original FFT embedding in JAX does not restore omitted scientific behavior elsewhere in the port.

The frozen benchmark port has inactive `sv`, whereas Fortran implements a trial-level drift multiplier. An existing correction on another branch also addresses reference-specific RT CDF and non-crossing behavior. These corrections were **not included in the comparisons above**, and must be reviewed and validated before use. The omission predates these FFT200 experiments; there is no verified basis for calling it a deliberate scientific simplification.

Original Fortran can serve as a CPU reference; it does not need to run on the GPU to establish scientific agreement. However, compilation dependencies and the fitting driver need attention before a new matched original-code run. No such new Fortran fit is established by these results.

## 10. What is fixed, and what is still open?

| Question | Current evidence/status |
| --- | --- |
| Could the old mode cap silently miss its variance target? | Yes. Guarded KL detects this and falls back to FFT. |
| Did capped KL preserve behavior at `sig=5`? | No. Both capped variants materially suppressed low-support wins. |
| Does a higher KL retention target help? | Diagnostic results support it, but additional modes can remove the speed advantage. |
| Does FFT200 preserve required noise covariance? | Passed the recorded covariance and PSD audits over the tested domain. |
| Is FFT200 faster on the measured RTX 4060 workloads? | Yes, including approximately 20% less replay time than FFT199. |
| Is full FFT200 behavioral equivalence certified? | No; formal qualification remained inconclusive. |
| Is FFT200 better at original-model parameter recovery? | Not established by the KL-generated archived case. |
| Is the frozen JAX model fully equivalent to original Fortran? | No such claim is supported; known port differences remain. |

The proposed 5% mean/10% worst recovery screens and behavioral margins are engineering acceptance criteria, **not thresholds calibrated from original Fortran recovery**. See [the repository's validation guidance](../../CLAUDE.md).

## 11. The next fair comparison

The next experiment should separate absolute recovery, reference agreement, and computational cost:

1. Restore and validate the relevant reference-model behavior identically for all methods before fitting.
2. Generate a fresh synthetic dataset with the full original-size FFT reference, preserving known generating parameters and provenance.
3. Fit that same dataset with **FFT199, FFT200, and repaired KL**, using matched initial conditions, optimizer settings, simulation budgets, and stopping rules.
4. Compare estimates with both generating truth and the FFT199 fit. Evaluate predictions independently through a common FFT reference as well as each method's native simulator.
5. Report training time, initialization, verification cost, failures, and time to an acceptable fit. Repeat datasets and starts before drawing broad recovery conclusions.

The [saved comparison plan](../plans/2026-10-27-fft-reference-recovery-comparison.md) provides the proposed seeds and budgets. **The subsequent user decision was fidelity first:** validate the model corrections before long fits, even if that delays results. This supersedes the draft plan's permissive fallback to the uncorrected frozen port. This next protocol is a plan, not a completed experiment reported here.

Real participant data is useful later for comparing fitted predictions and agreement with original-code estimates. It cannot establish absolute parameter recovery because participants' true parameters are unknown. Original-code estimates are a reference, not ground truth.

## 12. What “behavior-aware KL” would mean, and the paper angle

“Behavior-aware KL” is a **proposed approach, not an implemented or validated result** in this history. It would choose the retained rank using predefined tolerances on winning probabilities, RT distributions, and spatial outcomes, rather than relying only on a global variance percentage.

One possible design is to calibrate rank offline over a specified parameter domain, validate on held-out settings and seeds, and fall back to FFT200 where a sufficiently faithful KL representation is slower. Calibration cost must be included in the performance argument. It must not select favorable results after observing test outcomes.

KL makes a larger change to the noise representation than FFT200's embedding adjustment. However, standard KL truncation and factor-friendly FFT sizes are established techniques; neither alone establishes novelty. A stronger research contribution would require a defensible method for preserving first-passage behavior while reducing cost, supported by independent validation and a literature review.

For now, FFT200 is a strong practical baseline. The evidence supports optimizing around it, while treating behavioral qualification and a fair reference-generated recovery experiment as unfinished scientific work.

## Evidence index

- [Initial KL winner and RT study](../results/2026-10-25-kl-winner-fidelity/README.md)
- [Guarded KL correctness and speed screen](../results/2026-10-25-guarded-kl-screen/README.md)
- [FFT200 construction, covariance audit, and initial timing](../results/2026-10-25-fft200-screen/README.md)
- [RTX 4060 comparison and timing replay](../results/2026-10-26-fft200-comparison/README.md)
- [Behavioral assessment: formal look 10](../results/2026-10-26-behavioral-assessment-20-runs/look-10/README.md)
- [Archived dataset fitting comparison](../results/2026-10-26-fft200-archived-fit-comparison/README.md)
- [Proposed FFT-reference recovery study; subject to the fidelity-first amendment above](../plans/2026-10-27-fft-reference-recovery-comparison.md)

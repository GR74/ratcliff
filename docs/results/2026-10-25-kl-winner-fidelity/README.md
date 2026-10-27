# Model B: FFT versus truncated KL fidelity screen

Open `report.html` for the presentation-ready report and three figures. `metrics.csv`
contains all 36 condition/method summaries, with uncertainty and sensitivity metrics.
`trials.npz` contains the compact per-trial observations; `metadata.json` records
parameters, hardware, seeds, versions, validation results, and simulation source hashes.

The study ran 1,024 paired trials in each of 3 correlation settings × 3 drift
conditions × 4 methods (36,864 evaluated trajectories, with intentionally shared
random inputs within each correlation setting). The production simulator was not changed.

## Main finding

The nominal 99% KL setting is capped at 2,000 complex modes. At `sig=5`, this
retains 75.29% of the spectral variance. In the baseline-drift condition:

- Low-support wins: FFT 23.05%, current KL 13.28%; difference −9.77 percentage
  points, conservative 95% paired interval [−14.40, −5.05].
- Mean reaction time: +21.23 ms, paired bootstrap 95% interval [20.25, 22.22].
- Uncapped 99.9% retention (7,399 modes): differences shrink to +0.49 percentage
  points and +0.20 ms, respectively.

At `sig=10`, current KL retains 99.00% variance. The baseline-drift low-support
win difference is +0.78 percentage points [−1.83, +3.39], and mean RT changes
by +1.10 ms [0.61, 1.61]. The winning-probability interval is too wide to establish
equivalence within ±1 percentage point. This is a screening study, not universal
validation across the fitting domain.

“Low support” means baseline drift ≤10% of the peak; it is a fixed geographic
mask, also used in the zero-drift control. A threshold crossing is required.
These outcomes are not automatically psychological errors. The full report
explains the conservative paired intervals, additional cutoff sensitivity,
shared randomness, and use of masked FFTs to sample the production KL law.

## Reproduce

From the repository root in a Python environment with JAX, NumPy, SciPy,
Matplotlib and pytest:

```sh
python scripts/check_kl_winner_fidelity.py --trials 1024 --chunk 1 --seed 20261025
python scripts/report_kl_winner_fidelity.py
JAX_PLATFORMS=cpu python -m pytest model_b/tests/test_winner_fidelity_diagnostics.py -q
```

GPU run: NVIDIA GeForce RTX 4060 Laptop GPU; Python 3.12.3; JAX 0.10.2;
NumPy 2.5.4; 237.3 seconds including cross-checks. The diagnostic uses highest
matrix-multiplication precision for numerical cross-checks. Plotting used
Matplotlib 3.11.2. Four focused tests passed, plus production observer and
KL-basis cross-checks at each of the three correlation parameters.

The first attempted batch exceeded available GPU memory. The successful run
used batch size 1 and a bounded allocator; failed attempts contributed no data.
The initial diagnostic's Fourier sign was corrected before any accepted results;
all three production-basis equality checks passed on the final run.

Next confirmatory work should use investigator-approved equivalence margins,
independent seeds, more trials, additional thresholds/drift shapes, fitted
parameter settings, and empirical-data fits. Achieved variance should be checked
explicitly before selecting a KL mode cap. This study does not implement that
production change or claim that variance retention alone guarantees fidelity.

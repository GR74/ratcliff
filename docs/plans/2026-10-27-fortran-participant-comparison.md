# Original Fortran versus JAX on real participant data

The user proposed fitting the real participants with the original Fortran and
comparing our returned parameters with those estimates. This is a reference-fit
agreement study, not parameter recovery: neither method's estimates are known
ground truth. It complements the synthetic recovery experiments.

## Readiness findings

- `data/twod3datanew` contains 32 nonempty condition rows, organized as 16
  two-condition participant records, with five response categories and five RT
  quantiles. SHA-256: `2f369e3e398e263b7ffad85bf9d89e4b83422c071ecc9612746e14cfa2fc9ada`.
- The original source is
  `reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`. It reads that filename
  and `parinp`, whose values are starting parameters, not known participant truth.
- The included `SIMPLX` routine exists, but both calls to it in the main program
  are commented out. As shipped this executable evaluates the objective at
  supplied parameters; it does not perform the proposed new fitting study.
- The main program selects one record. A separate measurement driver must
  select participant records explicitly, activate fitting, and preserve the
  original source and simulator. Do not simply run it and call the result a fit.
- WSL preflight found no `ifx`, `ifort`, `mpiifort`, `mpiifx`, `gfortran`, or
  `mpifort` on PATH, and no `/opt/intel/oneapi` installation. The reference needs
  MKL VSL/DFTI and MPI. No compiler installation or new Fortran run has occurred.
- Fortran uses trial drift multiplier `1 + sv * (u - 0.5)`; the frozen JAX port
  does not implement this effect. Current-version agreement must label this
  model difference. A computational-fidelity claim needs a separately validated
  correction, not silently disabling it in the original reference.

## Proposed execution sequence

1. Establish a compatible compiler/MKL/MPI environment, archive its versions,
   and build an isolated copy. Retain all source changes as an explicit patch.
2. Run one fixed-parameter objective/simulation for the first participant record
   and verify input ordering, response regions, time units, normalization,
   non-crossing behavior, and output capture. Measure runtime before selecting
   a full-study budget. Validate any driver adaptation against the archived
   executable behavior and raw reference outputs where appropriate.
3. Fit that same participant's summaries with original Fortran, FFT200 and KL.
   Lock starts, trial counts, bounds and budget accounting before running.
   Report each algorithm's stopping rules; do not assume the Fortran simplex
   implementation equals SciPy Nelder-Mead. Separate simulator differences from
   optimizer differences through fixed-parameter and cross-evaluation checks.
4. Repeat starts/random streams to measure Fortran's own estimate variability.
   Compare parameter-specific differences with that variability and report
   response probabilities, RT quantiles, spatial predictions and fitting cost.
   The available data are summaries; fresh held-out participant trials cannot
   be manufactured by splitting these quantiles. Independent simulator seeds
   assess Monte Carlo stability, not held-out human predictive performance.
5. Expand to the remaining records after the pilot establishes runtime and
   measurement validity. Predeclare scientifically meaningful differences and
   uncertainty rules; do not tune thresholds to make a candidate pass.

A matching Fortran estimate establishes reference agreement, not that either
estimate is the participant's true psychological parameter. The complementary
Fortran-generated synthetic study is still needed for recovery calibration.

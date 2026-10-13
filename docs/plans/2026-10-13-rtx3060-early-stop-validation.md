# Testing early stopping on the laptop

The aim is to test improvement 1 more broadly and compare observed time to a
specified fitting-quality target. A faster simulation alone does not establish
equally accurate fitting or human-data validity.

## Hardware and environment

Live detection: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB VRAM, Windows
driver 596.36, Ubuntu under WSL2. GPU-enabled JAX 0.10.2, NumPy 2.5.4 and
SciPy 1.18.1 match the main library versions from the H100 experiment. CUDA
wheel dependencies/driver and batch size differ, so cross-machine timings are
workload comparisons rather than an isolated comparison of GPU silicon.

The isolated environment and outputs live in `D:/ratcliff-3060/`. WSL's Windows
drive bridge stalled pip extraction; downloaded Linux wheel payloads were
extracted with native Windows file I/O. `pip check` passed and JAX completed a
GPU calculation. No system Python, WSL lifetime or NVIDIA driver was changed.
Wheel names/hashes are saved in `control/wheel-manifest.json` on D:.

JAX preallocation is disabled and its allocator limit is 50% of total VRAM,
about 3 GB. Start with chunk size 8 and a K–L basis-cache capacity of 2 entries
for both variants; the production cache capacity of 64 could hold about 16 GB.
The experiment's cache setting affects rebuilding cost but not retained modes.
The simulator/library defaults remain unchanged.

[JAX installation](https://docs.jax.dev/en/latest/installation.html) and
[memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
describe the WSL GPU path and allocation controls used here.

## Comparisons

```text
Three true parameter settings
       |
       +-- Original full-horizon simulator
       +-- Full-horizon blocked control -- same indexed noise -- early stop
       |
       v
Starting points: two for baseline, one for each changed setting in the first pass
       |
       +-- Original fitter
       +-- Early-stop fitter
       |
       v
Saved training checkpoints --> Independent scoring --> Time/quality report
```

Settings vary the decision boundary, noise correlation and drifts. Starting
guesses vary within ±20% and are clamped into the simulator's valid domain.
The same observed dataset, starts, trial count, fitting seed and tolerance are
used by both fitting variants. Variant order reverses for the second start.
Ideas 2–5 remain disabled, including the active-only optimizer.

First calibrate at 512 trials per condition, then measure the 9,000-trial
workload if it fits the runtime budget. After the pilot, choose the fitting
trial count using measured costs, and record that choice before fitting.
Fresh worker processes isolate cases. Warm-up is separate from synchronized
simulation/fitting time. RT and choice equality are checked against the paired
full-horizon blocked control across settings; category/quantile screens are
recorded separately.

## Fitting quality and stopping

Allow up to 1,000 simplex iterations, with the existing tolerance `1e-5`.
Each case also has a recorded wall-time cap. A time-limited case must remain
labelled incomplete; an iteration limit must not be called convergence.

The provisional target requires both:

- Mean error over the 12 active, effective/clamped parameters at most 5%.
- Median independent G² at most the true-parameter median plus 10% of its
  absolute magnitude (minimum baseline magnitude 1), plus 1 G² unit.

This is a declared synthetic screening target, not a validated scientific
equivalence criterion. The unused `sv` is excluded from recovery metrics but
remains in both 13-coordinate searches. Preserve raw parameter vectors too.

Save the best training-loss estimate every 100 objective evaluations and at
termination. Score eligible saved checkpoints using the original legacy path
at three independent keys, 999/5001/5002. At most eight eligible checkpoints,
selected evenly in order, plus the final estimate are scored. Checkpoints that
fail the parameter-error requirement cannot meet the joint target and need no
scoring; the final estimate is always scored. Scoring never feeds back into
the fitter or changes its random seed.

Report the first **scored checkpoint** that passes, its training elapsed time,
the scoring trial count and all per-seed scores. This is an observed upper bound
on time to the target, not an exact threshold crossing. A missed target between
checkpoints, Monte Carlo variability and repeated checkpoint testing remain
limitations. Also report final recovery, G², iteration/evaluation counts and
convergence flags. Cases with no observed hit are censored, not assigned a
fabricated speedup. No successful-fit-only averaging.

## Run and preserve

`scripts/run_rtx3060_validation.sh` reuses the verified environment and stores
temporary/driver-cache files on D:. `scripts/validate_early_stopping.py` writes
settings, samples, all proposal traces, checkpoints, source fingerprints,
allocator peaks and partial-case status. Use a fresh output directory each run.
Compact results and plots should be committed to our existing experiment branch;
raw backups should be preserved with hashes as in the H100 experiment.

The runtime allowance was asked separately. Until a longer allowance is given,
use a bounded first pass of up to two hours, including scoring and a shutdown
reserve. This concerns only the local GPU process; do not stop WSL or other apps.

## Pilot and declared first study

The 512-trial pilot completed on the GPU: original median 4.8816 s, blocked
full-horizon 5.7353 s and early-stop 0.4853 s. Each used chunk 8, three warmed
timing repetitions and one simulator seed. Allocator peaks were 1.134/0.621/0.621
GB respectively. Measure 9,000-trial timings separately before reporting an
H100 comparison; the initial scaling estimate is not a measured hardware ratio.

For the first two-hour study, declare **128 fitting trials per condition**,
9,000 observed trials, 2,048 independent scoring trials, three simulator seeds,
three warmed timing repetitions and a 600-second fitting cap per case. Allow
up to 1,000 iterations. Use two starts for baseline and one start each for the
lower/higher-boundary settings: eight fits with their scoring passes. This
deliberately smaller fitting budget makes the laptop study practical. It does
not settle the earlier 9,000-trial fitting question. Preserve every time-limited
or incomplete case and stop at the overall two-hour deadline.

Score at most two eligible checkpoints, selected evenly in order, plus the final
estimate in this first pass. This coarser checkpoint resolution must accompany
any reported observed time to quality. The three-key independent score remains
unchanged; do not relax the target after seeing results.

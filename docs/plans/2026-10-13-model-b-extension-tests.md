# Testing improvements 1–5 on one H100

The rental limit is **three hours from deployment**, extended by the user on
October 13. The pod was created at 15:21:35 UTC; the shutdown guard is set for
18:20:45 UTC (2:20 p.m. Eastern). Workload deadlines reserve time to download
results and stop GPU billing. Stop earlier if all useful work is complete.

## The questions

| Idea | Change being tested | What would count as useful? |
|---|---|---|
| 1 | Stop/compact finished trials in time blocks | Less work, faster completed simulation, preserved RT/choice behavior |
| 2 | Fit with increasing trial budgets | Faster fitting at comparable final recovery/independent loss |
| 3 | Search 12 active Model B parameters; keep unused `sv` fixed | Less wasted optimization, comparable or better recovery |
| 4 | Pad K–L modes to a small set of capacities | Smaller matrix products without discarding retained modes |
| 5 | Accumulate K–L weights before reconstructing fields | Less large-array accumulation with decisions checked at every step |

The learned starting guess is a separate sixth idea and is outside this run.
Ideas 3–5 target Model B's K–L path. Model A is included in the first two ideas
and profiling, but does not use Model B's spatial K–L representation.

```text
Original comparisons
   Legacy → Blocks/full horizon → Blocks/early stop
                     + fixed versus staged fitting
                              |
                              v
Extension comparisons (same early-stop control)
   Control ─┬─ Remove unused search coordinate only
            ├─ Smaller K–L capacities only
            ├─ Cumulative K–L weights only
            └─ All three together
```

## 1. Correctness before timing claims

The regression tests check crossings within a block, a rise followed by a fall,
the final padded block, fallback responses and stable indexed trial randomness.
Extension tests check retained modes/variance and covariance, the linear
cumulative-weight identity at every time step, and the full 13-parameter
result/callback interface while `sv` stays fixed.

K–L capacities are 512, 1,024, 1,536 and 2,000 complex modes. At the tested
`sig=10`, 1,325 modes remain in both variants: padded real columns fall from
4,000 to 3,072. This is 23.2% fewer columns for the matrix product, not a promised
23.2% whole-program speedup. Draws keep a common 4,000-column random source and
slice its prefix, so changing padding preserves retained modes' random weights.

```text
Current:       weights → fields → cumulative fields → every-step crossings
Cumulative:    weights → cumulative weights → fields → every-step crossings
```

Linear operations commute in exact arithmetic. fp32 operation order can change
rounding and boundary crossings, so numerical identity alone is insufficient.
Matched samples measure exact RT/choice agreement and distribution screens.
The independent legacy-versus-blocked screen remains a separate check.

## 2. Synchronized speed measurements

Use one H100 SXM, unchanged software/precision and fresh worker processes.
Separate first calls/compilation from warmed measurements and wait for completed
GPU work. Run profilers in their own windows, with no benchmark overlap.

Extension simulator settings: 9,000 trials per condition, chunk 64, time blocks
32, seven warmed timing repetitions and simulator seeds 0–5. The production
baseline comparison used seeds 0–11. Record source commit/hash, GPU/driver,
versions, seeds, basis sizes, variance, allocated work and device allocator peaks.
Check the new control against previously saved indexed-noise samples too.

Use timing bootstrap intervals as conditional screening summaries. Seven calls
on one machine do not quantify machine-to-machine or long-run variability.

## 3. Recovery and fitting time

Use exactly the same 9,000-trial synthetic observed data for all extensions.
Use three starting points (seeds 0–2, ±10% from the true parameters), a fixed
fitting random key 321, 9,000 trials per condition and 40 simplex iterations.
Rotate variant order between starts. Record fit time, warm-up time, objective
evaluations, convergence flags, active parameter errors and an independent
legacy-path G² score at key 999. Keep `sv` out of error metrics for every variant.

The extension control uses fixed-budget fitting and early stopping, with the
original 13-coordinate search. This isolates ideas 3–5 from staged fitting.
Each feature is tested alone and in a combined case. The earlier experiment
separately compared fixed versus staged fitting.

Provisional recovery screens permit at most +2 percentage points of mean active
parameter error and +10% independent G² for every paired start. Choice/RT screens
use ±1 percentage point and ±2% quantiles, with sparse categories labelled
unscored. These are declared screening margins, not formal global equivalence.

The original fits also received five additional independent scoring keys
5001–5005. Report this variation separately; do not replace the original score
or tune parameters against the held-out keys. Repeat such scoring for extensions
if the remaining rental permits it.

## 4. Diagnose and decide

cProfile shows Python time by function; line_profiler shows time by source
line. Both preserve JIT and include explicit result waits. At 9,000 trials,
Model B's legacy objective is dominated by simulation. A high G² line time can
include queued simulation work, so it is not a direct measurement of pure
aggregation. Cold K–L basis building is separately measured from reuse.

Publish speed **and** quality, plus incomplete cases. Fixed 40-iteration fits
do not measure time to convergence. Retain existing defaults and make changes
opt-in until stronger recovery and human-data validation support adoption.
Save compact reports/plots/JSON/CSV and compressed raw samples/logs/profiles to
our own GitHub branch, as requested. Preserve local copies too. Exclude account
credentials and pod-control files from the backup.

Run the extensions with `scripts/benchmark_model_b_extensions.py`; provide the
baseline `data-b.npz`, a fresh output directory and a deadline based on rental
time remaining. This experiment is scoped to one synthetic parameter setting,
near-truth starting points, K–L simulation and the recorded H100 environment.

## Completed run

The pod stopped at **18:14:36 UTC**, about **2 hours 53 minutes** after creation,
within the three-hour limit. All saved artifacts were downloaded before shutdown.
The original larger pass completed all 26 cases. The extension pass recorded
20 cases: 19 workers completed; one timed out after its optimizer finished.
Its parameters were recovered from the saved loss trace and independently
scored, while its runtime remains a lower bound. cProfile and line_profiler
completed small-budget and 9,000-trial passes for both models.

Read the [original results](../results/2026-10-13-h100/README.md),
[extension results](../results/2026-10-13-h100/EXTENSION_RESULTS.md) and
[raw backup manifest](../results/2026-10-13-h100/raw/README.md).

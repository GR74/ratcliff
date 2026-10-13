# Finding where the DDM code spends time

Use `scripts/profile_speedups.py` to diagnose both models' Python drivers with
**cProfile** and **line_profiler**. This is separate from the speedup benchmark.
Do not run profiling at the same time as scientific GPU timing measurements.

```text
New process for each model + variant
                  |
                  v
Record basis preparation, data generation, first calls and warm-up
                  |
                  v
Run the same workload three separate ways
       /                  |                   \
No profiler           cProfile           line_profiler
normal duration       time by function   time by source line
       \                  |                   /
                  v
Save profiles, source locations, hardware and a readable report
```

The script profiles actual production functions. It preserves JIT compilation;
registering their Python source does not make JAX run them eagerly. It waits for
JAX results after each workload, including conversion of simulation samples to
NumPy, so a timer includes completed work rather than just dispatching work.

For Model B's K-L path it also separates **building a basis with an empty Python
basis cache** from **reusing that basis**. Each fit pass clears this cache and
warms its starting objective and stage trial counts, so the function and line
passes start from comparable basis-cache states. Parameter changes inside the
fit still build new bases as normal.

## A short CPU check

Install the optional tool into your environment:

```powershell
.venv\Scripts\python.exe -m pip install line_profiler==5.0.2
.venv\Scripts\python.exe scripts/profile_speedups.py --allow-cpu --models a --variants legacy,combined --nsim 8 --data-nsim 32 --chunk-a 4 --block-size 4 --repeats 2 --fit-iters 3 --output .benchmark-results/profile-cpu-a --budget-seconds 180 --worker-timeout 90
```

`--fit-iters 0` skips fitting. `--noise fft` exercises Model B's FFT generator;
the default is K-L, matching the H100 speedup experiment. `--nsim` is trials per
condition, not total trials. The default ordered models are `b,a` so Model B has
priority within the deadline.

## A dedicated H100 window of less than five minutes

Run only after the scientific benchmark process has exited:

```bash
.venv/bin/python -m pip install line_profiler==5.0.2
JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
timeout 270s .venv/bin/python scripts/profile_speedups.py \
  --models b --variants legacy,combined --nsim 128 --data-nsim 256 \
  --chunk-b 64 --block-size 32 --repeats 2 --fit-iters 6 \
  --budget-seconds 250 --worker-timeout 120 \
  --output .benchmark-results/profile-h100
```

This includes three real small fits per case: no profiler, cProfile, and
line_profiler. They have the same initial parameters and random seed. The
combined variant runs staged refinement and blocked early stopping. A worker
can time out; completed phases are checkpointed. Read `cases.json` and
`worker.log` before treating a case as complete. Installation time is outside
the command above; the pod's separate two-hour rental deadline still applies.

For a larger simulation or fit, increase `--nsim`, `--data-nsim`, and
`--fit-iters` only when the remaining GPU budget allows it.

## Reading the output

All raw output lives in `.benchmark-results/`, which Git ignores:

- `PROFILE_REPORT.md`: durations and the highest recorded Python line times.
- `cases.json`: completed, failed, and timed-out cases.
- Each case's `profile.json`: first-call/warm-up durations, profiler summaries,
  source hashes and locations, seed, settings, package versions and hardware.
- `*.pstats` and `*-cprofile.txt`: raw and readable function profiles.
- `*.lprof` and `*-lines.txt`: raw and readable line profiles.

In cProfile, **self time** is time inside that function excluding its calls;
**cumulative time** includes its calls. Cumulative times overlap, so do not add
them together. In line_profiler, **hits** counts executions of a Python line.

```text
Python calls JAX  -->  GPU works asynchronously  -->  Python waits for result
      |                                                    |
often small line time                              can show large line time
```

A large time at `block_until_ready`, `float(loss)`, or `np.asarray` tells us
where Python waits. It does **not** tell us which GPU kernel is expensive.
Likewise, a JIT function can show zero Python line hits while doing most of the
GPU computation: its Python body ran during tracing, and XLA executes the
compiled program afterward. Kernel diagnosis needs JAX/XProf or Nsight.

Use these reports to identify costly basis rebuilds, repeated host operations,
objective evaluation counts, and waits. Use the separate uninstrumented H100
benchmarks to claim a speedup. The tiny synthetic fits here do not establish
parameter recovery or convergence.

Sources: [Python profiling documentation](https://docs.python.org/3.11/library/profile.html)
and [line_profiler documentation](https://kernprof.readthedocs.io/en/latest/auto/line_profiler.line_profiler.html).

## Completed H100 diagnostics: October 13, 2026

All four cases completed: Model A and Model B, each with legacy and combined
paths. Their workload summaries were identical across the uninstrumented,
cProfile, and line_profiler passes. The final GPU profiling command exited
before 16:21 UTC; a subsequent process read found no remaining profiler
processes. Scientific timing runs were paused during this window.

Hardware was **NVIDIA H100 80GB HBM3**, driver **580.126.09**. The environment
used Python 3.12.3, JAX/jaxlib 0.10.2 with CUDA 12 plugin/PJRT 0.10.2, NumPy
2.5.4, SciPy 1.18.1 and line_profiler 5.0.2. JAX x64 was enabled and matrix
multiplication precision was `default`. The checkout was
`b4172448654b6d105714e364c4291c47ed7f3ac0` plus the copied profiling script;
each registered source file's SHA-256 is recorded in `profile.json`.

Model B used K-L, 128 trials per condition, chunk size 64, time blocks of 32,
and synthetic data with 256 trials per condition. Model A used 256 trials per
condition and chunk size 256, with the same block size/data budget. Both used
root random seed 321 and two repeats for simulation/objective profiling.

| Workload | Ordinary duration (s) | cProfile duration (s) | line_profiler duration (s) |
|---|---:|---:|---:|
| B legacy: two objective calls | 0.480 | 0.598 | 0.805 |
| B combined: two objective calls | 0.341 | 0.540 | 0.737 |
| B legacy: tiny fit, 21 evaluations | 11.107 | 12.107 | 13.813 |
| B combined: tiny fit, 45 evaluations | 11.139 | 14.985 | 18.411 |
| A legacy: two objective calls | 0.205 | 0.317 | 0.413 |
| A combined: two objective calls | 0.201 | 0.312 | 0.413 |

These are diagnostic timings at small trial budgets, with only two objective
repeats. **They are not a statistical speedup comparison.** Both B fits used a
six-iteration cap and did not converge. The combined fit used stage trial
counts 32, 64 and 128, with iteration budgets 1, 1 and 4; restarting the
simplex increased objective evaluation count.

The two clearest places to investigate next are:

1. **K-L basis rebuilding when `sig` changes.** One uncached basis took about
   0.84–0.91 seconds; a cache hit took about 1.2–1.3 milliseconds per call.
   The legacy fit's line profile recorded eight new bases taking 5.90 seconds
   of its 13.81-second instrumented duration. The NumPy cosine/sine lines
   accounted for 3.90 seconds of that. The combined fit recorded five new
   bases taking 3.32 seconds of its 18.41-second instrumented duration.
2. **G² aggregation and repeated JAX host operations.** In the combined B
   fit, the `g2_vmap(...)` call in `model_b/objective.py:153` recorded 13.46
   seconds. In the A objective profile, the corresponding aggregation call
   recorded 0.368 seconds of 0.413 seconds. cProfile also showed many calls
   through JAX's `apply_primitive` and `_pjit_call_impl`. Compiling the pure
   aggregation calculation as one JAX program is a candidate to test.

```text
Parameter proposal
       |
       +--> Did sig change? --> Rebuild K-L basis with NumPy cos/sin
       |
       +--> Simulate trials on the GPU
       |
       +--> G² aggregation --> Many host-side JAX operations
       |
       v
Return loss to the simplex optimizer
```

The aggregation line's time can include waits for previously dispatched GPU
work. It does not prove that a particular aggregation kernel is slow. These
shares change with trial count and are affected by instrumentation. A basis
refactor must preserve the covariance model, and any aggregation refactor
must preserve the original loss. Neither change was made in this profiling
task; each needs an independent correctness and timing experiment.

Raw Model B files are in `.benchmark-results/profile-h100/`; Model A files on
the pod are in `.benchmark-results/profile-h100-a/`. Each contains the
readable report, JSON provenance, pstats and line profiles. CPU smoke checks
also completed both legacy/combined paths and real bounded fits; they were
used to verify the profiling harness, not to establish GPU performance.

## Follow-up: 9,000-trial objective profiles

After the scientific 9,000-trial benchmark and held-out scoring pass finished,
we profiled all four model/variant cases at **9,000 trials per condition**,
using synthetic data with 9,000 trials per condition. The H100 environment,
original `b417244` checkout, random seed, chunk sizes and block size were the
same as above. All cases finished by **17:16:19 UTC**, before the 17:19 cutoff;
a subsequent process read found no profiler processes. No scientific GPU
measurement overlapped this window.

These profiles cover simulations and objectives, with **no fitting**. The
combined variant therefore exercises blocked early stopping; staged fitting
is only exercised in the earlier bounded fit diagnostic. All workload
summaries matched across the three profiler passes. Fourteen pstats and
fourteen line profiles were downloaded and verified locally.

All durations below are **totals for two completed calls**, after warm-up:

| Case | Ordinary simulation (s) | Ordinary objective (s) | cProfile objective (s) | line_profiler objective (s) |
|---|---:|---:|---:|---:|
| B legacy | 7.861 | 7.966 | 8.017 | 8.219 |
| B combined | 0.727 | 1.029 | 1.250 | 1.403 |
| A legacy | 0.117 | 0.328 | 0.424 | 0.482 |
| A combined | 0.125 | 0.310 | 0.413 | 0.515 |

The larger workload changes the interpretation of the earlier tiny fits:

- **Large B legacy objectives are dominated by simulation.** Its objective
  duration was almost the same as simulation alone. However, line_profiler
  recorded 8.176 seconds at the G² aggregation call. This demonstrates why
  that source line cannot be read as pure aggregation cost: it can wait for
  the simulation work dispatched earlier. The earlier high aggregation-line
  shares also include dispatch/waits and must not be presented as isolated
  G² computation.
- **After B early stopping, objective overhead becomes more visible.** The
  combined path took 0.727 seconds for simulation and 1.029 seconds for the
  complete objective. For A, the complete objective also took substantially
  longer than simulation alone. These gaps identify a useful next experiment:
  compile the pure G² aggregation and measure the complete objective again.
  Subtracting separately measured workloads is only an approximate diagnostic;
  the objective and standalone simulator have different surrounding dispatch
  and condition handling.
- **K-L rebuilding remains a separate fitting concern.** An uncached basis
  still took about 0.88–0.90 seconds. These production profiles kept parameters
  fixed and reused the basis, so they do not establish its share of a real
  9,000-trial fit. The earlier changing-`sig` fit diagnostic showed actual
  rebuilds; its percentage shares cannot be extrapolated to larger fits.

```text
B legacy, 9,000 trials:       simulation dominates objective duration
                                         |
                                         v
B with early stopping:       less simulation work
                                         |
                                         v
                              remaining objective overhead is more visible

Changing sig during a fit:    can add a separate K-L basis rebuild
```

Only two repeats were used per workload, and instrumentation adds overhead,
especially to the faster paths. This follow-up establishes the profiling
diagnostics at the production trial count; use the separate repeated
scientific benchmarks for statistical speedup and fitting-quality claims.
Full provenance, first-call/warm-up timings and raw profiles are in
`.benchmark-results/profile-h100-production/` and its downloaded tar archive.

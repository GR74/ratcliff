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

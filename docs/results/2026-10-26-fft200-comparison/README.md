# FFT200 follow-up: stopped early at the user's request

The benchmark process has ended and the GPU was verified idle. The launcher
exited through its cleanup path, releasing its temporary sleep-prevention request.

**24 complete paired timing rounds** were preserved, out of 30
planned. The user requested stopping after reviewing the accumulating speed
results; the original protocol and all completed raw records remain unchanged.
The interrupted replay is excluded from timing estimates. Its last reported
position is preserved in `interrupted-status.json` in the raw directory and in
`replay-summary.json`. Per-call timings from the interrupted replay were not
flushed; its progress and interruption are retained in `run.log`.

| Method | Complete timed replays | Median seconds per 30-vector replay |
| --- | --- | --- |
| fft199 | 24 | 207.280 |
| fft200 | 24 | 165.891 |
| repaired_kl | 24 | 262.355 |
| bucketed_kl | 24 | 216.619 |

FFT200 used **19.97% less objective time** than FFT199. The
whole-replay paired bootstrap ratio interval is [0.7960, 0.8017].
Because stopping followed inspection of results, this is a **descriptive**
interval, not the originally planned fixed-sample confirmatory analysis or a
sequentially adjusted confidence interval. This is one frozen workload and one
GPU session; no general fitting-speed claim follows.

Observer parity passed for all four methods. The 17-case behavioral pilot is
complete, but qualification remains inconclusive: the precision planner needs
about 2.97 million trials per condition for its hardest estimable case, and
some zero-drift conditional RT groups are too sparse to plan reliably. This
exceeds the bounded design. See `pilot.md` and `pilot-summary.json`.

**Parameter recovery and matched fits were not run.** No production defaults
were changed. Exploratory recovery would be a separately identified follow-up;
it must not be represented as proof of behavioral equivalence.

Raw source snapshots, trials, complete replays, telemetry and logs remain under
`.benchmark-results/2026-10-26-fft200-comparison/`. GPU memory/power were sampled;
energy efficiency and true peak memory have not been analyzed. The earlier
covariance audit remains at `docs/results/2026-10-25-fft200-screen/`.

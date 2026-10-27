"""Record the user-authorized early stop without changing the frozen protocol."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.fft200_comparison_metrics import replay_summary


def main():
    root = Path(__file__).resolve().parents[1]
    raw = root / ".benchmark-results/2026-10-26-fft200-comparison"
    public = root / "docs/results/2026-10-26-fft200-comparison"
    previous = json.loads((raw / "status.json").read_text())
    interrupted = raw / "interrupted-status.json"
    if not interrupted.exists():
        interrupted.write_text(json.dumps(previous, indent=2)+"\n")
    original = json.loads(interrupted.read_text())
    records = [json.loads(p.read_text()) for p in (raw / "replays").glob("*.json") if not p.name.startswith("warmup")]
    result = replay_summary(records)
    result["fixed_sample_gate_status"] = result.pop("status")
    result.update(status="stopped_early_by_user", planned_pairs=30,
                  stopping_note="User requested stopping after reviewing the accumulating speed results. Intervals are descriptive, not a confirmatory fixed-sample or sequentially adjusted analysis.",
                  partial_replay_excluded=dict(method=original.get("method"), repetition=original.get("repetition"), vector=original.get("vector")),
                  controls={m: replay_summary(records, control=m) for m in ("repaired_kl", "bucketed_kl")})
    for comparison in result["controls"].values():
        comparison["fixed_sample_gate_status"] = comparison.pop("status")
        comparison["status"] = "descriptive_after_user_stop"
    previous.update(status="stopped_by_user", phase="finished_early",
                    updated_utc=datetime.now(timezone.utc).isoformat(),
                    termination="SIGINT requested by user; process exit and idle GPU verified.")
    previous["gates"].update(replay_speed="promising descriptive result; stopped before planned 30 rounds",
                             matched_fits="not run; behavioral qualification remains inconclusive")
    for directory in (raw, public):
        for name, value in [("status.json", previous), ("replay-summary.json", result)]:
            (directory / name).write_text(json.dumps(value, indent=2)+"\n")
    (public / "replay-progress.json").write_text(json.dumps(result, indent=2)+"\n")
    table = []
    for method in ("fft199", "fft200", "repaired_kl", "bucketed_kl"):
        values = [r["seconds"] for r in records if r["repeat"] >= 0 and r["method"] == method]
        table.append(f"| {method} | {len(values)} | {np.median(values):.3f} |")
    ratio, interval = result["ratio"], result["interval"]
    text = f"""# FFT200 follow-up: stopped early at the user's request

The benchmark process has ended and the GPU was verified idle. The launcher
exited through its cleanup path, releasing its temporary sleep-prevention request.

**{result['pairs']} complete paired timing rounds** were preserved, out of 30
planned. The user requested stopping after reviewing the accumulating speed
results; the original protocol and all completed raw records remain unchanged.
The interrupted replay is excluded from timing estimates. Its last reported
position is preserved in `interrupted-status.json` in the raw directory and in
`replay-summary.json`. Per-call timings from the interrupted replay were not
flushed; its progress and interruption are retained in `run.log`.

| Method | Complete timed replays | Median seconds per 30-vector replay |
| --- | --- | --- |
""" + "\n".join(table) + f"""

FFT200 used **{100*(1-ratio):.2f}% less objective time** than FFT199. The
whole-replay paired bootstrap ratio interval is [{interval[0]:.4f}, {interval[1]:.4f}].
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
"""
    (public / "README.md").write_text(text, encoding="utf-8")
    (public / "replay.md").write_text(text, encoding="utf-8")
    (raw / "early-stop-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

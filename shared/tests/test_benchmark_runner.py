"""The parent keeps failure records without needing a GPU worker."""
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_speedups


def test_worker_crash_is_checkpointed_and_dependent_fits_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_speedups.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1))
    args = benchmark_speedups.parser().parse_args([
        "--models", "a", "--allow-cpu", "--output", str(tmp_path),
        "--recovery-seeds", "1",
    ])
    assert benchmark_speedups.main(args) == 1
    rows = json.loads((tmp_path / "results.json").read_text())
    assert [r["status"] for r in rows] == ["error"] * 4 + ["skipped"] * 5
    assert (tmp_path / "REPORT.md").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["execution"]["completed_cases"] == 0


def test_existing_results_are_preserved(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("[]")
    args = benchmark_speedups.parser().parse_args(["--output", str(tmp_path)])
    with pytest.raises(ValueError, match="fresh output directory"):
        benchmark_speedups.main(args)
    assert path.read_text() == "[]"


def test_fit_only_followup_runs_selected_variants(tmp_path, monkeypatch):
    actions = []

    def failed_worker(command, **kwargs):
        action = command[command.index("--action") + 1]
        actions.append(action)
        # Keep the data prerequisite present to exercise fit dispatch.
        if action == "data":
            (tmp_path / "data-a.npz").touch()
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(benchmark_speedups.subprocess, "run", failed_worker)
    args = benchmark_speedups.parser().parse_args([
        "--models", "a", "--allow-cpu", "--output", str(tmp_path),
        "--fit-variants", "legacy,combined", "--skip-micro", "--recovery-seeds", "1",
    ])
    assert benchmark_speedups.main(args) == 1
    assert actions == ["data", "fit", "fit"]
    rows = json.loads((tmp_path / "results.json").read_text())
    assert [r["variant"] for r in rows] == ["legacy", "legacy", "combined"]

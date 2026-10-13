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

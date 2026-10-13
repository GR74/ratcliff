"""Do not turn a fast/finished fit, a CPU fallback or an empty run into a win."""
import json
from types import SimpleNamespace

import pytest

from scripts import validate_early_stopping as validation


def test_low_parameter_error_does_not_override_worse_independent_predictions():
    assert validation.meets_target(.04, 105., 100.)
    assert not validation.meets_target(.04, 134., 100.)
    assert not validation.meets_target(.08, 80., 100.)
    assert not validation.meets_target(.04, float("nan"), 100.)


def test_negative_g2_baseline_uses_an_additive_margin():
    assert validation.meets_target(.04, -9.5, -10.)
    assert not validation.meets_target(.04, -7., -10.)


def test_cpu_fallback_is_rejected_before_benchmarking(monkeypatch):
    import jax
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="required"):
        validation.worker(SimpleNamespace(allow_cpu=False, gpu_kind="3060"))


def test_budget_exhaustion_is_saved_without_claiming_completed_cases(tmp_path):
    args = validation.parser().parse_args([
        "--output", str(tmp_path), "--budget-seconds", "1", "--reserve-seconds", "2"])
    validation.main(args)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["recorded_cases"] == 0
    assert summary["planned_cases"] == 36
    assert summary["stopped_by_budget"]
    assert summary["fits"] == summary["scores"] == []


def test_memory_sized_study_keeps_extra_baseline_start_inside_saved_plan(tmp_path):
    args = validation.parser().parse_args([
        "--output", str(tmp_path), "--budget-seconds", "1", "--reserve-seconds", "2",
        "--starts", "1", "--baseline-extra-starts", "1"])
    validation.main(args)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["planned_cases"] == 28
    assert summary["stopped_by_budget"]

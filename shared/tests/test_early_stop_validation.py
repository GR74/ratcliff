"""Do not turn a fast/finished fit, a CPU fallback or an empty run into a win."""
import json
import hashlib
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


def test_repaired_run_plans_one_exact_paired_start_and_gpu_verification(tmp_path):
    args = validation.parser().parse_args([
        "--output", str(tmp_path), "--budget-seconds", "1", "--reserve-seconds", "2",
        "--scenarios", "baseline", "--start-seeds", "1", "--nsim", "9000",
        "--stable-kl-noise", "--score-nsim", "9000"])
    validation.main(args)
    plan = json.loads((tmp_path / "planned-jobs.json").read_text())
    assert plan[0]["action"] == "verify"
    fits = [p for p in plan if p["action"] == "fit"]
    assert [(p["start_seed"], p["variant"]) for p in fits] == [(1, "early_stop"), (1, "legacy")]
    config = json.loads((tmp_path / "run-config.json").read_text())
    assert config["nsim"] == config["score_nsim"] == 9000 and config["stable_kl_noise"]


def test_reusing_observations_checks_fingerprints_and_trial_count(tmp_path):
    source, out = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    out.mkdir()
    (source / "run-config.json").write_text(json.dumps(dict(data_nsim=9000)))
    payload = b"example-observations"
    (source / "data-baseline.npz").write_bytes(payload)
    record = dict(status="completed", scenario="baseline", data_nsim=9000,
                  data_seed=123, data_sha256=hashlib.sha256(payload).hexdigest())
    (source / "data-baseline-0-legacy.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="trial count"):
        validation.reuse_data(source, out, ["baseline"], 512)
    validation.reuse_data(source, out, ["baseline"], 9000)
    assert (out / "data-baseline.npz").read_bytes() == payload
    (source / "data-baseline.npz").write_bytes(b"corrupted-observations")
    with pytest.raises(ValueError, match="mismatched"):
        validation.reuse_data(source, out, ["baseline"], 9000)


def test_watchdog_matches_only_exact_benchmark_and_output():
    from scripts.guard_laptop_validation import matches_run
    script = "/repo/scripts/validate_early_stopping.py"
    output = "/results/ours"
    assert matches_run(["python", script, "--output", output], "/repo", script, output)
    assert matches_run(["python", "scripts/validate_early_stopping.py", "--output", output],
                       "/repo", script, output)
    assert not matches_run(["python", script, "--output", "/results/other"], "/repo", script, output)
    assert not matches_run(["python", "other.py", "--output", output], "/repo", script, output)
    assert not matches_run(["python", "scripts/validate_early_stopping.py", "--output", output],
                           "/another/repo", script, output)

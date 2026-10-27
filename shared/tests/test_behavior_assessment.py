"""Sequential coverage accounting, stopping and restart invariants."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import behavior_assessment_stats as stats
from scripts import run_behavior_assessment as runner


@pytest.fixture
def protocol():
    return json.loads((Path(__file__).resolve().parents[2]/"docs/plans/2026-10-26-behavioral-assessment-20-runs.json").read_text(encoding="utf-8-sig"))


def sample(n=1000, position=5):
    return dict(rt=np.linspace(200, 600, n), cat=np.arange(n)%5+1,
                crossed=np.ones(n, dtype=bool), positions=np.full(n, position, dtype=int), steps=np.ones(n, dtype=int))


def test_alpha_spending_and_no_unscheduled_peeks(protocol):
    assert sum(stats.look_alpha(protocol, n)*594 for n in (5, 10, 15, 20)) == pytest.approx(.05)
    with pytest.raises(ValueError, match="registered"):
        stats.look_alpha(protocol, 6)


def test_new_seeds_unique_across_round_method_case(protocol):
    seeds = [stats.seed_for(protocol, r, m, c) for r in range(20) for m in range(4) for c in range(17)]
    assert len(set(seeds)) == 20*4*17
    assert min(seeds) >= 2026102700 and max(seeds) < 2**32


def test_spatial_lower_detects_large_shift_and_handles_empty():
    a, b = sample(10000, 0), sample(10000, 15999)
    lo = stats.spatial_lower(a, b, 1e-5)
    distance = stats.base.spatial_distance(a, b, stats.base.projection_geometry())
    assert 2 < lo < distance
    b["crossed"][:] = False
    assert stats.spatial_lower(a, b, 1e-5) == 0


def test_spatial_lower_identical_positions_zero():
    assert stats.spatial_lower(sample(), sample(), .001) == 0


def test_all_endpoints_retained_and_sensitivity_cannot_stop(protocol):
    a, b = sample(), sample()
    b["cat"][:] = 1
    rows = stats.compare(a, b, [np.ones(16000, dtype=bool)]*3,
                         stats.look_alpha(protocol, 5), stats.base.projection_geometry())
    assert len(rows) == 35 and sum(r["registered"] for r in rows) == 33
    assert all(r["status"] == "diagnostic_only" for r in rows if r["metric"] in stats.SENSITIVITY)
    assert any(r["status"] == "inconclusive" for r in rows if r["metric"].startswith("rt_region_2"))
    assert stats.cell_status(rows) == "material_mismatch"


def test_stop_requires_every_core_cell_decisive(protocol):
    rows = [dict(case=c, condition=i, endpoints=[dict(registered=True, status="inconclusive") for _ in range(33)])
            for c in protocol["core_case_names"] for i in range(2)]
    rows[0]["endpoints"][0]["status"] = "fail"
    decision = stats.stop_decision(rows, protocol["core_case_names"])
    assert decision["status"] == "material_mismatch_in_core_scope" and not decision["stop"]
    for r in rows[1:]:
        for e in r["endpoints"]: e["status"] = "pass"
    assert stats.stop_decision(rows, protocol["core_case_names"])["stop"]
    with pytest.raises(ValueError): stats.stop_decision(rows[:-1], protocol["core_case_names"])


def test_all_pass_is_scoped_equivalence(protocol):
    rows = [dict(case=c, condition=i, endpoints=[dict(registered=True, status="pass") for _ in range(33)])
            for c in protocol["core_case_names"] for i in range(2)]
    assert stats.stop_decision(rows, protocol["core_case_names"])["status"] == "equivalent_in_core_scope"


def test_atomic_batch_roundtrip_and_corruption_rejection(tmp_path):
    path = tmp_path/"batch.npz"
    values = {k: np.stack([v, v]) for k, v in sample(10).items()}
    runner.save_batch(path, values, dict(seed=123, n=10))
    restored = runner.read_batch(path, 123, 10)
    for k in values: np.testing.assert_array_equal(restored[k], values[k])
    with pytest.raises(ValueError, match="different"): runner.read_batch(path, 124, 10)
    with pytest.raises(ValueError, match="overwrite"): runner.save_batch(path, values, dict(seed=123, n=10))
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Corrupt"): runner.read_batch(path, 123, 10)


def test_interrupted_uncommitted_batch_preserved_on_retry(tmp_path):
    path = tmp_path/"batch.npz"
    path.write_bytes(b"partial")
    assert runner.read_batch(path, 1, 2) is None
    runner.save_batch(path, dict(rt=np.array([1, 2])), dict(seed=1, n=2))
    assert len(list(tmp_path.glob("*.orphan-*.npz"))) == 1
    np.testing.assert_array_equal(runner.read_batch(path, 1, 2)["rt"], [1, 2])


def test_round_markers_require_contiguous_completion(tmp_path):
    assert runner.completed_rounds(tmp_path) == 0
    runner.write(tmp_path/"rounds/round-01.json", {})
    assert runner.completed_rounds(tmp_path) == 1
    runner.write(tmp_path/"rounds/round-03.json", {})
    with pytest.raises(ValueError, match="gap"): runner.completed_rounds(tmp_path)


def test_invalid_samples_fail_instead_of_disappearing():
    values = {k: np.stack([v, v]) for k, v in sample(10).items()}
    stats.validate_sample(values, 10)
    values["rt"][0, 1] = np.nan
    with pytest.raises(ValueError, match="Invalid RT"): stats.validate_sample(values, 10)


def test_figures_render_sparse_intervals_and_partial_history(tmp_path, protocol):
    from scripts import behavior_assessment_plots as plots
    root = Path(__file__).resolve().parents[2]
    plt = plots.initialize(root/".benchmark-results/kl-report-deps")
    a, b = sample(100), sample(100, 6)
    b["cat"][:] = 1
    endpoints = stats.compare(a, b, [np.ones(16000, dtype=bool)]*3,
                              stats.look_alpha(protocol, 5), stats.base.projection_geometry())
    records = [dict(case="sig5_ordinary", condition=0, method="fft200", rounds=5,
                    trials=100, endpoints=runner.clean(endpoints))]
    plots.case_figures(plt, a, b, tmp_path/"maps.png", "Rendering check")
    plots.comparison_panels(plt, records, tmp_path, "Sparse intervals rendering check")
    plots.precision_history(plt, [dict(records=records)], protocol["core_case_names"], tmp_path/"precision.png")
    assert all((tmp_path/name).stat().st_size > 1000 for name in ("maps.png", "effects.png", "precision.png"))

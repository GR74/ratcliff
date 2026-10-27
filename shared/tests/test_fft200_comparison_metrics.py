"""Check inferential edge cases and exact discrete spatial calculations."""
import numpy as np
from scipy.stats import wasserstein_distance

from scripts import fft200_comparison_metrics as m


def test_binary_extremes_and_difference_sign():
    assert m.binary_interval(0, 100, .05)[0] == 0
    assert m.binary_interval(100, 100, .05)[1] == 1
    interval = m.interval_difference(m.binary_interval(100, 1000, .025),
                                     m.binary_interval(500, 1000, .025))
    assert interval[0] > .3
    assert m.interval_status(interval, .005) == "fail"


def test_uncertain_is_not_pass_and_quantile_sparse_is_unbounded():
    assert m.interval_status((-.01, .01), .005) == "inconclusive"
    assert m.interval_status((-.001, .001), .005) == "pass"
    assert m.quantile_interval([], .5, .05) == (-np.inf, np.inf)
    assert not np.isfinite(m.quantile_interval([1., 2.], .9, .0001)).all()


def test_quantile_order_statistic_coverage_by_exact_binomial_probability():
    # For uniform observations, the true q is enclosed iff the binomial count
    # lies between the selected order-statistic indices (ties have zero mass).
    from scipy.stats import binom
    n, q, alpha = 1000, .9, .001
    lo = int(binom.ppf(alpha/2, n, q))
    hi = int(binom.ppf(1-alpha/2, n, q))
    coverage = binom.cdf(hi, n, q)-binom.cdf(lo-1, n, q)
    assert coverage >= 1-alpha


def test_spatial_grid_histogram_matches_direct_wasserstein():
    a = dict(positions=np.array([0, 1, 1, 161, 350, 15999]), crossed=np.array([1, 1, 1, 1, 1, 0], dtype=bool))
    b = dict(positions=np.array([5, 5, 160, 400]), crossed=np.ones(4, dtype=bool))
    angles = np.arange(64)*np.pi/64
    xy = lambda s: 2*np.column_stack([s["positions"][s["crossed"]] % 160, s["positions"][s["crossed"]] // 160])
    expected = np.mean([wasserstein_distance(xy(a) @ [np.cos(t), np.sin(t)], xy(b) @ [np.cos(t), np.sin(t)]) for t in angles])
    np.testing.assert_allclose(m.spatial_distance(a, b, m.projection_geometry()), expected, atol=1e-12)


def test_spatial_bound_respects_missing_samples_and_precision():
    assert np.isinf(m.spatial_upper(0, [0, 100], .01))
    small = m.spatial_upper(.1, [1000, 1000], .01)
    big = m.spatial_upper(.1, [1000000, 1000000], .01)
    assert .1 < big < 2 < small


def test_replay_uses_complete_pairs_and_excludes_first_replay():
    records = [dict(method=meth, repeat=i, seconds=s) for i in range(-1, 30)
               for meth, s in [("fft199", 10.), ("fft200", 8. if i >= 0 else 999.)]]
    result = m.replay_summary(records)
    assert result["pairs"] == 30 and result["status"] == "pass"
    assert result["ratio"] == .8
    assert m.replay_summary(records[:-1])["status"] == "inconclusive"


def test_low_support_excludes_no_crossing_placeholder():
    sample = dict(cat=np.array([1, 5]), crossed=np.array([True, False]), positions=np.array([0, 0]))
    vectors = m.probability_vectors(sample, [np.array([True])]*3)
    assert vectors["low_support_0.1"].tolist() == [True, False]
    assert vectors["non_crossing"].tolist() == [False, True]


def test_matched_fit_orchestration_keeps_common_data_and_scores_every_arm(tmp_path):
    """Exercise the fit/scoring plumbing with a tiny deterministic fake model."""
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[2] / "scripts/run_fft200_comparison.py"
    spec = importlib.util.spec_from_file_location("comparison_driver_test", source)
    driver = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path[:] = before
    truth = np.arange(1., 14.)
    protocol = dict(truth=truth.tolist(), x0=(truth*1.2).tolist(), data_seed=71,
                    calibration_seeds=list(range(20)), validation_seeds=list(range(20, 40)),
                    scoring_seconds_cap=60)
    out, public = tmp_path / "raw", tmp_path / "public"
    out.mkdir(); public.mkdir()
    calls, cold = [], []
    def sample(method, p, seed, n):
        calls.append((method, seed, n))
        # Tests orchestration, not sampling precision: all fake observations
        # have all five categories, matching quantiles and identical locations.
        cat = np.tile(np.arange(1, 6), 20)[None].repeat(2, axis=0)
        return dict(cat=cat, rt=(cat*10.+300), crossed=np.ones_like(cat, dtype=bool),
                    positions=cat.astype(int), steps=cat, work=np.zeros(2))
    def objective_call(method, p, data):
        assert data["quant"].shape == (2, 5, 5)
        return float(np.sum((p-truth)**2))
    def minimize(loss, x0, **kwargs):
        loss(np.array(x0)); loss(truth)
        return SimpleNamespace(success=True, message="test convergence")
    state = dict(gates={})
    driver.run_fits(out, public, protocol, state, lambda *a, **k: None,
                    sample, objective_call, SimpleNamespace(clamp_b=np.asarray), np, np,
                    minimize, None, SimpleNamespace(QUANTILES=m.QUANTILES, spatial_distance=lambda *a: 0.),
                    driver.write, lambda: cold.append(True))
    results = json.loads((public / "matched-fit-summary.json").read_text())
    assert [r["method"] for r in results] == list(driver.METHODS)
    assert len(cold) == 4
    assert all(r["first_acceptable_checkpoint"]["joint_pass"] for r in results)
    assert all(r["time_to_quality"] >= r["fit"]["training_seconds"] for r in results)
    assert all(c[0] == "fft199" for c in calls)
    assert len(calls) == 2+20+4*20  # data, reference, calibration, all four final scores

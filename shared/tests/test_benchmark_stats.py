import numpy as np
import pytest

from shared.benchmark_stats import bootstrap_speedup, compare_distributions


def test_constant_timing_ratio_and_paired_bootstrap():
    out = bootstrap_speedup([4., 4., 4.], [2., 2., 2.], paired=True)
    assert out["median_speedup"] == 2.
    assert out["bootstrap_95_ci"] == [2., 2.]


def test_invalid_timings_rejected():
    with pytest.raises(ValueError):
        bootstrap_speedup([0.], [1.])


def test_identical_paired_samples_have_zero_differences():
    rt = np.arange(1000.) + 200.
    cat = np.tile([1, 2], 500)
    out = compare_distributions(rt, cat, rt, cat, 2, paired=True)
    assert out["all_proportion_screens_pass"]
    assert out["all_rt_screens_pass"]
    assert all(r["proportion_difference_95_ci"] == [0., 0.]
               for r in out["by_category"])


def test_changed_choices_fail_and_empty_categories_are_not_certified():
    rt = np.arange(1000.) + 200.
    out = compare_distributions(rt, np.ones(1000), rt, np.full(1000, 2), 2,
                                 paired=True)
    assert not out["all_proportion_screens_pass"]
    assert not out["all_rt_screens_pass"]
    assert all(r["rt_screen"] == "insufficient category samples"
               for r in out["by_category"])


def test_large_rt_shift_is_detected():
    rt = np.arange(1000.) + 200.
    cat = np.ones(1000)
    out = compare_distributions(rt, cat, rt * 1.2, cat, 1, paired=True)
    assert not out["all_rt_screens_pass"]

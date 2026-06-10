"""Checkpoint 1 tests — DDM agents decide sensibly. Pure NumPy, runs in <1s."""
import numpy as np
import pytest

from cognitive_society.agent import DDMAgent, DDMParams, make_population


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_style_labels():
    assert DDMParams(boundary=1.6).style() == "cautious"
    assert DDMParams(boundary=0.5).style() == "decisive"
    assert DDMParams(boundary=1.0).style() == "balanced"


def test_decide_returns_choice_and_rt():
    a = DDMAgent(DDMParams())
    choice, rt = a.decide(0.5, _rng())
    assert choice in (0, 1)
    assert rt >= DDMParams().ndt


def test_positive_evidence_favors_choice_1():
    a = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0))
    choices, _ = a.decide_batch(1.0, 2000, _rng(1))
    assert choices.mean() > 0.8, "strong positive evidence should mostly pick 1"


def test_negative_evidence_favors_choice_0():
    a = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0))
    choices, _ = a.decide_batch(-1.0, 2000, _rng(2))
    assert choices.mean() < 0.2, "strong negative evidence should mostly pick 0"


def test_zero_evidence_is_unbiased():
    a = DDMAgent(DDMParams(boundary=1.0))
    choices, _ = a.decide_batch(0.0, 4000, _rng(3))
    assert 0.4 < choices.mean() < 0.6, "ambiguous evidence ~50/50"


def test_decisive_is_faster_than_cautious():
    decisive = DDMAgent(DDMParams(boundary=0.5))
    cautious = DDMAgent(DDMParams(boundary=1.6))
    _, rt_d = decisive.decide_batch(0.5, 2000, _rng(4))
    _, rt_c = cautious.decide_batch(0.5, 2000, _rng(5))
    assert rt_d.mean() < rt_c.mean(), "lower boundary commits faster"


def test_cautious_is_more_accurate_than_decisive():
    # With the same nonzero evidence (truth = choice 1), higher boundary should
    # accumulate more evidence before committing -> higher accuracy.
    decisive = DDMAgent(DDMParams(boundary=0.5))
    cautious = DDMAgent(DDMParams(boundary=1.6))
    acc_d = decisive.decide_batch(0.4, 3000, _rng(6))[0].mean()
    acc_c = cautious.decide_batch(0.4, 3000, _rng(7))[0].mean()
    assert acc_c > acc_d, "more cautious -> more accurate at the same evidence"


def test_rt_never_below_ndt():
    a = DDMAgent(DDMParams(ndt=0.3))
    _, rts = a.decide_batch(0.2, 1000, _rng(8))
    assert (rts >= 0.3 - 1e-9).all()


def test_deterministic_for_same_seed():
    a = DDMAgent(DDMParams())
    c1, r1 = a.decide_batch(0.3, 500, _rng(11))
    c2, r2 = a.decide_batch(0.3, 500, _rng(11))
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(r1, r2)


def test_batch_matches_single_in_aggregate():
    a = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0))
    # batch
    batch_choices, _ = a.decide_batch(0.6, 1500, _rng(20))
    # singles
    rng = _rng(21)
    single = np.array([a.decide(0.6, rng)[0] for _ in range(1500)])
    # same agent, same evidence -> similar choice proportion (different seeds,
    # so allow a tolerance)
    assert abs(batch_choices.mean() - single.mean()) < 0.08


def test_make_population_assembles_heterogeneous_agents():
    pop = make_population([
        ("decisive", DDMParams(boundary=0.5)),
        ("cautious", DDMParams(boundary=1.6)),
        DDMParams(boundary=1.0),
    ])
    assert len(pop) == 3
    assert pop[0].params.style() == "decisive"
    assert pop[1].params.style() == "cautious"
    assert pop[2].agent_id == 2

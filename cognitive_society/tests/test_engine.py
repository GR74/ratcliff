"""Engine-interface tests — the 1D and 2D engines are interchangeable.

The fast conformance checks run in <1s. The actual 2D-spatial run is marked slow
(it invokes the JAX simulator) so it never blocks routine test runs.
"""
import numpy as np
import pytest

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.engine import DecisionEngine, SpatialDDMEngine, is_decision_engine


SPATIAL_PARAMS = dict(
    ter=200.0, st=50.0, cr=10.0, crsd=2.0,
    av1=15.0, av2=10.0, av3=8.0,
    sis=12.0, sig=10.0, si=6.0,
)


def test_1d_agent_satisfies_engine_contract():
    a = DDMAgent(DDMParams())
    assert is_decision_engine(a)
    assert isinstance(a, DecisionEngine)  # runtime_checkable Protocol
    assert a.n_choices == 2


def test_spatial_engine_satisfies_engine_contract():
    eng = SpatialDDMEngine(SPATIAL_PARAMS)
    assert is_decision_engine(eng)
    assert isinstance(eng, DecisionEngine)
    assert eng.n_choices == 5


def test_spatial_engine_rejects_missing_params():
    bad = {k: v for k, v in SPATIAL_PARAMS.items() if k != "sig"}
    with pytest.raises(ValueError, match="missing"):
        SpatialDDMEngine(bad)


def test_society_code_can_be_engine_agnostic():
    """A function written against the contract runs on either engine without
    knowing which it got (here we only exercise the 1D engine for speed)."""
    def run_one(engine, rng):
        # depends only on the contract
        choice, rt = engine.decide(0.5, rng)
        assert 0 <= choice < engine.n_choices
        assert rt >= 0
        return choice, rt

    run_one(DDMAgent(DDMParams()), np.random.default_rng(0))


@pytest.mark.slow
def test_spatial_engine_actually_decides():
    """Real 2D-spatial decisions map to the (choice 0..4, rt seconds) contract."""
    eng = SpatialDDMEngine(SPATIAL_PARAMS, chunk_size=8, use_kl=False)
    choices, rts = eng.decide_batch(None, 16, np.random.default_rng(0))
    assert choices.shape == (16,)
    assert ((choices >= 0) & (choices < 5)).all()
    assert (rts > 0).all()
    assert np.isfinite(rts).all()

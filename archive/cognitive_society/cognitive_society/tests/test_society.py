"""Checkpoints 3+4 tests — the integrated society. Pure NumPy, fast."""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import (
    Society, SocietyConfig, cfg_flat, cfg_outcome_trust, cfg_full,
)


def _honest_and_adversary():
    honest = [
        DDMAgent(DDMParams(boundary=1.0, drift_scale=1.1, ndt=0.2)),
        DDMAgent(DDMParams(boundary=1.3, drift_scale=1.0, ndt=0.22)),
        DDMAgent(DDMParams(boundary=0.9, drift_scale=1.2, ndt=0.19)),
    ]
    adversary = DDMAgent(DDMParams(boundary=0.5, drift_scale=-1.3, ndt=0.16))
    return honest + [adversary]


def test_society_constructs():
    soc = Society(_honest_and_adversary(), config=cfg_full(), rng_seed=0)
    assert soc.K == 4
    assert len(soc.trust) == 4


def test_cognitive_map_flags_adversary_as_incompetent():
    soc = Society(_honest_and_adversary(), config=cfg_full(), rng_seed=1)
    comp = soc.build_cognitive_maps()
    # honest agents (indices 0-2) more competent than the adversary (index 3)
    assert comp[:3].mean() > comp[3]
    assert comp[3] < 0.5, "confidently-wrong adversary has sub-chance accuracy"


def test_round_returns_expected_keys():
    soc = Society(_honest_and_adversary(), config=cfg_full(), rng_seed=2)
    soc.build_cognitive_maps()
    r = soc.round(0.4)
    assert set(r) >= {"final", "majority", "truth", "correct"}
    assert r["truth"] == 1
    assert r["majority"] in (0, 1)


def test_full_beats_flat_under_adversaries():
    """The headline: cognitive-map-grounded + gated trust is more robust than
    flat broadcast when confidently-wrong agents are present."""
    agents_spec = lambda: _honest_and_adversary() + [
        DDMAgent(DDMParams(boundary=0.5, drift_scale=-1.3, ndt=0.16))
    ]  # 3 honest + 2 adversaries
    rng = np.random.default_rng(5)
    evidences = rng.choice([-0.4, -0.3, 0.3, 0.4], size=30).tolist()

    flat_acc = Society(agents_spec(), config=cfg_flat(), rng_seed=5).run(evidences)[
        "collective_accuracy"]
    full_acc = Society(agents_spec(), config=cfg_full(), rng_seed=5).run(evidences)[
        "collective_accuracy"]
    assert full_acc > flat_acc, f"full {full_acc:.2f} should beat flat {flat_acc:.2f}"


def test_trust_down_weights_adversary_after_rounds():
    soc = Society(_honest_and_adversary(), config=cfg_full(), rng_seed=7)
    soc.build_cognitive_maps()
    for ev in [0.4, -0.3, 0.3, 0.4, -0.4, 0.3]:
        soc.round(ev)
    # observer 0's trust in the adversary (index 3) should be low
    t0 = soc.trust[0].trust()
    assert t0[3] < t0[:3].mean(), "adversary should end up less trusted than honest peers"


def test_named_configs_set_expected_flags():
    assert cfg_flat().use_trust_weights is False
    assert cfg_outcome_trust().use_trust_weights is True
    assert cfg_outcome_trust().use_competence_prior is False
    assert cfg_full().use_competence_prior is True
    assert cfg_full().adaptive is True

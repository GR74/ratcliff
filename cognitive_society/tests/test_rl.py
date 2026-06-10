"""Checkpoint 4 (RL) tests — the learned deference policy. Pure NumPy.

The REINFORCE *convergence* sanity tests run many iterations and are marked slow;
the fast suite keeps the cheap range/integration/safety checks.
"""
import numpy as np
import pytest

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.rl import DeferencePolicy
from cognitive_society.society import Society, cfg_full, cfg_rl


def test_action_in_range_and_neutral_at_init():
    pol = DeferencePolicy(m_max=2.0, seed=0)
    phi = pol.features(0.5, 0.5, 0.5)
    # untrained (theta = 0) -> greedy action is the neutral midpoint m_max/2
    assert abs(pol.act_greedy(phi) - 1.0) < 1e-9
    for _ in range(50):
        m = pol.act(phi)
        assert 0.0 < m < 2.0


def test_features_shape():
    phi = DeferencePolicy.features(0.3, 0.7, 0.1)
    assert phi.shape == (DeferencePolicy.N_FEATURES,)
    assert phi[-1] == 1.0  # bias term


@pytest.mark.slow
def test_reinforce_increases_a_rewarded_action():
    """Contextual-free bandit where reward rises with the action -> the policy
    should learn to push its action up (REINFORCE gradient sign is correct)."""
    pol = DeferencePolicy(m_max=2.0, sigma=0.5, lr=0.1, seed=1)
    phi = pol.features(0.0, 0.0, 0.0)  # only the bias feature is active
    start = pol.act_greedy(phi)
    for _ in range(400):
        m = pol.act(phi)
        pol.update([m / pol.m_max])  # reward = normalized action in [0,1]
    end = pol.act_greedy(phi)
    assert end > start + 0.3, f"action should rise toward m_max: {start:.2f}->{end:.2f}"


@pytest.mark.slow
def test_reinforce_learns_context_dependence():
    """Reward = defer-more-when-trust-high. The policy should end up using a
    LARGER action when mean_trust is high than when it is low."""
    pol = DeferencePolicy(m_max=2.0, sigma=0.5, lr=0.1, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(1500):
        trust = float(rng.uniform(0, 1))
        phi = pol.features(0.5, trust, 0.5)
        m = pol.act(phi)
        # good policy: action m should track trust. reward high when m and trust agree.
        target = trust * pol.m_max
        reward = 1.0 - abs(m - target) / pol.m_max  # in [0,1], max when m == target
        pol.update([reward])
    hi = pol.act_greedy(pol.features(0.5, 0.9, 0.5))
    lo = pol.act_greedy(pol.features(0.5, 0.1, 0.5))
    assert hi > lo + 0.2, f"defer more under high trust: lo={lo:.2f} hi={hi:.2f}"


def test_update_requires_aligned_rewards():
    pol = DeferencePolicy(seed=3)
    pol.act(pol.features(0.5, 0.5, 0.5))
    pol.act(pol.features(0.5, 0.5, 0.5))
    try:
        pol.update([1.0])  # 1 reward for 2 buffered actions
        assert False, "should have raised on misaligned rewards"
    except ValueError:
        pass


def _society(use_rl, seed=0):
    agents = [DDMAgent(DDMParams(boundary=1.0, drift_scale=1.1, ndt=0.2)) for _ in range(5)]
    # small map_trials: these are integration/safety checks, not accuracy checks.
    if use_rl:
        return Society(agents, config=cfg_rl(map_trials=300), rng_seed=seed,
                       policy=DeferencePolicy(seed=seed))
    return Society(agents, config=cfg_full(map_trials=300), rng_seed=seed)


def test_society_with_rl_runs_and_learns():
    soc = _society(use_rl=True, seed=4)
    theta0 = soc.policy.theta.copy()
    res = soc.run([0.4, -0.3, 0.3, 0.4, -0.4] * 4, learn=True)
    assert 0.0 <= res["collective_accuracy"] <= 1.0
    # learning happened -> the policy parameters moved off their zero init
    assert not np.allclose(soc.policy.theta, theta0)


def test_rl_eval_mode_does_not_learn():
    soc = _society(use_rl=True, seed=5)
    theta0 = soc.policy.theta.copy()
    soc.run([0.4, -0.3, 0.3] * 5, learn=False)
    assert np.allclose(soc.policy.theta, theta0), "learn=False must not update the policy"


def test_rl_flag_off_leaves_policy_untouched():
    # cfg_full (use_rl_policy False) with a policy attached must ignore it entirely
    agents = [DDMAgent(DDMParams(boundary=1.0, drift_scale=1.1, ndt=0.2)) for _ in range(5)]
    pol = DeferencePolicy(seed=6)
    theta0 = pol.theta.copy()
    soc = Society(agents, config=cfg_full(), rng_seed=6, policy=pol)
    soc.run([0.4, -0.3, 0.3, 0.4], learn=True)
    assert np.allclose(pol.theta, theta0), "policy must be untouched when use_rl_policy is False"

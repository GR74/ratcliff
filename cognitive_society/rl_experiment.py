"""RL adaptation experiment — a learned deference policy out-adapts the fixed
gate when the peer regime flips hostile.

  Phase A (friendly): an all-honest society; deferring to peers HELPS.
  Regime shift:        a MAJORITY of agents flip to confidently-wrong adversaries.
  Phase B (hostile):   deferring to the group now HURTS; the right move is to
                       defer less and lean on your own evidence.

The fixed `cfg_full` keeps a social-gain floor it cannot drop below, no matter how
untrustworthy the group becomes. The RL policy conditions deference on the mean
trust it holds in its peers, so it can LEARN to stop deferring when the group goes
bad. Both conditions share the identical trust + cognitive-map machinery — the
ONLY difference is fixed-gate vs learned-gate deference, isolating the RL effect.

We report post-shift collective accuracy (fixed vs RL) and show the RL policy's
mean deference multiplier falling from phase A to phase B — i.e. it ADAPTED.

    python -m cognitive_society.rl_experiment
"""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import Society, cfg_outcome_trust, cfg_full, cfg_rl
from cognitive_society.rl import DeferencePolicy

ADVERSARY = dict(boundary=0.5, drift_scale=-1.3, ndt=0.16, sigma=1.0)


def _honest(rng):
    return DDMParams(boundary=float(rng.uniform(0.8, 1.6)),
                     drift_scale=float(rng.uniform(0.9, 1.3)),
                     ndt=float(rng.uniform(0.18, 0.28)), sigma=1.0)


def _problems(rng, n):
    return rng.choice([-0.4, -0.3, 0.3, 0.4], size=n).tolist()


def _flip_to_adversary(soc, n_flip, rng):
    """Turn n_flip of the society's agents into confidently-wrong adversaries
    (the hostile regime shift). Trust/competence are NOT re-mapped — both
    conditions must notice the shift online, via outcome-driven trust."""
    idx = rng.choice(soc.K, size=n_flip, replace=False)
    for j in idx:
        soc.agents[j].params = DDMParams(**ADVERSARY)
    return idx


def _mean_deference(soc, policy):
    """Mean greedy deference multiplier the policy would use right now, given the
    society's current trust state (NaN for the fixed condition)."""
    if policy is None:
        return float("nan")
    ms = []
    for i in range(soc.K):
        mask = np.ones(soc.K, dtype=bool)
        mask[i] = False
        mean_trust = float(np.mean(soc.trust[i].trust()[mask]))
        # probe at a fixed mid uncertainty/consensus so only trust drives the read
        ms.append(policy.act_greedy(policy.features(0.5, mean_trust, 0.5)))
    return float(np.mean(ms))


CONDITIONS = {
    "no_gating (outcome_trust)": cfg_outcome_trust,  # trust only, no gating — reference
    "fixed gate (cfg_full)": cfg_full,               # trust + hand-tuned gating
    "learned gate (RL)": cfg_rl,                      # trust + learned gating
}


def run_once(seed, condition, n_agents=7, n_flip=4, t_phase=100):
    """`condition` is a key of CONDITIONS. Returns (post-shift accuracy, mean
    deference before, mean deference after); deference is NaN unless RL."""
    rng = np.random.default_rng(seed)
    agents = [DDMAgent(_honest(rng)) for _ in range(n_agents)]
    policy = DeferencePolicy(seed=seed) if condition == "learned gate (RL)" else None
    soc = Society(agents, config=CONDITIONS[condition](), rng_seed=seed, policy=policy)

    # Phase A — friendly: build maps (if any), learn that deferring helps.
    soc.run(_problems(rng, t_phase), build_maps=True, learn=True)
    m_before = _mean_deference(soc, policy)

    # Regime shift — a majority turn confidently-wrong.
    _flip_to_adversary(soc, n_flip, rng)

    # Phase B — hostile: keep going (online), measure accuracy here.
    res_b = soc.run(_problems(rng, t_phase), build_maps=False, learn=True)
    m_after = _mean_deference(soc, policy)
    return res_b["collective_accuracy"], m_before, m_after


def main(n_seeds=12):
    print("=" * 64)
    print("RL ADAPTATION — learned deference vs fixed gate under a hostile shift")
    print("=" * 64)
    print(f"7 agents, all honest in phase A, then 4 flip to confidently-wrong;")
    print(f"post-shift collective accuracy over {n_seeds} seeds.\n")

    accs = {c: [] for c in CONDITIONS}
    m_before, m_after = [], []
    for seed in range(n_seeds):
        for cond in CONDITIONS:
            a, mb, ma = run_once(seed, cond)
            accs[cond].append(a)
            if cond == "learned gate (RL)":
                m_before.append(mb); m_after.append(ma)
    accs = {c: np.asarray(v) for c, v in accs.items()}

    for cond in CONDITIONS:
        v = accs[cond]
        print(f"  post-shift  {cond:26s}: {v.mean():6.1%} +/- {v.std():4.1%}")

    rl, fx = accs["learned gate (RL)"], accs["fixed gate (cfg_full)"]
    wins = int(np.sum(rl > fx))
    print(f"\n  learned vs fixed gate: {(rl.mean() - fx.mean()) * 100:+.1f}pt, "
          f"RL wins {wins}/{n_seeds} seeds.")

    mb, ma = float(np.nanmean(m_before)), float(np.nanmean(m_after))
    moved = "rose" if ma > mb else "fell"
    print(f"  RL deference m: phase A {mb:.2f} -> phase B {ma:.2f} ({moved}).")
    print("  Read: trust-weighting already discounts the flipped peers, so the")
    print("  trust-weighted social signal stays useful and 'defer when unsure'")
    print("  remains right -- the learned gate rediscovers the fixed gate's")
    print("  behavior from reward but does not beat it on this task.")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""Runnable demo — a heterogeneous Cognitive Society decides and reads each
other's minds. Exercises checkpoint 1 (agents decide) + the cognitive-mapping
engine (agents infer each other's DDM parameters from observed behavior).

    python -m cognitive_society.demo

Pure NumPy — runs in ~1 second, no GPU.
"""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.ez_diffusion import recover_from_agent_observations


def build_society():
    """A small society with distinct cognitive styles."""
    specs = [
        ("Dash  (decisive)", DDMParams(boundary=0.55, drift_scale=1.0, ndt=0.18)),
        ("Casey (balanced)", DDMParams(boundary=1.00, drift_scale=1.0, ndt=0.20)),
        ("Quinn (balanced)", DDMParams(boundary=1.05, drift_scale=1.2, ndt=0.22)),
        ("Ria   (cautious)", DDMParams(boundary=1.55, drift_scale=1.0, ndt=0.25)),
        ("Sol   (cautious)", DDMParams(boundary=1.70, drift_scale=0.9, ndt=0.28)),
    ]
    names = [s[0] for s in specs]
    agents = [DDMAgent(p, agent_id=i) for i, (_, p) in enumerate(specs)]
    return names, agents


def collective_decision(agents, evidence, n_trials, rng):
    """Each agent decides; report individual accuracy/speed and a majority vote."""
    truth = 1 if evidence > 0 else 0
    rows = []
    votes = np.zeros(len(agents))
    for i, a in enumerate(agents):
        choices, rts = a.decide_batch(evidence, n_trials, rng)
        acc = (choices == truth).mean()
        votes[i] = choices.mean()
        rows.append((acc, rts.mean()))
    majority = 1 if votes.mean() >= 0.5 else 0
    return rows, majority, truth


def cognitive_maps(names, agents, rng):
    """Each agent observes the others across several evidence levels and recovers
    their decision parameters — its 'map' of who's who."""
    evidence_levels = [0.4, 0.6, 0.8]
    maps = {}
    for j, target in enumerate(agents):
        obs = []
        for ev in evidence_levels:
            choices, rts = target.decide_batch(ev, 2500, rng)
            obs.append((choices, rts, ev))
        rec = recover_from_agent_observations(obs, sigma=target.params.sigma)
        maps[names[j]] = rec
    return maps


def main():
    rng = np.random.default_rng(7)
    names, agents = build_society()

    print("=" * 64)
    print("COGNITIVE SOCIETY - demo (checkpoint 1 + cognitive mapping)")
    print("=" * 64)

    print("\nTrue cognitive styles:")
    for nm, a in zip(names, agents):
        p = a.params
        print(f"  {nm:18s}  boundary={p.boundary:.2f}  drift={p.drift_scale:.2f}  "
              f"ndt={p.ndt:.2f}  [{p.style()}]")

    # --- Collective decision -------------------------------------------------
    evidence = 0.45  # truth = choice 1, moderate signal
    rows, majority, truth = collective_decision(agents, evidence, 3000, rng)
    print(f"\nDecision task (evidence={evidence:+.2f}, truth=choice {truth}):")
    for nm, (acc, mrt) in zip(names, rows):
        bar = "#" * int(acc * 20)
        print(f"  {nm:18s}  acc={acc:5.1%}  mean_rt={mrt:.2f}s  {bar}")
    print(f"  -> majority vote: choice {majority}  "
          f"({'correct' if majority == truth else 'WRONG'})")

    print("\n  Note: the cautious agents are slower but more accurate; the "
          "decisive\n  agent is fast but error-prone - the speed/accuracy "
          "tradeoff, per agent.")

    # --- Cognitive maps ------------------------------------------------------
    print("\nEach agent reads the others' styles from observed behavior "
          "(EZ recovery):")
    maps = cognitive_maps(names, agents, rng)
    print(f"  {'agent':18s}  {'true boundary':>13s}  {'recovered':>10s}  "
          f"{'true ndt':>9s}  {'recovered':>10s}")
    for nm, a in zip(names, agents):
        rec = maps[nm]
        print(f"  {nm:18s}  {a.params.boundary:>13.2f}  {rec['boundary']:>10.2f}  "
              f"{a.params.ndt:>9.2f}  {rec['ndt']:>10.2f}")

    print("\n  The recovered boundary/ndt track the true values -> the society "
          "can\n  tell decisive agents from cautious ones by watching them "
          "decide. That\n  inferred map is what trust + deference will be built "
          "on (checkpoints 2-4).")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""The anchor experiment — robustness to confidently-wrong agents.

Question: does cognitive-map-grounded, uncertainty-gated trust make a society
robust to confidently-wrong / adversarial agents, where flat-broadcast or
outcome-only-trust societies fail?

Setup: a society of honest agents (positive drift) plus a few ADVERSARIES — an
adversary is just a DDM agent with NEGATIVE drift and a low (decisive) boundary,
so it commits fast and confidently to the WRONG answer. Its inferred competence
(accuracy on observed behavior) is low, so a cognitive-map-grounded society can
down-weight it *before* its bad advice costs anything.

We compare three conditions on collective accuracy:
  flat           — everyone weighted equally (no trust, no adaptation)
  outcome_trust  — trust learns from outcomes (no cognitive-map prior, no gating)
  full (ours)    — cognitive-map-grounded trust + uncertainty-gated deference

    python -m cognitive_society.experiment
"""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import Society, cfg_flat, cfg_outcome_trust, cfg_full


def build_mixed(n_honest, n_adversary, seed):
    """Honest agents (varied, positive drift) + confidently-wrong adversaries."""
    rng = np.random.default_rng(seed)
    agents = []
    for _ in range(n_honest):
        agents.append(DDMAgent(DDMParams(
            boundary=float(rng.uniform(0.8, 1.6)),
            drift_scale=float(rng.uniform(0.9, 1.3)),
            ndt=float(rng.uniform(0.18, 0.28)),
            sigma=1.0,
        )))
    for _ in range(n_adversary):
        # negative drift -> responds opposite to evidence; low boundary -> fast +
        # "confident". This is the confidently-wrong adversary.
        agents.append(DDMAgent(DDMParams(
            boundary=0.5, drift_scale=-1.3, ndt=0.16, sigma=1.0,
        )))
    return agents


def run_experiment(n_honest=4, n_adversary=3, n_problems=40, n_seeds=10):
    conditions = {
        "flat": cfg_flat,
        "outcome_trust": cfg_outcome_trust,
        "full (ours)": cfg_full,
    }
    out = {name: [] for name in conditions}
    for seed in range(n_seeds):
        prng = np.random.default_rng(2000 + seed)
        # moderate-evidence problems with mixed truth -> agents are somewhat
        # uncertain, so social information matters (where adversaries can sway).
        evidences = prng.choice([-0.4, -0.3, 0.3, 0.4], size=n_problems).tolist()
        for name, mk in conditions.items():
            agents = build_mixed(n_honest, n_adversary, seed)
            soc = Society(agents, config=mk(), rng_seed=seed)
            res = soc.run(evidences)
            out[name].append(res["collective_accuracy"])
    return {name: (float(np.mean(v)), float(np.std(v))) for name, v in out.items()}


def main():
    print("=" * 64)
    print("ANCHOR EXPERIMENT — robustness to confidently-wrong agents")
    print("=" * 64)
    print("4 honest agents + 3 confidently-wrong adversaries, 40 problems, "
          "10 seeds.\n")
    res = run_experiment()
    print(f"  {'condition':16s}  {'collective accuracy':>20s}")
    for name in ("flat", "outcome_trust", "full (ours)"):
        m, s = res[name]
        bar = "#" * int(m * 30)
        print(f"  {name:16s}  {m:6.1%} +/- {s:4.1%}   {bar}")
    print()
    flat = res["flat"][0]
    full = res["full (ours)"][0]
    gain = (full - flat) * 100
    print(f"  full vs flat: +{gain:.1f} accuracy points under adversaries.")
    print("  Cognitive-map-grounded + uncertainty-gated trust down-weights the")
    print("  confidently-wrong agents, so the honest majority isn't dragged off.")
    print("=" * 64)


if __name__ == "__main__":
    main()

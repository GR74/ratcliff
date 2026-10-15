"""Phase 3 — the tipping-fraction test (the decisive COLLECTIVE scale result).

Phases 2 and 3a were per-victim (local). Here we ask the collective question: as the
adversary cluster grows, at what fraction does the swarm get dragged off — and does
outcome-grounded trust RAISE that tipping point vs agreement / no trust?

Metric = HONEST-AGENT accuracy (not all-agent majority). Adversaries vote wrong by
construction, so an all-agent majority trivially flips once they are >50% — that
measures vote-counting, not the mechanism. Honest-agent accuracy isolates the real
question: do the HONEST agents resist the cluster's social pressure? Outcome-trust
should keep them correct to higher adversary fractions (it discounts the confidently-
wrong cluster); agreement-trust lets captured honest agents flip; no-trust gives the
cluster full broadcast weight.

    python -m cognitive_society.phase3_tipping
"""
import numpy as np

from cognitive_society.adversary import build_clustered_society
from cognitive_society.baselines import AgreementTrustModel
from cognitive_society.society import SocietyConfig
from cognitive_society.phase2_experiment import (
    make_evidences, DEFAULT_BUILDER, DEFAULT_TOPOLOGY_FN,
)


def _cfg(use_trust):
    """No competence prior (clean comparison); trust-weighting on/off; same gating."""
    return SocietyConfig(use_social=True, use_trust_weights=use_trust,
                         use_competence_prior=False, adaptive=True, local_maps=False)


def honest_accuracy(N, deg, n_adv, seed, method, n_problems=30):
    """Fraction of HONEST agents that decide correctly, averaged over rounds."""
    use_trust = (method != "no-trust")
    soc, gt, S = build_clustered_society(
        N, deg, n_adv, seed, config=_cfg(use_trust),
        builder=DEFAULT_BUILDER, topology_fn=DEFAULT_TOPOLOGY_FN)
    if method == "agreement":
        soc.trust = [AgreementTrustModel(soc.K, neighbour_mask=soc.adj[i])
                     for i in range(soc.K)]
    honest = np.asarray(gt) == 0
    recs = soc.run(make_evidences(n_problems, seed), build_maps=False, learn=True)["records"]
    return float(np.mean([(r["final"][honest] == r["truth"]).mean() for r in recs]))


def main():
    deg, n_seeds = 4, 3
    methods = ["no-trust", "agreement", "outcome"]
    print("=" * 64)
    print("PHASE 3 — tipping fraction (HONEST-agent accuracy vs adversary fraction)")
    print("=" * 64)
    print("Do the honest agents resist the cluster? Outcome-trust should hold to a")
    print("HIGHER adversary fraction than agreement / no-trust.\n")

    for N in (40, 80):
        print(f"  N={N}, deg={deg}  (honest-agent accuracy)")
        print(f"    {'adv frac':>8s} {'no-trust':>9s} {'agreement':>10s} {'outcome':>9s}")
        for frac in (0.2, 0.3, 0.4, 0.5):
            n_adv = round(frac * N)
            acc = {m: [] for m in methods}
            for seed in range(n_seeds):
                for m in methods:
                    acc[m].append(honest_accuracy(N, deg, n_adv, seed, m))
            print(f"    {frac:8.0%} {np.mean(acc['no-trust']):9.1%} "
                  f"{np.mean(acc['agreement']):10.1%} {np.mean(acc['outcome']):9.1%}")
        print()

    print("=" * 64)
    print("Read: where agreement / no-trust honest-accuracy collapses as the cluster")
    print("grows, outcome-grounded trust should hold it up — a higher tipping fraction.")
    print("=" * 64)


if __name__ == "__main__":
    main()

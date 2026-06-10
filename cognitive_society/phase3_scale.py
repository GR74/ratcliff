"""Phase 3 — does the Phase-2 per-victim isolation hold as the swarm SCALES?

Phase 2 (per captured victim, no truth-prior on either arm): outcome-grounded
trust isolates the colluding cluster at AUC ~1.0 while agreement-based trust
INVERTS to ~0.0 once a node is majority-captured. That is a LOCAL property — each
victim scores its OWN trust over its OWN neighbours — so the prediction is that it
is ~N-INVARIANT: growing the swarm at fixed degree + fixed adversary fraction
should not change the per-victim verdict. This experiment tests that prediction,
which is the honest core of "does it scale": the mechanism is local, so it should
hold at any N (and that's a feature — no global controller, scale-free).

Sweep N at fixed degree (4) and a fixed contiguous-cluster fraction; report the
per-victim outcome vs agreement AUC at each N (averaged over seeds), with the count
of captured victims (which should GROW with N as the cluster boundary grows).

    python -m cognitive_society.phase3_scale
"""
import numpy as np

from cognitive_society.phase2_per_victim import build_and_run, victim_aucs


def per_victim_at_N(N, deg, n_adv, n_seeds, n_problems=30):
    o_aucs, a_aucs, nv = [], [], 0
    for seed in range(n_seeds):
        soc_o, gt, vic = build_and_run(N, deg, n_adv, seed, agreement=False,
                                       prior=False, n_problems=n_problems)
        if not vic:
            continue
        nv += len(vic)
        o_aucs += [auc for _, auc in victim_aucs(soc_o, gt, vic)]
        soc_a, _, _ = build_and_run(N, deg, n_adv, seed, agreement=True,
                                    prior=False, n_problems=n_problems)
        a_aucs += [auc for _, auc in victim_aucs(soc_a, gt, vic)]
    return o_aucs, a_aucs, nv


def main():
    deg = 4
    frac = 0.25          # fixed contiguous-cluster fraction
    n_seeds = 4
    grid = [24, 48, 96, 144]

    print("=" * 64)
    print("PHASE 3 — does the per-victim isolation hold as N scales?")
    print("=" * 64)
    print(f"fixed degree {deg}, fixed cluster fraction {frac:.0%}, {n_seeds} seeds, "
          f"no truth-prior on either arm.\n")
    print(f"  {'N':>4s} {'n_adv':>6s} {'victims':>8s} {'outcome AUC':>12s} "
          f"{'agreement AUC':>14s} {'gap':>6s}")

    rows = []
    for N in grid:
        n_adv = max(6, round(frac * N))
        o, a, nv = per_victim_at_N(N, deg, n_adv, n_seeds)
        om = float(np.mean(o)) if o else float("nan")
        am = float(np.mean(a)) if a else float("nan")
        rows.append((N, om, am))
        print(f"  {N:4d} {n_adv:6d} {nv:8d} {om:12.2f} {am:14.2f} {om - am:+6.2f}")

    oms = [r[1] for r in rows if r[1] == r[1]]
    ams = [r[2] for r in rows if r[2] == r[2]]
    print()
    if oms and ams:
        o_inv = max(oms) - min(oms)
        print(f"  outcome AUC across N: {min(oms):.2f}-{max(oms):.2f} "
              f"(spread {o_inv:.2f})  agreement AUC: {min(ams):.2f}-{max(ams):.2f}")
        invariant = o_inv < 0.1 and min(oms) > 0.9
        print(f"  [{'SCALE-INVARIANT' if invariant else 'CHECK'}] outcome isolation "
              f"{'holds' if invariant else 'varies'} as the swarm grows; the inversion is")
        print("  a LOCAL effect, so it is scale-free by construction — no global controller.")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""Phase 4 — ITERATIVE trust-weighted consensus: outcome-trust stops the cascade.

The Phase-2/3 society had ONE-SHOT communication (a single social-drift injection per
round), so opinions could not PROPAGATE — no cascade formed and the collective effect
was a near-null. This adds the missing mechanism: iterative belief propagation on the
graph (DeGroot / Laplacian-coupled, trust-weighted), T steps per problem, with a
STUBBORN confidently-wrong adversary CLUSTER that can drag the swarm via a cascade.

Trust gates the propagation:
  - no-trust   : uniform neighbour weights (a naive DeGroot consensus).
  - agreement  : trust neighbours who agree with the LOCAL consensus (the SOTA flavour).
  - outcome    : trust neighbours whose past beliefs matched the resolved TRUTH.

Honest positioning: the cascade itself — stubborn agents / zealots dominating a naive
consensus — is a KNOWN result (Mobilia; Acemoglu et al.); that is NOT the contribution.
The contribution is that OUTCOME-grounded trust kills the cascade while AGREEMENT-based
trust cannot (near the cluster the local consensus IS the cluster, so it trusts the
zealots). docs/notes/2026-06-05-ddm-coupled-comms-design-notes.md asserted *by
construction* that agreement-driven trust is "*the* mechanism that produces echo
chambers", and flagged a "Laplacian / mean-removing difference form" as a possible v2.
This module IS that v2, and turns the by-construction claim into an empirical demonstration.

Caveats: (1) this is a richer model than the one-shot DDM society — the missing
mechanism, added on purpose; (2) outcome-trust needs OBSERVABLE outcomes (the standing
assumption); (3) the collapse magnitude depends on the confidence asymmetry (confident
adversaries vs uncertain honest agents) — realistic but a modelling choice.

    python -m cognitive_society.phase4_consensus
"""
import numpy as np

from cognitive_society import topology as topo
from cognitive_society.adversary import build_clustered_society
from cognitive_society.society import cfg_outcome_trust
from cognitive_society.phase2_experiment import DEFAULT_BUILDER, DEFAULT_TOPOLOGY_FN


def clustered_graph(N, deg, n_adv, seed):
    soc, gt, _ = build_clustered_society(N, deg, n_adv, seed, config=cfg_outcome_trust(),
                                         builder=DEFAULT_BUILDER, topology_fn=DEFAULT_TOPOLOGY_FN)
    return soc.adj, np.asarray(gt)


def honest_graph(N, deg, seed):
    """All-honest control graph (no cluster) — the sanity check."""
    adj = topo.watts_strogatz(N, deg, 0.3, np.random.default_rng(500 + seed))
    return adj, np.zeros(N, dtype=int)


def run(adj, gt, method, n_problems=40, alpha=0.3, T=20, lr=0.5,
        honest_conf=0.25, adv_conf=0.9, seed=0):
    """Iterative trust-weighted consensus; returns honest-agent accuracy (2nd half).

    honest_conf / adv_conf: initial belief magnitude for honest (uncertain) vs
    adversary (confident, stubborn) nodes.
    """
    rng = np.random.default_rng(1000 + seed)
    N = adj.shape[0]
    honest = gt == 0
    adv = ~honest
    hi = np.nonzero(honest)[0]
    nb = [np.nonzero(adj[i])[0] for i in range(N)]
    e = np.zeros((N, N))   # trust logit per (observer, neighbour)
    accs = []
    for _ in range(n_problems):
        truth = 1.0 if rng.random() < 0.5 else -1.0
        b = np.zeros(N)
        b[honest] = np.clip(truth * honest_conf + rng.normal(0, 0.5, int(honest.sum())), -1, 1)
        b[adv] = -truth * adv_conf
        if method == "no-trust":
            W = adj.astype(float).copy()
        else:
            W = adj.astype(float) * (1.0 / (1.0 + np.exp(-np.clip(e, -30, 30))))
        for _ in range(T):
            den = W.sum(1)
            nbrm = np.where(den > 0, (W @ b) / np.maximum(den, 1e-9), b)
            b = np.clip(b + alpha * (nbrm - b), -1, 1)
            if adv.any():
                b[adv] = -truth * adv_conf
        dec = np.where(b >= 0, 1.0, -1.0)
        accs.append(float(np.mean(dec[honest] == truth)))
        if method == "outcome":
            right = np.where(dec == truth, 1.0, -1.0)
            for i in hi:
                e[i, nb[i]] += lr * right[nb[i]]
        elif method == "agreement":
            for i in hi:
                if nb[i].size:
                    loc = 1.0 if np.mean(dec[nb[i]]) >= 0 else -1.0
                    e[i, nb[i]] += lr * np.where(dec[nb[i]] == loc, 1.0, -1.0)
        if method != "no-trust":
            np.clip(e, -5, 5, out=e)
    return float(np.mean(accs[len(accs) // 2:]))


METHODS = ("no-trust", "agreement", "outcome")


def _avg(graph_fn, method, n_seeds):
    return float(np.mean([run(*graph_fn(s), method, seed=s) for s in range(n_seeds)]))


def main():
    N, deg, n_seeds = 60, 4, 6
    print("=" * 66)
    print("PHASE 4 — iterative trust-weighted consensus (cascade regime)")
    print("=" * 66)

    print("\nSANITY — all honest (no cluster): every method should sit ~high")
    print(f"  {'no-trust':>9s} {'agreement':>10s} {'outcome':>9s}")
    san = {m: _avg(lambda s: honest_graph(N, deg, s), m, n_seeds) for m in METHODS}
    print(f"  {san['no-trust']:9.1%} {san['agreement']:10.1%} {san['outcome']:9.1%}")
    ok = min(san.values()) > 0.85
    print(f"  [{'PASS' if ok else 'FAIL — bug'}] all-honest swarm converges to truth "
          f"regardless of trust rule.")

    print(f"\nCASCADE — contiguous confidently-wrong cluster (N={N}, deg={deg}, {n_seeds} seeds)")
    print(f"  {'adv frac':>8s} {'no-trust':>9s} {'agreement':>10s} {'outcome':>9s} {'out - best-other':>16s}")
    for frac in (0.2, 0.3, 0.4, 0.45, 0.5):
        n_adv = round(frac * N)
        acc = {m: _avg(lambda s: clustered_graph(N, deg, n_adv, s), m, n_seeds) for m in METHODS}
        bo = max(acc["no-trust"], acc["agreement"])
        print(f"  {frac:8.0%} {acc['no-trust']:9.1%} {acc['agreement']:10.1%} "
              f"{acc['outcome']:9.1%} {acc['outcome'] - bo:+16.1%}")

    print("\n" + "=" * 66)
    print("Read: with PROPAGATION, a stubborn cluster cascades and no-trust/agreement")
    print("collapse (zealot domination, a known effect). OUTCOME-grounded trust kills the")
    print("cascade — the contribution is the trust GATING, not the coupling.")
    print("=" * 66)


if __name__ == "__main__":
    main()

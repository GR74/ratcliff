"""Swarm Phase 0 — topology-only control (ALL honest, no adversaries).

Before adding adversaries (Phase 2), confirm the decentralized port is sound: with
EVERY agent honest, does collective accuracy survive moving from the fully-connected
society to LOCAL topologies (ring / small-world / random-regular / scale-free)? If a
topology collapses accuracy or injects bias even with no adversaries, that's the
Lorenz / topology-distortion trap (social drift on a graph manufacturing false
confidence) — and it must be caught HERE, before it can contaminate any adversary
result.

Pass = accuracy stays high and comparable across topologies, so any later
global->local gap is attributable to adversaries, not the graph itself.

    python -m cognitive_society.swarm_experiment
"""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import Society, cfg_outcome_trust
from cognitive_society import topology as topo


def honest_pop(K, rng):
    return [DDMAgent(DDMParams(
        boundary=float(rng.uniform(0.8, 1.6)),
        drift_scale=float(rng.uniform(0.9, 1.3)),
        ndt=float(rng.uniform(0.18, 0.28)), sigma=1.0)) for _ in range(K)]


def topologies(K, degree, rng):
    return {
        "complete": topo.complete(K),
        f"ring (deg {degree})": topo.ring_lattice(K, degree),
        f"small-world (deg {degree}, p=.2)": topo.watts_strogatz(K, degree, 0.2, rng),
        f"random-regular (deg {degree})": topo.random_regular(K, degree, rng),
        "scale-free (m=3)": topo.barabasi_albert(K, 3, rng),
    }


def phase0_control(K=49, degree=6, n_problems=30, n_seeds=5):
    rows = {}
    for seed in range(n_seeds):
        graph_rng = np.random.default_rng(1000 + seed)
        prng = np.random.default_rng(2000 + seed)
        # HARD evidence on purpose: a large honest swarm trivially saturates to
        # 100% on easy problems (no headroom for topology to distort), so a
        # meaningful topology-only control needs low evidence where the collective
        # sits below ceiling and a topology effect, if any, can actually show.
        evidences = prng.choice([-0.15, -0.1, 0.1, 0.15], size=n_problems).tolist()
        graphs = topologies(K, degree, graph_rng)
        for name, adj in graphs.items():
            # identical honest population per seed, so only the topology varies
            agents = honest_pop(K, np.random.default_rng(1000 + seed))
            soc = Society(agents, config=cfg_outcome_trust(), rng_seed=seed, topology=adj)
            acc = soc.run(evidences)["collective_accuracy"]
            r = rows.setdefault(name, {"acc": [], "deg": [], "conn": []})
            r["acc"].append(acc)
            r["deg"].append(topo.mean_degree(adj))
            r["conn"].append(topo.is_connected(adj))
    return rows


def main():
    K, degree, n_seeds, n_problems = 29, 6, 5, 30
    print("=" * 70)
    print("SWARM PHASE 0 — topology-only control (ALL honest, no adversaries)")
    print("=" * 70)
    print(f"{K} honest agents, target degree ~{degree}, {n_problems} HARD problems, "
          f"{n_seeds} seeds.\n")
    rows = phase0_control(K=K, degree=degree, n_problems=n_problems, n_seeds=n_seeds)

    print(f"  {'topology':34s} {'accuracy':>13s} {'mean deg':>9s} {'conn':>6s}")
    accs = {}
    for name, d in rows.items():
        a = np.asarray(d["acc"])
        accs[name] = a.mean()
        conn = "yes" if all(d["conn"]) else "some-no"
        print(f"  {name:34s} {a.mean():6.1%} +/- {a.std():4.1%} {np.mean(d['deg']):9.1f} {conn:>6s}")

    spread = (max(accs.values()) - min(accs.values())) * 100
    print(f"\n  spread across topologies: {spread:.1f} accuracy points")
    verdict = "PASS" if spread < 8.0 else "CHECK"
    print(f"  [{verdict}] all-honest accuracy is {'comparable' if spread < 8.0 else 'NOT comparable'} "
          f"across graphs -> topology alone {'does not' if spread < 8.0 else 'DOES'} distort the swarm.")
    print("  Next (Phase 1): per-agent LOCAL maps; (Phase 2): confident adversary clusters.")
    print("=" * 70)


if __name__ == "__main__":
    main()

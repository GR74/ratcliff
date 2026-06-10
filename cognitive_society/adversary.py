"""Phase 2 — the contiguous confident-colluding adversary CLUSTER.

The anchor experiment (experiment.py) sprinkles a few confidently-wrong
adversaries into a fully-connected society and shows cognitive-map-grounded
trust down-weights them. Phase 2 escalates to the regime where *agreement-based*
defenses (W-MSR / median-reputation / bounded-confidence) provably FAIL but
outcome-grounded trust still works: a CONTIGUOUS, mutually-agreeing block of
confidently-wrong agents on a LOCAL topology, sized so that — in at least one
honest node's neighbourhood — the adversaries are the LOCAL MAJORITY.

Why contiguity matters (the load-bearing fact, from LeBlanc/Sundaram W-MSR
theory). A W-MSR(F) node discards the F largest and F smallest neighbour values
before averaging. A confidently-wrong block that is internally consistent does
NOT look extreme to a victim it has captured — it looks like the local
consensus. Once a victim has > F (i.e. >= F+1) such neighbours, every dissenting
*honest* value can be thrown out as one of the F extremes, the victim is sealed
inside the cluster's wrong value, and no agreement filter can recover the truth.
The per-NEIGHBOURHOOD adversary count exceeding F is what breaks the filter; the
global adversary fraction is irrelevant (a scattered same-count placement keeps
every neighbourhood <= F and the filters survive).

This module is pure construction + verification, built against the EXISTING
public surface (topology.py, agent.py, society.py) with ZERO edits to any of
them:

  * ``bfs_ball_cluster`` / ``densest_cluster`` grow a CONNECTED set S of node ids
    directly off the adjacency matrix, so the cluster is contiguous on the graph.
  * ``verify_fbound_exceeded`` asserts the decisive regime (some honest node's
    neighbourhood is majority-adversary) BEFORE any experiment runs, so a null
    result is attributable to trust-signal quality, never to a mis-sized cluster.
  * ``build_clustered_society`` places the confidently-wrong DDMParams agents
    EXACTLY on S and varied honest agents elsewhere, then hands the SAME ``adj``
    to ``Society`` as ``topology=`` — so the adversary agent ids coincide with the
    contiguous graph cluster. (``experiment.build_mixed`` appends adversaries at
    the END of the id list but on NO particular graph node, so we build the
    population ourselves here; we reuse only its exact adversary DDMParams.)

Pure NumPy, no GPU, no scipy. Mirrors experiment.py / swarm_experiment.py style.
"""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import Society, cfg_swarm
from cognitive_society import topology as topo


# ---------------------------------------------------------------------------
# (1) Contiguous-cluster builders — both return a CONNECTED set of node ids.
# ---------------------------------------------------------------------------
def bfs_ball_cluster(adj: np.ndarray, seed_node: int, n_adv: int) -> list:
    """BFS-grow a CONNECTED set of ``n_adv`` node ids outward from ``seed_node``.

    A BFS ball is connected by construction, and on locally-clustered graphs
    (ring / small-world / random-regular) it saturates the neighbourhoods of the
    honest nodes on its boundary — exactly the pressure that makes some honest
    victim majority-adversary. Returns the node ids sorted ascending.
    """
    adj = np.asarray(adj, dtype=bool)
    K = adj.shape[0]
    n_adv = int(min(n_adv, K))
    seed_node = int(seed_node)
    S = [seed_node]
    seen = {seed_node}
    frontier = [seed_node]
    while len(S) < n_adv and frontier:
        nxt = []
        for u in frontier:
            for v in np.nonzero(adj[u])[0]:
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    S.append(v)
                    nxt.append(v)
                    if len(S) >= n_adv:
                        break
            if len(S) >= n_adv:
                break
        frontier = nxt
    return sorted(S)


def densest_cluster(adj: np.ndarray, seed_node: int, n_adv: int) -> list:
    """Greedily grow a connected set by repeatedly adding the node with the most
    edges INTO the current set (max internal-degree / greedy densest-subgraph).

    Maximising internal degree makes S internally dense — high mutual agreement,
    hard to split by an extreme-value filter — and concentrates its external
    pressure on a few victims, which maximises the local f-bound violation.
    Returns the node ids sorted ascending.
    """
    adj = np.asarray(adj, dtype=bool)
    K = adj.shape[0]
    n_adv = int(min(n_adv, K))
    S = {int(seed_node)}
    while len(S) < n_adv:
        members = list(S)
        # internal-degree of every non-member = #edges from it into S.
        internal = adj[:, members].sum(axis=1)
        internal[members] = -1                 # never re-pick a member
        # only consider nodes actually adjacent to S (keeps the set connected).
        adjacent = adj[members].any(axis=0)
        internal[~adjacent] = -1
        best = int(np.argmax(internal))
        if internal[best] <= 0:                # no connected candidate left
            break
        S.add(best)
    return sorted(S)


# ---------------------------------------------------------------------------
# (2) Decisive-regime verification — read only adj + S.
# ---------------------------------------------------------------------------
def cluster_is_connected(adj: np.ndarray, S) -> bool:
    """BFS restricted to S — True iff the cluster itself (not just the whole
    graph) is a single connected component. A disconnected 'cluster' does not
    produce the local-consensus camouflage the decisive regime depends on."""
    adj = np.asarray(adj, dtype=bool)
    S = [int(x) for x in S]
    if not S:
        return True
    Sset = set(S)
    seen = {S[0]}
    stack = [S[0]]
    while stack:
        u = stack.pop()
        for v in np.nonzero(adj[u])[0]:
            v = int(v)
            if v in Sset and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(Sset)


def verify_fbound_exceeded(adj: np.ndarray, S):
    """For every honest node touching S, compute the per-neighbourhood adversary
    fraction and the local W-MSR bound F, and flag victims that exceed it.

    For a victim v: ``F_local(v) = floor((deg(v) - 1) / 2)`` is the largest
    adversary count W-MSR / a simple majority filter can survive (it needs
    >= F+1 honest *outside* values to leave one un-trimmed). The cluster is in
    the DECISIVE regime iff ``worst_fraction > 0.5`` — at least one honest node's
    neighbourhood is majority-adversary, so its truth-bearing dissent can be
    discarded as the F extremes and no agreement filter can move it.

    Returns ``(worst_fraction, victims)`` where ``victims`` is a list of
    ``(v, adv_count_in_nbhd, deg)`` for every honest v with adv_count > F_local(v).
    """
    adj = np.asarray(adj, dtype=bool)
    K = adj.shape[0]
    Sset = set(int(x) for x in S)
    worst = 0.0
    victims = []
    for v in range(K):
        if v in Sset:
            continue
        nbr = set(int(x) for x in np.nonzero(adj[v])[0])
        if not nbr:
            continue
        inter = nbr & Sset
        if not inter:
            continue
        deg = len(nbr)
        a = len(inter)
        f_local = (deg - 1) // 2               # local majority / W-MSR F s.t. deg>=2F+1
        frac = a / deg
        if frac > worst:
            worst = frac
        if a > f_local:
            victims.append((v, a, deg))
    return worst, victims


# ---------------------------------------------------------------------------
# (3) Agent personalities — match experiment.build_mixed exactly.
# ---------------------------------------------------------------------------
def adversary_params() -> DDMParams:
    """The confidently-wrong adversary: commits fast to the WRONG sign and so
    AGREES with sibling adversaries.

    DDMParams(boundary=0.5, drift_scale=-1.3, ndt=0.16, sigma=1.0) — negative
    drift answers OPPOSITE the evidence; the low boundary makes it fast and
    'confident'. Byte-identical to experiment.build_mixed's adversary so Phase 2
    results are comparable to the anchor experiment.
    """
    return DDMParams(boundary=0.5, drift_scale=-1.3, ndt=0.16, sigma=1.0)


def honest_params(rng: np.random.Generator) -> DDMParams:
    """A varied honest agent (positive drift), drawn to match
    experiment.build_mixed / swarm_experiment.honest_pop:
    boundary~U(0.8,1.6), drift_scale~U(0.9,1.3), ndt~U(0.18,0.28), sigma=1.0."""
    return DDMParams(
        boundary=float(rng.uniform(0.8, 1.6)),
        drift_scale=float(rng.uniform(0.9, 1.3)),
        ndt=float(rng.uniform(0.18, 0.28)),
        sigma=1.0,
    )


# ---------------------------------------------------------------------------
# (4) Build the clustered society — adversaries SIT on the graph cluster.
# ---------------------------------------------------------------------------
def build_clustered_society(K: int, deg: int, n_adv: int, seed: int,
                            config=None, builder=bfs_ball_cluster,
                            topology_fn=None):
    """Build a Society whose adversary agents occupy a CONTIGUOUS graph cluster.

    Steps:
      1. Build ``adj`` (``random_regular(K, deg, rng)`` by default; pass
         ``topology_fn(K, deg, rng) -> bool KxK`` for ring/small-world/scale-free).
      2. Grow the connected cluster S off ``adj`` via ``builder`` (BFS ball or
         greedy densest), seeded at the max-degree node (hub capture on scale-free;
         arbitrary-but-deterministic on a regular graph).
      3. Place ``adversary_params()`` on every node in S and ``honest_params()``
         elsewhere, so adversary agent ids COINCIDE with the graph cluster.
      4. Hand the SAME ``adj`` to ``Society`` as ``topology=``; default config is
         ``cfg_swarm()`` (local per-agent outcome-grounded maps, decentralized).

    Returns ``(soc, gt, S)`` where ``gt`` is an int array of length K
    (1 = adversary, 0 = honest) and ``S`` is the sorted list of adversary ids.
    """
    rng = np.random.default_rng(seed)
    if topology_fn is None:
        adj = topo.random_regular(K, deg, rng)
    else:
        adj = np.asarray(topology_fn(K, deg, rng), dtype=bool)

    # Seed the cluster at the highest-degree node (a hub on scale-free graphs;
    # ties broken to the lowest id by argmax — deterministic on regular graphs).
    seed_node = int(np.argmax(adj.sum(axis=1)))
    S = builder(adj, seed_node, n_adv)
    Sset = set(S)

    agents = []
    for i in range(K):
        params = adversary_params() if i in Sset else honest_params(rng)
        agents.append(DDMAgent(params, agent_id=i))

    cfg = config if config is not None else cfg_swarm()
    soc = Society(agents, config=cfg, rng_seed=seed, topology=adj)
    gt = np.array([1 if i in Sset else 0 for i in range(K)], dtype=int)
    return soc, gt, S

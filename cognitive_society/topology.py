"""Graph topologies for the decentralized swarm (Phase 0).

Each generator returns a boolean K x K adjacency matrix: symmetric, with a False
diagonal (no self-edges). A `True` at [i, j] means agents i and j are neighbors
and can exchange social drift. The society restricts each agent's coupling, trust
and (later) cognitive map to its neighbor set, so global order must EMERGE from
local interaction — no agent ever sees the whole.

Families (per the research synthesis, 2026-06-10):
  complete         every agent connected to all others (= the old global society; sanity)
  ring_lattice     k-regular ring; clean degree control, high clustering, large diameter
  watts_strogatz   small-world: ring + random rewiring (realistic; short paths + clustering)
  random_regular   d-regular random graph (primary: clean degree + (r,s)-robustness story)
  barabasi_albert  scale-free: preferential attachment (stress-test hub capture)

All pure NumPy, deterministic given the passed Generator.
"""
import numpy as np


def _blank(K: int) -> np.ndarray:
    return np.zeros((K, K), dtype=bool)


def _finish(adj: np.ndarray) -> np.ndarray:
    """Symmetrize and clear the diagonal (defensive)."""
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    return adj


def complete(K: int) -> np.ndarray:
    """Every agent connected to all others (the fully-connected global society)."""
    return _finish(~np.eye(K, dtype=bool))


def ring_lattice(K: int, degree: int) -> np.ndarray:
    """k-regular ring: each node joined to `degree/2` neighbors on each side."""
    if degree % 2 != 0:
        raise ValueError("ring_lattice degree must be even")
    if degree >= K:
        return complete(K)
    adj = _blank(K)
    half = degree // 2
    for i in range(K):
        for d in range(1, half + 1):
            j = (i + d) % K
            adj[i, j] = adj[j, i] = True
    return _finish(adj)


def watts_strogatz(K: int, degree: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Watts-Strogatz small-world: start from a ring lattice, rewire each
    clockwise edge to a random target with probability `p` (no self-loops or
    duplicate edges)."""
    adj = ring_lattice(K, degree)
    half = degree // 2
    for i in range(K):
        for d in range(1, half + 1):
            if rng.random() >= p:
                continue
            j = (i + d) % K
            # pick a new target not equal to i and not already a neighbor
            for _ in range(2 * K):
                t = int(rng.integers(K))
                if t != i and not adj[i, t]:
                    adj[i, j] = adj[j, i] = False
                    adj[i, t] = adj[t, i] = True
                    break
    return _finish(adj)


def random_regular(K: int, degree: int, rng: np.random.Generator,
                   swaps_per_edge: int = 10) -> np.ndarray:
    """Random d-regular graph by degree-preserving randomization: start from a
    deterministic d-regular graph, then apply many random double-edge swaps
    ((a,b),(c,d) -> (a,c),(b,d)). Each swap preserves every degree, so the result
    is always a valid simple d-regular graph (unlike the pairing model, which
    restarts on any conflict and rarely succeeds at scale)."""
    if degree >= K:
        return complete(K)
    if (K * degree) % 2 != 0:
        raise ValueError("random_regular needs K*degree even")
    if degree % 2 == 0:
        adj = ring_lattice(K, degree)
    else:
        if K % 2 != 0:
            raise ValueError("odd degree requires even K")
        adj = ring_lattice(K, degree - 1)
        for i in range(K // 2):  # add the diameter matching for the odd +1
            j = i + K // 2
            adj[i, j] = adj[j, i] = True
    edges = [(int(i), int(j)) for i in range(K) for j in range(i + 1, K) if adj[i, j]]
    for _ in range(swaps_per_edge * len(edges)):
        e1, e2 = int(rng.integers(len(edges))), int(rng.integers(len(edges)))
        if e1 == e2:
            continue
        a, b = edges[e1]
        c, d = edges[e2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or adj[a, c] or adj[b, d]:
            continue
        adj[a, b] = adj[b, a] = False
        adj[c, d] = adj[d, c] = False
        adj[a, c] = adj[c, a] = True
        adj[b, d] = adj[d, b] = True
        edges[e1] = (min(a, c), max(a, c))
        edges[e2] = (min(b, d), max(b, d))
    return _finish(adj)


def barabasi_albert(K: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Scale-free graph by Barabasi-Albert preferential attachment: each new node
    attaches to `m` existing nodes chosen with probability proportional to degree."""
    if m < 1 or m >= K:
        raise ValueError("barabasi_albert needs 1 <= m < K")
    adj = _blank(K)
    # seed: a small connected clique of the first m nodes
    for i in range(m):
        for j in range(i):
            adj[i, j] = adj[j, i] = True
    repeated = list(range(m)) * m  # degree-proportional bag (each appears ~deg times)
    for new in range(m, K):
        chosen = set()
        while len(chosen) < m:
            chosen.add(int(repeated[int(rng.integers(len(repeated)))]))
        for t in chosen:
            adj[new, t] = adj[t, new] = True
            repeated.append(new)
            repeated.append(t)
    return _finish(adj)


def degrees(adj: np.ndarray) -> np.ndarray:
    return adj.sum(axis=1)


def mean_degree(adj: np.ndarray) -> float:
    return float(adj.sum(axis=1).mean())


def is_connected(adj: np.ndarray) -> bool:
    """BFS from node 0 — True if every node is reachable (one component)."""
    K = adj.shape[0]
    seen = np.zeros(K, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        n = stack.pop()
        for j in np.nonzero(adj[n])[0]:
            if not seen[j]:
                seen[j] = True
                stack.append(int(j))
    return bool(seen.all())

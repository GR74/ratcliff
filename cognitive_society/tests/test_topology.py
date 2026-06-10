"""Topology tests — graph generators + society integration. Pure NumPy, fast."""
import numpy as np

from cognitive_society import topology as topo
from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import Society, cfg_outcome_trust


def _valid(adj, K):
    assert adj.shape == (K, K)
    assert adj.dtype == bool
    assert np.array_equal(adj, adj.T), "adjacency must be symmetric"
    assert not np.any(np.diag(adj)), "no self-edges"


def test_complete_is_all_but_self():
    K = 8
    adj = topo.complete(K)
    _valid(adj, K)
    assert (adj.sum(1) == K - 1).all()


def test_ring_lattice_is_regular_connected():
    K, deg = 20, 6
    adj = topo.ring_lattice(K, deg)
    _valid(adj, K)
    assert (adj.sum(1) == deg).all(), "ring lattice is exactly d-regular"
    assert topo.is_connected(adj)


def test_ring_lattice_rejects_odd_degree():
    try:
        topo.ring_lattice(10, 3)
        assert False, "odd degree should raise"
    except ValueError:
        pass


def test_random_regular_is_regular_connected():
    rng = np.random.default_rng(0)
    K, deg = 30, 6
    adj = topo.random_regular(K, deg, rng)
    _valid(adj, K)
    assert (adj.sum(1) == deg).all(), "exactly d-regular"
    assert topo.is_connected(adj)


def test_watts_strogatz_valid_connected_preserves_edges():
    rng = np.random.default_rng(1)
    K, deg = 30, 6
    adj = topo.watts_strogatz(K, deg, 0.2, rng)
    _valid(adj, K)
    # rewiring moves edges, never adds/removes -> mean degree preserved
    assert abs(topo.mean_degree(adj) - deg) < 1e-9
    assert topo.is_connected(adj)


def test_barabasi_albert_is_scale_free():
    rng = np.random.default_rng(2)
    K, m = 50, 3
    adj = topo.barabasi_albert(K, m, rng)
    _valid(adj, K)
    assert topo.is_connected(adj)
    deg = topo.degrees(adj)
    # preferential attachment -> heavy-tailed degree: a hub well above the mean
    assert deg.max() > 2 * deg.mean(), f"expected a hub, max={deg.max()} mean={deg.mean():.1f}"


def test_society_topology_restricts_to_neighbors():
    K = 4
    agents = [DDMAgent(DDMParams(boundary=1.0, drift_scale=1.1, ndt=0.2)) for _ in range(K)]
    adj = topo.ring_lattice(K, 2)  # 4-cycle 0-1-2-3-0
    soc = Society(agents, config=cfg_outcome_trust(), rng_seed=0, topology=adj)
    assert soc.adj[0, 1] and soc.adj[0, 3]
    assert not soc.adj[0, 2]      # opposite node is NOT a neighbour
    assert not soc.adj[0, 0]      # no self-edge
    res = soc.run([0.4, -0.3, 0.3, 0.4])
    assert 0.0 <= res["collective_accuracy"] <= 1.0


def test_society_default_topology_is_complete():
    K = 5
    agents = [DDMAgent(DDMParams(boundary=1.0)) for _ in range(K)]
    soc = Society(agents, rng_seed=0)
    assert np.array_equal(soc.adj, ~np.eye(K, dtype=bool))


def test_society_rejects_wrong_shaped_topology():
    agents = [DDMAgent(DDMParams()) for _ in range(4)]
    try:
        Society(agents, topology=np.ones((3, 3), dtype=bool))
        assert False, "wrong-shaped topology should raise"
    except ValueError:
        pass

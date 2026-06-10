"""Swarm tests — per-agent LOCAL cognitive maps (Phase 1). Pure NumPy, fast."""
import numpy as np

from cognitive_society import topology as topo
from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.society import Society, cfg_swarm, cfg_full


def _pop(K, seed=0):
    rng = np.random.default_rng(seed)
    return [DDMAgent(DDMParams(boundary=float(rng.uniform(0.8, 1.5)),
                               drift_scale=float(rng.uniform(0.9, 1.3)),
                               ndt=0.2)) for _ in range(K)]


def test_cfg_swarm_sets_local_maps():
    assert cfg_swarm().local_maps is True
    assert cfg_full().local_maps is False


def test_local_map_only_maps_neighbours():
    K = 8
    adj = topo.ring_lattice(K, 2)  # 8-cycle: each node has exactly 2 neighbours
    soc = Society(_pop(K), config=cfg_swarm(local_obs_trials=150), rng_seed=0, topology=adj)
    soc.build_cognitive_maps()
    for i in range(K):
        non_nb = ~adj[i]
        non_nb[i] = True  # self is also not mapped
        assert np.allclose(soc.competence[i][non_nb], 0.5), "non-neighbours stay neutral"
        assert not np.allclose(soc.competence[i][adj[i]], 0.5), "neighbours are estimated"


def test_local_maps_are_subjective():
    # observers that share a target generally estimate it differently (their own
    # noisy observations) -> the map is per-observer, not one global vector.
    K = 8
    soc = Society(_pop(K), config=cfg_swarm(local_obs_trials=120), rng_seed=1,
                  topology=topo.complete(K))
    soc.build_cognitive_maps()
    est_of_0 = soc.competence[1:, 0]  # each non-self observer's estimate of agent 0
    assert est_of_0.std() > 1e-9, "subjective local estimates should differ across observers"


def test_global_map_is_shared_and_returns_vector():
    K = 6
    soc = Society(_pop(K), config=cfg_full(), rng_seed=0)
    comp = soc.build_cognitive_maps()
    assert comp.shape == (K,), "global path returns a single shared vector"
    for i in range(K):
        assert np.allclose(soc.competence[i], comp), "every observer holds the same global map"


def test_swarm_society_runs_on_topology():
    K = 12
    adj = topo.watts_strogatz(K, 4, 0.2, np.random.default_rng(2))
    soc = Society(_pop(K, seed=2), config=cfg_swarm(local_obs_trials=150),
                  rng_seed=2, topology=adj)
    res = soc.run([0.3, -0.3, 0.4, -0.4, 0.3])
    assert 0.0 <= res["collective_accuracy"] <= 1.0
    assert soc.mapped

"""Phase 2 tests — pin the contracts for the clustered-adversary stress test.

Fast, pure NumPy (no GPU, no scipy). Mirrors test_swarm.py style. Covers:
  * cluster contiguity + correct size (bfs_ball_cluster, cluster_is_connected),
  * decisive-regime detection (verify_fbound_exceeded flags a sized contiguous
    cluster but NOT a scattered same-count placement),
  * W-MSR self-value retention + bounded (convex) behaviour, and the cluster
    sealing W-MSR to the WRONG bit when local count exceeds F,
  * sparsemax simplex + hard zeros,
  * A-RepC trusting a colluding cluster on the poisoned-median regime,
  * AgreementTrustModel ignoring the outcome (truth), differing from base TrustModel,
  * oracle beating no-trust on the decisive cluster,
  * AUC / Spearman edge cases,
  * an end-to-end smoke run of the outcome-vs-agreement ablation.
"""
import numpy as np

from cognitive_society import topology as topo
from cognitive_society.comms import TrustModel
from cognitive_society.adversary import (
    bfs_ball_cluster, densest_cluster, cluster_is_connected,
    verify_fbound_exceeded, adversary_params, honest_params,
    build_clustered_society,
)
from cognitive_society.baselines import (
    wmsr_step, wmsr_aggregator, sparsemax_masked, ARepCTrust, arepc_aggregator,
    notrust_aggregator, oracle_aggregator, AgreementTrustModel,
)
from cognitive_society.swarm_metrics import trust_auc, spearman, trust_gap
from cognitive_society.phase2_experiment import run_ablation, make_evidences


# ---------------------------------------------------------------------------
# (1) cluster contiguity + size.
# ---------------------------------------------------------------------------
def test_bfs_cluster_is_connected_and_sized():
    K, deg, n_adv = 20, 4, 6
    adj = topo.ring_lattice(K, deg)
    S = bfs_ball_cluster(adj, seed_node=0, n_adv=n_adv)
    assert len(S) == n_adv, "BFS ball must return exactly n_adv ids"
    assert S == sorted(S), "ids returned sorted"
    assert len(set(S)) == n_adv, "ids are distinct"
    assert cluster_is_connected(adj, S), "the cluster itself must be one component"


def test_densest_cluster_is_connected_and_sized():
    K, deg, n_adv = 24, 6, 7
    adj = topo.ring_lattice(K, deg)
    S = densest_cluster(adj, seed_node=0, n_adv=n_adv)
    assert len(S) == n_adv
    assert cluster_is_connected(adj, S), "densest cluster only adds adjacent nodes -> connected"


# ---------------------------------------------------------------------------
# (2) decisive-regime detection: contiguous crosses 0.5, scattered does not.
# ---------------------------------------------------------------------------
def test_verify_fbound_flags_decisive_regime():
    # A pure ring arc caps at worst-frac EXACTLY 0.5; the small-world graph lets a
    # contiguous densest cluster ENCLOSE a victim on a majority of its neighbourhood,
    # crossing 0.5 — that is the decisive regime the experiment runs on.
    K, deg, n_adv = 28, 6, 6
    adj = topo.watts_strogatz(K, deg, 0.3, np.random.default_rng(0))
    seed_node = int(np.argmax(adj.sum(axis=1)))
    S = densest_cluster(adj, seed_node, n_adv)
    worst_c, victims_c = verify_fbound_exceeded(adj, S)
    assert worst_c > 0.5, "a sized contiguous small-world cluster must cross worst-frac 0.5"
    assert len(victims_c) >= 1, "decisive regime has >= 1 victim (local count > F)"

    # SCATTERED same-count placement: spread evenly around the index range -> no
    # neighbourhood is a STRICT adversary-majority, so worst-frac stays <= 0.5 and
    # the regime is NOT decisive. (Same global count; only the CONTIGUITY differs —
    # that is what breaks agreement filters, per the claim.) The decisive criterion
    # is worst_fraction > 0.5 (verify_fbound_exceeded's docstring); the contiguous
    # cluster crosses it, the scattered one does not.
    scattered = sorted(set(int(round(x)) % K
                           for x in np.linspace(0, K, n_adv, endpoint=False)))
    worst_s, _ = verify_fbound_exceeded(adj, scattered)
    assert worst_s <= 0.5, "a scattered same-count placement must NOT cross worst-frac 0.5"
    assert worst_c > worst_s, "contiguity raises the worst-frac above the scattered placement"


def test_adversary_and_honest_params():
    p = adversary_params()
    assert p.drift_scale < 0, "confidently-wrong adversary has NEGATIVE drift"
    assert p.boundary == 0.5 and p.ndt == 0.16, "byte-identical to experiment.build_mixed"
    h = honest_params(np.random.default_rng(0))
    assert h.drift_scale > 0, "honest agent has positive drift"


# ---------------------------------------------------------------------------
# (3) W-MSR self-value retention + boundedness; cluster wins when local > F.
# ---------------------------------------------------------------------------
def test_wmsr_keeps_own_value_and_is_bounded():
    # convexity: every normal node's next value stays within [min, max] of all values.
    K, deg = 12, 4
    adj = topo.ring_lattice(K, deg)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=K)
    lo, hi = x.min(), x.max()
    x1 = wmsr_step(x, adj, F=1)
    assert np.all(x1 >= lo - 1e-9) and np.all(x1 <= hi + 1e-9), "W-MSR step is a convex average"

    # single-neighbour sanity: a node with one neighbour always keeps its own value
    # (the convex average of self + survivors always includes self).
    two = np.zeros((2, 2), dtype=bool)
    two[0, 1] = two[1, 0] = True
    y = np.array([3.0, -7.0])
    y1 = wmsr_step(y, two, F=1)
    # node 0 trims its single greater/less neighbour, falls back to self only
    assert y1[0] == 3.0, "self value retained when the only neighbour is trimmed"


def test_wmsr_cluster_wins_when_local_count_exceeds_F():
    # On the decisive small-world cluster (truth=1 -> honest lean +1), the pinned
    # adversary extreme seals the captured victims and W-MSR returns the WRONG bit.
    K, deg, n_adv = 28, 6, 6
    adj = topo.watts_strogatz(K, deg, 0.3, np.random.default_rng(0))
    seed_node = int(np.argmax(adj.sum(axis=1)))
    S = densest_cluster(adj, seed_node, n_adv)
    worst, _ = verify_fbound_exceeded(adj, S)
    assert worst > 0.5  # decisive regime precondition
    gt = np.array([1 if i in set(S) else 0 for i in range(K)])
    normal = (gt == 0)
    leanings = np.where(normal, 1.0, -1.0)  # honest lean +1 (truth), adversaries -1
    bit = wmsr_aggregator(leanings, adj, normal, F=1)
    assert bit == 0, "agreement filter is sealed inside the cluster -> WRONG bit"


# ---------------------------------------------------------------------------
# (4) sparsemax: simplex with hard zeros, ignores non-mask.
# ---------------------------------------------------------------------------
def test_sparsemax_is_simplex_with_hard_zeros():
    z = np.array([5.0, 4.9, -3.0, 100.0])
    mask = np.array([True, True, True, False])  # last entry excluded entirely
    p = sparsemax_masked(z, mask)
    assert abs(p[mask].sum() - 1.0) < 1e-9, "weights sum to 1 over the mask"
    assert np.all(p >= -1e-12), "weights are non-negative"
    assert p[3] == 0.0, "non-mask entry forced to 0 regardless of its score"
    assert p[2] == 0.0, "low-score neighbour gets EXACTLY zero (hard truncation)"


# ---------------------------------------------------------------------------
# (5) A-RepC trusts the colluding cluster on a poisoned-median neighbourhood.
# ---------------------------------------------------------------------------
def test_arepc_trusts_colluding_cluster():
    # Observer 0 has 6 neighbours; 4 are adversaries leaning -1, 2 honest leaning +1.
    # The local median is -1 (poisoned), so A-RepC gives the adversaries full weight
    # and sparsemax-truncates the honest minority to zero.
    K = 7
    mask = np.zeros(K, dtype=bool)
    mask[1:7] = True  # neighbours 1..6
    leanings = np.array([1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0])  # 4 adv -1, 2 honest +1
    rep = ARepCTrust(K, mask, eta=4.0, lam=0.9)
    for _ in range(5):
        p = rep.update_and_weights(leanings)
    adv_idx = [1, 2, 3, 4]
    hon_idx = [5, 6]
    assert p[adv_idx].sum() > 0.99, "A-RepC gives the colluding cluster ~all the weight"
    assert np.allclose(p[hon_idx], 0.0), "honest minority truncated to zero (poisoned median)"


# ---------------------------------------------------------------------------
# (6) AgreementTrustModel ignores the outcome; differs from base TrustModel.
# ---------------------------------------------------------------------------
def test_agreement_trustmodel_ignores_outcome():
    K = 5
    peer_leanings = np.array([1, 1, 1, -1, -1])  # local majority is +1
    a = AgreementTrustModel(K)
    a.update(peer_leanings, outcome=+1)
    e_pos = a.e.copy()
    b = AgreementTrustModel(K)
    b.update(peer_leanings, outcome=-1)  # opposite truth
    e_neg = b.e.copy()
    assert np.allclose(e_pos, e_neg), "AgreementTrustModel.update IGNORES the outcome (truth)"

    # base TrustModel DOES use the outcome -> opposite outcomes give opposite e.
    t1 = TrustModel(K); t1.update(peer_leanings, outcome=+1)
    t2 = TrustModel(K); t2.update(peer_leanings, outcome=-1)
    assert not np.allclose(t1.e, t2.e), "base TrustModel uses the outcome (differs from agreement)"

    # THE distinguishing case: when the TRUTH disagrees with the local majority,
    # the agreement model (targets +1, the majority) and the outcome model (targets
    # -1, the truth) move in OPPOSITE directions — that is the whole ablation.
    agr = AgreementTrustModel(K); agr.update(peer_leanings, outcome=-1)  # truth = -1
    out = TrustModel(K); out.update(peer_leanings, outcome=-1)           # truth = -1
    assert not np.allclose(agr.e, out.e), (
        "when truth contradicts the local majority, agreement and outcome trust diverge")
    assert np.allclose(agr.e, e_pos), "agreement model always targets the local majority (ignores truth)"


# ---------------------------------------------------------------------------
# (7) oracle beats no-trust on the decisive cluster.
# ---------------------------------------------------------------------------
def test_oracle_beats_notrust_on_cluster():
    # Honest majority leans +1 (truth=1); a colluding cluster of -1 votes is large
    # enough to flip the flat majority but the honest-only majority stays correct.
    honest = np.ones(5)            # 5 honest lean +1
    cluster = -np.ones(6)          # 6 adversaries lean -1 (drag the flat vote wrong)
    leanings = np.concatenate([honest, cluster])
    normal = np.array([True] * 5 + [False] * 6)
    assert oracle_aggregator(leanings, normal) == 1, "honest-only majority recovers truth"
    assert notrust_aggregator(leanings) == 0, "flat majority is dragged to the WRONG bit"


# ---------------------------------------------------------------------------
# (8) AUC / Spearman edge cases.
# ---------------------------------------------------------------------------
def test_trust_auc_and_spearman_edges():
    gt = np.array([0, 0, 1, 1])  # 0=honest, 1=adversary
    perfect = np.array([0.9, 0.8, 0.2, 0.1])  # honest strictly above adversary
    assert trust_auc(perfect, gt) == 1.0, "perfect separation -> AUC 1.0"
    identical = np.array([0.5, 0.5, 0.5, 0.5])
    assert trust_auc(identical, gt) == 0.5, "no separation -> AUC 0.5"
    inverted = np.array([0.1, 0.2, 0.8, 0.9])  # cluster looks MORE trusted
    assert trust_auc(inverted, gt) == 0.0, "inversion -> AUC 0.0 (failure signature)"

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])  # strictly monotone
    assert abs(spearman(x, y) - 1.0) < 1e-9, "monotone pair -> Spearman +1"
    assert trust_gap(perfect, gt) > 0.0, "positive trust gap when honest out-rank adversary"


# ---------------------------------------------------------------------------
# (9) end-to-end smoke of the outcome-vs-agreement ablation.
# ---------------------------------------------------------------------------
def test_ablation_smoke_runs():
    K, deg, n_adv, seed = 18, 6, 6, 0
    evidences = make_evidences(20, seed)
    res = run_ablation(K, deg, n_adv, seed, evidences)
    assert set(res) == {"outcome", "agreement", "delta"}
    for arm in ("outcome", "agreement"):
        sc = res[arm]
        assert 0.0 <= sc["accuracy"] <= 1.0
        assert "auc" in sc and "prf" in sc
    # Directional smoke (loose): outcome-grounding should not separate WORSE than
    # agreement on the clustered regime. nan-safe comparison.
    ao, aa = res["outcome"]["auc"], res["agreement"]["auc"]
    if ao == ao and aa == aa:  # both non-nan
        assert ao >= aa - 1e-9, "outcome arm AUC should be >= agreement arm AUC (directional)"


def test_build_clustered_society_places_adversaries_on_cluster():
    K, deg, n_adv, seed = 18, 6, 6, 0
    soc, gt, S = build_clustered_society(K, deg, n_adv, seed)
    assert soc.K == K and soc.adj.shape == (K, K)
    assert set(np.nonzero(gt == 1)[0].tolist()) == set(S), "gt mask matches the graph cluster S"
    assert cluster_is_connected(soc.adj, S), "adversary ids sit on a contiguous graph cluster"

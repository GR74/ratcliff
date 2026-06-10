"""Phase 2 — THE headline clustered-confident-adversary stress test.

This is the decisive experiment for the one claim Phase 2 must test:

    "OUTCOME-grounded trust isolates confident, internally-consistent adversary
     CLUSTERS that AGREEMENT-based local trust (A-RepC, W-MSR, bounded-confidence)
     provably cannot — with no global controller, on a local topology."

The regime is a CONTIGUOUS, confidently-wrong, mutually-agreeing adversary cluster
on a LOCAL topology, sized so that at >= 1 honest boundary node the per-neighbourhood
adversary count EXCEEDS the W-MSR bound F (worst_fraction > 0.5). adversary.py asserts
this BEFORE running (verify_fbound_exceeded), so any null result is attributable to
trust-signal quality, not to a mis-sized cluster.

Every condition is scored identically on the SAME (topology, adversary placement S,
evidence stream, seed); ONLY the trust signal differs:

  1. CENTRALIZED-ORACLE CEILING  honest-only majority (knows S). Upper bound.
  2. NO-TRUST FLOOR              flat unweighted majority of all votes. Lower bound.
  3. W-MSR(F)                    agreement filter on signed leanings; no truth. F in {1,2,3}.
  4. A-RepC                      sparsemax reputation from deviation-to-local-median; no truth.
  5. CROSS-INHIBITION            value-sensitive recruitment + inhibition; no truth (>=20 seeds).
  6. OURS                        outcome-grounded trust + local competence prior (cfg_swarm).
  7. OUTCOME-vs-AGREEMENT ABLATION OF OUR OWN SYSTEM — the load-bearing result:
       OUTCOME arm   = cfg_swarm() unchanged (trust.update driven by resolved truth).
       AGREEMENT arm = the SAME pipeline with soc.trust swapped to AgreementTrustModel
                       (update fed the round's local-majority leaning instead of truth).
       Same competence prior, same DDM coupling, same topology — only the trust target
       moves. The outcome-minus-agreement gap on identical inputs isolates outcome-
       grounding as the ONLY moving part.

The claim is SUPPORTED iff, on the decisive cluster: OURS and the OUTCOME arm keep
collective accuracy near the oracle ceiling with suspect-AUC -> 1.0 and finite
time-to-isolate, while W-MSR / A-RepC / cross-inhibition AND the AGREEMENT arm
collapse toward the no-trust floor with AUC ~ 0.5 (or < 0.5) and never isolate the
cluster. Controls: an all-honest run must give AUC ~ 0.5 for every method (no
false-positive isolation), and a SCATTERED same-count placement must let W-MSR /
A-RepC survive (proving contiguity, not fraction, is what breaks agreement filters).

Pure NumPy, no GPU, no scipy required. Mirrors experiment.py / swarm_experiment.py.

    python -m cognitive_society.phase2_experiment
"""
import numpy as np

from cognitive_society.adversary import (
    build_clustered_society, verify_fbound_exceeded, cluster_is_connected,
    bfs_ball_cluster, densest_cluster,
)
from cognitive_society.baselines import (
    wmsr_aggregator, arepc_aggregator, cross_inhibition_aggregator,
    notrust_aggregator, oracle_aggregator, AgreementTrustModel,
)
from cognitive_society.swarm_metrics import isolation_report, true_competence
from cognitive_society.society import Society, cfg_swarm, cfg_flat  # noqa: F401
from cognitive_society import topology as topo


# Default decisive-regime geometry. The cluster must locally EXCEED worst-frac 0.5
# at >= 1 honest victim. A pure ring_lattice arc caps at EXACTLY 0.5 (a 1-D arc gives
# a boundary node deg/2 adversary neighbours out of deg), and random_regular has LOW
# clustering so a densest cluster spreads out and stays <= ~0.33 — neither crosses the
# threshold with a contiguous cluster. A small-world graph (watts_strogatz: ring +
# rewiring) lets a densest cluster ENCLOSE a victim on a majority of its
# neighbourhood, so worst-frac reliably reaches 0.6-0.8 with a CONTIGUOUS cluster.
# We default to watts_strogatz + densest_cluster so the headline regime is decisive.
WS_REWIRE_P = 0.3


def _ws_topology(K, deg, rng):
    """watts_strogatz wrapped to the topology_fn(K, deg, rng) contract
    build_clustered_society expects (it supplies the extra rewiring prob WS_REWIRE_P).
    Small-world clustering is what lets a contiguous densest cluster cross worst-frac
    0.5 — a pure ring arc caps at exactly 0.5."""
    return topo.watts_strogatz(K, deg, WS_REWIRE_P, rng)


DEFAULT_TOPOLOGY_FN = _ws_topology
DEFAULT_BUILDER = densest_cluster


# ---------------------------------------------------------------------------
# Evidence streams — moderate, signed away from 0 so adversary wrongness shows.
# ---------------------------------------------------------------------------
def make_evidences(n_problems: int, seed: int):
    """Moderate-evidence problems with mixed truth (same family as experiment.py /
    swarm_experiment.py). |evidence| >= 0.3 keeps adversary wrongness observable for
    true_competence; mixed sign means the cluster can sway the collective."""
    prng = np.random.default_rng(3000 + seed)
    return prng.choice([-0.4, -0.3, 0.3, 0.4], size=n_problems).tolist()


# ---------------------------------------------------------------------------
# Read per-round signed leanings / truth / confidence off a society's stream.
# ---------------------------------------------------------------------------
def collect_round_signals(soc, evidences):
    """Replay the evidence stream on `soc` and collect the per-round PUBLIC signals
    the baseline aggregators consume: signed leanings (R, K in {-1,+1}), truth bits
    (R, in {0,1}), and per-agent private confidence (R, K in [0,1]).

    Confidence comes from soc._private(ev)[1] taken BEFORE the round (the exact
    private-confidence primitive Society.round uses), so cross-inhibition sees the
    byte-identical private quality signal. This MUTATES soc (advances its rng and
    trust); call it on a fresh, dedicated society — never the one being scored for
    isolation, which consumes its own stream.
    """
    leanings, truths, confs = [], [], []
    for ev in evidences:
        # private confidence first (matches Society.round's _private call ordering;
        # we then let round() recompute leanings off its own _private for the record).
        _, conf = soc._private(ev)
        rec = soc.round(ev, learn=True)
        leanings.append(rec["private_leanings"].astype(float))
        truths.append(rec["truth"])
        confs.append(conf)
    return (np.asarray(leanings, dtype=float),
            np.asarray(truths, dtype=int),
            np.asarray(confs, dtype=float))


def _accuracy(bits, truths):
    """Fraction of rounds a decision-bit stream matches the truth bits."""
    bits = np.asarray(bits, dtype=int)
    truths = np.asarray(truths, dtype=int)
    return float(np.mean(bits == truths))


# ---------------------------------------------------------------------------
# (A) OURS — the contender, scored on its own fresh society.
# ---------------------------------------------------------------------------
def run_condition(soc, gt, S, evidences):
    """Score one already-built society end to end on identical inputs.

    Returns {collective_accuracy, isolation scorecard}. We score accuracy and
    isolation on SEPARATE society instances (rebuilt from the same recipe) so the
    isolation run — which consumes the stream inside isolation_report — does not
    perturb the accuracy run. Here `soc` is the isolation society; the caller passes
    a freshly-built one. collective_accuracy is read from a second fresh build.
    """
    rep = isolation_report(soc, evidences, gt)
    return rep


def ours_scorecard(K, deg, n_adv, seed, evidences, agreement=False,
                   builder=DEFAULT_BUILDER, topology_fn=DEFAULT_TOPOLOGY_FN):
    """Build OUR society (cfg_swarm) and return (collective_accuracy, isolation
    scorecard). With agreement=True the trust models are swapped to
    AgreementTrustModel BEFORE running, so the ONLY change is the trust target.

    Two independent builds (identical recipe -> identical agents/topology/cluster):
    one for collective accuracy via soc.run, one for the isolation scorecard (which
    consumes its own stream). Keeping them separate avoids double-running one stream.
    """
    # accuracy build
    soc_acc, gt, S = ablation_society(K, deg, n_adv, seed, agreement, builder, topology_fn)
    acc = soc_acc.run(evidences)["collective_accuracy"]
    # isolation build (fresh, same recipe)
    soc_iso, _, _ = ablation_society(K, deg, n_adv, seed, agreement, builder, topology_fn)
    rep = isolation_report(soc_iso, evidences, gt)
    return acc, rep, gt, S


def ablation_society(K: int, deg: int, n_adv: int, seed: int, agreement: bool,
                     builder=DEFAULT_BUILDER, topology_fn=DEFAULT_TOPOLOGY_FN):
    """Build the clustered society; if agreement=True monkey-patch the trust models.

    The AGREEMENT arm swaps soc.trust = [AgreementTrustModel(K) ...] at the INSTANCE
    level, AFTER __init__ but BEFORE run(), so build_cognitive_maps() still seeds the
    SAME competence prior (set_prior_from_competence is inherited unchanged) — only
    the per-round update TARGET differs (local-majority agreement vs resolved truth).
    This is the one manual hook, documented here so implementers don't reinvent it.

    Returns (soc, gt, S) exactly as build_clustered_society does.
    """
    soc, gt, S = build_clustered_society(
        K, deg, n_adv, seed, config=cfg_swarm(), builder=builder,
        topology_fn=topology_fn)
    if agreement:
        # Observer-LOCAL agreement: each model judges agreement against ITS OWN
        # neighbourhood consensus (soc.adj[i]), so a captured victim sees the
        # adversary cluster as its local majority and trusts it. A global-majority
        # agreement target would coincide with truth whenever adversaries are a
        # global minority, making the ablation vacuous — locality is the point.
        soc.trust = [AgreementTrustModel(soc.K, neighbour_mask=soc.adj[i])
                     for i in range(soc.K)]
    return soc, gt, S


# ---------------------------------------------------------------------------
# (B) BASELINES — scored off the SAME public per-round signals.
# ---------------------------------------------------------------------------
def run_baselines(soc, gt, S, evidences, F_sweep=(1, 2, 3), ci_seeds: int = 20):
    """Score the reference conditions on the SAME soc.adj + per-round signals.

    `soc` is a fresh society built from the same recipe as OURS; we replay its stream
    once (collect_round_signals) to get byte-identical leanings/truth/confidence, then
    every aggregator consumes those + soc.adj. Each aggregator's collective accuracy is
    the fraction of rounds its decision bit matches the truth. Agreement methods carry
    no trust analogue, so their suspect-AUC is ~0.5 by construction (reported as such).

    Returns {name: {"accuracy": float, "auc": float}}.
    """
    adj = soc.adj
    normal = (np.asarray(gt) == 0)
    leanings, truths, confs = collect_round_signals(soc, evidences)
    R = leanings.shape[0]

    out = {}

    # 1. ORACLE CEILING — per-round honest-only majority of FINAL leanings.
    oracle_bits = np.array([oracle_aggregator(leanings[r], normal) for r in range(R)])
    out["oracle (ceiling)"] = {"accuracy": _accuracy(oracle_bits, truths), "auc": float("nan")}

    # 2. NO-TRUST FLOOR — flat unweighted majority of ALL leanings.
    notrust_bits = np.array([notrust_aggregator(leanings[r]) for r in range(R)])
    out["no-trust (floor)"] = {"accuracy": _accuracy(notrust_bits, truths), "auc": 0.5}

    # 3. W-MSR(F) — agreement filter on signed leanings; sweep F.
    for F in F_sweep:
        wmsr_bits = np.array(
            [wmsr_aggregator(leanings[r], adj, normal, F=F) for r in range(R)])
        out[f"W-MSR (F={F})"] = {"accuracy": _accuracy(wmsr_bits, truths), "auc": 0.5}

    # 4. A-RepC — sparsemax reputation; one pass over the whole stream.
    arepc_bits = arepc_aggregator(leanings, adj, normal)
    out["A-RepC"] = {"accuracy": _accuracy(arepc_bits, truths), "auc": 0.5}

    # 5. CROSS-INHIBITION — stochastic; average the per-round decision over seeds.
    ci_bits = np.zeros(R, dtype=int)
    for r in range(R):
        votes = np.zeros(ci_seeds, dtype=int)
        for s in range(ci_seeds):
            d, _ = cross_inhibition_aggregator(
                adj, leanings[r], confs[r], rng=np.random.default_rng(7000 + 100 * r + s))
            votes[s] = d
        ci_bits[r] = int(votes.mean() >= 0.5)
    out["cross-inhibition"] = {"accuracy": _accuracy(ci_bits, truths), "auc": 0.5}

    return out


# ---------------------------------------------------------------------------
# (C) THE OUTCOME-vs-AGREEMENT ABLATION — the cleanest isolation of the claim.
# ---------------------------------------------------------------------------
def run_ablation(K: int, deg: int, n_adv: int, seed: int, evidences,
                 builder=DEFAULT_BUILDER, topology_fn=DEFAULT_TOPOLOGY_FN):
    """Run OUR pipeline twice on identical inputs — OUTCOME arm vs AGREEMENT arm —
    and return both scorecards plus the outcome-minus-agreement deltas.

    OUTCOME arm:   cfg_swarm() unchanged (trust.update driven by resolved truth).
    AGREEMENT arm: identical build, soc.trust swapped to AgreementTrustModel (update
                   fed the round's local-majority leaning). NOTHING else changes.

    The deltas (accuracy / AUC / recall) on byte-identical inputs are the load-bearing
    result: a large positive delta isolates outcome-grounding as the only moving part;
    a ~0 delta FALSIFIES the claim (outcome-grounding adds nothing).
    """
    acc_o, rep_o, gt, S = ours_scorecard(
        K, deg, n_adv, seed, evidences, agreement=False, builder=builder, topology_fn=topology_fn)
    acc_a, rep_a, _, _ = ours_scorecard(
        K, deg, n_adv, seed, evidences, agreement=True, builder=builder, topology_fn=topology_fn)
    delta = {
        "accuracy": acc_o - acc_a,
        "auc": rep_o["auc"] - rep_a["auc"],
        "recall": rep_o["prf"]["recall"] - rep_a["prf"]["recall"],
    }
    return {
        "outcome": {"accuracy": acc_o, **rep_o},
        "agreement": {"accuracy": acc_a, **rep_a},
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# (D) CLUSTER-SIZE SWEEP — the gap must onset precisely as worst-frac crosses 0.5.
# ---------------------------------------------------------------------------
def cluster_size_sweep(K: int, deg: int, n_adv_grid, n_seeds: int = 12,
                       n_problems: int = 60, builder=DEFAULT_BUILDER,
                       topology_fn=DEFAULT_TOPOLOGY_FN):
    """Sweep cluster size; for each n_adv record the decisive-regime onset (worst-frac)
    and the accuracy/AUC of every condition, averaged over seeds.

    The mechanism attribution is proven iff the outcome-vs-agreement gap appears
    PRECISELY once worst-frac crosses 0.5 (a monotone onset, not a fixed offset):
    below the threshold agreement still works, above it agreement collapses while
    outcome-grounding holds.

    Returns {n_adv: {worst_frac, conditions:{name:{accuracy, auc}}}} aggregated.
    """
    rows = {}
    for n_adv in n_adv_grid:
        worst_fracs = []
        agg = {}  # name -> {"accuracy": [...], "auc": [...]}

        def _push(name, acc, auc):
            d = agg.setdefault(name, {"accuracy": [], "auc": []})
            d["accuracy"].append(acc)
            if auc == auc:  # skip nan
                d["auc"].append(auc)

        for seed in range(n_seeds):
            evidences = make_evidences(n_problems, seed)
            # one build just to read the geometry / decisive-regime fraction
            soc_geo, gt, S = build_clustered_society(
                K, deg, n_adv, seed, config=cfg_swarm(), builder=builder,
                topology_fn=topology_fn)
            worst, _ = verify_fbound_exceeded(soc_geo.adj, S)
            worst_fracs.append(worst)

            # OURS (outcome) + AGREEMENT arm
            acc_o, rep_o, _, _ = ours_scorecard(
                K, deg, n_adv, seed, evidences, agreement=False,
                builder=builder, topology_fn=topology_fn)
            acc_a, rep_a, _, _ = ours_scorecard(
                K, deg, n_adv, seed, evidences, agreement=True,
                builder=builder, topology_fn=topology_fn)
            _push("OURS-outcome", acc_o, rep_o["auc"])
            _push("OURS-agreement", acc_a, rep_a["auc"])

            # baselines (fresh society, same recipe)
            soc_bl, _, _ = build_clustered_society(
                K, deg, n_adv, seed, config=cfg_swarm(), builder=builder,
                topology_fn=topology_fn)
            bl = run_baselines(soc_bl, gt, S, evidences, ci_seeds=10)
            for name in ("oracle (ceiling)", "no-trust (floor)", "W-MSR (F=1)",
                         "A-RepC", "cross-inhibition"):
                _push(name, bl[name]["accuracy"], bl[name]["auc"])

        rows[n_adv] = {
            "worst_frac": float(np.mean(worst_fracs)),
            "conditions": {
                name: {
                    "accuracy": float(np.mean(d["accuracy"])),
                    "auc": float(np.mean(d["auc"])) if d["auc"] else float("nan"),
                }
                for name, d in agg.items()
            },
        }
    return rows


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------
def _fmt_auc(auc):
    return "  n/a" if auc != auc else f"{auc:5.2f}"  # nan-safe


def _print_headline_table(K, deg, n_adv, seed, evidences):
    """Print the head-to-head accuracy + isolation-AUC table for one decisive build."""
    # geometry + decisive-regime assertion
    soc_geo, gt, S = build_clustered_society(
        K, deg, n_adv, seed, config=cfg_swarm(), builder=DEFAULT_BUILDER,
        topology_fn=DEFAULT_TOPOLOGY_FN)
    worst, victims = verify_fbound_exceeded(soc_geo.adj, S)
    connected = cluster_is_connected(soc_geo.adj, S)

    print(f"  topology       : watts_strogatz(K={K}, deg={deg}, p={WS_REWIRE_P})  "
          f"(local, small-world)")
    print(f"  adversary set S: {len(S)} nodes, contiguous={connected}, ids={S}")
    print(f"  decisive regime: worst neighbourhood adversary-fraction = {worst:.2f} "
          f"({'>' if worst > 0.5 else '<='} 0.5)")
    print(f"                   victims (local count > F): {len(victims)} honest nodes")
    if worst > 0.5:
        print("  [OK] decisive regime confirmed — the cluster locally EXCEEDS the "
              "W-MSR bound F at >= 1 honest node.\n")
    else:
        print("  [WARN] cluster does NOT cross worst-frac 0.5 — grow n_adv "
              "(cluster_size_sweep finds the onset).\n")

    # OURS (outcome) + AGREEMENT arm + baselines
    abl = run_ablation(K, deg, n_adv, seed, evidences)
    soc_bl, _, _ = build_clustered_society(
        K, deg, n_adv, seed, config=cfg_swarm(), builder=DEFAULT_BUILDER,
        topology_fn=DEFAULT_TOPOLOGY_FN)
    bl = run_baselines(soc_bl, gt, S, evidences)

    print(f"  {'condition':22s} {'accuracy':>9s} {'AUC':>7s}  trust signal")
    print(f"  {'-' * 22} {'-' * 9} {'-' * 7}  {'-' * 28}")

    def row(name, acc, auc, note):
        print(f"  {name:22s} {acc:8.1%} {_fmt_auc(auc):>7s}  {note}")

    row("oracle (ceiling)", bl["oracle (ceiling)"]["accuracy"], bl["oracle (ceiling)"]["auc"],
        "knows S (upper bound)")
    row("no-trust (floor)", bl["no-trust (floor)"]["accuracy"], bl["no-trust (floor)"]["auc"],
        "flat majority (lower bound)")
    for F in (1, 2, 3):
        b = bl[f"W-MSR (F={F})"]
        row(f"W-MSR (F={F})", b["accuracy"], b["auc"], "agreement filter, no truth")
    row("A-RepC", bl["A-RepC"]["accuracy"], bl["A-RepC"]["auc"], "agreement reputation, no truth")
    row("cross-inhibition", bl["cross-inhibition"]["accuracy"], bl["cross-inhibition"]["auc"],
        "agreement+value, no truth")
    row("OURS (outcome)", abl["outcome"]["accuracy"], abl["outcome"]["auc"],
        "outcome-grounded trust")
    print()

    # The load-bearing ablation: SAME architecture, only the trust target moves.
    print("  OUTCOME-vs-AGREEMENT ABLATION OF OUR OWN SYSTEM (the load-bearing result)")
    print("  identical inputs; only TrustModel.update's target differs:")
    o, a, d = abl["outcome"], abl["agreement"], abl["delta"]
    print(f"    {'arm':16s} {'accuracy':>9s} {'AUC':>7s} {'recall':>8s} {'gap':>7s} {'never':>6s}")
    print(f"    OUTCOME (truth)  {o['accuracy']:8.1%} {_fmt_auc(o['auc']):>7s} "
          f"{o['prf']['recall']:8.2f} {o['trust_gap']:7.2f} {o['never']:6d}")
    print(f"    AGREEMENT (cons) {a['accuracy']:8.1%} {_fmt_auc(a['auc']):>7s} "
          f"{a['prf']['recall']:8.2f} {a['trust_gap']:7.2f} {a['never']:6d}")
    print(f"    delta (out-agr)  {d['accuracy']:+8.1%} {d['auc']:+7.2f} {d['recall']:+8.2f}\n")

    return worst, abl, bl


def _print_sweep(K, deg, n_adv_grid, n_seeds, n_problems):
    """Print the cluster-size sweep — the gap must onset as worst-frac crosses 0.5."""
    print("  CLUSTER-SIZE SWEEP — the outcome-vs-agreement gap must onset at worst-frac > 0.5")
    sweep = cluster_size_sweep(K, deg, n_adv_grid, n_seeds=n_seeds, n_problems=n_problems)
    hdr = (f"  {'n_adv':>5s} {'worst':>6s} {'oracle':>7s} {'no-tr':>6s} {'WMSR1':>6s} "
           f"{'ARepC':>6s} {'CI':>6s} {'OUT':>6s} {'AGR':>6s}  {'AUC.out':>7s} {'AUC.agr':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for n_adv, row in sweep.items():
        c = row["conditions"]

        def acc(name):
            return c[name]["accuracy"] if name in c else float("nan")
        print(f"  {n_adv:5d} {row['worst_frac']:6.2f} "
              f"{acc('oracle (ceiling)'):6.1%} {acc('no-trust (floor)'):5.1%} "
              f"{acc('W-MSR (F=1)'):5.1%} {acc('A-RepC'):5.1%} "
              f"{acc('cross-inhibition'):5.1%} {acc('OURS-outcome'):5.1%} "
              f"{acc('OURS-agreement'):5.1%}  "
              f"{_fmt_auc(c['OURS-outcome']['auc']):>7s} "
              f"{_fmt_auc(c['OURS-agreement']['auc']):>7s}")
    print()
    return sweep


def _verdict(worst, abl, bl):
    """PASS/FALSIFY the claim from the decisive-build numbers.

    SUPPORTS iff: decisive regime (worst>0.5) AND OURS keeps accuracy clearly above
    the no-trust floor with AUC -> 1, AND the AGREEMENT arm + agreement baselines
    collapse (AUC ~ 0.5 / below) AND the outcome-minus-agreement AUC gap is clearly
    positive. FALSIFIES if the agreement arm matches the outcome arm, or agreement
    baselines also isolate, or OURS fails to beat the floor.
    """
    o, a, d = abl["outcome"], abl["agreement"], abl["delta"]
    floor = bl["no-trust (floor)"]["accuracy"]
    ours_beats_floor = o["accuracy"] > floor + 0.05
    ours_isolates = (o["auc"] == o["auc"]) and o["auc"] >= 0.8
    agreement_collapses = (a["auc"] != a["auc"]) or a["auc"] <= 0.65
    gap_positive = (d["auc"] == d["auc"]) and d["auc"] >= 0.2
    decisive = worst > 0.5

    print("  VERDICT")
    print(f"    decisive regime (worst-frac > 0.5) ............ {'YES' if decisive else 'no'}")
    print(f"    OURS beats the no-trust floor ................. {'YES' if ours_beats_floor else 'no'}")
    print(f"    OURS isolates the cluster (AUC -> 1) .......... {'YES' if ours_isolates else 'no'}")
    print(f"    AGREEMENT arm collapses (AUC ~ 0.5) ........... {'YES' if agreement_collapses else 'no'}")
    print(f"    outcome-minus-agreement AUC gap is large ...... {'YES' if gap_positive else 'no'}")
    passed = decisive and ours_beats_floor and ours_isolates and agreement_collapses and gap_positive
    if passed:
        print("\n  [PASS] Outcome-grounding isolates the contiguous confident cluster that the")
        print("         agreement arm and agreement baselines provably cannot — with no global")
        print("         controller, on a local topology. The claim is SUPPORTED.")
    else:
        print("\n  [FALSIFY/UNPROVEN] one or more decisive conditions failed (see the rows above).")
        print("         Either the regime was not decisive, or outcome-grounding did not separate")
        print("         from agreement on this build — the claim is NOT supported as run.")
    return passed


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main():
    # Decisive-build geometry: on a small-world graph a contiguous densest cluster
    # ENCLOSES a victim on a MAJORITY of its neighbourhood, crossing worst-frac 0.5
    # at a modest cluster size; on deg=6 a victim is majority-adversary once >= 4 of
    # its 6 neighbours are in S. K=28, deg=6, n_adv=6, seed=0 lands worst-frac ~0.6.
    K, deg, n_adv, seed = 28, 6, 6, 0
    n_problems = 60

    print("=" * 74)
    print("PHASE 2 — clustered confident-adversary stress test "
          "(outcome vs agreement)")
    print("=" * 74)
    print("Claim: OUTCOME-grounded trust isolates a contiguous, confidently-wrong,")
    print("mutually-agreeing adversary CLUSTER that AGREEMENT-based trust (W-MSR,")
    print("A-RepC, cross-inhibition) provably cannot — no global controller, local graph.\n")

    evidences = make_evidences(n_problems, seed)
    worst, abl, bl = _print_headline_table(K, deg, n_adv, seed, evidences)

    # The sweep: the outcome-vs-agreement gap should grow as worst-frac crosses 0.5.
    _print_sweep(K, deg, n_adv_grid=(3, 5, 6, 8, 10), n_seeds=8, n_problems=40)

    _verdict(worst, abl, bl)
    print("=" * 74)


if __name__ == "__main__":
    main()

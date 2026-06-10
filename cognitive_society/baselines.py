"""Phase 2 — agreement-based / reference aggregators (the comparison set).

These are the conditions the outcome-grounded Society must BEAT. Each is a pure,
self-contained NumPy function that consumes ONLY public state — `soc.adj` (bool
KxK), the per-round signed leanings (`{-1,+1}`) and truth bit read off
`Society.round`, the `normal` mask, per-agent confidence — and returns a
collective decision bit (or a per-round bit stream). They NEVER touch Society
internals, so W-MSR / A-RepC / cross-inhibition see byte-identical inputs to OURS
on the same (topology, adversary placement, evidence stream, seed).

The family and what each proves on the decisive contiguous-cluster regime
(some honest node's neighbourhood is adversary-majority, > the W-MSR bound F):

  oracle_aggregator       CENTRALIZED-ORACLE CEILING. Honest-only majority (knows
                          the `normal` mask). Upper bound on achievable accuracy.
  notrust_aggregator      NO-TRUST FLOOR. Flat unweighted majority of all votes
                          (= cfg_flat's decision rule). The cluster drags it wrong.
  wmsr_aggregator         FAITHFUL W-MSR(F) (LeBlanc/Zhang/Koutsoukos/Sundaram,
                          IEEE JSAC 2013). Agreement filter on signed leanings,
                          no truth signal. Sweep F in {1,2,3}.
  arepc_aggregator        A-REPC (sparsemax reputation from deviation-to-local-
                          median loss; arXiv 2605.11357). Agreement, no truth.
  cross_inhibition_aggregator  Value-sensitive recruitment + cross-inhibition
                          (arXiv 2509.07561). Agreement + value, no truth. Average
                          over >=20 stochastic seeds.

Plus AgreementTrustModel: a duck-typed drop-in for comms.TrustModel whose
`update(peer_leanings, outcome)` IGNORES the truth-derived outcome and instead
targets the LOCAL MAJORITY sign of the peer leanings. Swapping
`soc.trust = [AgreementTrustModel(K) for _ in range(K)]` after __init__ but before
run() turns OUR pipeline into its agreement-trust ablation — the ONLY moving part
is the trust target, nothing else (same competence prior, same DDM coupling, same
topology). That outcome-vs-agreement gap on our OWN system is the load-bearing
result of Phase 2.

Why agreement defeats every method here: each one penalizes a neighbour for
deviating from the LOCAL median/consensus, never from RESOLVED TRUTH. A contiguous
colluding cluster that locally exceeds F IS the local majority, so its members
have ~zero deviation and stay fully trusted while the honest minority — now the
deviator — gets crushed. Outcome-grounded trust scores a peer by whether its past
leanings matched the truth, which is independent of agreement, so the confident
wrong cluster accrues outcome-penalty even at zero agreement-penalty.

Pure NumPy, no GPU, no scipy. Honors social_drift's |drift|<=social_gain bound
(A-RepC reputations live on the simplex). Zero edits to comms.py / society.py.
"""
import numpy as np

from cognitive_society.comms import TrustModel, social_drift  # noqa: F401  (contract anchors)
from cognitive_society.topology import degrees, mean_degree  # noqa: F401  (diagnostics)


# ====================================================================== #
#  W-MSR(F) — faithful Weighted-Mean-Subsequence-Reduced consensus       #
#  LeBlanc/Zhang/Koutsoukos/Sundaram, IEEE JSAC 31(4):766-781, 2013.     #
# ====================================================================== #
def wmsr_step(x: np.ndarray, adj: np.ndarray, F: int) -> np.ndarray:
    """One faithful W-MSR(F) consensus step on scalar beliefs `x` over `adj`.

    Each node i, w.r.t. its OWN value x_i:
      - removes up to F neighbour values STRICTLY GREATER than x_i (all of them if
        fewer than F are strictly greater — never a non-extreme value);
      - removes up to F neighbour values STRICTLY LESS than x_i (same rule);
      - takes a UNIFORM convex average of self + the surviving neighbours,
        w = 1/(1 + d_i - |R_i|) >= alpha > 0.

    The "w.r.t. own value" trimming and the ALWAYS-keep-self property are what
    distinguish W-MSR from plain MSR; both are required for the paper's safety
    lemma (every update stays within [min, max] of the NORMAL nodes' values).

    x   : (K,) float — current scalar beliefs of ALL nodes (normal + adversary).
    adj : (K,K) bool — adj[i, j] True iff j is an in-neighbour of i (Society.adj).
    F   : int        — W-MSR parameter (max adversaries tolerated per neighbourhood).
    """
    x = np.asarray(x, dtype=float)
    K = x.shape[0]
    x_next = x.copy()  # adversaries overwrite their own entries after the step
    for i in range(K):
        nb = np.nonzero(adj[i])[0]
        if nb.size == 0:
            continue  # isolated node keeps its own value (self-weight = 1)
        vals = x[nb]
        xi = x[i]
        greater = nb[vals > xi]
        smaller = nb[vals < xi]
        # descending by value -> the F strictly-largest; slice is safe if < F exist
        g_sorted = greater[np.argsort(-x[greater])]
        s_sorted = smaller[np.argsort(x[smaller])]
        removed = set(g_sorted[:F].tolist()) | set(s_sorted[:F].tolist())
        keep = [int(j) for j in nb.tolist() if j not in removed]
        used = keep + [i]  # ALWAYS retain own value (the W-MSR property)
        w = 1.0 / len(used)  # = 1/(1 + d_i - |R_i|); uniform, all weights >= alpha
        x_next[i] = w * x[used].sum()
    return x_next


def wmsr_aggregator(leanings: np.ndarray, adj: np.ndarray, normal: np.ndarray,
                    F: int = 1, adv_value: float = None, steps: int = 200) -> int:
    """Drive W-MSR(F) to consensus on SIGNED leanings -> a collective decision bit.

    Adversaries are stubborn MALICIOUS nodes pinned to an EXTREME `adv_value` and
    re-asserted after every step (so they never get averaged away). The binary
    readout is a post-hoc sign threshold on the mean of the NORMAL nodes' settled
    scalars — feeding magnitude-bearing leanings (not raw 0/1 votes) is what makes
    the extreme-trimming meaningful.

    leanings  : (K,) signed private leanings in {-1, +1} (the W-MSR initial state).
    adj       : (K,K) bool neighbour matrix (Society.adj).
    normal    : (K,) bool — True for honest nodes; adversaries are driven externally.
    F         : W-MSR per-neighbourhood tolerance. Sweep {1, 2, 3}.
    adv_value : the stubborn wrong-side scalar; default = -sign(normal-majority) * B
                with B = max|leanings| (confident extreme on the opposite side).
    steps     : averaging iterations (W-MSR is asymptotic; ~200 ample for K<=50).

    Returns 1 if the normals settle positive, else 0.
    """
    leanings = np.asarray(leanings, dtype=float)
    normal = np.asarray(normal, dtype=bool)
    K = leanings.shape[0]
    x = leanings.copy()

    B = float(np.max(np.abs(leanings))) if np.any(leanings) else 1.0
    if adv_value is None:
        # Default: the cluster holds the EXTREME scalar OPPOSITE the honest tendency
        # -> the malicious "hold max/min" construction (Prop 2 / Thm 1 necessity).
        if np.any(normal):
            honest_mean = float(np.mean(leanings[normal]))
        else:
            honest_mean = 0.0
        adv_sign = -1.0 if honest_mean >= 0.0 else 1.0
        adv_value = adv_sign * B

    adversary = ~normal
    x[adversary] = adv_value
    for _ in range(steps):
        x = wmsr_step(x, adj, F)
        x[adversary] = adv_value  # malicious nodes re-assert their fixed value
    consensus = float(np.mean(x[normal])) if np.any(normal) else float(np.mean(x))
    return int(consensus > 0.0)


# ====================================================================== #
#  A-RepC — sparsemax reputation from deviation-to-local-median loss      #
#  arXiv:2605.11357 (Martins & Astudillo 2016 sparsemax projection).     #
# ====================================================================== #
def sparsemax_masked(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact sparsemax (Martins & Astudillo 2016) restricted to `mask==True`.

    sparsemax(z) = argmax_{p in simplex} (p . z - 1/2 ||p||^2). Unlike softmax it
    has a flat boundary, so low-score entries get EXACTLY zero weight (hard
    truncation) rather than a small positive weight. Non-mask entries are excluded
    entirely (forced to 0). Returns simplex weights summing to 1 over the mask.

    z    : (K,) scores (here -eta * accumulated deviation loss).
    mask : (K,) bool — the neighbour set this projection ranges over.
    """
    z = np.asarray(z, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    p = np.zeros_like(z)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return p
    zi = z[idx]
    zs = np.sort(zi)[::-1]  # descending
    cssv = np.cumsum(zs) - 1.0
    rng = np.arange(1, zs.size + 1)
    cond = (zs - cssv / rng) > 0
    k = int(rng[cond][-1])  # support size (>=1 always; the top entry passes)
    tau = cssv[cond][-1] / k  # self-adapting threshold from the simplex projection
    p[idx] = np.maximum(zi - tau, 0.0)  # HARD truncation: scores <= tau -> 0
    return p


class ARepCTrust:
    """A-RepC per-agent local reputation: agreement-based, never reads truth.

    One instance per honest observer i, keyed to i's neighbour mask. Each round it
    measures every neighbour's deviation from the LOCAL coordinate-wise median
    (for {-1,+1} leanings this is the SIGN of the neighbour-majority leaning),
    accumulates that deviation with a forgetting factor, and projects -eta * (loss)
    onto the simplex via sparsemax -> reputation weights over neighbours.

    THE FAILURE MODE this exhibits: on a colluding cluster that locally exceeds the
    median breakdown bound, the local median IS the adversarial leaning, so every
    adversary's deviation is ~0 (full reputation) while the honest minority now
    deviates from the poisoned median and is sparsemax-truncated to zero weight.
    Agreement cannot tell an internally-consistent wrong cluster from the truth.
    """

    def __init__(self, K: int, neighbour_mask: np.ndarray, eta: float = 4.0,
                 lam: float = 0.9):
        self.K = K
        self.mask = np.asarray(neighbour_mask, dtype=bool)
        self.eta = eta  # inverse temperature (sparsemax sharpness)
        self.lam = lam  # forgetting factor for the accumulated deviation loss
        self.L = np.zeros(K, dtype=float)  # accumulated deviation loss per neighbour
        self.p = np.zeros(K, dtype=float)  # current reputation weights (simplex)

    def _coordwise_median(self, leanings: np.ndarray) -> float:
        """Local robust reference = sign of the neighbour-majority leaning.

        median of {-1,+1} values is in {-1, 0, +1}; a 0 (even split) is broken
        deterministically to +1 so the deviation loss is always well defined.
        """
        vals = leanings[self.mask]
        if vals.size == 0:
            return 1.0
        m = float(np.median(vals))
        return float(np.sign(m)) if m != 0.0 else 1.0

    def update_and_weights(self, leanings: np.ndarray) -> np.ndarray:
        """Accumulate this round's deviation loss and return reputation weights.

        Reads ONLY `leanings` — never truth. The 1-D binary specialization of the
        sup-norm loss: l_j = |leaning_j - cm| in {0, 2} (0 iff j agrees with the
        local majority). Returns weights summing to 1 over the neighbour set.
        """
        leanings = np.asarray(leanings, dtype=float)
        cm = self._coordwise_median(leanings)
        l = np.abs(leanings - cm)
        l = np.where(self.mask, l, 0.0)
        self.L = self.lam * self.L + l  # accumulate with forgetting
        z = np.where(self.mask, -self.eta * self.L, -np.inf)
        self.p = sparsemax_masked(z, self.mask)
        return self.p


def arepc_aggregator(leanings_per_round: np.ndarray, adj: np.ndarray,
                     normal: np.ndarray, eta: float = 4.0,
                     lam: float = 0.9) -> np.ndarray:
    """Run one ARepCTrust per honest node across a round stream -> per-round bits.

    For each round, every honest node forms a reputation-weighted vote over its
    neighbours (sum_j p_ij * leaning_j; the weights live on the simplex so this is
    in [-1, 1]); the collective decision is the majority sign of those weighted
    votes across honest nodes. Agreement-based, no truth.

    leanings_per_round : (R, K) signed leanings in {-1, +1}, one row per round.
    adj                : (K, K) bool neighbour matrix.
    normal             : (K,) bool — honest mask (only honest nodes vote/decide).

    Returns (R,) array of collective decision bits in {0, 1}.
    """
    leanings_per_round = np.asarray(leanings_per_round, dtype=float)
    if leanings_per_round.ndim == 1:
        leanings_per_round = leanings_per_round[None, :]
    R, K = leanings_per_round.shape
    normal = np.asarray(normal, dtype=bool)
    reps = {i: ARepCTrust(K, adj[i], eta=eta, lam=lam)
            for i in range(K) if normal[i]}
    bits = np.zeros(R, dtype=int)
    for r in range(R):
        leanings = leanings_per_round[r]
        votes = []
        for i, rep in reps.items():
            p = rep.update_and_weights(leanings)
            votes.append(float(np.sum(p * leanings)))  # reputation-weighted leaning
        if not votes:
            bits[r] = int(np.mean(leanings) > 0.0)
            continue
        bits[r] = int(np.mean(np.sign(votes)) >= 0.0)
    return bits


# ====================================================================== #
#  CROSS-INHIBITION — value-sensitive recruitment + inhibition           #
#  arXiv:2509.07561 ("antifragile" collective decision). No truth signal. #
# ====================================================================== #
def cross_inhibition_aggregator(adj: np.ndarray, leanings: np.ndarray,
                                conf: np.ndarray, eta: float = 0.1,
                                eta_a: float = 0.5, T: int = 300,
                                quorum: float = 0.75,
                                rng: np.random.Generator = None):
    """One cross-inhibition decision run over `adj` (binary collective aggregator).

    Three states per agent: A (=choice 1), B (=choice 0), U (uncommitted).
      - VALUE-SENSITIVE recruitment: an uncommitted agent meeting a committed
        neighbour adopts that option with probability scaled by its OWN private
        confidence as the option "quality" (recruitment floor keeps it unstuck).
      - CROSS-INHIBITION: a committed agent meeting an OPPOSING committed neighbour
        resets to U (opinions cancel rather than flip — the antifragile core that
        suppresses minority noise).
      - ASOCIAL term `eta` (zealots / corrupted msgs / self-discovery), biased by
        `eta_a` toward option A.
    Uses NO truth/outcome signal — that is the point: on the decisive colluding
    cluster a confidently-wrong adversary injects HIGH apparent quality for the
    wrong option, so agreement+value cannot isolate it. Average over >=20 seeds.

    leanings : (K,) signed private leanings in {-1, +1} (seed the committed state).
    conf     : (K,) private confidence in [0, 1] (the option quality).

    Returns (decision bit in {0, 1}, committed fraction). The caller compares
    `committed fraction >= quorum` for a "consensus reached" flag.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    adj = np.asarray(adj, dtype=bool)
    leanings = np.asarray(leanings, dtype=float)
    conf = np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
    K = adj.shape[0]

    state = np.where(leanings > 0, 0, 1).astype(int)  # 0=A(choice1), 1=B(choice0)
    U = np.zeros(K, dtype=bool)                        # start all committed
    qA = np.where(leanings > 0, conf, 0.0)             # value-sensitive quality
    qB = np.where(leanings < 0, conf, 0.0)
    floor = 0.05  # recruitment floor: avoids an all-U deadlock under disagreement

    for _ in range(T):
        i = int(rng.integers(K))
        nb = np.nonzero(adj[i])[0]
        if nb.size == 0:
            continue
        j = int(rng.choice(nb))
        if rng.random() < eta:  # asocial: zealot / corruption / self-discovery
            if not U[i]:
                state[i] = 0 if rng.random() < eta_a else 1
            continue
        if U[i]:  # recruitment (value-sensitive)
            if not U[j]:
                want = state[j]
                q = qA[i] if want == 0 else qB[i]
                if rng.random() < max(q, floor):
                    state[i] = want
                    U[i] = False
        else:  # cross-inhibition: opposing committed neighbour -> reset to U
            if (not U[j]) and state[j] != state[i]:
                U[i] = True

    committed = ~U
    n_comm = int(committed.sum())
    if n_comm == 0:
        return 1, 0.0
    nA = int((state[committed] == 0).sum())
    nB = int((state[committed] == 1).sum())
    frac = max(nA, nB) / n_comm
    decision = 1 if nA >= nB else 0  # majority committed option
    return decision, frac


# ====================================================================== #
#  NO-TRUST FLOOR and CENTRALIZED-ORACLE CEILING                         #
# ====================================================================== #
def notrust_aggregator(votes: np.ndarray) -> int:
    """NO-TRUST FLOOR: flat unweighted majority of ALL agents' votes/leanings.

    The decision rule of cfg_flat with no down-weighting — the lower bound that the
    colluding cluster drags toward the wrong answer. Accepts either {0,1} votes or
    {-1,+1} leanings (thresholds at the midpoint either way).

    Returns 1 if the (signed-or-binary) majority is positive, else 0.
    """
    votes = np.asarray(votes, dtype=float)
    if votes.size == 0:
        return 0
    # {-1,+1} leanings threshold at 0; {0,1} votes threshold at 0.5. Detect range.
    if votes.min() < 0.0:
        return int(np.mean(votes) > 0.0)
    return int(np.mean(votes) >= 0.5)


def oracle_aggregator(votes: np.ndarray, normal: np.ndarray) -> int:
    """CENTRALIZED-ORACLE CEILING: majority over HONEST agents only.

    A god aggregator that KNOWS the adversary set (the `normal` mask) and discards
    every adversary vote — the upper bound on achievable collective accuracy, which
    establishes the headroom the trust methods are competing for. Accepts {0,1}
    votes or {-1,+1} leanings (same midpoint detection as notrust_aggregator).

    Returns 1 if the honest majority is positive, else 0.
    """
    votes = np.asarray(votes, dtype=float)
    normal = np.asarray(normal, dtype=bool)
    if not np.any(normal):
        return notrust_aggregator(votes)
    return notrust_aggregator(votes[normal])


# ====================================================================== #
#  AgreementTrustModel — the outcome-vs-agreement ablation of OUR system  #
# ====================================================================== #
class AgreementTrustModel(TrustModel):
    """Duck-typed drop-in for comms.TrustModel that ignores the truth signal.

    Identical to TrustModel in every respect (same logit accumulator `e`, same
    logistic `trust()`, same competence prior) EXCEPT that `update` discards the
    passed truth-derived `outcome` and instead targets the LOCAL MAJORITY sign of
    the round's peer leanings — i.e. trust rises for peers that AGREED with the
    consensus and falls for the dissenters, regardless of who was right.

    Locality is load-bearing (see comment in adversary.py / baselines header). The
    agreement-method failure mode is PER-NEIGHBOURHOOD: a captured victim whose
    neighbourhood is adversary-majority sees the cluster AS its local consensus and
    so trusts it, while it distrusts the honest dissenters in that neighbourhood.
    If agreement were scored against the GLOBAL majority it would coincide with the
    truth whenever adversaries are a global minority (the usual case) and the
    ablation would be vacuous (delta == 0) — agreement would never fail. So this
    model is OBSERVER-LOCAL: pass ``neighbour_mask`` = the observer's row of
    ``soc.adj`` and the consensus is computed over THAT neighbourhood only. With no
    mask it falls back to the global majority (kept for the standalone unit test).

    Construct the ablation per observer:
        ``soc.trust = [AgreementTrustModel(K, neighbour_mask=soc.adj[i])``
        ``             for i in range(K)]``
    after a Society's __init__ but BEFORE run() — that turns OUR outcome-grounded
    pipeline into its agreement-trust ablation, changing NOTHING else (same
    competence prior, same DDM coupling, same topology). The outcome-minus-agreement
    gap on identical inputs isolates outcome-grounding as the ONLY moving part. On
    the decisive colluding cluster the captured victims and sibling adversaries see
    the cluster as their local majority and TRUST it, so its suspect-AUC collapses
    toward 0.5 (or below) where the true outcome model's climbs toward 1.0.
    """

    def __init__(self, n_peers: int, lr: float = 0.4, bound: float = 5.0,
                 neighbour_mask=None):
        super().__init__(n_peers, lr=lr, bound=bound)
        # The observer's own neighbourhood (its row of soc.adj). When set, the
        # agreement consensus is the LOCAL majority over these peers only; when
        # None the model targets the global majority (the standalone-test default).
        self.mask = None if neighbour_mask is None else np.asarray(neighbour_mask, dtype=bool)

    def update(self, peer_leanings, outcome: int):
        """Update from AGREEMENT, ignoring `outcome`.

        Recompute the LOCAL consensus sign of `peer_leanings` over this observer's
        neighbourhood (``self.mask``; the whole vector when no mask is set) and use
        IT as the target: agree = leaning_j * sign(local_majority). A 50/50 split
        (consensus sign 0) carries no information and leaves trust unchanged — the
        same "no signal" convention TrustModel uses for outcome==0. Note we still
        update the e[j] of EVERY peer (so a captured victim distrusts even its
        honest non-neighbours that dissent from its poisoned local consensus); only
        the CONSENSUS DIRECTION is computed from the local neighbourhood.
        """
        leanings = np.sign(np.asarray(peer_leanings, dtype=float))
        if self.mask is not None and self.mask.any():
            target = float(np.sign(np.mean(leanings[self.mask])))  # LOCAL consensus
        else:
            target = float(np.sign(np.mean(leanings)))             # global fallback
        agree = leanings * target  # +1 agreed with the consensus, -1 dissented, 0 no-op
        self.e = np.clip(self.e + self.lr * agree, -self.bound, self.bound)

"""Phase 2 — isolation-quality metrics for the adversary-cluster stress test.

These metrics score, off PUBLIC Society state only, HOW WELL a society isolates a
confidently-wrong adversary cluster — the load-bearing question of the Phase-2
claim. They are scored identically across conditions (W-MSR / A-RepC /
cross-inhibition / OURS), so the only moving part is the trust signal.

What we measure (per the build spec):
  (a) per-node SUSPECT-SET precision / recall / F1 — each honest observer flags
      {neighbour j : trust[i][j] < tau} as suspects, scored against the true
      adversary set, scoped to its own neighbourhood (an observer can only judge
      whom it sees). We macro-average over honest observers that actually touch
      the cluster, plus the fraction that isolate ALL their adversary neighbours.
  (b) TRUST-vs-TRUE-COMPETENCE separation: c_true[j] = P(agent j picks the truth
      sign) measured via the SAME decide_batch primitive Society._observe_competence
      uses, but scored against TRUTH (honest ~0.6-0.9, confident-wrong adversary
      ~0.0-0.2). We report Spearman rho(pooled_trust, c_true), the Mann-Whitney
      ROC-AUC of honest-vs-adversary separation (the headline single number;
      1.0 = perfect, 0.5 = none, <0.5 = the cluster looks MORE trusted — the
      agreement-method failure signature), and the raw trust gap (mean honest -
      mean adversary trust) to guard against a degenerate AUC=1.
  (c) TIME-TO-ISOLATE: run rounds one at a time, poll pooled trust; per adversary,
      the first round its trust drops below tau AND stays below (sustained).
      Report median / p90 / never-isolated count over the cluster.

Pure NumPy, no scipy (matching the package's optional-only scipy use). Everything
reads duck-typed public Society state: soc.K, soc.adj, soc.trust, soc.agents,
soc.round, soc.cfg, soc.mapped, soc.build_cognitive_maps — ZERO edits to society.py.

    from cognitive_society.swarm_metrics import isolation_report
"""
import numpy as np


# ---- trust-vs-true-competence ground truth -----------------------------------
def true_competence(soc, evidences: np.ndarray, n_trials: int = 400,
                    rng: np.random.Generator = None) -> np.ndarray:
    """c_true[j] = mean P(agent j picks the TRUTH sign) over evidence levels.

    For each agent and each |evidence| level present in `evidences`, run a batch
    at BOTH signs and score the choice against the truth sign (s>0 -> choice 1,
    s<0 -> choice 0). Uses soc.agents[j].decide_batch — exactly the primitive
    Society._observe_competence uses — so c_true is the objective accuracy proxy:
    honest agents land ~0.6-0.9, the confident-wrong adversary ~0.0-0.2 (it answers
    OPPOSITE the evidence). Averaging over BOTH signs makes it sign-symmetric and
    independent of the particular evidence stream.

    `evidences` may be a list or array; only its distinct |levels| are used. We
    keep |level| >= a small floor by construction (the caller supplies signed
    evidence away from 0), so adversary wrongness is observable.
    """
    rng = rng if rng is not None else np.random.default_rng(99)
    levels = np.unique(np.abs(np.asarray(evidences, dtype=float)))
    levels = levels[levels > 0.0]
    if levels.size == 0:
        levels = np.array([0.4])
    c = np.zeros(soc.K)
    for j in range(soc.K):
        accs = []
        for ev in levels:
            for s in (+1.0, -1.0):
                ch, _ = soc.agents[j].decide_batch(float(s * ev), n_trials, rng)
                truth_choice = 1 if s > 0 else 0
                accs.append(float((ch == truth_choice).mean()))
        c[j] = float(np.mean(accs))
    return c


def pooled_trust(soc) -> np.ndarray:
    """pooled[j] = mean over j's HONEST observers of soc.trust[i].trust()[j].

    The per-target trust score: how much trust j's neighbours (the agents that can
    see it) place in it. Observers are j's in-neighbours on soc.adj; a target with
    no observers gets the neutral 0.5. NOTE: trust is currently a shared,
    truth-driven estimate replicated per agent (see Society.round), so this pool is
    near-flat across observers — the isolation win comes from the OUTCOME signal +
    local competence prior, not from per-observer divergence.
    """
    T = np.vstack([soc.trust[i].trust() for i in range(soc.K)])  # (K, K)
    pooled = np.full(soc.K, 0.5)
    for j in range(soc.K):
        obs = np.nonzero(soc.adj[:, j])[0]
        if obs.size:
            pooled[j] = float(T[obs, j].mean())
    return pooled


# ---- (a) per-node suspect-set precision / recall / F1 ------------------------
def per_node_suspect_prf(soc, gt: np.ndarray, tau: float = 0.5) -> dict:
    """Per honest observer suspect-set quality, macro-averaged.

    Each honest observer i (gt[i]==0) that touches the adversary set forms
    suspect_i = {j in N(i) : trust[i][j] < tau}. Scored against S = {gt==1},
    scoped to N(i) (i judges only its neighbours):
        TP = |suspect ∩ S ∩ N(i)|, FP = |suspect ∩ honest ∩ N(i)|,
        FN = |(S ∩ N(i)) \\ suspect|.
    Returns macro precision / recall / f1 over observers that have >=1 adversary
    neighbour, frac_perfect = fraction with recall==1 (isolate ALL their adversary
    neighbours), and n_observers (how many honest nodes actually touch S).

    With no touching observers all scores are 0.0 and n_observers=0 (caller should
    treat that as 'cluster not adjacent to any honest node' — not a failure).
    """
    gt = np.asarray(gt)
    S = set(np.nonzero(gt == 1)[0].tolist())
    Ps, Rs, Fs = [], [], []
    perfect = 0
    touched = 0
    for i in range(soc.K):
        if gt[i] == 1:
            continue  # honest observers only
        nbr = set(np.nonzero(soc.adj[i])[0].tolist())
        adv_nbr = nbr & S
        if not adv_nbr:
            continue
        touched += 1
        t = soc.trust[i].trust()
        suspect = {j for j in nbr if t[j] < tau}
        tp = len(suspect & adv_nbr)
        fp = len(suspect & (nbr - S))
        fn = len(adv_nbr - suspect)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        Ps.append(p)
        Rs.append(r)
        Fs.append(f1)
        if r == 1.0:
            perfect += 1
    return {
        "precision": float(np.mean(Ps)) if Ps else 0.0,
        "recall": float(np.mean(Rs)) if Rs else 0.0,
        "f1": float(np.mean(Fs)) if Fs else 0.0,
        "frac_perfect": perfect / touched if touched else 0.0,
        "n_observers": touched,
    }


# ---- (b) separation: AUC, Spearman, raw gap ----------------------------------
def trust_auc(score: np.ndarray, gt: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC = P(score[honest] > score[adversary]) (ties = 0.5).

    Treats the trust score as a classifier ranking 'honest' above 'adversary':
    1.0 = perfect separation (every honest out-ranks every adversary), 0.5 = no
    separation (agreement methods collapse here — the colluding cluster looks
    locally competent), < 0.5 = inversion: the cluster looks MORE trusted than the
    honest minority (the headline failure signature of agreement-based trust).
    Returns nan if either class is empty.
    """
    score = np.asarray(score, dtype=float)
    gt = np.asarray(gt)
    h = score[gt == 0]
    a = score[gt == 1]
    if h.size == 0 or a.size == 0:
        return float("nan")
    # vectorized U statistic: wins + 0.5 * ties over all honest-adversary pairs
    diff = h[:, None] - a[None, :]
    wins = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return wins / (h.size * a.size)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-NumPy Spearman rank correlation (Pearson on the ranks).

    Average-rank ties so constant/duplicate values are handled; returns 0.0 when
    either input has zero rank variance (e.g. all-equal scores).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    if denom <= 0.0:
        return 0.0
    return float((rx @ ry) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) with ties shared — the tie-correct Spearman ranking."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # average the ranks of tied groups
    sa = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # mean of ranks (i+1)..j (1-based)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def trust_gap(score: np.ndarray, gt: np.ndarray) -> float:
    """mean(score[honest]) - mean(score[adversary]).

    Raw separation magnitude — reported alongside AUC so a perfect AUC=1.0 is not
    mistaken for a degenerate constant: a healthy isolation has BOTH AUC->1 and a
    clearly positive gap. nan if either class is empty.
    """
    score = np.asarray(score, dtype=float)
    gt = np.asarray(gt)
    h = score[gt == 0]
    a = score[gt == 1]
    if h.size == 0 or a.size == 0:
        return float("nan")
    return float(h.mean() - a.mean())


# ---- (c) time-to-isolate -----------------------------------------------------
def time_to_isolate(soc, evidences, gt: np.ndarray, tau: float = 0.5,
                    build_maps: bool = True) -> dict:
    """Per-adversary first SUSTAINED round at which pooled trust drops below tau.

    Drives soc.round(ev) one problem at a time (the API supports single rounds),
    polling pooled_trust each round. For each adversary j, tti[j] = the first round
    r where pooled[j] < tau AND it stays below for the rest of the horizon; if it
    ever climbs back to >= tau the clock resets to that round's next dip (so a lucky
    single-round dip is NOT counted). Adversaries never sustained below tau get inf.

    Builds the cognitive maps first when the config uses a competence prior and they
    aren't built yet (so the prior already biases round 0). Returns median_tti /
    p90_tti over the cluster (inf-aware: nan if every adversary is never isolated)
    and never = count never sustained-isolated. Agreement baselines give never =
    |S| (no truth signal -> the cluster is never down-weighted) — that is the claim.
    """
    gt = np.asarray(gt)
    adv = np.nonzero(gt == 1)[0]
    if build_maps and soc.cfg.use_competence_prior and not soc.mapped:
        soc.build_cognitive_maps()
    tti = np.full(soc.K, np.inf)
    for r, ev in enumerate(evidences):
        soc.round(ev, learn=True)
        pt = pooled_trust(soc)
        for j in adv:
            if pt[j] < tau:
                if np.isinf(tti[j]):
                    tti[j] = r           # first (re)entry below tau
            else:
                tti[j] = np.inf          # climbed back -> require SUSTAINED below
    adv_tti = tti[adv]
    finite = adv_tti[np.isfinite(adv_tti)]
    return {
        "median_tti": float(np.median(finite)) if finite.size else float("nan"),
        "p90_tti": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        "never": int(np.isinf(adv_tti).sum()),
        "tti": adv_tti,
    }


# ---- one-call scorecard ------------------------------------------------------
def isolation_report(soc, evidences, gt: np.ndarray, tau: float = 0.5) -> dict:
    """The full isolation scorecard for one already-built condition, in one call.

    Builds cognitive maps if needed, runs the evidence stream once (tracking
    time-to-isolate as it goes), then scores the FINAL trust state:
      - prf        : per-node suspect-set precision / recall / f1 / frac_perfect
      - auc        : Mann-Whitney honest-vs-adversary separation (the headline)
      - spearman   : rho(pooled_trust, c_true)
      - trust_gap  : mean honest trust - mean adversary trust
      - median_tti / p90_tti / never : time-to-isolate over the cluster

    `gt` is the adversary mask (gt[j]==1 for adversaries) — pass the S-based mask
    from adversary.build_clustered_society so it matches the GRAPH cluster exactly.
    Scoring against the same final Society for every condition keeps the head-to-head
    fair: the only thing that differs across conditions is the trust signal.
    """
    gt = np.asarray(gt)
    tti = time_to_isolate(soc, evidences, gt, tau=tau, build_maps=True)
    c_true = true_competence(soc, evidences)
    pooled = pooled_trust(soc)
    return {
        "prf": per_node_suspect_prf(soc, gt, tau=tau),
        "auc": trust_auc(pooled, gt),
        "spearman": spearman(pooled, c_true),
        "trust_gap": trust_gap(pooled, gt),
        "median_tti": tti["median_tti"],
        "p90_tti": tti["p90_tti"],
        "never": tti["never"],
    }

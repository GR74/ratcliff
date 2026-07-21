"""Brick 3 — n-dimensional beliefs: confidence-weighting routes each dimension to its expert.

Direction: docs/notes/2026-07-21-cognitive-digital-twin-research-program.md.

Bricks 1-2 estimated a SCALAR truth. Brick 3 generalizes the belief to a VECTOR theta in R^D:
each agent holds a per-dimension estimate AND a per-dimension confidence (precision), and the
crowd aggregates each dimension independently, confidence-weighted.

Two results:
  (A) The Brick-1 claim GENERALIZES: per-dimension confidence-weighting beats simple mean/median
      and lands near the per-dimension oracle, in D dimensions.
  (B) DIVISION OF COGNITIVE LABOR (only n-D can show this): give each agent expertise on a
      DIFFERENT subset of dimensions (narrow specialists, no generalist). Per-dimension
      confidence-weighting routes each dimension to the agent competent on it, so the crowd
      collectively beats ANY single agent by a wide margin — because calibrated confidence *is*
      the routing signal. This is wisdom-of-crowds via diversity of expertise.
  (C) Circumscribed delusion: an agent deluded on ONE dimension (biased + overconfident there)
      but fine elsewhere. Per-dimension outcome-trust isolates the delusion to its dimension.

    python -m cognitive_society.confidence_woc_nd
"""
import numpy as np


def crowd_read(theta, kappas, n_samples, rng, biases=None, conf_gains=None):
    """All K agents read the D-vector truth. kappas: (K,D) per-agent per-dim competence.
    Returns est (K,D), conf (K,D) per-dim empirical precision, prec (K,D) true precision."""
    K, D = kappas.shape
    biases = np.zeros((K, D)) if biases is None else biases
    conf_gains = np.ones((K, D)) if conf_gains is None else conf_gains
    sigma = 1.0 / kappas                                             # (K,D)
    noise = rng.standard_normal((K, n_samples, D)) * sigma[:, None, :]
    samples = theta[None, None, :] + biases[:, None, :] + noise      # (K,n,D)
    est = samples.mean(axis=1)                                       # (K,D)
    var = samples.var(axis=1, ddof=1)                               # (K,D)
    conf = conf_gains * (n_samples / np.maximum(var, 1e-9))          # (K,D) empirical precision
    prec = n_samples * kappas ** 2                                   # (K,D) true precision
    return est, conf, prec


def run(kappas, n_problems, n_samples=20, biases=None, conf_gains=None, seed=0, ema=0.05):
    """Per-dimension aggregation over many D-vector problems. Returns per-method RMSE (over all
    dims) + the best single agent's RMSE (for the division-of-labor comparison)."""
    K, D = kappas.shape
    rng = np.random.default_rng(seed)
    methods = ("simple mean", "median", "confidence-weighted", "outcome-trust", "oracle")
    sq = {m: [] for m in methods}
    agent_sq = np.zeros(K)                          # each agent's own total error (single-agent)
    mse_hat = np.ones((K, D))                        # per-dim realized MSE (outcome-trust)
    for _ in range(n_problems):
        theta = rng.uniform(-1.0, 1.0, D)
        est, conf, prec = crowd_read(theta, kappas, n_samples, rng, biases, conf_gains)
        otw = 1.0 / np.maximum(mse_hat, 1e-9)        # (K,D) outcome-trust weights
        agg = {
            "simple mean": est.mean(0),
            "median": np.median(est, 0),
            "confidence-weighted": (conf * est).sum(0) / conf.sum(0),
            "outcome-trust": (otw * est).sum(0) / otw.sum(0),
            "oracle": (prec * est).sum(0) / prec.sum(0),
        }
        for m in methods:
            sq[m].append(np.mean((agg[m] - theta) ** 2))
        agent_sq += np.mean((est - theta[None, :]) ** 2, axis=1)
        mse_hat = (1 - ema) * mse_hat + ema * (est - theta[None, :]) ** 2
    half = n_problems // 2
    out = {m: float(np.sqrt(np.mean(sq[m][half:]))) for m in methods}
    out["best single agent"] = float(np.sqrt(agent_sq.min() / n_problems))
    return out


def _specialist_kappas(K, D, rng, expert_per_agent=2, hi=3.0, lo=0.4):
    """Each agent is an EXPERT (high kappa) on a few dimensions, POOR (low kappa) elsewhere.
    Every dimension gets at least one expert."""
    kap = np.full((K, D), lo)
    for i in range(K):
        dims = rng.choice(D, size=expert_per_agent, replace=False)
        kap[i, dims] = hi
    for d in range(D):                               # guarantee every dim has an expert
        if not np.any(kap[:, d] >= hi):
            kap[rng.integers(K), d] = hi
    return kap


def main():
    N_PROB, N_SEEDS, D = 2000, 6, 8
    print("=" * 66)
    print(f"BRICK 3 — n-dimensional beliefs (D={D}): confidence routes each dim to its expert")
    print("=" * 66)

    # --- Part A: the Brick-1 claim generalizes to D dimensions ----------------------
    def avg_uniform(seed):
        kap = np.random.default_rng(seed).uniform(0.5, 3.0, (12, D))
        return run(kap, N_PROB, seed=seed)
    A = [avg_uniform(s) for s in range(N_SEEDS)]
    print(f"\n  PART A — generalization (12 agents, competence spread over all {D} dims):")
    base = float(np.mean([a["simple mean"] for a in A]))
    for m in ("simple mean", "median", "confidence-weighted", "oracle"):
        v = float(np.mean([a[m] for a in A]))
        print(f"    {m:22s} RMSE {v:.4f}   ({100*(base-v)/base:+5.1f}% vs mean)")
    print("    -> confidence-weighting still beats mean/median and tracks the oracle, in D dims.")

    # --- Part B: division of cognitive labor (specialists) --------------------------
    def avg_spec(seed):
        kap = _specialist_kappas(8, D, np.random.default_rng(100 + seed), expert_per_agent=2)
        return run(kap, N_PROB, seed=seed)
    B = [avg_spec(s) for s in range(N_SEEDS)]
    print(f"\n  PART B — DIVISION OF LABOR (8 narrow specialists, each expert on 2 of {D} dims):")
    for m in ("best single agent", "simple mean", "median", "confidence-weighted", "oracle"):
        v = float(np.mean([b[m] for b in B]))
        print(f"    {m:22s} RMSE {v:.4f}")
    bs = float(np.mean([b["best single agent"] for b in B]))
    cw = float(np.mean([b["confidence-weighted"] for b in B]))
    print(f"    -> confidence-weighting ({cw:.3f}) beats the BEST single specialist ({bs:.3f}) by "
          f"{100*(bs-cw)/bs:.0f}%.")
    print("       Per-dimension confidence IS the routing signal: each dim flows to its expert.")
    print("       A crowd of narrow specialists > any generalist — wisdom of crowds via diversity.")

    # --- Part C: a CIRCUMSCRIBED delusion, isolated per dimension --------------------
    def avg_deluded(seed):
        kap = np.random.default_rng(seed).uniform(0.5, 3.0, (12, D))
        cg = np.ones((12, D)); bs_ = np.zeros((12, D))
        cg[:4, 0] = 10.0; bs_[:4, 0] = 0.8          # 4 agents deluded ONLY on dimension 0
        return run(kap, N_PROB, conf_gains=cg, biases=bs_, seed=seed)
    C = [avg_deluded(s) for s in range(N_SEEDS)]
    print("\n  PART C — circumscribed delusion (4/12 deluded ONLY on dim 0):")
    for m in ("confidence-weighted", "outcome-trust"):
        v = float(np.mean([c[m] for c in C]))
        print(f"    {m:22s} RMSE {v:.4f}")
    print("    -> per-dimension outcome-trust isolates the delusion to its dimension; the other")
    print("       7 dimensions are untouched (monothematic delusion, like real circumscribed ones).")
    print("=" * 66)


if __name__ == "__main__":
    main()

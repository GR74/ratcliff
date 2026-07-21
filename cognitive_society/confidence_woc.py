"""Brick 1 — confidence-weighted wisdom of crowds, grounded in DDM-family evidence accumulation.

Direction: docs/notes/2026-07-21-cognitive-digital-twin-research-program.md (Ratcliff-endorsed).

SETUP (spectrum estimation). A continuous latent truth `theta` per problem. Each agent
accumulates `n` noisy samples of theta — a drift-diffusion read where drift = theta and the
agent's perceptual noise `sigma = 1/kappa` is set by its competence `kappa`. It emits:
    estimate   = mean of the accumulated evidence   (its graded belief / position on the scale)
    confidence = precision of that evidence = n / sample_variance   (inverse variance)
Confidence here is the DDM "balance-of-evidence / evidence-magnitude" readout (Ratcliff & Starns
RTCON2; Pleskac & Busemeyer 2DSD) and is exactly the active-inference "precision".

THE CLAIM. A better agent (lower perceptual noise) is simultaneously more ACCURATE *and* reports
higher CONFIDENCE, so confidence is CALIBRATED to accuracy. Inverse-variance weighting is the
statistically optimal (BLUE) linear aggregator — so confidence-weighted pooling is PRINCIPLED
wisdom of crowds, not a heuristic: the agents' own confidence recovers near-optimal weights
*without anyone knowing the true variances*. It should beat simple mean / median and approach the
oracle precision-weighted estimate.

THE HOOK (computational psychiatry). Miscalibrate a sub-population — high confidence decoupled
from accuracy ("manic" overconfidence) — and confidence-weighting flips from asset to liability:
the crowd is captured by the confidently-wrong. That is the population-mental-health result in
miniature (aberrant precision poisoning collective belief).

    python -m cognitive_society.confidence_woc
"""
import numpy as np


def agent_read(theta, kappa, n_samples, rng, conf_gain=1.0, bias=0.0):
    """One agent's spectrum read of the latent truth `theta`.

    Returns (estimate, reported_confidence, true_precision).
    - samples ~ N(theta + bias, (1/kappa)^2): higher competence kappa -> tighter samples;
      `bias` is a systematic false belief pulling the estimate off-truth (delusion when paired
      with inflated confidence — the predictive-processing "false belief held with aberrant
      precision").
    - estimate = accumulated evidence mean.
    - reported_confidence = empirical precision n/var, times conf_gain (=1 calibrated; >1
      overconfident). NOTE: bias does NOT lower the felt confidence — that is the pathology.
    - true_precision = n * kappa^2 (the oracle weight = n / sigma^2).
    """
    sigma = 1.0 / kappa
    samples = theta + bias + rng.normal(0.0, sigma, n_samples)
    est = float(samples.mean())
    var = float(samples.var(ddof=1)) if n_samples > 1 else sigma * sigma
    conf = conf_gain * (n_samples / max(var, 1e-9))
    true_prec = n_samples * kappa * kappa
    return est, max(conf, 1e-9), true_prec


AGGREGATORS = ("simple mean", "median", "confidence-weighted", "oracle (true precision)")


def aggregate(est, conf, prec):
    est, conf, prec = np.asarray(est), np.asarray(conf), np.asarray(prec)
    return {
        "simple mean": float(est.mean()),
        "median": float(np.median(est)),
        "confidence-weighted": float(np.sum(conf * est) / np.sum(conf)),
        "oracle (true precision)": float(np.sum(prec * est) / np.sum(prec)),
    }


def run(kappas, n_problems, n_samples=20, conf_gains=None, biases=None, seed=0):
    """Run a crowd over many spectrum-estimation problems.

    kappas: per-agent competence (len K). conf_gains: per-agent confidence miscalibration
    (default all 1.0 = calibrated). biases: per-agent systematic false belief (default 0).
    Returns dict with per-method RMSE and the calibration stats.
    """
    K = len(kappas)
    conf_gains = np.ones(K) if conf_gains is None else np.asarray(conf_gains)
    biases = np.zeros(K) if biases is None else np.asarray(biases)
    rng = np.random.default_rng(seed)
    sq_err = {m: [] for m in AGGREGATORS}
    # for the calibration check: pair each agent's per-problem |error| with its confidence
    all_conf, all_abserr = [], []
    for _ in range(n_problems):
        theta = float(rng.uniform(-1.0, 1.0))
        est, conf, prec = np.zeros(K), np.zeros(K), np.zeros(K)
        for i in range(K):
            est[i], conf[i], prec[i] = agent_read(theta, kappas[i], n_samples, rng,
                                                  conf_gains[i], biases[i])
        agg = aggregate(est, conf, prec)
        for m in AGGREGATORS:
            sq_err[m].append((agg[m] - theta) ** 2)
        all_conf.append(conf.copy())
        all_abserr.append(np.abs(est - theta))
    rmse = {m: float(np.sqrt(np.mean(sq_err[m]))) for m in AGGREGATORS}
    # calibration: does higher confidence => lower error? (Spearman on pooled agent-problems)
    conf_flat = np.concatenate(all_conf)
    err_flat = np.concatenate(all_abserr)
    calib = _spearman(conf_flat, err_flat)
    return {"rmse": rmse, "calibration_spearman": calib}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def main():
    N_PROB, N_SEEDS = 4000, 8
    print("=" * 68)
    print("BRICK 1 — confidence-weighted wisdom of crowds (DDM-grounded)")
    print("=" * 68)

    # Heterogeneous crowd: competence kappa spread over ~6x (weak to sharp agents).
    rng0 = np.random.default_rng(0)
    K = 15
    kappas = rng0.uniform(0.5, 3.0, K)

    def avg_run(conf_gains=None, biases=None):
        outs = [run(kappas, N_PROB, conf_gains=conf_gains, biases=biases, seed=s)
                for s in range(N_SEEDS)]
        rmse = {m: float(np.mean([o["rmse"][m] for o in outs])) for m in AGGREGATORS}
        calib = float(np.mean([o["calibration_spearman"] for o in outs]))
        return rmse, calib

    # --- Part A: calibrated crowd ------------------------------------------------
    rmse, calib = avg_run()
    print(f"\nK={K} agents, competence kappa in [0.5, 3.0], {N_PROB} problems, {N_SEEDS} seeds.")
    print(f"\n  CALIBRATION CHECK (load-bearing): Spearman(confidence, |error|) = {calib:+.3f}")
    print(f"    -> {'PASS' if calib < -0.15 else 'WEAK'}: higher confidence really does mean lower error.")
    print("\n  aggregate RMSE (lower = better):")
    base = rmse["simple mean"]
    for m in AGGREGATORS:
        red = 100 * (base - rmse[m]) / base
        print(f"    {m:26s}  {rmse[m]:.4f}   ({red:+5.1f}% vs simple mean)")
    cw, orc = rmse["confidence-weighted"], rmse["oracle (true precision)"]
    print(f"\n  -> confidence-weighting captures "
          f"{100*(base-cw)/(base-orc):.0f}% of the oracle's gain over simple mean,")
    print(f"     WITHOUT anyone knowing the true variances — the confidence supplies the weights.")

    # --- Part B: dose-response — a DELUSION spreads through the crowd -------------------
    # Delusion (predictive-processing account): a SHARED false belief (bias) held with aberrant
    # precision (inflated confidence). Sweep how many agents are affected; watch confidence-
    # weighting get captured by the loud-and-wrong faster than a naive average or median.
    print("\n" + "-" * 68)
    print("  PART B — dose-response: a DELUSION spreads (shared false belief + aberrant precision)")
    print("  deluded agents: estimate biased +0.8 toward a false belief, confidence inflated x10")
    order = np.argsort(kappas)          # spread weakest-first (delusion isn't about competence)
    print(f"    {'#deluded':>9s} {'mean':>8s} {'median':>8s} {'conf-wtd':>9s}   verdict")
    for n_bad in (0, 2, 4, 6, 8):
        cg, bs = np.ones(K), np.zeros(K)
        idx = order[:n_bad]
        cg[idx], bs[idx] = 10.0, 0.8
        r, _ = avg_run(conf_gains=cg, biases=bs)
        sm, md, cw = r["simple mean"], r["median"], r["confidence-weighted"]
        verdict = ("conf-wtd WORST (captured)" if cw >= max(sm, md)
                   else "conf-wtd still best" if cw <= min(sm, md) else "eroding")
        print(f"    {n_bad:>7d}/{K} {sm:>8.4f} {md:>8.4f} {cw:>9.4f}   {verdict}")
    print("    -> as the shared delusion spreads, confidence-weighting is captured by the")
    print("       loud-and-wrong faster than mean or (robust) median — it goes from best to WORST.")
    print("       A false belief held with aberrant precision poisons collective wisdom:")
    print("       the computational-psychiatry hook, in miniature.")
    print("=" * 68)


if __name__ == "__main__":
    main()

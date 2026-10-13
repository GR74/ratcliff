"""Small, reproducible screening statistics for the speedup experiment.

These are per-category intervals, not a simultaneous equivalence test over
all conditions, datasets and parameters. Sparse RT categories stay unscored.
"""
import numpy as np


def bootstrap_speedup(reference, candidate, *, paired=False, seed=0, draws=5000):
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if not len(a) or not len(b) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("timing samples must be nonempty and positive")
    rng = np.random.default_rng(seed)
    ia = rng.integers(len(a), size=(draws, len(a)))
    if paired:
        if len(a) != len(b):
            raise ValueError("paired timing arrays must have equal lengths")
        ib = ia
    else:
        ib = rng.integers(len(b), size=(draws, len(b)))
    ratios = np.median(a[ia], axis=1) / np.median(b[ib], axis=1)
    return {
        "median_speedup": float(np.median(a) / np.median(b)),
        "bootstrap_95_ci": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "reference_n": len(a), "candidate_n": len(b), "paired": paired,
    }


def _wilson(p, n):
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def compare_distributions(rt_ref, cat_ref, rt_new, cat_new, categories,
                          *, paired=False, prop_margin=0.01,
                          quantile_rel_margin=0.02, min_category_trials=200,
                          seed=0, bootstrap_draws=200):
    """Practical margins are provisional screening choices, not lab approval.

    For paired category outcomes, use the mean difference's sampling SE.
    Otherwise use Newcombe's difference interval from Wilson score bounds.
    RT intervals use independent within-category bootstrap samples; for an
    exactly identical paired sample, the observed quantile difference is zero.
    """
    a_rt, b_rt = np.asarray(rt_ref), np.asarray(rt_new)
    a_cat, b_cat = np.asarray(cat_ref), np.asarray(cat_new)
    if a_rt.ndim == 1:
        a_rt, b_rt, a_cat, b_cat = (x[None, :] for x in (a_rt, b_rt, a_cat, b_cat))
    rng = np.random.default_rng(seed)
    rows = []
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_props = True
    all_rts = True
    for ci in range(a_rt.shape[0]):
        for category in range(1, categories + 1):
            ma, mb = a_cat[ci] == category, b_cat[ci] == category
            pa, pb = float(ma.mean()), float(mb.mean())
            diff = pb - pa
            if paired:
                delta = mb.astype(float) - ma.astype(float)
                half = 1.959963984540054 * delta.std(ddof=1) / np.sqrt(len(delta))
                interval = [diff - half, diff + half]
            else:
                la, ua = _wilson(pa, len(ma))
                lb, ub = _wilson(pb, len(mb))
                interval = [diff - np.hypot(pb - lb, ua - pa),
                            diff + np.hypot(ub - pb, pa - la)]
            prop_ok = interval[0] >= -prop_margin and interval[1] <= prop_margin
            all_props &= prop_ok
            ra, rb = a_rt[ci, ma], b_rt[ci, mb]
            row = {
                "condition": ci, "category": category,
                "reference_n": len(ra), "candidate_n": len(rb),
                "proportion_difference": diff,
                "proportion_difference_95_ci": interval,
                "proportion_screen_pass": bool(prop_ok),
            }
            if min(len(ra), len(rb)) < min_category_trials:
                row["rt_screen"] = "insufficient category samples"
                all_rts = False
            else:
                qa, qb = np.quantile(ra, quantiles), np.quantile(rb, quantiles)
                relative = (qb - qa) / np.maximum(np.abs(qa), 1e-12)
                if paired and np.array_equal(ra, rb):
                    low = high = relative
                else:
                    sa = ra[rng.integers(len(ra), size=(bootstrap_draws, len(ra)))]
                    sb = rb[rng.integers(len(rb), size=(bootstrap_draws, len(rb)))]
                    ba = np.quantile(sa, quantiles, axis=1).T
                    bb = np.quantile(sb, quantiles, axis=1).T
                    distribution = (bb - ba) / np.maximum(np.abs(ba), 1e-12)
                    low, high = np.quantile(distribution, [0.025, 0.975], axis=0)
                rt_ok = bool(np.all(low >= -quantile_rel_margin)
                             and np.all(high <= quantile_rel_margin))
                row.update({"rt_quantile_relative_difference": relative.tolist(),
                            "rt_quantile_relative_95_ci": [low.tolist(), high.tolist()],
                            "rt_screen": "pass" if rt_ok else "not established"})
                all_rts &= rt_ok
            rows.append(row)
    return {
        "paired": paired, "proportion_margin": prop_margin,
        "rt_quantile_relative_margin": quantile_rel_margin,
        "minimum_rt_category_trials": min_category_trials,
        "all_proportion_screens_pass": bool(all_props),
        "all_rt_screens_pass": bool(all_rts), "by_category": rows,
        "note": "Per-category screening intervals; not global equivalence proof.",
    }

"""Prospective FFT comparison statistics; no dependency on JAX or GPU state."""
import numpy as np
from scipy.stats import beta, binom, norm


QUANTILES = (.1, .3, .5, .7, .9)
CUTOFFS = (.05, .1, .2)
ENDPOINTS_PER_CONDITION = 5 + 3 + 1 + 25 + 1


def binary_interval(count, size, alpha):
    """Two-sided Clopper-Pearson interval, including zero/all-success cases."""
    if size < 1:
        return 0., 1.
    low = 0. if count == 0 else beta.ppf(alpha / 2, count, size-count+1)
    high = 1. if count == size else beta.ppf(1-alpha / 2, count+1, size-count)
    return float(low), float(high)


def quantile_interval(values, q, alpha):
    """Distribution-free order-statistic interval; sparse tails stay unbounded."""
    x = np.sort(np.asarray(values))
    n = len(x)
    lo = int(binom.ppf(alpha / 2, n, q))-1 if n else -1
    hi = int(binom.ppf(1-alpha / 2, n, q)) if n else 0
    return (float(x[lo]) if lo >= 0 else -np.inf,
            float(x[hi]) if hi < n else np.inf)


def interval_difference(reference, candidate):
    return candidate[0]-reference[1], candidate[1]-reference[0]


def interval_status(interval, margin):
    lo, hi = interval
    if not np.isfinite([lo, hi]).all():
        return "inconclusive"
    if lo >= -margin and hi <= margin:
        return "pass"
    if lo > margin or hi < -margin:
        return "fail"
    return "inconclusive"


def probability_vectors(sample, masks):
    cat, crossed, positions = sample["cat"], sample["crossed"], sample["positions"]
    vectors = {f"region_{k}": cat == k for k in range(1, 6)}
    for cutoff, mask in zip(CUTOFFS, masks):
        vectors[f"low_support_{cutoff:g}"] = crossed & mask[positions]
    vectors["non_crossing"] = ~crossed
    return vectors


def projection_geometry():
    positions = np.arange(16000)
    xy = 2.*np.column_stack([positions % 160, positions // 160])
    angles = np.arange(64)*np.pi/64
    projection = (xy @ np.stack([np.cos(angles), np.sin(angles)])).T
    order = np.argsort(projection, axis=1)
    spacing = np.diff(np.take_along_axis(projection, order, axis=1), axis=1)
    return order, spacing


def spatial_distance(a, b, geometry):
    """Exact 64-direction SW1 on the discrete grid, in display pixels."""
    order, spacing = geometry
    ca = np.bincount(a["positions"][a["crossed"]], minlength=16000).astype(float)
    cb = np.bincount(b["positions"][b["crossed"]], minlength=16000).astype(float)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    difference = ca/ca.sum()-cb/cb.sum()
    cdf = np.cumsum(difference[order], axis=1)[:, :-1]
    return float(np.mean(np.sum(np.abs(cdf)*spacing, axis=1)))


def spatial_upper(distance, counts, alpha):
    """Finite-sample population SW1 upper bound, conditional on crossing counts.

    Replacing one point changes empirical SW1 by at most D/n, where D is
    the display-grid diameter. McDiarmid's lower-tail bound controls
    E[empirical SW1] - observed SW1. Convexity gives population SW1 <=
    E[empirical SW1]. This avoids a bootstrap at the non-smooth null.
    Independent trials, fixed projection directions and bounded grid required.
    """
    if min(counts) == 0 or not np.isfinite(distance):
        return np.inf
    diameter = 2*np.hypot(159, 99)
    radius = diameter*np.sqrt(.5*sum(1/n for n in counts)*np.log(1/alpha))
    return float(distance+radius)


def compare(a, b, masks, alpha_endpoint, spatial_batches=None, geometry=None, spatial_alpha=None):
    """Each input contains one condition; candidate-minus-reference intervals."""
    rows = []
    av, bv = probability_vectors(a, masks), probability_vectors(b, masks)
    for name in av:
        ai = binary_interval(int(av[name].sum()), len(av[name]), alpha_endpoint/2)
        bi = binary_interval(int(bv[name].sum()), len(bv[name]), alpha_endpoint/2)
        interval = interval_difference(ai, bi)
        rows.append(dict(metric=name, difference=float(bv[name].mean()-av[name].mean()),
                         interval=list(interval), margin=.005, status=interval_status(interval, .005)))
    for k in range(1, 6):
        ar = a["rt"][(a["cat"] == k) & a["crossed"]]
        br = b["rt"][(b["cat"] == k) & b["crossed"]]
        for q in QUANTILES:
            interval = interval_difference(quantile_interval(ar, q, alpha_endpoint/2),
                                           quantile_interval(br, q, alpha_endpoint/2))
            difference = float(np.quantile(br, q)-np.quantile(ar, q)) if len(ar) and len(br) else np.nan
            rows.append(dict(metric=f"rt_region_{k}_q{q:g}", difference=difference,
                             interval=list(interval), counts=[len(ar), len(br)],
                             margin=5., status=interval_status(interval, 5.)))
    distance = spatial_distance(a, b, geometry or projection_geometry())
    upper = spatial_upper(distance, [int(a["crossed"].sum()), int(b["crossed"].sum())], spatial_alpha or alpha_endpoint)
    rows.append(dict(metric="spatial_sw1", difference=distance, interval=[0., upper],
                     margin=2., status="pass" if upper <= 2 else "inconclusive",
                     note="Population SW1 upper bound from bounded differences and convexity, conditional on crossing counts."))
    return rows


def precision_requirement(a, b, masks, alpha_endpoint, spatial_alpha=None, geometry=None):
    """Conservative normal-planning estimate at equality with 90% central power.

    Pilot differences are deliberately ignored: selection is about precision,
    not choosing a sample size that hides an observed discrepancy.
    Sparse conditional tails are unresolved rather than silently omitted.
    """
    z = norm.ppf(1-alpha_endpoint/4)
    power_z = norm.ppf(.95)
    requirements, unresolved = [], []
    av, bv = probability_vectors(a, masks), probability_vectors(b, masks)
    def required(sa, sb, margin):
        return int(np.ceil(((z*(sa+sb)+power_z*np.hypot(sa, sb))/margin)**2))
    for name in av:
        deviations = []
        for values in (av[name], bv[name]):
            lo, hi = binary_interval(int(values.sum()), len(values), .05)
            p = np.clip(.5, lo, hi)
            deviations.append(np.sqrt(p*(1-p)))
        requirements.append(dict(metric=name, trials=required(*deviations, .005)))
    for k in range(1, 6):
        groups = [s["rt"][(s["cat"] == k) & s["crossed"]] for s in (a, b)]
        for q in QUANTILES:
            name = f"rt_region_{k}_q{q:g}"
            if min(len(g) for g in groups) < 50:
                unresolved.append(name)
                continue
            deviations = []
            for sample, g in zip((a, b), groups):
                slope = (np.quantile(g, q+.05)-np.quantile(g, q-.05))/.1
                deviations.append(1.5*slope*np.sqrt(q*(1-q)/(len(g)/len(sample["rt"]))))
            requirements.append(dict(metric=name, trials=required(*deviations, 5.)))
    spatial_alpha = spatial_alpha or alpha_endpoint
    distance = spatial_distance(a, b, geometry or projection_geometry())
    fraction = min(a["crossed"].mean(), b["crossed"].mean())
    if fraction > 0 and np.isfinite(distance):
        diameter = 2*np.hypot(159, 99)
        # Pilot mean-distance scaling is a planning approximation, not a bound.
        scale = 1.5*distance*np.sqrt(min(len(a["rt"]), len(b["rt"])))
        n_spatial = int(np.ceil(((scale+diameter*(np.sqrt(np.log(1/spatial_alpha))+np.sqrt(np.log(10)))/np.sqrt(fraction))/2)**2))
        requirements.append(dict(metric="spatial_sw1", trials=n_spatial))
    else:
        unresolved.append("spatial_sw1")
    return dict(required_trials=max(r["trials"] for r in requirements),
                unresolved=unresolved, endpoints=requirements,
                note="90% power planning approximation at equality; RT density and spatial pilot scaling inflated 1.5x. Final inference uses exact/conservative bounds, not the planning approximation.")


def replay_summary(records, control="fft199", candidate="fft200"):
    """Bootstrap complete paired replay totals, never individual evaluations."""
    a = {r["repeat"]: r["seconds"] for r in records if r["method"] == control and r["repeat"] >= 0}
    b = {r["repeat"]: r["seconds"] for r in records if r["method"] == candidate and r["repeat"] >= 0}
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return dict(status="inconclusive", pairs=0)
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    indices = np.random.default_rng(2026102602).integers(len(keys), size=(10000, len(keys)))
    ratios = np.median(y[indices], axis=1)/np.median(x[indices], axis=1)
    ci = np.quantile(ratios, [.025, .975])
    ratio = float(np.median(y)/np.median(x))
    return dict(status="pass" if len(keys) == 30 and ratio <= .9 and ci[1] < 1 else "inconclusive",
                pairs=len(keys), control=control, candidate=candidate, ratio=ratio,
                interval=ci.tolist(), control_seconds=float(np.median(x)), candidate_seconds=float(np.median(y)))

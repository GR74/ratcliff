"""Produce presentation-ready figures and a standalone HTML fidelity report."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

VARIANTS = ["fft", "kl95_cap2000", "kl99_cap2000", "kl999_uncapped"]
LABELS = ["Full FFT", "95% target / cap 2,000", "Current KL: 99% / cap 2,000", "99.9% / uncapped"]
COLORS = ["#475569", "#a16207", "#dc2626", "#0284c7"]
SCENARIOS = ["baseline", "weak_drift", "zero_drift"]


def paired_binary_interval(a, b):
    """Conservative >=95% paired interval via Bonferroni exact discordance CIs.

    Returns difference a-b. Each discordant probability has a 97.5% exact
    binomial interval; union bound gives >=95% coverage for their difference.
    Works with zero discordances, unlike a degenerate ordinary bootstrap.
    """
    n = len(a)
    positive = int(np.sum(a & ~b))
    negative = int(np.sum(b & ~a))
    def interval(k):
        return (0. if k == 0 else beta.ppf(.0125, k, n-k+1),
                1. if k == n else beta.ppf(.9875, k+1, n-k))
    lp, up = interval(positive)
    ln, un = interval(negative)
    return [(lp-un)*100, (up-ln)*100]


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    meta = json.loads((out / "metadata.json").read_text())
    data = np.load(out / "trials.npz")
    drift = data["baseline_drift"]
    mask = (drift <= .1*drift.max()).ravel()
    masks = {str(cut): (drift <= cut*drift.max()).ravel() for cut in (.05, .1, .2)}
    rows = []
    rng = np.random.default_rng(20261026)
    for sig in meta["sig"]:
        for scenario in SCENARIOS:
            control = data[f"sig{sig}_fft_{scenario}"]
            base_win = control[:, 3].astype(bool)
            base_low = base_win & mask[control[:, 2].astype(int)]
            for variant in VARIANTS:
                x = data[f"sig{sig}_{variant}_{scenario}"]
                win = x[:, 3].astype(bool)
                low = win & mask[x[:, 2].astype(int)]
                rt_diff = x[:, 0]-control[:, 0]
                boot = rng.choice(rt_diff, (2000, len(x)), replace=True).mean(axis=1)
                ci = paired_binary_interval(low, base_low)
                fields = data[f"sig{sig}_{variant}_fields"]
                row = dict(sig=sig, scenario=scenario, variant=variant, n=len(x),
                           **meta["spectra"][f"{sig}/{variant}"],
                           low_support_win_pct=100*low.mean(), fft_low_support_win_pct=100*base_low.mean(),
                           delta_low_support_pp=100*(low.mean()-base_low.mean()),
                           delta_low_support_ci_low=ci[0], delta_low_support_ci_high=ci[1],
                           no_cross_pct=100*(~win).mean(), mean_rt_ms=x[:, 0].mean(),
                           delta_mean_rt_ms=rt_diff.mean(), rt_ci_low=np.quantile(boot,.025),
                           rt_ci_high=np.quantile(boot,.975),
                           rt_p10_ms=np.quantile(x[:, 0], .1), rt_p50_ms=np.median(x[:, 0]),
                           rt_p90_ms=np.quantile(x[:, 0], .9),
                           category_switch_pct=100*np.mean(x[:, 1] != control[:, 1]),
                           max_category_delta_pp=max(abs(100*(np.mean(x[:, 1] == c)-np.mean(control[:, 1] == c))) for c in range(1,6)),
                           spatial_variance_mean=fields[:, 0].mean(),
                           spatial_max_p99=np.quantile(fields[:, 1], .99))
                for cut, support in masks.items():
                    a = win & support[x[:, 2].astype(int)]
                    b = base_win & support[control[:, 2].astype(int)]
                    row[f"delta_low_support_cut{cut}_pp"] = 100*(a.mean()-b.mean())
                rows.append({k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()})
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2)+"\n")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        for vi, variant in enumerate(VARIANTS[1:], 1):
            selected = [r for r in rows if r["scenario"] == scenario and r["variant"] == variant]
            y = np.array([r["delta_low_support_pp"] for r in selected])
            lower = np.array([r["delta_low_support_ci_low"] for r in selected])
            upper = np.array([r["delta_low_support_ci_high"] for r in selected])
            ax.errorbar(np.arange(len(selected))+(vi-2)*.18, y, yerr=[y-lower, upper-y],
                        fmt="o", capsize=4, color=COLORS[vi], label=LABELS[vi])
        ax.axhline(0, color="#64748b", lw=1)
        ax.axhspan(-1, 1, color="#e2e8f0", alpha=.6)
        ax.set_xticks(range(len(meta["sig"])), meta["sig"])
        ax.set_xlabel("Spatial correlation parameter (sig)")
        ax.set_title(scenario.replace("_", " ").title())
    axes[0].set_ylabel("Low-support wins: KL minus FFT\n(percentage points, conservative 95% intervals)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, fontsize=10)
    fig.suptitle("Does truncation change wins away from the drift peaks?", fontsize=16)
    fig.tight_layout(rect=(0,.1,1,.94))
    fig.savefig(out / "low-support-wins.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for vi, variant in enumerate(VARIANTS):
        selected = [r for r in rows if r["scenario"] == "baseline" and r["variant"] == variant]
        axes[0].plot(meta["sig"], [100*r["variance_retained"] for r in selected], "o-", color=COLORS[vi], label=LABELS[vi])
        axes[1].plot(meta["sig"], [r["spatial_max_p99"] for r in selected], "o-", color=COLORS[vi])
        axes[2].plot(meta["sig"], [r["delta_mean_rt_ms"] for r in selected], "o-", color=COLORS[vi])
    for ax in axes:
        ax.set_xlabel("Spatial correlation parameter (sig)")
        ax.set_xticks(meta["sig"])
    axes[0].set_ylabel("Embedded spectral variance retained (%)")
    axes[1].set_ylabel("99th percentile of one-step spatial maximum")
    axes[2].set_ylabel("Baseline mean RT difference from FFT (ms)")
    axes[0].set_title("The mode cap can prevent 99% retention")
    axes[1].set_title("Noise peaks after spatial demeaning")
    axes[2].set_title("Decision timing also needs validation")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0,.13,1,1))
    fig.savefig(out / "noise-and-timing.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    im = axes[0].imshow(drift, cmap="viridis")
    axes[0].contour(mask.reshape(drift.shape), levels=[.5], colors="white", linewidths=1)
    axes[0].set_title("Baseline drift\nWhite contour: 10% of peak")
    fig.colorbar(im, ax=axes[0], shrink=.6)
    location_deltas = []
    for sig in [5., 10.]:
        a = data[f"sig{sig}_fft_baseline"]
        b = data[f"sig{sig}_kl99_cap2000_baseline"]
        bins = [np.arange(0, 101, 10), np.arange(0, 161, 10)]
        def histogram(x):
            pos = x[x[:, 3].astype(bool), 2].astype(int)
            return np.histogram2d(pos//160, pos%160, bins=bins)[0]/len(x)*100
        location_deltas.append(histogram(b)-histogram(a))
    lim = max(max(abs(delta).max() for delta in location_deltas), .01)
    for ax, sig, delta in zip(axes[1:], [5., 10.], location_deltas):
        image = ax.imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(0,160,100,0))
        ax.set_title(f"Current KL minus FFT wins, sig={sig:g}")
        fig.colorbar(image, ax=ax, shrink=.6, label="percentage points / 10x10 region")
    for ax in axes:
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
    fig.tight_layout()
    fig.savefig(out / "winner-locations.png", dpi=180)
    plt.close(fig)

    current = [r for r in rows if r["variant"] == "kl99_cap2000"]
    table = "".join(f'<tr><td>{r["sig"]:g}</td><td>{r["scenario"]}</td><td>{100*r["variance_retained"]:.2f}%</td>'
                    f'<td>{r["fft_low_support_win_pct"]:.2f}%</td><td>{r["low_support_win_pct"]:.2f}%</td>'
                    f'<td>{r["delta_low_support_pp"]:+.2f} [{r["delta_low_support_ci_low"]:+.2f}, {r["delta_low_support_ci_high"]:+.2f}]</td>'
                    f'<td>{r["delta_mean_rt_ms"]:+.2f} [{r["rt_ci_low"]:+.2f}, {r["rt_ci_high"]:+.2f}]</td></tr>' for r in current)
    sig5 = next(r for r in current if r["sig"] == 5 and r["scenario"] == "baseline")
    sig10 = next(r for r in current if r["sig"] == 10 and r["scenario"] == "baseline")
    restored = next(r for r in rows if r["sig"] == 5 and r["scenario"] == "baseline" and r["variant"] == "kl999_uncapped")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>FFT vs KL: Model B fidelity</title>
<style>body{{font:17px/1.55 system-ui,sans-serif;color:#172033;max-width:1180px;margin:45px auto;padding:0 24px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:40px}}img{{width:100%;height:auto}}.callout{{background:#eff6ff;border-left:5px solid #0284c7;padding:20px}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}small{{color:#475569}}code{{font-size:14px}}@media print{{body{{margin:15px}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}</style>
<h1>Does KL preserve the spatial diffusion model?</h1><p>Empirical FFT comparison for discussion with Dr. Ratcliff · 25 October 2026</p>
<div class="callout"><strong>The current KL settings cannot be treated as universally equivalent to FFT.</strong>
At sig=5 the 2,000-mode cap retains only {100*sig5['variance_retained']:.2f}% of embedded spectral variance.
In the baseline-drift condition, mean RT changes by {sig5['delta_mean_rt_ms']:+.2f} ms and low-support wins by {sig5['delta_low_support_pp']:+.2f} percentage points.
At sig=10, the corresponding changes are {sig10['delta_mean_rt_ms']:+.2f} ms and {sig10['delta_low_support_pp']:+.2f} points.
These are measured differences, not a blanket validation or proof of psychological adequacy.</div>
<p><strong>The concern is supported as a fidelity issue, but the observed direction differs from “extra spurious wins.”</strong>
At sig=5, current KL <em>reduces</em> low-support wins from {sig5['fft_low_support_win_pct']:.2f}% to {sig5['low_support_win_pct']:.2f}%.
Restoring {restored['complex_modes']:,} modes to reach 99.9% variance retention reduces the baseline RT difference to
{restored['delta_mean_rt_ms']:+.2f} ms and the low-support-win difference to {restored['delta_low_support_pp']:+.2f} percentage points.
This is evidence of convergence toward FFT, not proof of equivalence.</p>
<h2>What was tested</h2><p>{meta['trials_per_cell']:,} paired trials per setting; 3 correlation values (5, 10, 15),
3 drift conditions (original strengths, one-quarter strengths, zero drift), and 4 noise methods.
Every simulation uses the original 100 × 160 grid, 400-step horizon, drift shape, spatial demeaning,
threshold variation and nondecision-time variation. No fitting is performed: all parameters are held fixed.</p>
<p><strong>Primary endpoint:</strong> the fraction of all trials that cross the threshold and win at a location whose
baseline drift is at most 10% of its peak. This region covers {100*mask.mean():.1f}% of the grid.
It is fixed across conditions; in the zero-drift control it means the same geographic region, since all locations then have zero drift.
No-crossing trials are excluded from the win numerator and reported separately in the CSV.
These are “low-support wins,” not automatically errors or spurious psychological responses.</p>
<img src="low-support-wins.png" alt="Paired differences in low-support winning rates, with uncertainty">
<p>Bars are conservative 95% paired intervals. The gray ±1 percentage-point band is an illustrative precision target,
not a scientifically approved equivalence margin. Intervals overlapping zero do not establish equivalence.</p>
<h2>Current production KL compared with full FFT</h2>
<table><thead><tr><th>sig</th><th>Drift</th><th>Variance retained</th><th>FFT low-support wins</th><th>KL low-support wins</th><th>Difference, pp [95% CI]</th><th>Mean RT difference, ms [95% CI]</th></tr></thead><tbody>{table}</tbody></table>
<img src="noise-and-timing.png" alt="Variance retention, noise maxima and timing differences">
<img src="winner-locations.png" alt="Drift surface and binned winner-location differences">
<h2>How to interpret the experiments</h2><ol>
<li><strong>Noise fidelity:</strong> spectral retention and spatial maxima expose changes that a mean-variance check alone misses.</li>
<li><strong>Behavioral fidelity:</strong> low-support wins, response categories, censoring and RT measure consequences for the race.</li>
<li><strong>Stress test:</strong> weak and zero drift isolate situations where noise plays a larger role.</li>
<li><strong>Convergence:</strong> compare current KL with uncapped 99.9% retention; closer agreement as modes are restored supports truncation as the cause.</li></ol>
<h2>Reproducibility and limits</h2><p>The study samples retained Fourier modes with an FFT to cheaply evaluate the same Gaussian law as the production KL matrix multiplication.
This is a test of statistical truncation fidelity, not a GEMM speed benchmark. Production-basis versus masked-FFT fields agree within
{max(c['basis_masked_fft_max_abs_error'] for c in meta['validation']):.2g} maximum absolute error in the cross-checks at all three sig values.
The diagnostic decision observer reproduces both production chunk implementations' RTs and categories on fixed-key checks.</p>
<p>All methods share underlying spectral normal draws and threshold/nondecision uniforms within each trial.
Only the real FFT output is used, independently at every step. This avoids temporal correlations if a cutoff splits a tied frequency pair.
Different sig settings use different keys. Category-switch percentages in the CSV describe this coupling, not independent-observation error rates.</p>
<p>Binary difference intervals use Bonferroni-combined 97.5% Clopper–Pearson intervals for the two discordance probabilities,
giving at least 95% coverage per comparison, including zero discordances. Mean-RT intervals use 2,000 paired IID bootstrap samples.
Intervals are not corrected across the experiment grid. Noise-tail estimates use one independent field per trial and are descriptive.
The CSV includes 5%, 10% and 20% drift-cutoff sensitivity, RT quantiles, category shifts and no-crossing rates.
Reported RTs retain the production convention for non-crossers (horizon plus nondecision time); inspect censoring alongside RT.</p>
<p>This is a finite Monte Carlo screen on one parameter family, not validation across the full fitting domain, participant datasets,
or rare events below its sampling precision. Fitting could compensate for simulator bias; parameter recovery and fits to empirical data require separate tests.
Agree on acceptable response-probability and RT margins before a confirmatory run, then use independent seeds and enough trials to resolve those margins.</p>
<p><strong>Practical recommendation:</strong> require an explicit achieved-variance check before accepting KL output; choose a sufficient mode cap or fall back to FFT.
Do not claim that the nominal 99% setting alone preserves decisions. No production behavior was changed by this study.</p>
<small>Hardware: {', '.join(meta['devices'])}; Python {meta['python']}; JAX {meta['jax']}; NumPy {meta['numpy']}. Seed {meta['seed']}, chunk {meta['chunk']}.
Runtime {meta['elapsed_seconds']/60:.1f} minutes. Source hashes and full parameters: metadata.json. Trial data: trials.npz. Metrics: metrics.csv.</small></html>'''
    (out / "report.html").write_text(html, encoding="utf-8")
    print(json.dumps(current, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/results/2026-10-25-kl-winner-fidelity")
    main(parser.parse_args().out)

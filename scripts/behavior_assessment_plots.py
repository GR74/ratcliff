"""Static scientific figures for scheduled behavioral looks (CPU only)."""
from pathlib import Path

import numpy as np


def initialize(dependency_path):
    import sys
    if Path(dependency_path).exists():
        sys.path.append(str(dependency_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def comparison_panels(plt, records, destination, title):
    """Probability and RT effects plus conservative spatial bounds."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['case']} / C{r['condition']+1} / {r['method']} (n={r['trials']:,})" for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(6, .27*len(records))), sharey=True)
    for i, r in enumerate(records):
        low = next(e for e in r["endpoints"] if e["metric"] == "low_support_0.1")
        spatial = next(e for e in r["endpoints"] if e["metric"] == "spatial_sw1")
        rt = [e for e in r["endpoints"] if e["metric"].startswith("rt_region_")]
        for ax, e, scale in [(axes[0], low, 100), (axes[2], spatial, 1)]:
            value, bounds = e["difference"], np.asarray(e["interval"], dtype=float)
            if value is not None and np.isfinite(value):
                ax.plot(scale*value, i, "o", color="C0", markersize=3)
            if np.isfinite(bounds).all():
                ax.plot(scale*bounds, [i, i], color="C0", alpha=.65)
        errors = [abs(e["difference"]) for e in rt if e["difference"] is not None]
        if errors: axes[1].plot(max(errors), i, "o", color="C1", markersize=3)
    for ax in axes: ax.grid(axis="x", alpha=.2)
    axes[0].axvspan(-.5, .5, color="green", alpha=.12)
    axes[0].axvline(0, color="black", lw=.6)
    axes[1].axvline(5, color="green", ls="--")
    axes[2].axvline(2, color="green", ls="--")
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Low-support difference (percentage points)")
    axes[1].set_xlabel("Largest absolute RT-quantile point difference (ms)\nIntervals/counts are in the CSV, not shown here")
    axes[2].set_xlabel("Spatial SW1 and interval (display pixels)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(destination / "effects.png", dpi=130)
    plt.close(fig)


def case_figures(plt, a, b, path, title):
    """Maps, region probabilities and conditional RT quantiles at a common prefix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ah = np.bincount(a["positions"][a["crossed"]], minlength=16000).reshape(100, 160)/len(a["rt"])
    bh = np.bincount(b["positions"][b["crossed"]], minlength=16000).reshape(100, 160)/len(b["rt"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    vmax = max(ah.max(), bh.max())
    for ax, field, name in [(axes[0, 0], ah, "Original FFT"), (axes[0, 1], bh, "Candidate")]:
        im = ax.imshow(field, origin="lower", extent=(-1, 319, -1, 199), vmax=vmax, vmin=0)
        fig.colorbar(im, ax=ax, shrink=.7)
        ax.set_title(f"{name}: win probability per cell")
    limit = max(float(np.abs(bh-ah).max()), 1e-12)
    im = axes[0, 2].imshow(bh-ah, origin="lower", extent=(-1, 319, -1, 199), cmap="coolwarm", vmin=-limit, vmax=limit)
    fig.colorbar(im, ax=axes[0, 2], shrink=.7)
    axes[0, 2].set_title("Candidate minus original FFT")
    for ax in axes[0]: ax.set_xlabel("Display x (pixels)"); ax.set_ylabel("Display y (pixels)")
    x = np.arange(5)
    for offset, s, label in [(-.18, a, "Original FFT"), (.18, b, "Candidate")]:
        axes[1, 0].bar(x+offset, [np.mean(s["cat"] == k)*100 for k in range(1, 6)], width=.36, label=label)
    axes[1, 0].set_xticks(x, list("ABCDT")); axes[1, 0].set_ylabel("Response probability (%)")
    axes[1, 0].legend(fontsize=8)
    quantiles = [.1, .3, .5, .7, .9]
    for k in range(1, 6):
        groups = [s["rt"][(s["cat"] == k) & s["crossed"]] for s in (a, b)]
        if all(len(g) for g in groups):
            axes[1, 1].plot(quantiles, np.quantile(groups[1], quantiles)-np.quantile(groups[0], quantiles), "o-", label="ABCDT"[k-1])
    axes[1, 1].axhspan(-5, 5, color="green", alpha=.12)
    axes[1, 1].set_xlabel("Conditional RT quantile"); axes[1, 1].set_ylabel("Candidate minus FFT (ms)")
    axes[1, 1].legend(fontsize=8)
    for s, label in [(a, "Original FFT"), (b, "Candidate")]:
        rt = np.sort(s["rt"][s["crossed"]])
        if len(rt):
            step = max(1, len(rt)//2000)
            axes[1, 2].plot(rt[::step], np.arange(1, len(rt)+1)[::step]/len(rt), label=label)
    axes[1, 2].set_xlabel("RT among crossings (ms)"); axes[1, 2].set_ylabel("Empirical CDF")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle(title+"\nDescriptive maps and point estimates; inferential bounds in tables")
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(path, dpi=120); plt.close(fig)


def precision_history(plt, history, core_cases, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, case in enumerate(core_cases):
        for condition in range(2):
            ax = axes[condition, col]
            for method in ("fft200", "repaired_kl", "bucketed_kl"):
                points = {}
                for look in history:
                    for r in look["records"]:
                        if r["case"] == case and r["condition"] == condition and r["method"] == method:
                            e = next(e for e in r["endpoints"] if e["metric"] == "low_support_0.1")
                            points[r["rounds"]] = e
                x = sorted(points)
                if not x: continue
                values = np.array([points[n]["difference"] for n in x])*100
                bounds = np.array([points[n]["interval"] for n in x], dtype=float)*100
                line, = ax.plot(x, values, "o-", label=method)
                ax.vlines(x, bounds[:, 0], bounds[:, 1], color=line.get_color(), alpha=.7)
            ax.axhspan(-.5, .5, color="green", alpha=.12)
            ax.axhline(0, color="black", lw=.6)
            ax.set_title(f"{case}, condition {condition+1}")
            ax.set_xlabel("Independent runs"); ax.set_ylabel("Low-support difference (pp)")
            ax.set_xticks([5, 10, 15, 20])
            if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    fig.suptitle("Registered-look intervals; stopped arms retain their original prefix")
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=130); plt.close(fig)

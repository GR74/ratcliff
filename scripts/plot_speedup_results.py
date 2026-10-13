"""Plot measured pilot timings and accuracy from benchmark_speedups.py outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("legacy", "blocked_full", "early_stop", "staged", "combined")
LABELS = ("Existing", "Blocks\nfull horizon", "Blocks\nearly stop",
          "Blocks\nstaged fit", "Both ideas")
COLORS = ("#65758b", "#8f8bc9", "#008b8b", "#db9525", "#3683d1")


def plot_results(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / "results.json").read_text())
    completed = [r for r in rows if r.get("status") == "completed"]
    fit_variants = [v for v in VARIANTS if any(r["action"] == "fit"
                    and r["variant"] == v for r in completed)] or list(VARIANTS)
    gpu_run = all(r.get("metadata", {}).get("backend") == "gpu" for r in completed)
    run_label = "H100 pilot" if gpu_run and completed else "CPU smoke (not GPU evidence)"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for mi, model in enumerate(("a", "b")):
        for col, (field, title, multiplier) in enumerate((
            ("fit_seconds", "Timed fitting (seconds)", 1),
            ("mean_relative_parameter_error", "Mean active parameter error (%)", 100),
            ("independent_full_budget_loss", "Independent G2 (lower is better)", 1),
        )):
            ax = axes[mi, col]
            for vi, variant in enumerate(fit_variants):
                values = [r[field] * multiplier for r in completed
                          if r["action"] == "fit" and r["model"] == model
                          and r["variant"] == variant]
                if values:
                    ax.bar(vi, np.median(values), color=COLORS[VARIANTS.index(variant)], alpha=.8)
                    offsets = np.linspace(-.13, .13, len(values))
                    ax.scatter(vi + offsets, values, color="#172233", s=22, zorder=3)
                    ax.text(vi, max(values), f" n={len(values)}", ha="center",
                            va="bottom", fontsize=8)
            ax.set_title(f"Model {model.upper()}: {title}")
            if not any(r["action"] == "fit" and r["model"] == model for r in completed):
                ax.text(.5, .5, "No complete fitting cases", transform=ax.transAxes,
                        ha="center", va="center", color="#65758b")
            ax.set_xticks(range(len(fit_variants)),
                          [LABELS[VARIANTS.index(v)] for v in fit_variants], fontsize=9)
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
    settings = next((r["settings"] for r in completed if r["action"] == "fit"), {})
    fig.suptitle(f"{run_label}: runtime and accuracy must be assessed together\n"
                 f"Bars = medians; dots = starting points. "
                 f"{settings.get('final_nsim', '?')} final trials/condition, "
                 f"{settings.get('fit_iters', '?')} iteration cap; convergence not assumed.",
                 fontsize=14)
    fig.savefig(output / "fit-runtime-and-accuracy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for mi, model in enumerate(("a", "b")):
        ax = axes[mi]
        for vi, variant in enumerate(VARIANTS[:3]):
            case = next((r for r in completed if r["action"] == "micro"
                         and r["model"] == model and r["variant"] == variant), None)
            if case:
                values = case["timings_seconds"]
                ax.bar(vi, np.median(values), color=COLORS[vi], alpha=.8)
                ax.scatter(vi + np.linspace(-.12, .12, len(values)), values,
                           color="#172233", s=20, zorder=3)
        ax.set_title(f"Model {model.upper()}: all-condition simulation")
        if not any(r["action"] == "micro" and r["model"] == model for r in completed):
            ax.text(.5, .5, "Not measured in this pass", transform=ax.transAxes,
                    ha="center", va="center", color="#65758b")
        ax.set_xticks(range(3), LABELS[:3])
        ax.set_ylabel("Synchronized seconds (compile excluded)")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(f"{run_label}: simulator timings; dots are repeated calls")
    fig.savefig(output / "simulation-runtime.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_results(args.source, args.output)

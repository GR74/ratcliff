"""Rebuild the teaching figures in docs/tutorials/assets using NumPy/Matplotlib.

These are seeded mathematical illustrations, not new JAX or H100 benchmarks.
Run from the repository root: python scripts/make_teaching_visuals.py
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
NAVY = "#17324d"
TEAL = "#007f83"
ORANGE = "#c56b24"
RED = "#b5434d"
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "text.color": NAVY,
    "axes.labelcolor": NAVY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def stopped_path(seed, v=0.5, sigma=1.0, boundary=1.0, dt=0.002):
    rng = np.random.default_rng(seed)
    x = np.r_[0.0, np.cumsum(v * dt + sigma * np.sqrt(dt) * rng.normal(size=6000))]
    hits = np.flatnonzero(np.abs(x) >= boundary)
    if not len(hits):
        raise RuntimeError("Illustration did not reach a boundary")
    k = int(hits[0])
    return np.arange(k + 1) * dt, x[:k + 1]


paths = [stopped_path(seed) for seed in range(40)]
upper = [p for p in paths if p[1][-1] > 0]
lower = [p for p in paths if p[1][-1] < 0]
fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
for index, (t, x) in enumerate([upper[0], upper[2], lower[0]]):
    color = [TEAL, ORANGE, RED][index]
    label = ["Trial 1: upper choice", "Trial 2: upper choice", "Trial 3: lower choice"][index]
    ax.plot(t, x, color=color, lw=2, label=label)
    ax.scatter(t[-1], x[-1], color=color, s=48, zorder=5)
    ax.annotate(f"{t[-1]:.2f} s", (t[-1], x[-1]), xytext=(5, -19 if x[-1] > 0 else 9),
                textcoords="offset points", color=color, fontsize=11)
ax.axhline(1, color=NAVY, ls="--", lw=1.4)
ax.axhline(-1, color=NAVY, ls="--", lw=1.4)
ax.axhline(0, color="#d0d7df", lw=1)
ax.set(title="Same signal, different noise: the first boundary hit ends the trial",
       xlabel="Decision time (seconds; nondecision time is added afterward)",
       ylabel="Accumulated evidence", ylim=(-1.28, 1.28))
ax.text(0.015, 0.93, "+a: choose upper", transform=ax.transAxes, fontsize=11)
ax.text(0.015, 0.02, "−a: choose lower", transform=ax.transAxes, fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, fontsize=10)
ax.grid(axis="x", alpha=0.15)
save(fig, "ddm-paths.png")


a = np.linspace(0.1, 1.8, 180)
v, sigma = 0.5, 1.0
p_upper = 1 / (1 + np.exp(-2 * v * a / sigma ** 2))
mean_time = a / v * np.tanh(a * v / sigma ** 2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
axes[0].plot(a, 100 * p_upper, color=TEAL, lw=3)
axes[0].set(title="More evidence required → more accurate", xlabel="Boundary distance a",
            ylabel="Upper-choice probability (%)", ylim=(50, 100))
axes[1].plot(a, mean_time, color=ORANGE, lw=3)
axes[1].set(title="More evidence required → longer wait", xlabel="Boundary distance a",
            ylabel="Mean decision time (seconds)")
for axis in axes:
    axis.grid(alpha=0.18)
fig.suptitle("Exact simple-DDM formulas: start at 0, drift = 0.5, noise = 1", fontsize=15)
save(fig, "ddm-threshold-tradeoff.png")


def embedded_spectrum(n=100, m=160, sig=10.0):
    dx, dy = np.arange(m)[None, :], np.arange(n)[:, None]
    def kernel(x, y):
        return (1 - x * x / sig ** 2 - x * y / sig ** 2 - y * y / sig ** 2) * np.exp(
            -(x * x + y * y) / sig ** 2
        )
    rows, cols = kernel(dx, dy), kernel(-dx, dy)
    embedded = np.zeros((2 * n - 1, 2 * m - 1))
    embedded[:n, :m] = rows
    embedded[:n, m:] = cols[:, 1:][:, ::-1]
    embedded[n:, :m] = cols[1:, :][::-1, :]
    embedded[n:, m:] = rows[1:, 1:][::-1, ::-1]
    spectrum = np.fft.fft2(embedded).real / embedded.size
    assert spectrum.min() > -1e-14
    return np.sqrt(np.maximum(spectrum, 0))


n, m = 100, 160
row, col = np.arange(n)[:, None] + 1, np.arange(m)[None, :] + 1
def bump(center, width):
    return np.exp(-((row - 50) ** 2 + (col - center) ** 2) / (2 * width ** 2)) / (
        width * np.sqrt(2 * np.pi)
    )

zone = np.full((n, m), 5)
b1, b2, b3 = [bump(c, 6) for c in (80, 30, 130)]
zone[b1 > 0.0003] = 2
zone[b1 > 0.03] = 1
zone[b2 > 0.0003] = 3
zone[b3 > 0.0003] = 4
drift = 15 * bump(80, 12) + 10 * bump(30, 12) + 8 * bump(130, 12)
lam = embedded_spectrum()
rng = np.random.default_rng(7)
field = np.zeros((n, m))
early = None
crossing_step = None
step = 0
while step < 400 and crossing_step is None:
    f = np.fft.fft2(lam * (rng.normal(size=lam.shape) + 1j * rng.normal(size=lam.shape)))
    for noise in (f[:n, :m].real, f[:n, :m].imag):
        step += 1
        delta = drift + noise
        field += delta - delta.mean()
        if step == 1:
            early = field.copy()
        if field.max() > 10:
            crossing_step = step
            break
assert crossing_step is not None
winner = np.unravel_index(np.argmax(field), field.shape)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
cmap = ListedColormap([TEAL, "#8dc8b6", ORANGE, "#6e82b8", "#e2e8ee"])
zone_img = axes[0].imshow(zone, cmap=cmap, vmin=0.5, vmax=5.5, origin="lower")
axes[0].set_title("Five response regions")
cb = fig.colorbar(zone_img, ax=axes[0], shrink=0.65, ticks=[1, 2, 3, 4, 5])
cb.set_label("Category")
limit = max(abs(field.min()), abs(field.max()))
for axis, array, title in zip(axes[1:], [early, field],
                            ["Evidence after 10 ms", f"First crossing at {10 * step} ms"]):
    im = axis.imshow(array, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
    axis.set_title(title)
    fig.colorbar(im, ax=axis, shrink=0.65, label="Evidence")
axes[2].scatter(winner[1], winner[0], s=115, facecolors="none", edgecolors="black", linewidths=2)
axes[2].set_title(f"First crossing at {10 * step} ms\nCategory {int(zone[winner])} wins; boundary 10", fontsize=11)
for axis in axes:
    axis.set(xlabel="Column", ylabel="Row")
fig.suptitle("Model B mechanism: many locations compete on one field", fontsize=16)
fig.supxlabel("Seeded NumPy illustration of the FFT mechanism; not a JAX benchmark", fontsize=10)
save(fig, "spatial-ddm.png")


fig, ax = plt.subplots(figsize=(11.8, 4.6), layout="constrained")
ax.set(xlim=(0, 12), ylim=(0, 5))
ax.axis("off")
def box(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
                              facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color="white", fontsize=12)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=17, lw=1.6, color=NAVY))
ax.text(0, 4.6, "Old recurrence: carry the accumulator through every step", fontsize=15, weight="bold")
for x, label in [(0.1, "Step 1\nadd + center"), (3.1, "Step 2\nadd + center"),
                 (6.1, "Step 3\nadd + center"), (9.1, "Keep going\n400 steps")]:
    box(x, 3.3, 2.6, 0.9, label, NAVY)
for x in [2.7, 5.7, 8.7]:
    arrow(x + 0.08, 3.75, x + 0.32, 3.75)
ax.text(0, 2.65, "Rewritten: prepare the increments, then use an optimized prefix sum", fontsize=15, weight="bold")
box(0.1, 1.2, 3.2, 0.95, "Generate all increments\nin one batch", TEAL)
box(4.4, 1.2, 3.2, 0.95, "Center each increment\nsubtract its spatial mean", TEAL)
box(8.7, 1.2, 3.0, 0.95, "cumsum over time\nthen find first crossing", TEAL)
arrow(3.38, 1.68, 4.3, 1.68)
arrow(7.68, 1.68, 8.6, 1.68)
ax.text(0.1, 0.35, "Example: increments [2, 3, −1, 4]  →  cumulative evidence [2, 5, 4, 8]",
        fontsize=14)
save(fig, "jax-prefix-sum.png")

# Verify the centering identity using the same input in both formulations.
deltas = np.random.default_rng(12).normal(size=(17, 9))
state = np.zeros(9)
loop = []
for delta in deltas:
    state = state + delta
    state -= state.mean()
    loop.append(state.copy())
prefix = np.cumsum(deltas - deltas.mean(axis=1, keepdims=True), axis=0)
assert np.allclose(loop, prefix, atol=1e-12)


y, x = np.mgrid[0:1:80j, 0:1:110j]
toy_modes = [np.cos(2 * np.pi * x), np.sin(2 * np.pi * y), np.cos(2 * np.pi * (x + y))]
weights = [0.8, -0.5, 1.1]
combined = sum(w * mode for w, mode in zip(weights, toy_modes))
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.5), layout="constrained")
for index, (axis, array) in enumerate(zip(axes, [*toy_modes, combined])):
    axis.imshow(array, origin="lower", cmap="RdBu_r", vmin=-2.4, vmax=2.4)
    axis.set_title(f"Shape {index + 1}\nweight {weights[index]:+.1f}" if index < 3 else "Add the weighted shapes\n→ one random field", fontsize=12)
    axis.set_xticks([])
    axis.set_yticks([])
fig.suptitle("Low-rank noise: keep useful shapes, draw fresh random weights", fontsize=15)
fig.supxlabel("Three toy shapes shown for teaching; the recorded run retained 1,325 complex modes", fontsize=10)
save(fig, "kl-patterns.png")


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
labels = ["FFT (Stage 5)", "KL first version", "KL cached + padded"]
for axis, values, title, xlabel in zip(
    axes, [[10.66, 6.59, 4.71], [30.3, 27.7, 20.1]],
    ["One full two-condition loss evaluation", "Entire synthetic recovery run"],
    ["Seconds (lower is faster)", "Minutes (lower is faster)"],
):
    bars = axis.barh(labels, values, color=[NAVY, ORANGE, TEAL], height=0.55)
    axis.invert_yaxis()
    axis.set(title=title, xlabel=xlabel, xlim=(0, max(values) * 1.19))
    for bar, value in zip(bars, values):
        axis.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2,
                  f"{value:g}", va="center", fontsize=12)
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
fig.suptitle("Recorded H100 results: June 5, 2026", fontsize=16)
fig.supxlabel("9,000 simulations per condition; different versions needed different evaluation counts", fontsize=10)
save(fig, "jax-benchmarks.png")


epsilon = np.array([0.5, -1.0, 0.7, 0.2, 1.4])
increments = 0.5 * 0.1 + 0.2 * np.sqrt(0.1) * epsilon
evidence = np.cumsum(increments)
assert np.all(evidence[:-1] < 0.3) and evidence[-1] > 0.3
numbers = {
    "walkthrough_evidence": evidence.tolist(),
    "walkthrough_rt_seconds": 0.5 + 0.2,
    "simulator_speedup_six_node": 11 / 4.7,
    "objective_speedup_six_node_estimate": 22 / 4.71,
    "objective_speedup_stage5": 10.66 / 4.71,
    "fit_speedup_stage5": 30.3 / 20.1,
    "kl_cache_padding_time_reduction_percent": 100 * (27.7 - 20.1) / 27.7,
    "spatial_example_crossing_step": int(crossing_step),
    "spatial_example_category": int(zone[winner]),
}
(OUT / "teaching-numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
print(f"Created 6 figures in {OUT}")
print(json.dumps(numbers, indent=2))

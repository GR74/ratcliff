"""CPU-only forensic replay of the archived RTX 3060 fitting failures.

The simulator is not rerun. Every proposed coordinate and returned loss is
checked against the saved trace before any optimizer diagnosis is accepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def unpack(destination):
    raw = ROOT / "docs/results/2026-10-13-rtx3060/raw"
    records = json.loads((raw / "manifest.json").read_text())
    record = next(r for r in records if r["archive"] == "six-hour-study.tar.gz")
    archive = raw / record["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == record["sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not member.isfile():
                raise ValueError(f"unsafe archive member: {member.name}")
        for member in handle.getmembers():
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    for entry in record["files"]:
        assert hashlib.sha256((destination / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    return destination / "results/study-512-six-hour"


def replay(source, fit):
    import numpy as np
    from scipy.optimize import minimize
    from shared.staged_fit import minimize_staged

    name = f'{fit["scenario"]}-{fit["start_seed"]}-{fit["variant"]}'
    path = source / f"trace-{name}.jsonl"
    trace = [json.loads(line) for line in path.read_text().splitlines()]
    cursor = 0

    class RecordedDeadline(Exception):
        pass

    def oracle(params, nsim=512):
        nonlocal cursor
        assert nsim == 512
        assert cursor < len(trace), (name, "trace exhausted", cursor)
        row = trace[cursor]
        assert row["eval"] == cursor + 1
        np.testing.assert_array_equal(params, row["params"], err_msg=f"{name}, eval {cursor + 1}")
        cursor += 1
        if fit["status"] == "fit_time_limit" and cursor == len(trace):
            raise RecordedDeadline()
        return row["loss"]

    x0 = np.asarray(fit["x0"])
    result = None
    try:
        result = minimize_staged(oracle, x0, nsim=512, maxiter=1000, tol=1e-5)
    except RecordedDeadline:
        assert fit["status"] == "fit_time_limit"
    assert cursor == len(trace)
    if result:
        np.testing.assert_array_equal(result.params, fit["params"])
        assert result.n_evals == fit["n_evals"]
        assert result.converged == fit["converged"]
        cursor = 0
        detail = minimize(oracle, x0, method="Nelder-Mead", options={
            "maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5})
        simplex, losses = detail.final_simplex
        final_simplex = dict(max_coordinate_span=float(np.max(np.ptp(simplex, axis=0))),
                             loss_span=float(np.ptp(losses)), unique_losses=len(set(losses.tolist())))
    else:
        final_simplex = None
    truth = np.asarray(fit["true_params"])
    active = [i for i in range(13) if i != 6]
    raw = np.asarray(fit["params"])
    # Use the production clamp rather than a separate copy of the bounds.
    import jax
    jax.config.update("jax_enable_x64", True)
    from model_b.objective import clamp_b
    effective = np.asarray(clamp_b(raw))
    error = np.mean(np.abs(effective[active] / truth[active] - 1))
    assert abs(error - fit["mean_active_effective_parameter_error"]) < 1e-12
    values = np.asarray([r["loss"] for r in trace])
    proposals = np.asarray([r["params"] for r in trace])
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(proposals))
    scoring = json.loads((source / f"score-{name}.json").read_text())
    np.testing.assert_array_equal(scoring["final_params"], fit["params"])
    assert scoring["target_reached"] is False
    best_indices = np.flatnonzero(np.r_[True, np.diff(np.minimum.accumulate(values)) < 0])
    return dict(case=name, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                exact_coordinate_replay=True, status=fit["status"], converged=fit["converged"],
                finite_proposals_and_losses=True, independently_scored_target_pass=False,
                evaluations=len(trace), final_simplex=final_simplex,
                initial_error=float(np.mean(np.abs(x0[active] / truth[active] - 1))),
                final_error=float(error), initial_loss=float(values[0]), final_loss=fit["loss"],
                best_updates=len(best_indices), last_strict_improvement_eval=int(best_indices[-1] + 1),
                final_loss_occurrences=int(np.count_nonzero(values == fit["loss"])),
                per_parameter_initial_error=(x0 / truth - 1).tolist(),
                per_parameter_final_error=(effective / truth - 1).tolist(),
                proposal_min=proposals.min(axis=0).tolist(), proposal_max=proposals.max(axis=0).tolist(),
                unused_sv_initial_simplex_loss_identical=trace[0]["loss"] == trace[7]["loss"],
                maximum_relative_excursion=np.max(np.abs(proposals / x0 - 1), axis=0).tolist(),
                effective_clamping_changes_final=not np.array_equal(raw, effective),
                params=raw.tolist(), truth=truth.tolist())


def probe_mode_order(output):
    """Compare the real production basis, then align only its mode labels.

    Alignment here is a diagnostic intervention with an unchanged retained
    set. It is not a deployed sampler and does not handle changing mode sets.
    """
    import numpy as np
    from model_b.grf import calc_LAM
    from model_b import grf_kl

    def spectrum(sig):
        # Match production rounding and the JAX square exactly. Last-bit
        # changes matter here because the diagnosis concerns tied ranks.
        sig = round(float(sig), 6)
        eigenvalues = np.asarray(calc_LAM(s1=sig, s2=sig) ** 2)
        indices = np.argsort(-eigenvalues.ravel())
        values = eigenvalues.ravel()[indices]
        k = min(int(np.searchsorted(np.cumsum(values) / values.sum(), .99)) + 1, 2000)
        return eigenvalues, indices[:k]

    rows = []
    # One physical location set is used for both members of every comparison.
    locations = np.random.default_rng(55).choice(16000, 128, replace=False)
    grf_kl._BASIS_CACHE_MAXSIZE = 1
    for sig in (10., 11.651022):
        e0, ids0 = spectrum(sig)
        e1, ids1 = spectrum(sig + 1e-6)
        v0_full, k0, _ = grf_kl.calc_kl_basis(sig, pad_to_k_max=True)
        v0 = np.asarray(v0_full[locations, :])
        del v0_full
        v1_full, k1, _ = grf_kl.calc_kl_basis(sig + 1e-6, pad_to_k_max=True)
        v1 = np.asarray(v1_full[locations, :])
        del v1_full
        assert k0 == k1 and set(ids0) == set(ids1), "alignment probe requires an unchanged set"
        lookup = {mode: index for index, mode in enumerate(ids1)}
        indices = np.asarray([lookup[mode] for mode in ids0])
        column_map = np.c_[2 * indices, 2 * indices + 1].ravel()
        aligned = np.zeros_like(v1)
        aligned[:, :2 * k0] = v1[:, column_map]
        z = np.random.default_rng(7).normal(size=(32, 4000)).astype(np.float32)
        original_fields, changed_fields, aligned_fields = z @ v0.T, z @ v1.T, z @ aligned.T
        row = dict(sig=sig, delta=1e-6, retained_modes=k0, retained_set_identical=True,
                   same_rank_fraction=float(np.mean(ids0 == ids1)),
                   relative_spectrum_change=float(np.linalg.norm(e1-e0) / np.linalg.norm(e0)),
                   expected_relative_noise_change=float(np.linalg.norm(v1-v0) / np.linalg.norm(v0)),
                   expected_relative_noise_change_after_alignment=float(np.linalg.norm(aligned-v0) / np.linalg.norm(v0)),
                   sampled_relative_noise_change=float(np.linalg.norm(changed_fields-original_fields) / np.linalg.norm(original_fields)),
                   sampled_relative_noise_change_after_alignment=float(np.linalg.norm(aligned_fields-original_fields) / np.linalg.norm(original_fields)),
                   permutation_covariance_preserved=bool(np.allclose(v1 @ v1.T, aligned @ aligned.T, atol=2e-6, rtol=2e-6)))
        assert row["expected_relative_noise_change"] > .5
        assert row["expected_relative_noise_change_after_alignment"] < 1e-5
        assert row["permutation_covariance_preserved"]
        rows.append(row)
    report = dict(scope="Production padded basis at 100x160; 128 spatial locations; CPU, x64 spectrum/fp32 basis; 32 fixed diagnostic normal draws.",
                  intervention="Reassign the same coefficients by physical Fourier mode, retaining the same modes and eigenvalues. Diagnostic only.", results=rows)
    (output / "mode-order-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def probe_trial_budget(source, output):
    """Conditional bootstrap of real saved samples using the production G2."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from model_b.objective import condition_g2_b

    archive = ROOT / "docs/results/2026-10-13-rtx3060/raw/pilots-and-interrupted-study.tar.gz"
    manifest = json.loads((archive.parent / "manifest.json").read_text())
    record = next(r for r in manifest if r["archive"] == archive.name)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == record["sha256"]
    with tarfile.open(archive) as handle:
        with handle.extractfile("results/pilot-9000/samples-baseline-legacy-0.npz") as stream:
            with np.load(stream) as saved:
                rt, cat = saved["rt"].copy(), saved["cat"].copy()
    with np.load(source / "data-baseline.npz") as saved:
        data = {key: jnp.asarray(saved[key]) for key in saved.files}
    score = jax.jit(jax.vmap(condition_g2_b, in_axes=(0, 0, 0, 0, 0)))
    batch_score = jax.jit(jax.vmap(lambda a, b: score(
        a, b, data["prop"], data["count"], data["quant"]).sum()))
    rows = []
    for nsim in (128, 512, 2048, 9000):
        indices = np.random.default_rng(766).integers(0, 9000, (128, 2, nsim))
        rt_sample, cat_sample = rt[np.arange(2)[None, :, None], indices], cat[np.arange(2)[None, :, None], indices]
        values = np.asarray(batch_score(jnp.asarray(rt_sample), jnp.asarray(cat_sample)))
        rows.append(dict(nsim=nsim, replicates=128, median_g2=float(np.median(values)),
                         p10_g2=float(np.quantile(values, .1)), p90_g2=float(np.quantile(values, .9))))
    report = dict(scope="Conditional bootstrap of one real 9000-trial pool at truth; not independent new simulations, fitting, or an accuracy claim.",
                  full_pool_g2=float(score(jnp.asarray(rt), jnp.asarray(cat), data["prop"], data["count"], data["quant"]).sum()),
                  pool_seed=0, observed_seed=123, bootstrap_seed=766, results=rows)
    (output / "trial-budget-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def publish(output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = ROOT / "docs/results/2026-10-15-model-b-fit-diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("replay.json", "mode-order-probe.json", "trial-budget-probe.json", "historical-start.json"):
        shutil.copyfile(output / name, destination / name)
    replayed = json.loads((output / "replay.json").read_text())
    budget = json.loads((output / "trial-budget-probe.json").read_text())
    modes = json.loads((output / "mode-order-probe.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    labels = []
    for index, case in enumerate(replayed["cases"]):
        labels.append(case["case"].replace("_boundary", "").replace("early_stop", "early"))
        axes[0].plot([case["initial_error"] * 100, case["final_error"] * 100], [index, index], color="#a5a5a5")
        axes[0].scatter(case["initial_error"] * 100, index, facecolor="white", edgecolor="#595959", s=45)
        axes[0].scatter(case["final_error"] * 100, index, color="#337f70", s=40)
    axes[0].axvline(5, color="#b04646", linestyle="--", label="5% target")
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, xlabel="Mean active parameter error (%)", title="Searches stayed far from the target", xlim=(0, 15))
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", frameon=False)
    rows = budget["results"]
    x = np.arange(len(rows))
    med = np.asarray([r["median_g2"] for r in rows])
    bounds = np.asarray([[r["p10_g2"] for r in rows], [r["p90_g2"] for r in rows]])
    axes[1].errorbar(x, med, yerr=np.vstack([med - bounds[0], bounds[1] - med]), fmt="o-", color="#337f70", capsize=4)
    axes[1].axhline(budget["full_pool_g2"], color="#595959", linestyle=":", label="Saved full-pool score")
    axes[1].set(xticks=x, xticklabels=[str(r["nsim"]) for r in rows], yscale="log", xlabel="Resampled trials per condition", ylabel="G2 mismatch at the true parameters", title="Small simulation budgets add score noise")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CPU diagnosis of archived Model B fits", fontsize=14)
    fig.savefig(destination / "fitting-diagnosis.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.5), constrained_layout=True)
    row = modes["results"][0]
    ax.barh(["Current ranked assignment", "Aligned by physical mode"],
            [row["expected_relative_noise_change"], row["expected_relative_noise_change_after_alignment"]],
            color=["#b36650", "#337f70"])
    ax.set(xscale="log", xlim=(1e-8, 2), xlabel="Relative RMS field change under the same normal draws",
           title="Changing noise width from 10 to 10.000001")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination / "mode-assignment.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Published compact CPU diagnostic evidence and figures.")


def main():
    # This process must remain CPU-only even on a future CUDA-enabled host.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np
    import scipy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".benchmark-results/fit-diagnosis")
    parser.add_argument("--require-quality", action="store_true", help="Exit nonzero on the historical failure.")
    parser.add_argument("--case", help="Replay just one archived case to minimize the reproduction.")
    parser.add_argument("--probes", action="store_true", help="Run bounded CPU mode-order and resampling probes.")
    parser.add_argument("--publish", action="store_true", help="Copy compact evidence and plots into docs/results.")
    args = parser.parse_args()
    source = unpack(args.output / "archived")
    rows = json.loads((source / "results.json").read_text())
    fits = [r for r in rows if r["action"] == "fit"]
    if args.case:
        fits = [r for r in fits if f'{r["scenario"]}-{r["start_seed"]}-{r["variant"]}' == args.case]
        if not fits:
            raise ValueError("unknown case")
    cases = [replay(source, fit) for fit in fits]
    import jax
    report = dict(scope="CPU optimizer replay; no new simulator or GPU work", numpy=np.__version__,
                  scipy=scipy.__version__, source_packages=fits[0]["metadata"]["packages"], cases=cases)
    report["probe_runtime"] = dict(jax=jax.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64))
    report["implementation_source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("model_b/grf.py", "model_b/grf_kl.py", "model_b/objective.py", "model_b/fit.py", "shared/staged_fit.py")}
    filename = f"replay-{args.case}.json" if args.case else "replay.json"
    (args.output / filename).write_text(json.dumps(report, indent=2) + "\n")
    active = [i for i in range(13) if i != 6]
    historical = np.random.RandomState(0).uniform(.9, 1.1, 13)
    history = dict(source="scripts/h100_stage6_kl_benchmark.py: np.random.seed(0), uniform(.9,1.1,13)",
                   june_initial_multipliers=historical.tolist(),
                   june_initial_mean_active_parameter_error=float(np.mean(abs(historical[active] - 1))),
                   reported_june_final_mean_active_parameter_error=.0298,
                   fitting_trials_per_condition_june=9000, fitting_trials_per_condition_october=512,
                   observations_per_condition_june=1024, observations_per_condition_october=9000,
                   note="June start was already below the current 5% mean-error criterion; its joint prediction gate was not tested here.")
    (args.output / "historical-start.json").write_text(json.dumps(history, indent=2) + "\n")
    for case in cases:
        print(case["case"], "EXACT REPLAY", case["evaluations"], "evaluations; error",
              f'{100 * case["final_error"]:.3f}%', "converged", case["converged"])
    if args.probes:
        print(json.dumps(probe_mode_order(args.output), indent=2))
        print(json.dumps(probe_trial_budget(source, args.output), indent=2))
    if args.publish:
        if args.case:
            raise ValueError("publish requires all cases")
        publish(args.output)
    if args.require_quality and any(c["final_error"] > .05 for c in cases):
        raise SystemExit(f"QUALITY FAIL reproduced: {len(cases)} archived fits exceed 5% parameter error.")


if __name__ == "__main__":
    main()

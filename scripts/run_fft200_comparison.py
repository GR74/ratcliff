"""Serial RTX 4060 comparison: pilot, locked fidelity study, replay, gated fits.

This is a new study of the frozen repaired simulator, not a production change.
The 27-fit-per-method study is deliberately not part of this runner.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".benchmark-results/2026-10-21-rtx4060-baseline"
FROZEN = BASE / "source-322bd1f"
METHODS = ("fft199", "fft200", "repaired_kl", "bucketed_kl")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(FROZEN))


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "default")
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    from model_b import grf, grf_kl, simulate, objective
    from shared import prng

    assert jax.default_backend() == "gpu" and "4060" in jax.devices()[0].device_kind
    active = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    others = [s for s in active.splitlines() if s.strip() and not s.startswith(str(os.getpid())+",")]
    if others:
        raise RuntimeError(f"GPU already occupied: {others}")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    public = ROOT / "docs/results/2026-10-26-fft200-comparison"
    public.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "source-manifest.json").read_text())
    for relative, sha in manifest["files"].items():
        assert digest(FROZEN / relative.replace("\\", "/")) == sha, relative
    tools_dir = out / "run-tools"
    tools_dir.mkdir(exist_ok=True)
    files = [ROOT / "scripts/run_fft200_comparison.py", ROOT / "scripts/fft200_comparison_metrics.py", ROOT / "scripts/start_fft200_comparison.ps1",
             ROOT / "scripts/model_b_baseline_observer.py", ROOT / "model_b/grf.py"]
    hashes = {}
    for path in files:
        target = tools_dir / path.name
        if target.exists() and digest(target) != digest(path):
            raise RuntimeError(f"Source changed: {path}; use a new study directory")
        if not target.exists():
            shutil.copyfile(path, target)
        hashes[path.name] = digest(target)
    metrics = load(tools_dir / "fft200_comparison_metrics.py", "comparison_metrics_frozen")
    observer_helper = load(tools_dir / "model_b_baseline_observer.py", "observer_helper_frozen")
    candidate = load(tools_dir / "grf.py", "fft200_frozen")
    prior_screen = json.loads((ROOT / "docs/results/2026-10-25-fft200-screen/screen.json").read_text())
    assert prior_screen["status"] == "complete" and hashes["grf.py"] == prior_screen["candidate_sha256"], "Candidate differs from the audited covariance builder"
    original_lam = grf.calc_LAM
    observer = observer_helper.load_observer(FROZEN)
    geometry = metrics.projection_geometry()
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    baseline_protocol = json.loads((BASE / "archived-case/protocol.json").read_text())
    truth = np.array(baseline_protocol["truth"])
    old_fit = json.loads((BASE / "archived-case/fit-baseline-1-early_stop.json").read_text())
    trace_path = BASE / "archived-case/trace-baseline-1-early_stop.jsonl"
    trace = [json.loads(s) for s in trace_path.read_text().splitlines()]
    vectors = [r["params"] for r in trace if len(r.get("params", [])) == 13 and np.isfinite(r["params"]).all()][:30]
    assert len(vectors) == 30
    x0 = truth*(1+np.random.default_rng(1).uniform(-.2, .2, 13))
    cases = []
    for sig in (5., 10., 15.):
        for scenario in ("ordinary", "quarter_drift", "zero_drift", "boundary_5", "boundary_20"):
            p = truth.copy(); p[5] = sig
            mask_p = p.copy()
            if scenario == "quarter_drift": p[7:] *= .25
            if scenario == "zero_drift": p[7:] = 0.
            if scenario.startswith("boundary_"): p[2] = float(scenario.split("_")[1])
            cases.append(dict(name=f"sig{sig:g}_{scenario}", params=p.tolist(), mask_params=mask_p.tolist(),
                              zero_drift_geographic_control=scenario == "zero_drift"))
    for name, p in [("archived_start", trace[0]["params"]), ("archived_fit", old_fit["params"])]:
        p = np.asarray(objective.clamp_b(jnp.asarray(p)))
        cases.append(dict(name=name, params=p.tolist(), mask_params=p.tolist(), zero_drift_geographic_control=False))
    family = len(cases)*2*3
    alpha_endpoint = .025/(family*34)
    alpha_spatial = .025/family
    protocol = dict(version=1, source_commit=manifest["commit"], source_hashes=hashes,
        frozen_manifest_sha256=digest(BASE / "source-manifest.json"), methods=METHODS,
        cases=cases, pilot_trials=4096, qualification_seed_count=20,
        qualification_min_trials_per_seed=9000, qualification_max_trials_per_seed=45000,
        behavior_seconds_cap=8*3600, replay_seconds_cap=12*3600, fits_seconds_cap_each=8100,
        scoring_seconds_cap=8*3600, trace_vectors=vectors, trace_sha256=digest(trace_path),
        alpha_endpoint=alpha_endpoint, alpha_spatial=alpha_spatial,
        family_note="All 17 parameter cases, both conditions and three contrasts; 0.025 allocated to 34 binary/RT endpoints, 0.025 to spatial endpoints.",
        inference="Exact binary and order-statistic quantile bounds; spatial bounded-difference upper bound; Bonferroni family control. Replaces the prospective bootstrap with a conservative analytic bound before execution.",
        precision_note="Separate pilot, normal-planning 90% power at equality, RT density inflated 1.5x. Sparse pilot tails or oversized requirements defer qualification and fits.",
        nsim_objective=9000, chunk=8, block=32, replay_pairs=30, replay_warmups=3,
        seed_base=202610260, data_seed=202610280, training_seed=321,
        validation_seeds=list(range(202610300, 202610320)), calibration_seeds=list(range(202610400, 202610420)),
        truth=truth.tolist(), x0=x0.tolist(), optimizer="Nelder-Mead", maxiter=1000,
        xatol=1e-5, fatol=1e-5, checkpoint_training_seconds=300,
        acceptance="Recovery mean <=5%, worst <=10% active relative error. Predictive thresholds calibrated before fits. sv inactive, retained and reported.",
        limits="Single matched case only; no 27-fit study; no default adoption; no independent Fortran qualification.",
        precision="x64 on, fp32 fields, default matmul", jax=jax.__version__, numpy=np.__version__)
    if (out / "protocol.json").exists():
        if json.loads((out / "protocol.json").read_text()) != clean(protocol):
            raise RuntimeError("Protocol changed; use a new output directory")
    else:
        write(out / "protocol.json", protocol)
        write(public / "protocol.json", protocol)
    write(out / "frozen-source-manifest.json", manifest)
    import platform
    import scipy
    write(out / "environment.json", dict(python=sys.version, platform=platform.platform(),
          jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__,
          gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"], text=True).strip()))
    started = time.perf_counter()
    stop = threading.Event()
    state = dict(status="running", phase="setup", utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                 device=jax.devices()[0].device_kind, raw_directory=str(out), gates={})
    if (out / "status.json").exists():
        previous = json.loads((out / "status.json").read_text())
        state["gates"] = previous.get("gates", {})
    def report():
        rows = "\n".join(f"| {k} | {v} |" for k, v in state["gates"].items())
        text = f"""# FFT200 full-comparison follow-up

Status: **{state['status']}**, phase **{state['phase']}**.
Last update: {state.get('updated_utc', state['utc'])}. Device: {state['device']}.

| Gate | Result |
| --- | --- |
{rows}

This run compares original FFT, FFT200, repaired KL and bucketed KL in frozen
source 322bd1f. Defaults remain unchanged. A passed numerical covariance audit
does not itself establish behavioral or fitted-prediction equivalence.

The sequence is a separate 4,096-trial pilot; a locked, precision-planned
behavior study when feasible; thirty paired 30-vector optimizer-trace replays;
and one matched fit per arm only after FFT200 passes the fidelity and speed gates.
Sparse data, exhausted budgets and confidence intervals crossing margins are
inconclusive. The multi-day 27-fit study is not launched by this runner.

All source hashes, cases, seeds, margins and caps are in `protocol.json`.
Raw trials, GPU telemetry, per-call timings, progress and fit checkpoints are in
`{out.as_posix()}`. Large raw archives are kept outside versioned documentation.

Detailed phase summaries appear beside this report as phases finish. Spatial
confidence bounds use bounded differences and convexity on the fixed display
grid, replacing the earlier proposed bootstrap before collecting this study.
"""
        (public / "README.md").write_text(text, encoding="utf-8")
    def status(phase, **extra):
        state.update(phase=phase, elapsed_seconds=time.perf_counter()-started,
                     updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write(out / "status.json", state); write(public / "status.json", state); report()
        print(json.dumps(clean(dict(phase=phase, **extra))), flush=True)
    def telemetry():
        with (out / "gpu-telemetry.csv").open("a", buffering=1) as handle:
            while not stop.is_set():
                r = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if r.returncode == 0: handle.write(r.stdout)
                stop.wait(5)
    threading.Thread(target=telemetry, daemon=True).start()
    def select(method):
        grf.calc_LAM = candidate.calc_LAM_even if method == "fft200" else original_lam
        return dict(use_kl=method.endswith("kl"), stable_kl_noise=True, kl_bucketed=method == "bucketed_kl") if method.endswith("kl") else dict(use_kl=False)
    def sample(method, params, seed, n, *, observed=True, early=True):
        options = select(method)
        module = observer if observed else simulate
        p = np.asarray(params)
        outputs = []
        for ci, indices in enumerate(objective.COND_MAP_B):
            outputs.append(module.simulate_b_blocked_with_work(
                prng.split_for_condition(jax.random.key(seed), ci), *p[:4], *p[list(indices)],
                p[4], float(p[5]), 6., n, 8, time_block_size=32, stop_early=early, compact=True, **options))
        jax.block_until_ready(outputs)
        names = ["rt", "cat", "work", "steps", "positions", "crossed"] if observed else ["rt", "cat", "work"]
        return {name: np.stack([np.asarray(o[i]) for o in outputs]) for i, name in enumerate(names)}
    def cached_sample(stage, case, method, index, n):
        path = out / stage / case["name"] / method / f"batch-{index:02d}.npz"
        if path.exists():
            return dict(np.load(path))
        phase_seed = {"pilot": 0, "qualification": 1000000}[stage]
        seed = protocol["seed_base"]+phase_seed+cases.index(case)*10000+METHODS.index(method)*1000+index
        status(stage, case=case["name"], method=method, batch=index, trials_per_condition=n)
        t = time.perf_counter()
        values = sample(method, case["params"], seed, n)
        assert np.isfinite(values["rt"]).all()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **values); temp.replace(path)
        write(path.with_suffix(".json"), dict(seed=seed, seconds=time.perf_counter()-t, n=n, work=values["work"].tolist()))
        return values
    def condition(sample_values, ci):
        return {k: v[ci] for k, v in sample_values.items() if k != "work"}
    def masks(case, ci):
        p = np.asarray(case["mask_params"])
        v = simulate.drift_bumps(p[4])
        drift = np.asarray(sum(p[i]*b for i, b in zip(objective.COND_MAP_B[ci], v))).ravel()
        return [drift <= c*drift.max() for c in metrics.CUTOFFS]
    def objective_call(method, p, data, seed=321):
        options = select(method)
        return float(objective.fofs_b_new(jnp.asarray(p), data, jax.random.key(seed), nsim=9000,
                    chunk_size=8, time_block_size=32, **options).block_until_ready())
    def gate_rows(rows):
        statuses = [r["status"] for r in rows]
        return "fail" if "fail" in statuses else "pass" if all(s == "pass" for s in statuses) else "inconclusive"
    try:
        # Validate observer against the real frozen API and full-horizon control.
        if not (out / "observer-parity.json").exists():
            checks = []
            for method in METHODS:
                status("observer_parity", method=method)
                a = sample(method, truth, 712, 9)
                b = sample(method, truth, 712, 9, observed=False)
                c = sample(method, truth, 712, 9, early=False)
                for name in ("rt", "cat", "work"): np.testing.assert_array_equal(a[name], b[name])
                for name in ("rt", "cat", "steps", "positions", "crossed"): np.testing.assert_array_equal(a[name], c[name])
                checks.append(dict(method=method, status="pass", trials=18))
            write(out / "observer-parity.json", checks); write(public / "observer-parity.json", checks)
        state["gates"]["observer"] = "pass"
        # Separate diagnostic sample; never reuse it in the final comparison.
        if not (out / "pilot-summary.json").exists():
            pilot_rows, plans = [], []
            for case in cases:
                all_samples = {m: cached_sample("pilot", case, m, 0, 4096) for m in METHODS}
                for ci in range(2):
                    a = condition(all_samples["fft199"], ci)
                    for method in METHODS[1:]:
                        b = condition(all_samples[method], ci)
                        rows = metrics.compare(a, b, masks(case, ci), alpha_endpoint, geometry=geometry, spatial_alpha=alpha_spatial)
                        pilot_rows.append(dict(case=case["name"], condition=ci, method=method, endpoints=rows))
                        if method == "fft200":
                            plans.append(dict(case=case["name"], condition=ci,
                                              **metrics.precision_requirement(a, b, masks(case, ci), alpha_endpoint, spatial_alpha=alpha_spatial, geometry=geometry)))
            required = max(p["required_trials"] for p in plans)
            unresolved = [dict(case=p["case"], condition=p["condition"], metrics=p["unresolved"]) for p in plans if p["unresolved"]]
            n_per_seed = max(9000, int(np.ceil(required/20/8))*8)
            precision = dict(required_trials_per_condition=required, trials_per_seed=n_per_seed, seeds=20,
                             feasible=not unresolved and n_per_seed <= 45000,
                             unresolved=unresolved, plans=plans,
                             note="No re-estimation from qualifying outcomes. The fixed 8-hour qualification cap can still leave a feasible design incomplete.")
            write(out / "pilot-summary.json", dict(rows=pilot_rows, precision=precision))
            write(public / "pilot-summary.json", dict(rows=pilot_rows, precision=precision))
        pilot = json.loads((out / "pilot-summary.json").read_text())
        precision = pilot["precision"]
        pilot_table = []
        for row in pilot["rows"]:
            endpoint = next(e for e in row["endpoints"] if e["metric"] == "low_support_0.1")
            lo, hi = endpoint["interval"]
            pilot_table.append(f"| {row['case']} | {row['condition']+1} | {row['method']} | {100*endpoint['difference']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
        pilot_text = f"""# Behavioral pilot

4,096 trials per condition, case and method. This separate pilot is not reused
for qualification. Differences below are candidate minus original FFT, in
percentage points. Intervals include the predeclared family adjustment.
For zero drift, the low-support mask is the frozen ordinary-drift geographic
mask; it is not interpreted as a zero-drift support classification.

Precision plan feasible: **{precision['feasible']}**.
Requested sample size: **{precision['required_trials_per_condition']:,} trials per condition**;
rounded design: 20 seeds x {precision['trials_per_seed']:,}.
Resource ceiling: 20 x 45,000 and eight hours of qualifying simulation.
Unresolved sparse-tail case/condition entries: {len(precision['unresolved'])}.
The precision calculation ignores observed differences and targets precision at
equality. See `pilot-summary.json` for each limiting endpoint, conditional RT
interval, spatial bound and diagnostic. No inconclusive endpoint is a pass.

| Case | Condition | Method | Low-support difference (pp) | Simultaneous interval (pp) |
| --- | --- | --- | --- | --- |
""" + "\n".join(pilot_table) + "\n"
        (public / "pilot.md").write_text(pilot_text)
        state["gates"]["precision_plan"] = "feasible" if precision["feasible"] else "inconclusive"
        # The canonical family contains all methods, but only candidate/reference
        # need expensive qualification. Unqualified KL contrasts never pass.
        if precision["feasible"] and not (out / "behavior-summary.json").exists():
            t0 = time.perf_counter(); complete = True
            for case in cases:
                for index in range(20):
                    for method in ("fft199", "fft200"):
                        if time.perf_counter()-t0 > protocol["behavior_seconds_cap"]:
                            complete = False; break
                        cached_sample("qualification", case, method, index, precision["trials_per_seed"])
                    if not complete: break
                if not complete: break
            if complete:
                results = []
                for case in cases:
                    batches = {m: [dict(np.load(out / "qualification" / case["name"] / m / f"batch-{i:02d}.npz")) for i in range(20)] for m in ("fft199", "fft200")}
                    for ci in range(2):
                        samples = {m: {k: np.concatenate([s[k][ci] for s in batches[m]]) for k in ("rt", "cat", "positions", "steps", "crossed")} for m in batches}
                        distances = [metrics.spatial_distance(condition(a, ci), condition(b, ci), geometry) for a, b in zip(batches["fft199"], batches["fft200"])]
                        rows = metrics.compare(samples["fft199"], samples["fft200"], masks(case, ci), alpha_endpoint,
                                               spatial_batches=distances, geometry=geometry, spatial_alpha=alpha_spatial)
                        results.append(dict(case=case["name"], condition=ci, spatial_batch_distances=distances, endpoints=rows))
                summary = dict(status=gate_rows([r for x in results for r in x["endpoints"]]), results=results,
                               note="Conservative spatial bound conditional on crossing counts. KL pilot contrasts are not qualified.")
            else:
                summary = dict(status="inconclusive", reason="Eight-hour behavioral budget exhausted; partial samples retained.")
            write(out / "behavior-summary.json", summary); write(public / "behavior-summary.json", summary)
        elif not precision["feasible"]:
            summary = dict(status="inconclusive", reason="Pilot cannot support the locked precision design within 20 x 45,000 trials per condition, or conditional tails are too sparse.")
            write(out / "behavior-summary.json", summary); write(public / "behavior-summary.json", summary)
        behavior = json.loads((out / "behavior-summary.json").read_text())
        state["gates"]["behavior"] = behavior["status"]
        # Timings are independent useful evidence even when fidelity is unresolved.
        data_path = BASE / "archived-case/data-baseline.npz"
        assert digest(data_path) == baseline_protocol["data_sha256"]
        data = {k: jnp.asarray(v) for k, v in dict(np.load(data_path)).items()}
        replay_dir = out / "replays"; replay_dir.mkdir(exist_ok=True)
        replay_started = time.perf_counter()
        for method in METHODS:
            for warmup in range(3):
                status("replay_warmup", method=method, warmup=warmup)
                before = time.perf_counter(); objective_call(method, vectors[0], data)
                write(replay_dir / f"warmup-{method}-{warmup}.json", dict(seconds=time.perf_counter()-before))
        capped = False
        for repeat in range(-1, 30):
            for method in (METHODS if repeat % 2 == 0 else METHODS[::-1]):
                destination = replay_dir / f"{repeat:02d}-{method}.json"
                if destination.exists(): continue
                if time.perf_counter()-replay_started > protocol["replay_seconds_cap"]:
                    capped = True; break
                grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()
                times, losses = [], []
                before = time.perf_counter()
                for i, p in enumerate(vectors):
                    status("replay", method=method, repetition=repeat, vector=i+1, total_vectors=30)
                    t = time.perf_counter(); value = objective_call(method, p, data)
                    assert np.isfinite(value)
                    times.append(time.perf_counter()-t); losses.append(value)
                write(destination, dict(method=method, repeat=repeat, seconds=sum(times), wall_seconds=time.perf_counter()-before,
                                         evaluation_seconds=times, losses=losses))
            if capped: break
            interim = [json.loads(p.read_text()) for p in replay_dir.glob("*.json") if not p.name.startswith("warmup")]
            write(public / "replay-progress.json", metrics.replay_summary(interim))
        records = [json.loads(p.read_text()) for p in replay_dir.glob("*.json") if not p.name.startswith("warmup")]
        replay = metrics.replay_summary(records)
        replay["controls"] = {m: metrics.replay_summary(records, control=m) for m in ("repaired_kl", "bucketed_kl")}
        replay["capped"] = capped
        write(out / "replay-summary.json", replay); write(public / "replay-summary.json", replay)
        state["gates"]["replay_speed"] = replay["status"]
        replay_rows = []
        for m in METHODS:
            durations = [r['seconds'] for r in records if r['method'] == m and r['repeat'] >= 0]
            if durations:
                replay_rows.append(f"| {m} | {len(durations)} | {np.median(durations):.3f} |")
        (public / "replay.md").write_text("# Thirty-vector trace replay\n\n"
            "One replicate is an entire immutable 30-vector trace. Host basis caches\n"
            "are empty at its start; normal caching is retained inside each replay.\n"
            "Methods alternate order; first-use calls and the first replay are excluded\n"
            "from steady timing and saved separately. Timed totals sum synchronized\n"
            "objective calls; status-file overhead is excluded and wall totals are saved.\n\n"
            "| Method | Complete timed replays | Median seconds per replay |\n| --- | --- | --- |\n"
            + "\n".join(replay_rows) + f"\n\nFFT200 speed gate: **{replay['status']}**. See `replay-summary.json` for paired whole-replay bootstrap intervals.\n")
        # Fits are deliberately gated. A negative KL control doesn't disqualify
        # an otherwise equivalent FFT200, and refitting cannot cure fidelity bias.
        if behavior["status"] != "pass" or replay["status"] != "pass":
            state["gates"]["matched_fits"] = "deferred: fidelity or speed not established"
            state["status"] = "complete_with_inconclusive_or_failed_gate"
            status("finished"); return
        def cold_fit():
            jax.clear_caches()
            grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()
        run_fits(out, public, protocol, state, status, sample, objective_call, objective, jnp, np, minimize, geometry, metrics, write, cold_fit)
        state["status"] = "complete" if state["gates"].get("matched_fits", "").startswith("single-case complete") else "complete_with_inconclusive_or_failed_gate"
        status("finished")
    except Exception as error:
        state["status"] = "failed"
        status("error", error=repr(error))
        raise
    finally:
        grf.calc_LAM = original_lam
        stop.set()


def run_fits(out, public, protocol, state, status, sample, objective_call, objective, jnp, np, minimize, geometry, metrics, write, cold_fit):
    """Matched single-case fits with independent predictive calibration/scoring."""
    fit_dir = out / "fits"; fit_dir.mkdir(exist_ok=True)
    truth = np.array(protocol["truth"])
    data_path = fit_dir / "fft-generated-data.npz"
    if not data_path.exists():
        status("generating_fft_observations")
        values = sample("fft199", truth, protocol["data_seed"], 9000)
        np.savez_compressed(fit_dir / "fft-observation-trials.npz", **values)
        count = np.array([[(c == k).sum() for k in range(1, 6)] for c in values["cat"]])
        quant = np.array([[np.quantile(values["rt"][ci][values["cat"][ci] == k], metrics.QUANTILES) for k in range(1, 6)] for ci in range(2)]).transpose(0, 2, 1)
        np.savez_compressed(data_path, prop=count/9000, count=count, quant=quant)
    data_np = dict(np.load(data_path)); data = {k: jnp.asarray(v) for k, v in data_np.items()}
    reference_path = fit_dir / "prediction-reference.npz"
    if not reference_path.exists():
        values = sample("fft199", truth, 202610500, 180000)
        np.savez_compressed(reference_path, **values)
    reference = dict(np.load(reference_path))
    def predictive(sample_values):
        probability, rt_error, distances = [], [], []
        for ci in range(2):
            for k in range(1, 6):
                a, b = reference["cat"][ci] == k, sample_values["cat"][ci] == k
                probability.append(abs(a.mean()-b.mean()))
                ar, br = reference["rt"][ci][a & reference["crossed"][ci]], sample_values["rt"][ci][b & sample_values["crossed"][ci]]
                if min(len(ar), len(br)) < 10: return [np.inf]*3
                rt_error.extend(abs(np.quantile(ar, metrics.QUANTILES)-np.quantile(br, metrics.QUANTILES)))
            a = {k: v[ci] for k, v in reference.items() if k != "work"}
            b = {k: v[ci] for k, v in sample_values.items() if k != "work"}
            distances.append(metrics.spatial_distance(a, b, geometry))
            probability.append(abs(reference["crossed"][ci].mean()-sample_values["crossed"][ci].mean()))
        return [float(max(probability)), float(max(rt_error)), float(max(distances))]
    # Frozen before optimization, calibrated at the exact validation sample size.
    calibration_path = fit_dir / "prediction-calibration.json"
    if not calibration_path.exists():
        calibration = []
        for seed in protocol["calibration_seeds"]:
            status("predictive_calibration", seed=seed)
            calibration.append(predictive(sample("fft199", truth, seed, 9000)))
        if not np.isfinite(calibration).all():
            state["gates"]["matched_fits"] = "inconclusive: sparse predictive calibration"; return
        thresholds = np.max(calibration, axis=0).tolist()
        write(calibration_path, dict(thresholds=thresholds, samples=calibration,
              note="Exploratory finite calibration: maxima of 20 truth replicates per metric; not population-wide predictive guarantees."))
    thresholds = np.array(json.loads(calibration_path.read_text())["thresholds"])
    outcomes = []
    scoring_seconds = 0.
    for method in METHODS:
        destination = fit_dir / f"{method}.json"
        trace_path = fit_dir / f"{method}-trace.jsonl"
        if not destination.exists():
            if trace_path.exists():
                raise RuntimeError(f"Interrupted fit trace retained at {trace_path}; explicit replacement run needed")
            cold_fit()
            status("fit_warmup", method=method)
            before = time.perf_counter(); objective_call(method, protocol["x0"], data)
            warmup_seconds = time.perf_counter()-before
            best, checkpoints = {"loss": float("inf"), "params": protocol["x0"]}, []
            started = time.perf_counter(); next_checkpoint = 300; evaluations = 0
            class FitCap(Exception): pass
            def loss(p):
                nonlocal next_checkpoint, evaluations
                if time.perf_counter()-started >= 8100: raise FitCap()
                value = objective_call(method, p, data); evaluations += 1
                effective = np.asarray(objective.clamp_b(jnp.asarray(p))).tolist()
                elapsed = time.perf_counter()-started
                if value < best["loss"]: best.update(loss=value, params=effective)
                row = dict(evaluation=evaluations, seconds=elapsed, loss=value, params=effective)
                with trace_path.open("a") as handle: handle.write(json.dumps(row)+"\n")
                if elapsed >= next_checkpoint:
                    checkpoints.append(dict(seconds=elapsed, **best)); next_checkpoint += 300
                    write(fit_dir / f"{method}-checkpoints.json", checkpoints)
                status("fit", method=method, evaluations=evaluations, training_seconds=elapsed, best_loss=best["loss"])
                return value
            try:
                result = minimize(loss, protocol["x0"], method="Nelder-Mead", options={"maxiter":1000, "xatol":1e-5, "fatol":1e-5})
                termination = dict(success=bool(result.success), message=str(result.message), capped=False)
            except FitCap:
                termination = dict(success=False, message="135-minute training cap", capped=True)
            elapsed = time.perf_counter()-started
            checkpoints.append(dict(seconds=elapsed, **best))
            write(destination, dict(method=method, training_seconds=elapsed, warmup_seconds=warmup_seconds,
                                   evaluations=evaluations, checkpoints=checkpoints, best=best, termination=termination))
        fit = json.loads(destination.read_text())
        scores = []
        for index, checkpoint in enumerate(fit["checkpoints"]):
            if scoring_seconds > protocol["scoring_seconds_cap"]:
                state["gates"]["matched_fits"] = "inconclusive: validation budget exhausted"; return
            path = fit_dir / f"{method}-checkpoint-{index:03d}-score.json"
            if path.exists():
                score = json.loads(path.read_text()); scores.append(score)
                scoring_seconds += score["verification_seconds"]
                continue
            p = np.array(checkpoint["params"])
            active = np.arange(13) != 6
            relative = np.abs(p[active]-truth[active])/np.abs(truth[active])
            recovery_pass = relative.mean() <= .05 and relative.max() <= .1
            # Evaluate every checkpoint, even when recovery failed; no cherry-picking.
            t = time.perf_counter(); predictions = []
            for seed in protocol["validation_seeds"]:
                status("independent_checkpoint_score", method=method, checkpoint=index, seed=seed)
                # The common full-spectrum scorer makes refitted KL predictions comparable.
                predictions.append(predictive(sample("fft199", p, seed, 9000)))
            predictions = np.array(predictions)
            prediction_pass = bool(np.isfinite(predictions).all() and np.all(np.quantile(predictions, .95, axis=0) <= thresholds))
            score = dict(checkpoint=index, training_seconds=checkpoint["seconds"], verification_seconds=time.perf_counter()-t,
                         mean_relative_error=float(relative.mean()), worst_relative_error=float(relative.max()),
                         signed_parameter_errors=(p-truth).tolist(), recovery_pass=bool(recovery_pass),
                         predictive_metrics=clean(predictions), prediction_pass=prediction_pass,
                         joint_pass=bool(recovery_pass and prediction_pass))
            write(path, score); scores.append(score)
            scoring_seconds += score["verification_seconds"]
        first = next((s for s in scores if s["joint_pass"]), None)
        outcomes.append(dict(method=method, fit=fit, scores=scores, first_acceptable_checkpoint=first,
                             time_to_quality=None if first is None else fit["warmup_seconds"]+first["training_seconds"]+sum(s["verification_seconds"] for s in scores if s["checkpoint"] <= first["checkpoint"])))
        write(public / "matched-fit-summary.json", outcomes)
    state["gates"]["matched_fits"] = "single-case complete; no across-case qualification"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".benchmark-results/2026-10-26-fft200-comparison")
    main(parser.parse_args())

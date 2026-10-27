"""Twenty independent behavioral batches with registered sequential looks."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/2026-10-26-behavioral-assessment-20-runs.json"
BASE = ROOT / ".benchmark-results/2026-10-21-rtx4060-baseline"
FROZEN = BASE / "source-322bd1f"


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read_batch(path, seed, n):
    """Only a data archive with its checksum commit record is reusable."""
    path = Path(path); meta = path.with_suffix(".json")
    if not meta.exists(): return None
    record = json.loads(meta.read_text())
    if record["seed"] != seed or record["n"] != n:
        raise ValueError("Saved batch belongs to a different seed or sample size")
    if not path.exists() or digest(path) != record["sha256"]:
        raise ValueError(f"Corrupt saved batch: {path}")
    return dict(np.load(path, allow_pickle=False))


def save_batch(path, values, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite a committed batch")
    if path.exists():
        # Preserve a data file left before the checksum commit on interruption.
        path.rename(path.with_name(path.stem+f".orphan-{time.time_ns()}.npz"))
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **values); temp.replace(path)
    write(path.with_suffix(".json"), dict(metadata, sha256=digest(path)))


def completed_rounds(out):
    markers = sorted(int(p.stem.split("-")[1]) for p in (Path(out)/"rounds").glob("round-*.json"))
    if markers != list(range(1, len(markers)+1)):
        raise ValueError("Completed-round markers contain a gap")
    return len(markers)


def main(args):
    protocol = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    assert not protocol["run_fits"] and not protocol["reuse_old_trials"]
    assert protocol["core_registered_endpoints"] == 3*3*2*33
    assert abs(sum(protocol["look_weights"])-1) < 1e-12
    out, public = ROOT / protocol["planned_raw_directory"], ROOT / protocol["planned_report_directory"]
    out.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    existing = (out/"manifest.json").exists()
    if existing and not args.resume:
        raise RuntimeError("Study already exists; use --resume explicitly to reuse verified batches")
    lock = out/"process.lock"
    if lock.exists():
        pid = int(json.loads(lock.read_text())["pid"])
        try: os.kill(pid, 0)
        except ProcessLookupError:
            if not args.resume: raise RuntimeError("Stale process lock; explicit resume required")
            lock.unlink()
        else: raise RuntimeError(f"Study is already running as PID {pid}")
    with lock.open("x") as handle: json.dump(dict(pid=os.getpid()), handle)
    try:
        run(protocol, out, public, args.resume)
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid(): lock.unlink()


def run(protocol, out, public, resume):
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(FROZEN))
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "default")
    import jax.numpy as jnp
    import scipy
    from model_b import grf, grf_kl, simulate, objective
    from shared import prng
    assert jax.default_backend() == "gpu" and "4060" in jax.devices()[0].device_kind
    active = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    others = [s for s in active.splitlines() if s.strip() and not s.startswith(str(os.getpid())+",")]
    if others: raise RuntimeError(f"GPU is occupied: {others}")
    baseline_manifest = json.loads((BASE/"source-manifest.json").read_text())
    for relative, sha in baseline_manifest["files"].items():
        if digest(FROZEN/relative.replace("\\", "/")) != sha: raise ValueError(f"Changed frozen source: {relative}")
    sources = [Path(__file__), ROOT/"scripts/behavior_assessment_stats.py", ROOT/"scripts/behavior_assessment_plots.py",
               ROOT/"scripts/fft200_comparison_metrics.py", ROOT/"scripts/model_b_baseline_observer.py",
               ROOT/"scripts/start_behavior_assessment.ps1", ROOT/"model_b/grf.py", PLAN,
               PLAN.with_suffix(".md")]
    snapshots = out/"run-tools"; snapshots.mkdir(exist_ok=True)
    hashes = {}
    for source in sources:
        target = snapshots/source.name
        if target.exists() and digest(target) != digest(source):
            raise ValueError(f"Study source changed: {source}; do not silently resume with changed code")
        if not target.exists(): shutil.copyfile(source, target)
        hashes[source.name] = digest(target)
    previous = json.loads((ROOT/"docs/results/2026-10-25-fft200-screen/screen.json").read_text())
    assert previous["status"] == "complete" and hashes["grf.py"] == previous["candidate_sha256"]
    manifest = dict(protocol=protocol, source_hashes=hashes, frozen_manifest_sha256=digest(BASE/"source-manifest.json"),
                    python=sys.version, jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__,
                    device=jax.devices()[0].device_kind, source_commit=baseline_manifest["commit"])
    if (out/"manifest.json").exists() and json.loads((out/"manifest.json").read_text()) != clean(manifest):
        raise ValueError("Manifest changed; a new study is required")
    write(out/"manifest.json", manifest); write(public/"manifest.json", manifest)
    stats = load_module(snapshots/"behavior_assessment_stats.py", "behavior_stats_frozen")
    plots = load_module(snapshots/"behavior_assessment_plots.py", "behavior_plots_frozen")
    plt = plots.initialize(ROOT/".benchmark-results/kl-report-deps")
    observer = load_module(snapshots/"model_b_baseline_observer.py", "behavior_observer_frozen").load_observer(FROZEN)
    candidate = load_module(snapshots/"grf.py", "behavior_fft200_frozen")
    old_lam = grf.calc_LAM
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    methods, cases = protocol["methods"], protocol["cases"]
    geometry = stats.base.projection_geometry()
    n = protocol["trials_per_condition_per_round"]
    state_path = out/"status.json"
    prior_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if prior_state.get("status") in ("completed", "early_stop_decisive"):
        print("Study already completed; no work restarted.", flush=True); return
    elapsed_before = prior_state.get("wall_seconds", 0.)
    began = time.perf_counter()
    state = dict(status="running", phase="preflight", pid=os.getpid(), device=manifest["device"],
                 completed_rounds=completed_rounds(out), maximum_rounds=20, stopped_candidates={}, last_look=None)
    for path in sorted((out/"looks").glob("look-*.json")):
        look = json.loads(path.read_text())
        state["last_look"] = look["round"]
        for method, decision in look["decisions"].items():
            if decision["stop"]:
                state["stopped_candidates"].setdefault(method, dict(round=look["round"], decision=decision))
    stop_telemetry = threading.Event()
    def render_status():
        latest = f"[Latest behavioral results](look-{state['last_look']:02d}/README.md)" if state["last_look"] else "No scheduled inferential look yet. First look is after five complete independent rounds."
        text = f"""# Twenty-run behavioral assessment

Status: **{state['status']}**. Phase: **{state['phase']}**.
Completed independent rounds: **{state['completed_rounds']} / 20**.
Updated: {state['updated_utc']}. Device: {state['device']}.

{latest}

Each round generates 9,000 fresh trials per condition, case and active method.
Compare original FFT199, FFT200, repaired KL and bucketed KL at identical
parameters. Three ordinary sig settings form the registered core; fourteen
stress cases remain separately labeled diagnostics. Every trial's RT, category,
crossing flag, step, location and coordinates is saved. No fitting is running.

Decisions occur only after rounds 5,10,15,20 with the predeclared endpoint and
look adjustment. A candidate stops early only when all six core cells have
decisive outcomes. Rare responses remain inconclusive. Reaching round 20 does
not automatically establish behavioral equivalence.

State detail: `{json.dumps(state.get('detail', {}))}`.
Stopped candidate records: `{json.dumps(state['stopped_candidates'])}`.

Raw batches, checksum records, telemetry, geometry and frozen sources are in
`.benchmark-results/2026-10-26-behavioral-assessment-20-runs/`.
The reference is the original FFT in the frozen JAX port; this does not newly
validate the entire port against Ratcliff's Fortran implementation or paper.
"""
        (public/"README.md").write_text(text, encoding="utf-8")
    def status(phase, **detail):
        state.update(phase=phase, detail=detail, wall_seconds=elapsed_before+time.perf_counter()-began,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state); write(public/"status.json", state); render_status()
        print(json.dumps(dict(phase=phase, completed_rounds=state["completed_rounds"], **detail)), flush=True)
    class BudgetExceeded(Exception): pass
    def check_budget():
        if elapsed_before+time.perf_counter()-began >= protocol["sampling_cap_seconds"]:
            raise BudgetExceeded("Six-hour cumulative wall budget reached")
    def telemetry():
        with (out/"gpu-telemetry.csv").open("a", buffering=1) as handle:
            while not stop_telemetry.is_set():
                r = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if r.returncode == 0: handle.write(r.stdout)
                stop_telemetry.wait(5)
    def select(method):
        grf.calc_LAM = candidate.calc_LAM_even if method == "fft200" else old_lam
        return dict(use_kl=True, stable_kl_noise=True, kl_bucketed=method == "bucketed_kl") if method.endswith("kl") else dict(use_kl=False)
    def sample(method, p, seed, count, observed=True, early=True):
        options = select(method); module = observer if observed else simulate
        arrays = []
        for ci, indices in enumerate(objective.COND_MAP_B):
            arrays.append(module.simulate_b_blocked_with_work(prng.split_for_condition(jax.random.key(seed), ci),
                *p[:4], *np.asarray(p)[list(indices)], p[4], float(p[5]), 6., count, protocol["chunk"],
                time_block_size=protocol["block"], stop_early=early, compact=True, **options))
        jax.block_until_ready(arrays)
        return arrays
    def unpack(arrays):
        return {name: np.stack([np.asarray(a[i]) for a in arrays]) for i, name in enumerate(("rt", "cat", "work", "steps", "positions", "crossed"))}
    def batch_path(r, ci, mi):
        return out/"trials"/f"round-{r+1:02d}"/cases[ci]["name"]/f"{methods[mi]}.npz"
    def batch(r, ci, mi):
        path = batch_path(r, ci, mi); seed = stats.seed_for(protocol, r, mi, ci)
        values = read_batch(path, seed, n)
        if values is not None:
            stats.validate_sample(values, n); return
        check_budget()
        status("sampling", round=r+1, case=cases[ci]["name"], method=methods[mi], trials_per_condition=n)
        before = time.perf_counter()
        arrays = sample(methods[mi], np.asarray(cases[ci]["params"]), seed, n)
        simulation_seconds = time.perf_counter()-before
        values = unpack(arrays); stats.validate_sample(values, n)
        xy = 2*np.stack([values["positions"] % 160, values["positions"] // 160], axis=-1).astype(np.float32)
        values["xy_pixels"] = np.where(values["crossed"][..., None], xy, np.nan)
        values["trial_index"] = np.arange(n, dtype=np.int32)
        save_batch(path, values, dict(seed=seed, n=n, round=r+1, method=methods[mi], case=cases[ci]["name"],
                   simulation_seconds=simulation_seconds, generated_trial_steps=values["work"].tolist(),
                   timing_note="Observed simulator with GPU synchronization; host transfer and file writes excluded; not the uninstrumented speed benchmark."))
    masks = {}; geometry_save = {}
    for ci, case in enumerate(cases):
        p = np.asarray(case["mask_params"]); bumps = simulate.drift_bumps(p[4])
        for condition, indices in enumerate(objective.COND_MAP_B):
            drift = np.asarray(sum(p[i]*b for i, b in zip(indices, bumps))).ravel()
            masks[ci, condition] = [drift <= cutoff*drift.max() for cutoff in stats.base.CUTOFFS]
            geometry_save[f"{ci}_{condition}_mask_drift"] = drift
            geometry_save[f"{ci}_{condition}_masks"] = np.asarray(masks[ci, condition])
    geometry_save["zones"] = np.asarray(simulate.zone_array(6.))
    angles = np.arange(64)*np.pi/64
    geometry_save["directions"] = np.column_stack([np.cos(angles), np.sin(angles)])
    if not (out/"geometry.npz").exists(): np.savez_compressed(out/"geometry.npz", **geometry_save)
    def pooled(rounds, ci, mi):
        pieces = [read_batch(batch_path(r, ci, mi), stats.seed_for(protocol, r, mi, ci), n) for r in range(rounds)]
        if any(v is None for v in pieces): raise ValueError("Cannot analyze an incomplete round prefix")
        return {k: np.concatenate([v[k] for v in pieces], axis=1) for k in ("rt", "cat", "steps", "positions", "crossed")}
    def analyze(rounds):
        check_budget()
        alpha = stats.look_alpha(protocol, rounds)
        look_dir = public/f"look-{rounds:02d}"; look_dir.mkdir(exist_ok=True)
        records, decisions = [], {}
        for mi, method in enumerate(methods[1:], 1):
            if method in state["stopped_candidates"]:
                stopped_at = state["stopped_candidates"][method]["round"]
                old = json.loads((out/"looks"/f"look-{stopped_at:02d}.json").read_text())
                records.extend(r for r in old["records"] if r["method"] == method)
                decisions[method] = old["decisions"][method]
                continue
            current = []
            for ci, case in enumerate(cases):
                check_budget(); status("analyzing", look=rounds, method=method, case=case["name"])
                reference, cand = pooled(rounds, ci, 0), pooled(rounds, ci, mi)
                core = case["name"] in protocol["core_case_names"]
                for condition in range(2):
                    a, b = stats.condition(reference, condition), stats.condition(cand, condition)
                    endpoints = stats.compare(a, b, masks[ci, condition], alpha if core else .05, geometry, core=core)
                    row = dict(method=method, case=case["name"], condition=condition, core=core,
                               rounds=rounds, trials=n*rounds, endpoints=endpoints,
                               status=stats.cell_status(endpoints) if core else "diagnostic_only")
                    current.append(row)
                    if core:
                        plots.case_figures(plt, a, b, look_dir/"figures"/f"{case['name']}-C{condition+1}-{method}.png",
                                           f"{method} vs FFT199 / {case['name']} / condition {condition+1} / {rounds} independent rounds")
            decisions[method] = stats.stop_decision(current, protocol["core_case_names"])
            records.extend(current)
        result = dict(round=rounds, endpoint_alpha=alpha, family_alpha=protocol["family_alpha"],
                      records=records, decisions=decisions, interval_note="Only core registered endpoints support sequential decisions; stress and mask sensitivities are diagnostic.")
        write(look_dir/"results.json", result)
        with (look_dir/"endpoints.csv").open("w", newline="") as handle:
            columns = ["method", "case", "condition", "rounds", "trials", "metric", "reference", "candidate", "difference", "lower", "upper", "margin", "status", "interval_scope", "reference_count", "candidate_count"]
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
            for r in records:
                for e in r["endpoints"]:
                    row = {k:r[k] for k in columns[:5]}
                    row.update({k:e.get(k) for k in ("metric", "reference", "candidate", "difference", "margin", "status", "interval_scope")})
                    row.update(lower=e["interval"][0], upper=e["interval"][1], reference_count=e.get("counts", [None, None])[0], candidate_count=e.get("counts", [None, None])[1])
                    writer.writerow(clean(row))
        plots.comparison_panels(plt, [r for r in records if r["core"]], look_dir/"core", f"Core comparison after look {rounds}: adjusted intervals; RT panel shows point differences")
        plots.comparison_panels(plt, [r for r in records if not r["core"]], look_dir/"stress", f"Stress diagnostics after look {rounds}: pointwise intervals, not qualification")
        table = []
        for r in records:
            if not r["core"]: continue
            e = next(x for x in r["endpoints"] if x["metric"] == "low_support_0.1")
            lo, hi = e["interval"]
            table.append(f"| {r['method']} | {r['case']} | {r['condition']+1} | {r['rounds']} | {100*e['difference']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] | {r['status']} |")
        (look_dir/"README.md").write_text(f"""# Behavioral look after {rounds} independent runs

Up to {n*rounds:,} trials per condition and method in each case. Earlier-stopped
candidates retain their frozen reference prefix and original look intervals;
sample counts are explicit. Core intervals adjust for all registered endpoints
and all four scheduled looks. Stress panels are separately labeled diagnostics.
No parameter recovery or fitting is part of this study.

| Method | Case | Condition | Runs | Low-support difference and interval (pp) | Core cell status |
| --- | --- | --- | --- | --- | --- |
"""+"\n".join(table)+"\n\n![Core effects](core/effects.png)\n\n![Stress diagnostics](stress/effects.png)\n\n"
            "All endpoint values, conditional sample counts and intervals are in [endpoints.csv](endpoints.csv). "
            "Maps, region probabilities, conditional RT quantiles and RT CDFs are in `figures/`. "
            "Sparse/unbounded intervals remain inconclusive. Maps and point-estimate comparisons alone are not tests of superiority.\n", encoding="utf-8")
        history = [json.loads(p.read_text()) for p in sorted((out/"looks").glob("look-*.json"))]+[clean(result)]
        plots.precision_history(plt, history, protocol["core_case_names"], look_dir/"precision.png")
        with (look_dir/"README.md").open("a", encoding="utf-8") as handle:
            handle.write("\n![Low-support precision across independent runs](precision.png)\n")
        # Only commit the look after its tables/figures are complete. On an
        # interrupted report, resume reuses samples and regenerates the same look.
        write(out/"looks"/f"look-{rounds:02d}.json", result)
        for method, decision in decisions.items():
            if decision["stop"]:
                state["stopped_candidates"].setdefault(method, dict(round=rounds, decision=decision))
        state["last_look"] = rounds
        write(public/"latest-results.json", result)
        status("look_complete", look=rounds, decisions=decisions)
    def interrupt(_signal, _frame): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    threading.Thread(target=telemetry, daemon=True).start()
    try:
        status("preflight")
        if not (out/"observer-parity.json").exists():
            checks = []
            ordinary = np.asarray(next(c["params"] for c in cases if c["name"] == "sig10_ordinary"))
            for method in methods:
                status("observer_parity", method=method)
                a = unpack(sample(method, ordinary, 202610277, 9))
                plain = sample(method, ordinary, 202610277, 9, observed=False)
                control = unpack(sample(method, ordinary, 202610277, 9, early=False))
                for i, name in enumerate(("rt", "cat", "work")):
                    np.testing.assert_array_equal(a[name], np.stack([np.asarray(v[i]) for v in plain]))
                for name in ("rt", "cat", "steps", "positions", "crossed"):
                    np.testing.assert_array_equal(a[name], control[name])
                stats.validate_sample(a, 9)
                checks.append(dict(method=method, status="pass", trials=18))
            write(out/"observer-parity.json", checks); write(public/"observer-parity.json", checks)
        if not (out/"kl-diagnostics.json").exists():
            grf.calc_LAM = old_lam
            diagnostics = []
            for sig in sorted({float(c["params"][5]) for c in cases}):
                for method in ("repaired_kl", "bucketed_kl"):
                    basis = grf_kl.calc_kl_basis(sig, pad_to_k_max=True, return_mode_ids=True,
                            mode_buckets=grf_kl.DEFAULT_MODE_BUCKETS if method == "bucketed_kl" else None)
                    diagnostics.append(dict(method=method, sig=sig, modes=int(basis[1]), retained_variance=float(basis[2]), basis_shape=list(basis[0].shape)))
            write(out/"kl-diagnostics.json", diagnostics); write(public/"kl-diagnostics.json", diagnostics)
        if state["completed_rounds"] in protocol["look_after_rounds"] and state["last_look"] != state["completed_rounds"]:
            analyze(state["completed_rounds"])
        for r in range(state["completed_rounds"], protocol["max_independent_rounds"]):
            active_methods = [m for m in methods if m == "fft199" or m not in state["stopped_candidates"]]
            if len(active_methods) == 1: break
            for ci in range(len(cases)):
                # Rotate ordering so every method shares warm/cold positions.
                ordering = methods[r % len(methods):]+methods[:r % len(methods)]
                for method in ordering:
                    if method in active_methods: batch(r, ci, methods.index(method))
            write(out/"rounds"/f"round-{r+1:02d}.json", dict(round=r+1, methods=active_methods, cases=len(cases), trials_per_condition=n))
            state["completed_rounds"] = r+1
            status("round_complete", round=r+1)
            if r+1 in protocol["look_after_rounds"]: analyze(r+1)
        state["status"] = "early_stop_decisive" if state["completed_rounds"] < 20 else "completed"
        status("finished")
    except BudgetExceeded as error:
        state["status"] = "capped_incomplete"; status("finished", reason=str(error))
    except KeyboardInterrupt:
        state["status"] = "interrupted"; status("stopped", reason="Interrupted; completed batches and looks preserved")
        raise
    except Exception as error:
        state["status"] = "failed"; status("error", error=repr(error)); raise
    finally:
        stop_telemetry.set(); grf.calc_LAM = old_lam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())

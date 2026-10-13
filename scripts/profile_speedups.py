"""Profile real DDM Python drivers without changing JAX execution semantics.

Each model/variant runs in a fresh process. Compilation/first calls and warm-up
are recorded before independent cProfile and line_profiler passes. Output is
diagnostic: profilers add overhead and cannot resolve individual GPU kernels.
"""
import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import linecache
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ("legacy", "blocked_full", "early_stop", "staged", "combined")
TRUE = {
    "a": [200., 50., 50., 10., 4., 5., .7, 20., 10., 60.],
    "b": [200., 50., 10., 2., 12., 10., .5, 15., 10., 8., 14., 11., 9.],
}
LIMITATIONS = [
    "cProfile and line_profiler measure the Python host; they do not time individual XLA/GPU kernels.",
    "Every workload waits for JAX results. A wait can absorb earlier asynchronous work, so its line is not a kernel location.",
    "JIT bodies execute as compiled code after tracing; zero line hits there do not mean zero compute cost.",
    "Unwrapping JIT functions only registers source mappings; this script never invokes their eager versions.",
    "Compilation/first calls are separate from steady-state passes. New shapes or basis operations during a fit can still compile.",
    "Instrumented durations include profiler overhead and are not speedup evidence; use benchmark_speedups.py for comparisons.",
    "Basis cold means an empty Python K-L basis cache, not an empty XLA or operating-system cache.",
    "Cumulative function times overlap and must not be added together.",
    "Fits use synthetic data, fixed seeds and a small iteration budget; these are bottleneck diagnostics, not convergence/recovery tests.",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def timed(call):
    start = time.perf_counter()
    value = call()
    return time.perf_counter() - start, value


def source_functions(modules):
    """Register actual module functions without invoking eager JIT bodies."""
    functions, seen = [], set()

    def visit(function):
        function = inspect.unwrap(function)
        if not isinstance(function, types.FunctionType) or function.__code__ in seen:
            return
        seen.add(function.__code__)
        functions.append(function)

    for module in modules:
        for _, function in inspect.getmembers(module):
            if callable(function) and getattr(function, "__module__", None) == module.__name__:
                visit(function)
    return functions


def relative_source(filename):
    try:
        return str(Path(filename).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return filename


def cprofile_rows(stats):
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"file": relative_source(filename), "line": line,
                     "function": name, "primitive_calls": primitive,
                     "calls": calls, "self_seconds": own,
                     "cumulative_seconds": cumulative})
    return sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)


def line_rows(profiler):
    stats = profiler.get_stats()
    rows = []
    for (filename, first_line, name), timings in stats.timings.items():
        for line, hits, ticks in timings:
            rows.append({"file": relative_source(filename), "function": name,
                         "function_line": first_line, "line": line,
                         "hits": hits, "seconds": ticks * stats.unit,
                         "source": linecache.getline(filename, line).strip()})
    return sorted(rows, key=lambda row: row["seconds"], reverse=True)


def profile_workload(name, call, functions, out, repeats, prepare=None):
    """Separate uninstrumented, function and line passes of the same work."""
    from line_profiler import LineProfiler

    def batch():
        result = None
        for _ in range(repeats):
            result = call()
        return result

    row = {"phase": name, "calls_per_pass": repeats}
    if prepare:
        prepare()
    row["uninstrumented_seconds"], row["result"] = timed(batch)
    if prepare:
        prepare()
    cprof = cProfile.Profile()
    row["cprofile_seconds"], row["cprofile_result"] = timed(lambda: cprof.runcall(batch))
    cprof.dump_stats(str(out / f"{name}.pstats"))
    with (out / f"{name}-cprofile.txt").open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(cprof, stream=handle)
        stats.sort_stats("cumulative").print_stats(60)
        stats.sort_stats("tottime").print_stats(40)
    row["top_cumulative_functions"] = cprofile_rows(stats)[:40]
    row["top_self_functions"] = sorted(cprofile_rows(stats), key=lambda item: item["self_seconds"], reverse=True)[:40]
    row["project_functions"] = [item for item in cprofile_rows(stats)
                                if item["file"].startswith(("model_a/", "model_b/", "shared/", "scripts/"))][:40]
    if prepare:
        prepare()
    line_prof = LineProfiler()
    for function in functions:
        line_prof.add_function(function)
    line_prof.enable_by_count()
    try:
        row["line_profiler_seconds"], row["line_profiler_result"] = timed(batch)
    finally:
        line_prof.disable_by_count()
    line_prof.dump_stats(str(out / f"{name}.lprof"))
    with (out / f"{name}-lines.txt").open("w", encoding="utf-8") as handle:
        line_prof.print_stats(stream=handle, output_unit=.001, stripzeros=True)
    row["top_python_lines"] = line_rows(line_prof)[:40]
    row["identical_results_across_passes"] = row["result"] == row["cprofile_result"] == row["line_profiler_result"]
    return row


class Workloads:
    def __init__(self, args):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from shared import prng
        from shared.staged_fit import TrialStage

        self.args, self.jax, self.jnp, self.np, self.prng = args, jax, jnp, np, prng
        self.params = jnp.asarray(TRUE[args.model])
        self.key = jax.random.key(args.seed)
        self.x0 = self.params * jnp.asarray(np.random.default_rng(args.seed).uniform(.9, 1.1, len(self.params)))
        self.chunk = args.chunk_a if args.model == "a" else args.chunk_b
        self.cfg = {"time_block_size": 0} if args.variant == "legacy" else {
            "time_block_size": args.block_size,
            "stop_early": args.variant in ("early_stop", "combined"), "compact": True}
        self.schedule = None
        if args.variant in ("staged", "combined") and args.fit_iters:
            quarter = max(1, args.fit_iters // 4)
            self.schedule = [TrialStage(args.nsim // 4, quarter),
                             TrialStage(args.nsim // 2, quarter),
                             TrialStage(args.nsim, args.fit_iters - 2 * quarter)]
        if args.model == "a":
            from model_a import fit, objective, simulate
            self.modules = [fit, objective, simulate]
            self.objective, self.fitter = objective.fofs_new, fit.fit_simplex
            self.clamp, self.conditions, self.categories = objective.clamp, objective.COND_MAP, 3
        else:
            from model_b import fit, objective, simulate, grf, grf_kl
            self.modules = [fit, objective, simulate, grf, grf_kl]
            self.objective, self.fitter = objective.fofs_b_new, fit.fit_simplex_b
            self.clamp, self.conditions, self.categories = objective.clamp_b, objective.COND_MAP_B, 5
        self.simulator = simulate
        self.data = None

    def clear_basis_cache(self):
        if self.args.model == "b":
            from model_b import grf_kl
            grf_kl._basis_cache.clear()
            grf_kl._basis_cache_order.clear()

    def build_basis(self, cold=False):
        if cold:
            self.clear_basis_cache()
        if self.args.model == "a":
            basis = self.simulator.chol_factor(self.params[5])
            self.jax.block_until_ready(basis)
            return {"shape": list(basis.shape), "dtype": str(basis.dtype)}
        from model_b import grf, grf_kl
        if self.args.noise == "kl":
            basis, modes, variance = grf_kl.calc_kl_basis(float(self.params[5]), pad_to_k_max=True)
            self.jax.block_until_ready(basis)
            return {"shape": list(basis.shape), "dtype": str(basis.dtype),
                    "complex_modes": modes, "variance_captured": variance}
        basis = grf.calc_LAM(s1=float(self.params[5]), s2=float(self.params[5]))
        self.jax.block_until_ready(basis)
        return {"shape": list(basis.shape), "dtype": str(basis.dtype)}

    def simulate_arrays(self, nsim=None, legacy=False):
        nsim = self.args.nsim if nsim is None else nsim
        p, sim = self.clamp(self.params), self.simulator
        cfg = {"time_block_size": 0} if legacy else self.cfg
        rt_list, cat_list = [], []
        for ci, condition in enumerate(self.conditions):
            key = self.prng.split_for_condition(self.key, ci)
            if self.args.model == "a":
                di, bi = condition
                values = (key, p[0], p[1], p[bi], p[3], p[4], p[5], p[di], nsim, self.chunk)
                rt, cat = (sim.simulate_blocked(*values, **cfg) if cfg["time_block_size"] else sim.simulate(*values))
            else:
                d1, d2, d3 = condition
                values = (key, p[0], p[1], p[2], p[3], p[d1], p[d2], p[d3], p[4], float(p[5]), 6., nsim, self.chunk, self.args.noise == "kl")
                rt, cat = (sim.simulate_b_blocked(*values, **cfg) if cfg["time_block_size"] else sim.simulate_b(*values))
            rt_list.append(rt)
            cat_list.append(cat)
        rt, cat = self.jnp.stack(rt_list), self.jnp.stack(cat_list)
        self.jax.block_until_ready((rt, cat))
        return self.np.asarray(rt), self.np.asarray(cat)

    def run_simulation(self):
        rt, cat = self.simulate_arrays()
        return {"trials": int(rt.size), "mean_rt": float(rt.mean()),
                "category_counts": self.np.bincount(cat.ravel(), minlength=self.categories + 1)[1:].tolist()}

    def make_observed_data(self):
        rt, cat = self.simulate_arrays(nsim=self.args.data_nsim, legacy=True)
        np = self.np
        props, counts, quantiles = [], [], []
        for rts, cats in zip(rt, cat):
            props.append([(cats == c).mean() for c in range(1, self.categories + 1)])
            counts.append([(cats == c).sum() for c in range(1, self.categories + 1)])
            q = np.zeros((5, self.categories))
            for c in range(1, self.categories + 1):
                selected = rts[cats == c]
                if len(selected) >= 5:
                    q[:, c - 1] = np.quantile(selected, [.1, .3, .5, .7, .9])
            quantiles.append(q)
        self.data = {k: self.jnp.asarray(v) for k, v in
                     {"prop": props, "count": counts, "quant": quantiles}.items()}
        self.jax.block_until_ready(self.data)
        return {"data_trials_per_condition": self.args.data_nsim}

    def run_objective(self, nsim=None, params=None):
        options = {"chunk_size": self.chunk, **self.cfg}
        if self.args.model == "b":
            options["use_kl"] = self.args.noise == "kl"
        loss = self.objective(self.params if params is None else params, self.data,
                              self.key, nsim=self.args.nsim if nsim is None else nsim, **options)
        self.jax.block_until_ready(loss)
        return {"loss": float(loss)}

    def prepare_fit(self):
        self.clear_basis_cache()
        budgets = [s.nsim for s in self.schedule] if self.schedule else [self.args.nsim]
        for nsim in budgets:
            self.run_objective(nsim=nsim, params=self.x0)

    def run_fit(self):
        options = {"chunk_size": self.chunk, **self.cfg}
        if self.args.model == "b":
            options["use_kl"] = self.args.noise == "kl"
        result = self.fitter(self.data, self.key, self.x0, nsim=self.args.nsim,
                             maxiter=self.args.fit_iters, tol=1e-5,
                             trial_schedule=self.schedule, **options)
        self.jax.block_until_ready(result.params)
        return {"loss": result.loss, "n_iters": result.n_iters,
                "n_evals": result.n_evals, "converged": result.converged,
                "stages": result.stages, "params": self.np.asarray(result.params).tolist()}


def metadata(jax, functions):
    packages = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "line_profiler", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources, mappings = {}, []
    for function in functions:
        filename = function.__code__.co_filename
        relative = relative_source(filename)
        mappings.append({"file": relative, "function": function.__qualname__,
                         "line": function.__code__.co_firstlineno})
        if Path(filename).is_file():
            sources[relative] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    def command(args):
        try:
            result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            return None
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "commit": command(["git", "rev-parse", "HEAD"]), "packages": packages,
            "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "device_kind": jax.devices()[0].device_kind, "x64": jax.config.jax_enable_x64,
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "source_sha256": sources, "registered_functions": mappings,
            "allocator": {name: os.environ.get(name) for name in
                          ("JAX_PLATFORM_NAME", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")}}


def worker(args):
    import jax
    from shared import blocked, staged_fit

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    if not args.allow_cpu and (jax.default_backend() != "gpu" or "H100" not in jax.devices()[0].device_kind):
        raise RuntimeError(f"Expected H100 GPU; got {jax.devices()}. CPU smoke needs --allow-cpu.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    work = Workloads(args)
    functions = source_functions(work.modules + [blocked, staged_fit, sys.modules[__name__]])
    # Class methods are real driver/synchronization lines as well.
    functions.extend(value for _, value in inspect.getmembers(Workloads, inspect.isfunction))
    report = {"model": args.model, "variant": args.variant, "noise": args.noise,
              "settings": vars(args), "limitations": LIMITATIONS,
              "metadata": metadata(jax, functions), "setup": {}, "phases": []}
    setup = report["setup"]
    for name, call in (("basis_first_call", work.build_basis),
                       ("synthetic_data_generation", work.make_observed_data),
                       ("simulation_first_call", work.run_simulation),
                       ("objective_first_call", work.run_objective)):
        seconds, value = timed(call)
        setup[name] = {"seconds": seconds, "result": value}
        write_json(out / "profile.json", report)
    seconds, _ = timed(lambda: (work.run_simulation(), work.run_objective()))
    setup["unprofiled_warmup_seconds"] = seconds
    setup["first_call_note"] = "Includes execution and any compilation/cache preparation remaining at this point; not a compiler-only timer."
    phases = [("simulation", work.run_simulation, args.repeats, None),
              ("objective", work.run_objective, args.repeats, None)]
    if args.model == "b" and args.noise == "kl":
        phases.extend([("basis_cold", lambda: work.build_basis(cold=True), 1, None),
                       ("basis_cache_hit", work.build_basis, args.repeats, None)])
    else:
        phases.append(("basis_factor", work.build_basis, args.repeats, None))
    if args.fit_iters:
        phases.append(("fit", work.run_fit, 1, work.prepare_fit))
    for name, call, repeats, prepare in phases:
        print(f"Profiling {args.model}/{args.variant}/{name}", flush=True)
        report["phases"].append(profile_workload(name, call, functions, out, repeats, prepare))
        write_json(out / "profile.json", report)
    report["status"] = "completed"
    write_json(out / "profile.json", report)
    return report


def write_report(out, cases):
    text = ["# Python profiling diagnostics", "", *[f"- {note}" for note in LIMITATIONS], ""]
    for case in cases:
        text.extend([f"## {case['model'].upper()} / {case['variant']}", ""])
        path = out / f"{case['model']}-{case['variant']}" / "profile.json"
        if not path.exists():
            text.extend([f"No profile checkpoint. Status: {case['status']}.", ""])
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        text.extend([f"Backend: {row['metadata']['backend']}; device: {row['metadata']['device_kind']}. Status: {case['status']}.", "",
                     "| Workload | Calls per pass | No profiler (s) | cProfile (s) | line_profiler (s) |",
                     "|---|---:|---:|---:|---:|"])
        for phase in row["phases"]:
            text.append(f"| {phase['phase']} | {phase['calls_per_pass']} | {phase['uninstrumented_seconds']:.4f} | {phase['cprofile_seconds']:.4f} | {phase['line_profiler_seconds']:.4f} |")
        text.append("")
        for phase in row["phases"]:
            text.extend([f"### {phase['phase']}: highest cProfile self times", "",
                         "| Function | Self seconds | Cumulative seconds | Calls |",
                         "|---|---:|---:|---:|"])
            for item in phase["top_self_functions"][:5]:
                text.append(f"| {item['file']}:{item['line']} ({item['function']}) | {item['self_seconds']:.4f} | {item['cumulative_seconds']:.4f} | {item['calls']} |")
            text.append("")
            text.append(f"Identical workload summaries across the three passes: {phase['identical_results_across_passes']}.")
            text.append("")
            text.extend([f"### {phase['phase']}: highest recorded Python line times", "",
                         "| Source | Seconds | Hits | Code |", "|---|---:|---:|---|"])
            lines = [item for item in phase["top_python_lines"]
                     if item["file"].startswith(("model_a/", "model_b/", "shared/"))]
            for item in (lines or phase["top_python_lines"])[:5]:
                source = item["source"].replace("|", "\\|")
                text.append(f"| {item['file']}:{item['line']} | {item['seconds']:.4f} | {item['hits']} | `{source}` |")
            text.append("")
    (out / "PROFILE_REPORT.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="b,a", help="Ordered comma-separated a,b cases")
    parser.add_argument("--variants", default="legacy,combined")
    parser.add_argument("--model", choices=("a", "b"), default="b")
    parser.add_argument("--variant", choices=VARIANTS, default="legacy")
    parser.add_argument("--noise", choices=("kl", "fft"), default="kl")
    parser.add_argument("--output", default=".benchmark-results/profiling")
    parser.add_argument("--nsim", type=int, default=128)
    parser.add_argument("--data-nsim", type=int, default=256)
    parser.add_argument("--chunk-a", type=int, default=32)
    parser.add_argument("--chunk-b", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fit-iters", type=int, default=0, help="0 skips fits; positive uses real scipy Nelder-Mead")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--matmul-precision", choices=("default", "high", "highest"), default="default")
    parser.add_argument("--budget-seconds", type=int, default=250)
    parser.add_argument("--worker-timeout", type=int, default=110)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.nsim, args.data_nsim, args.chunk_a, args.chunk_b, args.block_size, args.repeats,
           args.budget_seconds, args.worker_timeout) < 1 or args.fit_iters < 0:
        parser.error("Trial/chunk/time/repeat values must be positive; fit-iters must be nonnegative")
    if args.fit_iters and any(v in ("staged", "combined") for v in args.variants.split(",")) and (args.nsim < 4 or args.fit_iters < 3):
        parser.error("Staged/combined fits require nsim >= 4 and fit-iters >= 3")
    if args.worker:
        worker(args)
        return
    models, variants = args.models.split(","), args.variants.split(",")
    if any(m not in ("a", "b") for m in models) or any(v not in VARIANTS for v in variants):
        parser.error("Unknown model/variant")
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output folder already contains files; choose a new folder to preserve profile provenance")
    out.mkdir(parents=True, exist_ok=True)
    deadline, cases = time.monotonic() + args.budget_seconds, []
    for model in models:
        for variant in variants:
            remaining = deadline - time.monotonic()
            if remaining < 10:
                break
            folder = out / f"{model}-{variant}"
            folder.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", model,
                       "--variant", variant, "--variants", args.variants, "--output", str(folder)]
            for name in ("nsim", "data_nsim", "chunk_a", "chunk_b", "block_size", "repeats", "fit_iters",
                         "seed", "noise", "matmul_precision"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            if args.allow_cpu:
                command.append("--allow-cpu")
            case = {"model": model, "variant": variant}
            start = time.perf_counter()
            with (folder / "worker.log").open("w", encoding="utf-8") as handle:
                try:
                    result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                            timeout=min(args.worker_timeout, remaining - 2))
                    case["status"] = "completed" if result.returncode == 0 else "failed"
                    case["exit_code"] = result.returncode
                except subprocess.TimeoutExpired:
                    case["status"] = "timeout"
            case["elapsed_seconds"] = time.perf_counter() - start
            cases.append(case)
            write_json(out / "cases.json", cases)
            write_report(out, cases)
            print(json.dumps(case), flush=True)
    if not cases or not any(case["status"] == "completed" for case in cases):
        raise SystemExit("No profiling case completed; inspect worker.log/checkpoints")


if __name__ == "__main__":
    main()

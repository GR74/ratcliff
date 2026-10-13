"""Reconstruct a finished optimizer from its losses, then finish held-out scoring.

Use the exact NumPy/SciPy versions from the experiment. This keeps timeout
records intact, validates replay against completed cases and NEVER invents an
exact fitting duration: the last callback is only a lower bound.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    from scripts import benchmark_model_b_extensions as extensions

    source = Path(args.source).resolve()
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError("choose a fresh recovery output file")
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    settings = json.loads((source / "run-config.json").read_text(encoding="utf-8"))
    control = next(r for r in rows if r["experiment"] == "control" and r["action"] == "fit"
                   and r["seed"] == args.seed and r["status"] == "completed")
    packages = control["metadata"]["packages"]
    if packages["numpy"] != np.__version__ or packages["scipy"] != scipy.__version__:
        raise RuntimeError("replay requires the experiment's exact NumPy/SciPy versions")
    active = np.asarray([i for i in range(13) if i != 6])
    x0 = np.asarray(control["x0"])

    def replay(label):
        path = source / label / f"fit-progress-b-early_stop-{args.seed}.jsonl"
        trace = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if [r["eval"] for r in trace] != list(range(1, len(trace) + 1)):
            raise ValueError("trace evaluations must be consecutive")
        cursor = 0

        def oracle(params):
            nonlocal cursor
            if cursor >= len(trace):
                raise ValueError("optimizer requested an evaluation absent from the trace")
            loss = trace[cursor]["loss"]
            cursor += 1
            return loss

        fixed_sv = "--optimize-active-only" in extensions.FEATURES[label]
        res = minimize(oracle, x0[active] if fixed_sv else x0, method="Nelder-Mead",
                       options={"maxiter": settings["fit_iters"], "xatol": 1e-5,
                                "fatol": 1e-5, "disp": False})
        if cursor != len(trace):
            raise ValueError("trace contains unused evaluations; cannot establish exact replay")
        params = x0.copy() if fixed_sv else res.x.copy()
        if fixed_sv:
            params[active] = res.x
        return params, res, trace, path

    validated = []
    for row in rows:
        if row["action"] == "fit" and row["seed"] == args.seed and row["status"] == "completed":
            params, res, _, _ = replay(row["experiment"])
            if not np.array_equal(params, np.asarray(row["params"])) or res.nfev != row["n_evals"]:
                raise ValueError("completed case replay did not match exactly")
            validated.append(row["experiment"])
    params, res, trace, path = replay(args.variant)
    result = {"experiment": args.variant, "seed": args.seed,
              "fit_completed_from_trace": True, "validated_completed_cases": validated,
              "params": params.tolist(), "x0": x0.tolist(), "n_iters": int(res.nit),
              "n_evals": int(res.nfev), "converged": bool(res.success),
              "fit_seconds_lower_bound": trace[-1]["elapsed"],
              "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "data_sha256": hashlib.sha256((source / "data-b.npz").read_bytes()).hexdigest(),
              "timing_note": "Original worker timed out; exact duration is unavailable.",
              "packages": {"numpy": np.__version__, "scipy": scipy.__version__}}
    if args.score:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "default")
        from model_b.objective import fofs_b_new
        from scripts import benchmark_speedups as bench
        if jax.default_backend() != "gpu" or "H100" not in jax.devices()[0].device_kind:
            raise RuntimeError("held-out recovery scoring requires the experiment H100")
        with np.load(source / "data-b.npz") as a:
            data = {k: jnp.asarray(v) for k, v in a.items()}
        result["scores"] = [{"seed": seed, "g2": float(fofs_b_new(
            jnp.asarray(params), data, jax.random.key(seed), nsim=settings["nsim"],
            chunk_size=settings["chunk_b"], use_kl=True))} for seed in (999, 5001, 5002, 5003, 5004, 5005)]
        true = np.asarray(bench.TRUE["b"])
        result["mean_relative_parameter_error"] = float(np.mean(np.abs(params[active] - true[active]) / true[active]))
        result["metadata"] = bench.metadata(jax)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Recovered", args.variant, args.seed, "with", len(validated), "exact replay validations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="active_only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", action="store_true")
    main(parser.parse_args())

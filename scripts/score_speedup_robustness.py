"""Repeat held-out G2 scoring of saved fits; never change fitted parameters.

This diagnostic retains the original single-seed screening result and reports
additional independent random seeds separately. It is not human-data validation
or a significance test. Run only while other GPU workers are stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_speedups as bench


def main(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "default")
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError("choose a fresh output file")
    if min(args.nsim, args.chunk_a, args.chunk_b) < 1 or args.budget_seconds <= 20:
        raise ValueError("positive trials and a budget greater than 20 seconds required")
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("scoring seeds must be nonempty and unique")
    if set(seeds) & {123, 321, 999}:
        raise ValueError("additional scoring seeds must differ from original experiment seeds")
    if not args.allow_cpu and (jax.default_backend() != "gpu"
                              or "H100" not in jax.devices()[0].device_kind):
        raise RuntimeError("this GPU experiment requires an H100; --allow-cpu is smoke only")
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    variants = set(args.variants.split(","))
    fits = [r for r in rows if r["status"] == "completed" and r["action"] == "fit"
            and r["variant"] in variants]
    deadline = time.monotonic() + args.budget_seconds
    result = {"purpose": __doc__, "settings": vars(args), "cases": [],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fit_results_sha256": hashlib.sha256((source / "results.json").read_bytes()).hexdigest(),
              "source_data_sha256": {},
              "planned_scores": len(fits) * len(seeds), "completed_scores": 0,
              "actual_scorer_calls": 0, "cpu_smoke_only": args.allow_cpu}
    cache = {}
    for model in ("a", "b"):
        if model == "a":
            from model_a.objective import fofs_new as evaluate
            options = {"chunk_size": args.chunk_a}
        else:
            from model_b.objective import fofs_b_new as evaluate
            options = {"chunk_size": args.chunk_b, "use_kl": True}
        if not any(r["model"] == model for r in fits):
            continue
        with np.load(source / f"data-{model}.npz") as archive:
            data = {k: jnp.asarray(v) for k, v in archive.items()}
        result["source_data_sha256"][model] = hashlib.sha256(
            (source / f"data-{model}.npz").read_bytes()).hexdigest()
        for fitted in (r for r in fits if r["model"] == model):
            case = {"model": model, "variant": fitted["variant"], "start_seed": fitted["seed"],
                    "params": fitted["params"], "original_score_seed": fitted["score_seed"],
                    "original_score_nsim": fitted["score_nsim"],
                    "original_g2": fitted["independent_full_budget_loss"], "scores": []}
            result["cases"].append(case)
            for seed in seeds:
                if time.monotonic() >= deadline - 20:
                    break
                started = time.monotonic()
                identity = (model, tuple(fitted["params"]), seed, args.nsim)
                if identity in cache:
                    saved = cache[identity]
                    score = {"seed": seed, "g2": saved["g2"],
                             "seconds_including_first_compilation": None,
                             "reused_from": saved["case"]}
                else:
                    loss = float(evaluate(jnp.asarray(fitted["params"]), data,
                                          jax.random.key(seed), nsim=args.nsim,
                                          time_block_size=0, **options))
                    score = {"seed": seed, "g2": loss,
                             "seconds_including_first_compilation": time.monotonic() - started}
                    cache[identity] = {"g2": loss, "case": [model, fitted["variant"], fitted["seed"]]}
                    result["actual_scorer_calls"] += 1
                case["scores"].append(score)
                result["completed_scores"] += 1
                bench.write_json(output, result)
            print("SCORED", model, fitted["variant"], fitted["seed"],
                  len(case["scores"]), "seeds", flush=True)
            if time.monotonic() >= deadline - 20:
                break
        if time.monotonic() >= deadline - 20:
            break
    result["metadata"] = bench.metadata(jax)
    result["metadata"]["x64"] = bool(jax.config.jax_enable_x64)
    result["stopped_by_budget"] = result["completed_scores"] < result["planned_scores"]
    bench.write_json(output, result)
    print("Saved", output, result["completed_scores"], "/", result["planned_scores"], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nsim", type=int, default=9000)
    parser.add_argument("--seeds", default="5001,5002,5003,5004,5005")
    parser.add_argument("--variants", default="legacy,early_stop,combined")
    parser.add_argument("--chunk-a", type=int, default=256)
    parser.add_argument("--chunk-b", type=int, default=64)
    parser.add_argument("--budget-seconds", type=int, default=300)
    parser.add_argument("--allow-cpu", action="store_true")
    main(parser.parse_args())

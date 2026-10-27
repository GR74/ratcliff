"""Bounded opt-in guarded-KL integration and complete-objective timing screen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model_b import grf_kl, objective, simulate


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_files = [__file__, "model_b/grf_kl.py", "model_b/simulate.py",
                    "model_b/objective.py", "model_b/fit.py"]
    record = {"status": "running", "devices": [d.device_kind for d in jax.devices()],
              "jax": jax.__version__, "numpy": np.__version__, "nsim_per_condition": args.nsim,
              "chunk": 4, "block": 16, "seed": 20261026, "precision": str(jax.config.jax_default_matmul_precision),
              "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
              "policy_checks": [], "timings": [], "summary": [],
              "data_note": "Fixed synthetic scoring bins for timing only; not a recovery or empirical-data fit."}
    def save():
        record["elapsed_seconds"] = time.perf_counter()-started
        (out / "screen.json").write_text(json.dumps(record, indent=2)+"\n")
    def within_budget():
        if time.perf_counter()-started > args.seconds:
            record["status"] = "capped"
            save()
            return False
        return True
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    key = jax.random.key(20261026)
    params = jnp.array([200., 50., 10., 2., 12., 10., .5, 15., 10., 8., 14., 11., 9.])
    data = {"prop": jnp.full((2, 5), .2), "count": jnp.full((2, 5), 100),
            "quant": jnp.broadcast_to(jnp.array([250., 300., 350., 400., 450.])[None, :, None], (2, 5, 5))}
    policies = {"fft": dict(use_kl=False), "legacy_kl": dict(use_kl=True),
                "guarded_kl": dict(use_kl=True, guarded_kl=True)}
    for sig in [.2, 5., 8., 10., 12., 15., 17.]:
        record["policy_checks"].append(grf_kl.guarded_kl_diagnostics(sig))
    common = dict(ter=200., st=50., cr=10., crsd=2., av1=15., av2=10., av3=8.,
                  sis=12., si=6., nsim=3, chunk_size=1)
    for blocked in [False, True]:
        fn = simulate.simulate_b_blocked if blocked else simulate.simulate_b
        options = dict(time_block_size=16) if blocked else {}
        a = fn(key, **common, sig=5., use_kl=False, **options)
        b = fn(key, **common, sig=5., use_kl=True, guarded_kl=True, **options)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        record[f"fallback_exact_{'blocked' if blocked else 'full'}"] = True
    # Verify unblocked objective vmap policy propagation as well as blocked timings.
    full_control = objective.fofs_b_new(params.at[5].set(5.), data, key, nsim=2, chunk_size=1)
    full_guard = objective.fofs_b_new(params.at[5].set(5.), data, key, nsim=2, chunk_size=1,
                                     use_kl=True, guarded_kl=True)
    np.testing.assert_array_equal(full_control, full_guard)
    record["full_objective_fallback_exact"] = True
    save()
    print("Fallback integrations passed", flush=True)
    for sig in [5., 10., 15.]:
        p = params.at[5].set(sig)
        def evaluate(name):
            return objective.fofs_b_new(p, data, key, nsim=args.nsim, chunk_size=4,
                                        time_block_size=16, **policies[name])
        for name in policies:
            if not within_budget():
                return
            start = time.perf_counter()
            result = float(evaluate(name).block_until_ready())
            record["timings"].append(dict(sig=sig, method=name, phase="first_call", seconds=time.perf_counter()-start, loss=result))
            for _ in range(3):
                evaluate(name).block_until_ready()
        for repetition in range(args.repetitions):
            for name in (list(policies) if repetition % 2 == 0 else list(policies)[::-1]):
                if not within_budget():
                    return
                # Warm compiled executables, but host basis/policy cache misses each call.
                grf_kl._basis_cache.clear()
                grf_kl._basis_cache_order.clear()
                grf_kl._guarded_selection.cache_clear()
                start = time.perf_counter()
                value = float(evaluate(name).block_until_ready())
                record["timings"].append(dict(sig=sig, method=name, phase="cache_miss", repetition=repetition,
                                               seconds=time.perf_counter()-start, loss=value))
            save()
        for name in policies:
            durations = [r["seconds"] for r in record["timings"] if r["sig"] == sig and r["method"] == name and r["phase"] == "cache_miss"]
            record["summary"].append(dict(sig=sig, method=name, median_seconds=float(np.median(durations))))
        save()
        print(json.dumps(record["summary"][-3:]), flush=True)
    record["status"] = "complete"
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/results/2026-10-25-guarded-kl-screen")
    parser.add_argument("--nsim", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=600)
    main(parser.parse_args())

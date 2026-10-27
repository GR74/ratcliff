"""Twenty-minute initial FFT200 correctness/speed screen on the frozen harness.

Only the FFT covariance builder is injected into repaired source 322bd1f.
All arms use its unchanged blocking, PRNG, decision and objective code.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".benchmark-results/2026-10-21-rtx4060-baseline"
FROZEN = BASE / "source-322bd1f"
sys.path.insert(0, str(FROZEN))

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "default")
import jax.numpy as jnp
import numpy as np
from model_b import grf, grf_kl, simulate, objective


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    original_lam = grf.calc_LAM
    candidate = load(ROOT / "model_b/grf.py", "fft200_candidate")
    manifest = json.loads((BASE / "source-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((FROZEN / relative.replace("\\", "/")).read_bytes()).hexdigest() == expected
    active = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    # JAX may already have initialized its context; only this process is allowed.
    others = [line for line in active.splitlines() if line.strip() and not line.startswith(str(os.getpid()) + ",")]
    if others:
        raise RuntimeError("GPU is occupied: " + str(others))
    started = time.perf_counter()
    record = {"status": "running", "utc": datetime.now(timezone.utc).isoformat(),
              "devices": [d.device_kind for d in jax.devices()], "jax": jax.__version__,
              "numpy": np.__version__, "budget_seconds": args.seconds, "seed": 20261029,
              "chunk": 8, "block": 32, "precision": "default", "covariance": [],
              "fields": [], "timings": [], "integration": {}, "source_commit": "322bd1f",
              "candidate_sha256": hashlib.sha256((ROOT / "model_b/grf.py").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "candidate_grf.py").write_bytes((ROOT / "model_b/grf.py").read_bytes())
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    (out / "frozen-source-manifest.json").write_text(json.dumps(manifest, indent=2))
    def save(stage):
        record["stage"] = stage
        record["elapsed_seconds"] = time.perf_counter()-started
        temporary = out / "screen.tmp"
        temporary.write_text(json.dumps(record, indent=2)+"\n")
        temporary.replace(out / "screen.json")
        print(stage, round(record["elapsed_seconds"], 1), flush=True)
    def deadline():
        if time.perf_counter()-started > args.seconds:
            raise TimeoutError("Initial screen reached its soft runtime cap")
    protocol = json.loads((BASE / "archived-case/protocol.json").read_text())
    truth = np.asarray(protocol["truth"])
    data_path = BASE / "archived-case/data-baseline.npz"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == protocol["data_sha256"]
    record["data_sha256"] = protocol["data_sha256"]
    data = {k: jnp.asarray(v) for k, v in dict(np.load(data_path)).items()}
    trace_path = BASE / "archived-case/trace-baseline-1-early_stop.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()[:30]]
    record["trace_first30"] = trace
    record["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    named = [.2, 1., 3., 5., 8., 10., 12., 15., 17.]
    # Include archived trace sig values (full parameter vectors are preserved).
    for item in trace:
        for key in ["params", "x", "parameters"]:
            if isinstance(item.get(key), list) and len(item[key]) == 13:
                named.append(float(item[key][5]))
                break
    widths = sorted(set(np.linspace(.2, 17., 101).tolist() +
                        [s+d for s in named for d in [-1e-6, 0, 1e-6] if .2 <= s+d <= 17.]))
    dy, dx = np.arange(-99, 100)[:, None], np.arange(-159, 160)[None, :]
    try:
        for sig in widths:
            deadline()
            c = np.asarray(candidate.even_embedding(s1=sig, s2=sig))
            spectral = np.fft.fft2(c)/c.size
            target = (1-(dx*dx+dx*dy+dy*dy)/sig**2)*np.exp(-(dx*dx+dy*dy)/sig**2)
            new_lam = np.asarray(candidate.calc_LAM_even(s1=sig, s2=sig))
            old_lam = np.asarray(original_lam(s1=sig, s2=sig))
            old_c = np.fft.fft2(old_lam**2).real[dy % 199, dx % 319]
            new_c = np.fft.fft2(new_lam**2).real[dy % 200, dx % 320]
            new32 = np.fft.fft2(new_lam.astype(np.float32).astype(np.float64)**2).real[dy % 200, dx % 320]
            old32 = np.fft.fft2(old_lam.astype(np.float32).astype(np.float64)**2).real[dy % 199, dx % 319]
            row = dict(sig=sig, min_normalized_eigenvalue=float(spectral.real.min()),
                       max_spectral_imag=float(abs(spectral.imag).max()),
                       max_target_error=float(abs(new_c-target).max()),
                       old_target_error=float(abs(old_c-target).max()),
                       max_reference_error=float(abs(new_c-old_c).max()),
                       max_fp32_target_error=float(abs(new32-target).max()),
                       max_fp32_reference_error=float(abs(new32-old32).max()))
            row["demeaned_reference_error_upper_bound"] = 4*row["max_reference_error"]
            assert row["min_normalized_eigenvalue"] >= -100*np.finfo(float).eps*c.size*abs(spectral.real).max(), row
            assert row["max_target_error"] < 1e-10 and row["old_target_error"] < 1e-10, row
            assert row["demeaned_reference_error_upper_bound"] < 1e-10, row
            assert 4*max(row["max_fp32_target_error"], row["max_fp32_reference_error"]) < 2e-5, row
            assert row["max_spectral_imag"] < 1e-14, row
            record["covariance"].append(row)
        save("covariance_passed")

        # Exercise public current wrappers against the same frozen FFT core.
        grf.calc_LAM_even = candidate.calc_LAM_even
        current_sim = load(ROOT / "model_b/simulate.py", "current_sim_integration")
        common = dict(ter=200., st=50., cr=10., crsd=2., av1=15., av2=10., av3=8.,
                      sis=12., sig=10., si=6., nsim=3, chunk_size=1, use_kl=False)
        key = jax.random.key(20261029)
        for is_blocked in [False, True]:
            options = dict(time_block_size=32) if is_blocked else {}
            frozen_fn = simulate.simulate_b_blocked if is_blocked else simulate.simulate_b
            current_fn = current_sim.simulate_b_blocked if is_blocked else current_sim.simulate_b
            # Distinct compiled full-horizon function must trace the candidate once.
            grf.calc_LAM = candidate.calc_LAM_even
            expected = frozen_fn(key, **common, **options)
            actual = current_fn(key, **common, **options, fft200=True)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
            record["integration"]["blocked" if is_blocked else "full"] = True
        grf.calc_LAM = original_lam
        # Only blocked objectives are timed, so the full-horizon trace above cannot leak.
        save("integration_passed")

        @jax.jit
        def field_batch(k, lam):
            z = jax.random.normal(k, (32, 2, *lam.shape), dtype=jnp.float32)
            f = jnp.fft.fft2(lam*(z[:, 0]+1j*z[:, 1]))[:, :100, :160]
            fields = jnp.stack([f.real, f.imag], axis=1).reshape(64, 100, 160)
            anchor = fields[:, 50, 80]
            stats = jnp.stack([fields.var(axis=(-2,-1)), fields.max(axis=(-2,-1)),
                               anchor, fields[:, 50, 81], fields[:, 51, 80],
                               fields[:, 50, 85], fields[:, 55, 80]], axis=1)
            return stats
        field_arrays = {}
        for si, sig in enumerate([5., 10., 15.]):
            for mi, (name, builder) in enumerate([("fft199", original_lam), ("fft200", candidate.calc_LAM_even)]):
                lam = builder(s1=sig, s2=sig).astype(jnp.float32)
                pieces = []
                for batch in range(64):
                    deadline()
                    k = jax.random.fold_in(key, 100000+si*1000+mi*100+batch)
                    pieces.append(np.asarray(field_batch(k, lam)))
                a = np.concatenate(pieces)
                field_arrays[f"{sig}_{name}"] = a
                record["fields"].append(dict(sig=sig, method=name, count=len(a),
                    mean_spatial_variance=float(a[:, 0].mean()), maximum_p99=float(np.quantile(a[:, 1], .99)),
                    anchor_variance=float(a[:, 2].var()), covariance_lags=[float(np.cov(a[:, 2], a[:, j])[0,1]) for j in range(3,7)],
                    real_imag_anchor_correlation=float(np.corrcoef(a[::2, 2], a[1::2, 2])[0,1]),
                    successive_pair_anchor_correlation=float(np.corrcoef(a[:-2:2, 2], a[2::2, 2])[0,1])))
        np.savez_compressed(out / "field-statistics.npz", **field_arrays)
        save("field_checks_recorded")

        grf_kl._BASIS_CACHE_MAXSIZE = 2
        methods = {"fft199": dict(use_kl=False), "fft200": dict(use_kl=False),
                   "repaired_kl": dict(use_kl=True, stable_kl_noise=True),
                   "bucketed_kl": dict(use_kl=True, stable_kl_noise=True, kl_bucketed=True)}
        def call(name, nsim, p):
            grf.calc_LAM = candidate.calc_LAM_even if name == "fft200" else original_lam
            return objective.fofs_b_new(jnp.asarray(p), data, key, nsim=nsim,
                chunk_size=8, time_block_size=32, **methods[name])
        def timed(name, nsim, p, phase, repetition):
            deadline()
            before = time.perf_counter()
            value = float(call(name, nsim, p).block_until_ready())
            elapsed = time.perf_counter()-before
            assert np.isfinite(value)
            record["timings"].append(dict(method=name, nsim=nsim, sig=float(p[5]),
                phase=phase, repetition=repetition, seconds=elapsed, loss=value))
        # Initial cold/steady-state comparison at the production budget for KL controls.
        for name in ["repaired_kl", "bucketed_kl"]:
            timed(name, 9000, truth, "first_call", 0)
            for i in range(3):
                timed(name, 9000, truth, "warmup", i)
        for repetition in range(5):
            for name in (["repaired_kl", "bucketed_kl"] if repetition % 2 == 0 else ["bucketed_kl", "repaired_kl"]):
                timed(name, 9000, truth, "control_screen", repetition)
            save(f"kl_controls_{repetition+1}/5")
        # Paired FFT objective timings across trial budgets. All settings otherwise equal.
        for nsim in [128, 1024, 9000]:
            for name in ["fft199", "fft200"]:
                timed(name, nsim, truth, "first_call", 0)
                for i in range(3):
                    timed(name, nsim, truth, "warmup", i)
            for repetition in range(30):
                for name in (["fft199", "fft200"] if repetition % 2 == 0 else ["fft200", "fft199"]):
                    timed(name, nsim, truth, "steady", repetition)
                if repetition % 5 == 4:
                    save(f"fft_{nsim}_{repetition+1}/30")
        # Changing-width suite is separate from the cached truth measurements.
        for repetition in range(30):
            p = truth.copy()
            p[5] = [5., 8., 10., 12., 15., 17.][repetition % 6]
            for name in (["fft199", "fft200"] if repetition % 2 == 0 else ["fft200", "fft199"]):
                timed(name, 1024, p, "changing_sig", repetition)
            if repetition % 5 == 4:
                save(f"changing_sig_{repetition+1}/30")
        record["status"] = "complete"
        save("complete")
    except TimeoutError as error:
        record["status"] = "capped"
        record["failure"] = str(error)
        save("capped")
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = repr(error)
        save("failed")
        raise
    finally:
        grf.calc_LAM = original_lam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/results/2026-10-25-fft200-screen")
    parser.add_argument("--seconds", type=float, default=1200)
    main(parser.parse_args())

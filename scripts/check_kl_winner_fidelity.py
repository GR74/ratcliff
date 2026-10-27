"""Paired, full-grid Model B truncation study; no production code changes.

Run from the repository root with a JAX GPU Python environment. Masked FFT
sampling implements exactly the same Gaussian law as the real KL basis, while
sharing spectral draws with the full FFT control. Only real FFT outputs are
used so asymmetric cutoffs cannot correlate consecutive time increments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".55")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
from model_b import grf, grf_kl, simulate

SCENARIOS = {"baseline": 1.0, "weak_drift": 0.25, "zero_drift": 0.0}
VARIANTS = {"fft": (1.0, None), "kl95_cap2000": (.95, 2000),
            "kl99_cap2000": (.99, 2000), "kl999_uncapped": (.999, None)}
PARAMS = dict(ter=200., st=50., cr=10., crsd=2.,
              av1=15., av2=10., av3=8., sis=12., si=6.)


def spectrum(lam, threshold, cap):
    values = np.asarray(lam ** 2).ravel()
    order = np.argsort(-values)
    cumulative = np.cumsum(values[order]) / values.sum()
    count = len(values) if threshold == 1 else int(np.searchsorted(cumulative, threshold) + 1)
    if cap is not None:
        count = min(count, cap)
    mask = np.zeros(values.size, dtype=np.float32)
    mask[order[:count]] = 1
    return mask.reshape(lam.shape), count, float(cumulative[count - 1]), order[:count]


@jax.jit
def observe(fields, uniforms, drift, zones):
    """Production drift/demean/cumsum/crossing operations plus winner location."""
    increments = fields + drift[None, None]
    increments -= increments.mean(axis=(-2, -1), keepdims=True)
    paths = jnp.cumsum(increments, axis=1)
    thresholds = (PARAMS["cr"] + PARAMS["crsd"] * (uniforms[:, 4] - .5)).astype(jnp.float32)
    maxima = paths.max(axis=(-2, -1))
    crossed_steps = maxima > thresholds[:, None]
    crossed = crossed_steps.any(axis=1)
    steps = jnp.where(crossed, crossed_steps.argmax(axis=1) + 1, simulate.NSTEP)
    at_crossing = paths[jnp.arange(len(fields)), steps - 1].reshape(len(fields), -1)
    position = at_crossing.argmax(axis=1)
    category = jnp.where(crossed, zones.ravel()[position], 5)
    rt = steps * simulate.E + PARAMS["ter"] + PARAMS["st"] * (.5 - uniforms[:, 9])
    return jnp.stack([rt, category, position, crossed, steps], axis=1)


@jax.jit
def sample_fields(z, lam):
    return jnp.fft.fft2(lam * (z[:, :, 0] + 1j * z[:, :, 1]))[..., :100, :160].real


def validate_observer_and_basis(sig=10., variance_target=.99):
    """Check the observer against both actual production chunk implementations."""
    lam = grf.calc_LAM(s1=sig, s2=sig).astype(jnp.float32)
    bumps = simulate.drift_bumps(PARAMS["sis"])
    drift = sum(PARAMS[f"av{i+1}"] * b.astype(jnp.float32) for i, b in enumerate(bumps))
    zones = simulate.zone_array(PARAMS["si"])
    key = jax.random.key(923)
    ku, kg = jax.random.split(key)
    u = jax.random.uniform(ku, (2, 10))
    z = jax.random.normal(kg, (2, 200, 2, *lam.shape), dtype=jnp.float32)
    f = jnp.fft.fft2(lam * (z[:, :, 0] + 1j*z[:, :, 1]))[..., :100, :160]
    fields = jnp.stack([f.real, f.imag], axis=2).reshape(2, 400, 100, 160)
    args = [PARAMS[k] for k in ("ter", "st", "cr", "crsd", "av1", "av2", "av3")]
    expected = simulate._simulate_chunk_b(key, *args, lam, *bumps, zones, 2)
    actual = np.asarray(observe(fields, u, drift, zones))
    np.testing.assert_allclose(actual[:, 0], expected[0], atol=1e-8)
    np.testing.assert_array_equal(actual[:, 1], expected[1])
    basis, count, retained = grf_kl.calc_kl_basis(sig, variance_threshold=variance_target)
    zk = jax.random.normal(kg, (2, 400, 2*count), dtype=jnp.float32)
    fk = jnp.einsum("csk,Nk->csN", zk, basis).reshape(2, 400, 100, 160)
    expected_kl = simulate._simulate_chunk_b_kl(key, *args, basis, *bumps, zones, 2)
    actual_kl = np.asarray(observe(fk, u, drift, zones))
    np.testing.assert_allclose(actual_kl[:, 0], expected_kl[0], atol=1e-8)
    np.testing.assert_array_equal(actual_kl[:, 1], expected_kl[1])
    # Forward FFT uses exp(-i*phase): real((a+i*b)*exp(-i*phase))=a*cos+b*sin.
    exact_lam = grf.calc_LAM(s1=sig, s2=sig)
    _, _, _, indices = spectrum(exact_lam, variance_target, 2000)
    weights = jax.random.normal(jax.random.key(924), (8, 2*count), dtype=jnp.float32)
    spectral = jnp.zeros((8, lam.size), dtype=jnp.complex64)
    spectral = spectral.at[:, indices].set(weights[:, 0::2] + 1j*weights[:, 1::2])
    via_fft = jnp.fft.fft2(spectral.reshape(8, *lam.shape)*lam)[..., :100, :160].real
    via_basis = grf_kl.sample_kl_grf(basis, weights)
    error = float(jnp.max(jnp.abs(via_fft-via_basis)))
    assert error < 2e-5, error
    return {"sig": sig, "variance_target": variance_target, "observer_fft_and_kl": "passed", "basis_masked_fft_max_abs_error": error,
            "basis_modes": count, "retained_variance": retained}


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    checks = [validate_observer_and_basis(sig, .999 if args.guarded else .99) for sig in args.sig]
    grf_kl._basis_cache.clear()
    grf_kl._basis_cache_order.clear()
    jax.clear_caches()
    print(json.dumps({"validation": checks}), flush=True)
    bumps = simulate.drift_bumps(PARAMS["sis"])
    drift = sum(PARAMS[f"av{i+1}"] * b.astype(jnp.float32) for i, b in enumerate(bumps))
    zones = simulate.zone_array(PARAMS["si"])
    raw = {"baseline_drift": np.asarray(drift), "zones": np.asarray(zones)}
    configured_variants = dict(VARIANTS)
    if args.guarded:
        configured_variants["guarded_kl"] = (.999, 2000)
    metadata = {"utc": datetime.now(timezone.utc).isoformat(), "parameters": PARAMS,
                "scenarios": SCENARIOS, "variants": configured_variants, "trials_per_cell": args.trials,
                "seed": args.seed, "chunk": args.chunk, "steps": 400, "sig": args.sig,
                "jax": jax.__version__, "numpy": np.__version__, "python": platform.python_version(),
                "devices": [str(d) + " " + d.device_kind for d in jax.devices()],
                "validation": checks, "spectra": {}, "command": sys.argv,
                "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                  [__file__, "model_b/grf.py", "model_b/grf_kl.py", "model_b/simulate.py"]}}
    for si, sig in enumerate(args.sig):
        lam64 = grf.calc_LAM(s1=sig, s2=sig)
        variants = {}
        for name, (threshold, cap) in configured_variants.items():
            if name == "guarded_kl" and grf_kl.guarded_kl_diagnostics(sig)["generator"] == "fft":
                threshold, cap = 1., None
            mask, count, retained, _ = spectrum(lam64, threshold, cap)
            variants[name] = lam64.astype(jnp.float32)*jnp.asarray(mask)
            metadata["spectra"][f"{sig}/{name}"] = {"complex_modes": count, "variance_retained": retained}
        results = {(v, s): [] for v in variants for s in SCENARIOS}
        field_stats = {v: [] for v in variants}
        for start in range(0, args.trials, args.chunk):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.key(args.seed), si), start)
            ku, kg = jax.random.split(key)
            u = jax.random.uniform(ku, (args.chunk, 10))
            z = jax.random.normal(kg, (args.chunk, 400, 2, *lam64.shape), dtype=jnp.float32)
            for name, spectrum_lam in variants.items():
                fields = sample_fields(z, spectrum_lam)
                # One independent field per trial: avoid treating 400 fields as 400 trials.
                field = fields[:, 0] - fields[:, 0].mean(axis=(-2, -1), keepdims=True)
                field_stats[name].append(np.asarray(jnp.stack([field.var(axis=(-2, -1)),
                                          field.max(axis=(-2, -1))], axis=1)))
                for scenario, factor in SCENARIOS.items():
                    results[name, scenario].append(np.asarray(observe(fields, u, drift*factor, zones)))
            if start % 128 == 0:
                print(f"sig={sig} trials={start+args.chunk}/{args.trials} elapsed={time.time()-started:.1f}s", flush=True)
        for (name, scenario), arrays in results.items():
            raw[f"sig{sig}_{name}_{scenario}"] = np.concatenate(arrays)[:args.trials]
        for name, arrays in field_stats.items():
            raw[f"sig{sig}_{name}_fields"] = np.concatenate(arrays)[:args.trials]
        metadata["elapsed_seconds"] = time.time()-started
        np.savez_compressed(out / "trials.npz", **raw)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Completed: {out}, {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/results/2026-10-25-kl-winner-fidelity")
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--sig", type=float, nargs="+", default=[5., 10., 15.])
    parser.add_argument("--guarded", action="store_true", help="Include the production guarded-KL selection policy")
    main(parser.parse_args())

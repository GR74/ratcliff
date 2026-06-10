"""Track B SBI MVP — amortized neural posterior estimation for the 2D model.

Proves the whole amortized-inference thesis for ~$5 of GPU:
  1. Generate N simulated datasets from the 2D diffusion model (varying all 13
     params; FFT path so sig can vary freely).
  2. Train a neural posterior estimator (sbi / SNPE) on (theta, summary-stats).
  3. Validate: simulation-based calibration (SBC) + posterior recovery on
     held-out test data, and agreement with the known true params.

If SBC ranks are ~uniform and recovery is good, the approach works and is worth
scaling to the full training set (and to the K-L fast path / hierarchical fits).

Usage on a rented GPU (after setup — see docs):
    python scripts/sbi_mvp.py 2>&1 | tee /workspace/sbi_mvp_results.txt

Env knobs (optional):
    SBI_N=3000        number of training datasets
    SBI_NSIM=500      trials per dataset
    SBI_NTEST=200     held-out datasets for SBC + recovery
"""
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

print("=" * 70)
print("Track B SBI MVP — amortized posterior for the 2D diffusion model")
print("=" * 70)
print(f"JAX devices: {jax.devices()}  backend={jax.default_backend()}")

from model_b import simulate as sim_b
from model_b.objective import COND_MAP_B, clamp_b
from shared import prng

N_TRAIN = int(os.environ.get("SBI_N", "3000"))
NSIM = int(os.environ.get("SBI_NSIM", "500"))
N_TEST = int(os.environ.get("SBI_NTEST", "200"))
CHUNK = 64 if jax.default_backend() == "gpu" else 8

# 13-param vector: ter, st, cr, crsd, sis, sig, sv, av1c1, av2c1, av3c1, av1c2, av2c2, av3c2
PARAM_LOW = np.array([150., 10., 5., 0.5, 6., 4., 0.2, 5., 3., 3., 5., 3., 3.])
PARAM_HIGH = np.array([260., 90., 18., 5., 16., 16., 1.5, 22., 18., 16., 22., 18., 16.])


def summary_stats(rt, cat):
    """20-dim summary per dataset: per-condition handled by caller; here one
    condition -> 5 category proportions + 5 RT quantiles."""
    rt = np.asarray(rt); cat = np.asarray(cat)
    props = np.array([(cat == c).mean() for c in (1, 2, 3, 4, 5)])
    qs = np.quantile(rt, [0.1, 0.3, 0.5, 0.7, 0.9])
    return np.concatenate([props, qs])


def simulate_summary(theta, key_seed):
    """theta (13,) -> 20-dim summary stats over 2 conditions (10 each)."""
    p = clamp_b(jnp.asarray(theta))
    ter, st, cr, crsd, sis, sig = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
    si = 6.0
    feats = []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(jax.random.key(key_seed), ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, float(p[d1]), float(p[d2]), float(p[d3]),
            sis, sig, si, nsim=NSIM, chunk_size=CHUNK, use_kl=False,
        )
        feats.append(summary_stats(rt, cat))
    return np.concatenate(feats)  # 20-dim


def generate(n, seed0):
    rng = np.random.default_rng(seed0)
    thetas = rng.uniform(PARAM_LOW, PARAM_HIGH, size=(n, 13))
    xs = np.zeros((n, 20))
    t0 = time.perf_counter()
    for i in range(n):
        xs[i] = simulate_summary(thetas[i], seed0 + i)
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  generated {i+1}/{n}  ({el:.0f}s, {el/(i+1):.2f}s/dataset)", flush=True)
    return thetas.astype(np.float32), xs.astype(np.float32)


def main():
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    # NOTE: run_sbc/check_sbc live in sbi.diagnostics (moved from sbi.analysis in
    # sbi>=0.23). Imported lazily inside the SBC try-block below so any further API
    # drift there can't kill the run -- the recovery numbers are the core result.

    print(f"\nGenerating {N_TRAIN} training datasets (nsim={NSIM}, chunk={CHUNK}) ...")
    print("  (first call compiles the FFT path — 1-3 min — then it's fast)")
    theta_train, x_train = generate(N_TRAIN, seed0=0)

    print(f"\nGenerating {N_TEST} held-out test datasets ...")
    theta_test, x_test = generate(N_TEST, seed0=10_000)

    prior = BoxUniform(
        low=torch.tensor(PARAM_LOW, dtype=torch.float32),
        high=torch.tensor(PARAM_HIGH, dtype=torch.float32),
    )
    print("\nTraining SNPE neural posterior estimator ...")
    t0 = time.perf_counter()
    inference = SNPE(prior=prior)
    inference.append_simulations(
        torch.tensor(theta_train), torch.tensor(x_train)
    ).train()
    posterior = inference.build_posterior()
    print(f"  trained in {time.perf_counter()-t0:.0f}s")

    # --- Recovery on held-out test sets ---
    print("\nPosterior recovery on held-out test data:")
    errs = []
    for i in range(min(N_TEST, 50)):
        samples = posterior.sample((500,), x=torch.tensor(x_test[i]), show_progress_bars=False)
        post_mean = samples.mean(0).numpy()
        rel = np.abs(post_mean - theta_test[i]) / (np.abs(theta_test[i]) + 1e-6)
        errs.append(rel)
    errs = np.array(errs)
    labels = ["ter", "st", "cr", "crsd", "sis", "sig", "sv",
              "av1c1", "av2c1", "av3c1", "av1c2", "av2c2", "av3c2"]
    print(f"  mean relative error per param (over {len(errs)} test sets):")
    for j, lab in enumerate(labels):
        print(f"    {lab:7s} {errs[:, j].mean()*100:5.1f}%")
    print(f"  OVERALL mean relative error: {errs.mean()*100:.1f}%")

    # --- Simulation-based calibration ---
    print("\nSimulation-based calibration (SBC) — ranks should be ~uniform:")
    try:
        from sbi.diagnostics import run_sbc, check_sbc  # sbi>=0.23 location
        ranks, dap = run_sbc(
            torch.tensor(theta_test), torch.tensor(x_test), posterior,
            num_posterior_samples=500, use_batched_sampling=False,
        )
        stats = check_sbc(ranks, torch.tensor(theta_test), dap, num_posterior_samples=500)
        # check_sbc return-keys have drifted across sbi versions; print whatever
        # it returns rather than hard-coding 'c2st'/'ks_pvals'.
        for k, v in stats.items():
            try:
                val = float(np.asarray(v, dtype=float).mean())
                print(f"  {k}: mean={val:.3f}")
            except Exception:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"  SBC step error (non-fatal): {e}")

    print("\n" + "=" * 70)
    print("DONE. If overall recovery error is modest (<~15%) and SBC is roughly")
    print("calibrated, the amortized-inference thesis holds — scale it up.")
    print("=" * 70)


if __name__ == "__main__":
    main()

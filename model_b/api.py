"""
model_b/api.py — high-level wrapper functions used by the FastAPI backend.

Three responsibilities:
1. Device-aware defaults: CPU gets FFT + small nsim; GPU gets K-L + production nsim.
2. Stable JSON-shaped inputs/outputs so the backend never sees JAX types.
3. A fit callback hook so the UI can render live progress.

See docs/plans/2026-06-05-stage-7-design.md for the full design.
"""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import fit as fit_b
from model_b import grf as grf_circulant
from model_b import simulate as sim_b
from model_b.objective import COND_MAP_B, clamp_b


# Preview nsim: small enough for snappy slider feedback on CPU. The forward
# preview is for shape intuition, not publishable numbers — that's what the
# "Run full" button (nsim=9000) is for.
PREVIEW_NSIM_CPU = 128

# Tiny LRU cache so a slider returning to a prior value answers instantly.
_PREVIEW_CACHE_MAXSIZE = 256
_preview_cache: dict[tuple, dict] = {}
_preview_cache_order: list[tuple] = []

PARAM_KEYS = ("ter", "st", "cr", "crsd", "sis", "sig", "av1", "av2", "av3")

# sig (GRF correlation length) must stay below the positive-definite-embedding
# ceiling (~17.95 at 100x160). Above it, calc_LAM silently clips negative
# spectral values to zero and returns a degenerate field — silently wrong
# output, which is worse than an error. clamp_b uses 17.0; we validate to 17.5.
SIG_MAX = 17.5


def _validate_params(params: dict) -> None:
    """Raise ValueError with a clear message if any param is out of valid range.

    Catches the silently-wrong cases (notably sig past the PD ceiling) so the
    API returns an actionable 400 instead of a degenerate result.
    """
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"missing params: {missing}")
    sig = float(params["sig"])
    if not (0.2 <= sig <= SIG_MAX):
        raise ValueError(
            f"sig (GRF correlation length) must be in [0.2, {SIG_MAX}]; got {sig}. "
            f"Above ~17.95 the circulant embedding is not positive-definite."
        )
    if float(params["sis"]) <= 0:
        raise ValueError(f"sis (drift bump width) must be > 0; got {params['sis']}")
    if float(params["cr"]) < 1:
        raise ValueError(f"cr (threshold) must be >= 1; got {params['cr']}")
    for k in ("av1", "av2", "av3"):
        if float(params[k]) < 0:
            raise ValueError(f"{k} (drift amplitude) must be >= 0; got {params[k]}")


def _preview_cache_key(params: dict, key_seed: int) -> tuple:
    return (key_seed,) + tuple(round(float(params[k]), 4) for k in PARAM_KEYS)


def _get_device_defaults(backend: str | None = None) -> dict:
    """Return sensible nsim/chunk_size/use_kl based on JAX device.

    backend : "cpu", "gpu", or None to auto-detect via jax.default_backend().
    """
    if backend is None:
        backend = jax.default_backend()
    if backend == "gpu":
        return {"use_kl": True, "nsim": 9000, "chunk_size": 64}
    # CPU: chunk_size=8 halves the lax.map iteration count vs 4 (faster) while
    # keeping the per-chunk FFT noise tensor well under 1 GB.
    return {"use_kl": False, "nsim": 512, "chunk_size": 8}


def forward_sim_preview(params: dict, key_seed: int = 0) -> dict:
    """Fast small-nsim simulation for slider previews.

    params : dict with keys ter, st, cr, crsd, sis, sig, av1, av2, av3.
             A single condition's worth of drift bumps.
    Returns: {"rt": list[float], "cat": list[int]}, JSON-friendly.

    Cached on rounded params so a slider returning to a prior value answers
    instantly without re-running the simulator.
    """
    _validate_params(params)
    ck = _preview_cache_key(params, key_seed)
    hit = _preview_cache.get(ck)
    if hit is not None:
        return hit

    defaults = _get_device_defaults()
    nsim_preview = min(defaults["nsim"], PREVIEW_NSIM_CPU)
    chunk_preview = min(defaults["chunk_size"], 4)

    key = jax.random.key(key_seed)
    rt, cat = sim_b.simulate_b(
        key,
        ter=params["ter"], st=params["st"],
        cr=params["cr"], crsd=params["crsd"],
        av1=params["av1"], av2=params["av2"], av3=params["av3"],
        sis=params["sis"], sig=params["sig"], si=6.0,
        nsim=nsim_preview, chunk_size=chunk_preview,
        use_kl=defaults["use_kl"],
    )
    out = {
        "rt": [float(x) for x in np.asarray(rt)],
        "cat": [int(c) for c in np.asarray(cat)],
    }
    _preview_cache[ck] = out
    _preview_cache_order.append(ck)
    if len(_preview_cache_order) > _PREVIEW_CACHE_MAXSIZE:
        evicted = _preview_cache_order.pop(0)
        _preview_cache.pop(evicted, None)
    return out


def warmup() -> None:
    """Trigger JIT compilation of the preview path so the first real request is
    fast. Safe to call from a background thread at server startup.
    """
    forward_sim_preview(
        {"ter": 200.0, "st": 50.0, "cr": 10.0, "crsd": 2.0,
         "sis": 12.0, "sig": 10.0, "av1": 15.0, "av2": 10.0, "av3": 8.0},
        key_seed=0,
    )


def forward_sim_full(params: dict, nsim: int = 9000,
                     chunk_size: int | None = None,
                     key_seed: int = 0) -> dict:
    """Production-scale single-condition simulation."""
    _validate_params(params)
    defaults = _get_device_defaults()
    cs = chunk_size if chunk_size is not None else defaults["chunk_size"]

    key = jax.random.key(key_seed)
    rt, cat = sim_b.simulate_b(
        key,
        ter=params["ter"], st=params["st"],
        cr=params["cr"], crsd=params["crsd"],
        av1=params["av1"], av2=params["av2"], av3=params["av3"],
        sis=params["sis"], sig=params["sig"], si=6.0,
        nsim=nsim, chunk_size=cs,
        use_kl=defaults["use_kl"],
    )
    return {
        "rt": [float(x) for x in np.asarray(rt)],
        "cat": [int(c) for c in np.asarray(cat)],
    }


def fit_model(data: dict, x0: list[float],
              nsim: int | None = None,
              chunk_size: int | None = None,
              maxiter: int = 100,
              on_update=None) -> dict:
    """Run fit_simplex_b with optional per-eval callback.

    data : {"prop": (2,5), "count": (2,5), "quant": (2,5,5)} — lists or arrays
    x0   : 13-element list of floats
    on_update(eval_n, loss, x_list) callable, optional — receives plain Python types
    Returns: {"params": list, "loss": float, "n_iters": int, "converged": bool}
    """
    defaults = _get_device_defaults()
    nsim_eff = nsim or defaults["nsim"]
    cs_eff = chunk_size or defaults["chunk_size"]

    data_jax = {
        "prop": jnp.asarray(data["prop"]),
        "count": jnp.asarray(data["count"], dtype=jnp.int64),
        "quant": jnp.asarray(data["quant"]),
    }
    x0_jax = jnp.asarray(x0)
    key = jax.random.key(1)

    def wrapped_cb(eval_n, loss, x):
        if on_update:
            on_update(eval_n, loss, x.tolist())

    res = fit_b.fit_simplex_b(
        data_jax, key, x0_jax,
        nsim=nsim_eff, maxiter=maxiter, chunk_size=cs_eff,
        use_kl=defaults["use_kl"],
        on_update=wrapped_cb,
    )
    return {
        "params": [float(p) for p in np.asarray(res.params)],
        "loss": float(res.loss),
        "n_iters": int(res.n_iters),
        "converged": bool(res.converged),
    }


def predict_from_params(params_full: list[float],
                        n_conditions: int = 2,
                        nsim: int = 1024,
                        key_seed: int = 0) -> dict:
    """Generate per-condition RT distributions from fitted params.

    params_full : 13-element list (ter, st, cr, crsd, sis, sig, sv, av1c1, av2c1,
                  av3c1, av1c2, av2c2, av3c2).
    Returns: {"by_condition": [{"rt": [...], "cat": [...], "props": [...]}, ...]}
    """
    p = clamp_b(jnp.asarray(params_full))
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    out = []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B[:n_conditions]):
        sub = forward_sim_full(
            params={"ter": float(ter), "st": float(st),
                    "cr": float(cr), "crsd": float(crsd),
                    "sis": float(sis), "sig": float(sig),
                    "av1": float(p[d1]), "av2": float(p[d2]), "av3": float(p[d3])},
            nsim=nsim, key_seed=key_seed + ci,
        )
        cat_arr = np.asarray(sub["cat"])
        sub["props"] = [float((cat_arr == c).mean()) for c in (1, 2, 3, 4, 5)]
        out.append(sub)
    return {"by_condition": out}


def field_snapshots(params: dict,
                    mode: str = "single",
                    n_frames: int = 48,
                    n_trials_mean: int = 64,
                    grid_stride: int = 2,
                    key_seed: int = 0) -> dict:
    """Generate accumulator-field snapshots for the three.js Field view.

    Returns the evidence field a(i, j, t) — the cumulative demeaned increment
    that the diffusion model accumulates toward threshold — sampled at n_frames
    timesteps and downsampled on the spatial grid by grid_stride.

    params : dict with keys ter, st, cr, crsd, sis, sig, av1, av2, av3 (single
             condition; ter/st/crsd unused here, only the spatial+drift params
             shape the field).
    mode   : "single" (one trial, noisy and intuitive) or "mean" (average over
             n_trials_mean trials, smoother).
    Returns JSON-friendly dict:
        {
          "frames": list of (n_rows, n_cols) float grids, one per sampled step,
          "steps":  the timestep index of each frame,
          "threshold": cr (the decision threshold plane height),
          "n", "m": downsampled grid dimensions,
          "nstep": total timesteps in the model,
        }
    """
    if mode not in ("single", "mean"):
        raise ValueError(f"mode must be 'single' or 'mean', got {mode!r}")
    _validate_params(params)

    N, M, NSTEP = sim_b.N, sim_b.M, sim_b.NSTEP
    n_trials = 1 if mode == "single" else int(n_trials_mean)

    LAM = grf_circulant.calc_LAM(s1=params["sig"], s2=params["sig"])
    v1, v2, v3 = sim_b.drift_bumps(sis=params["sis"])
    n_pad, m_pad = LAM.shape
    n_fft = (NSTEP + 1) // 2

    key = jax.random.key(key_seed)
    _, kg = jax.random.split(key)

    # Generate the GRF path via the same FFT F1/F2 trick the simulator uses.
    z = jax.random.normal(kg, (n_trials, n_fft, 2, n_pad, m_pad), dtype=jnp.float32)
    LAM_f32 = LAM.astype(jnp.float32)
    X = LAM_f32 * (z[:, :, 0, :, :] + 1j * z[:, :, 1, :, :])
    F = jnp.fft.fft2(X)
    F1 = F[:, :, :N, :M].real
    F2 = F[:, :, :N, :M].imag
    grf_path = jnp.stack([F1, F2], axis=2).reshape(
        n_trials, n_fft * 2, N, M
    )[:, :NSTEP, :, :]

    drift_const = (
        jnp.float32(params["av1"]) * v1.astype(jnp.float32)
        + jnp.float32(params["av2"]) * v2.astype(jnp.float32)
        + jnp.float32(params["av3"]) * v3.astype(jnp.float32)
    )
    incr = drift_const[None, None, :, :] + grf_path
    incr = incr - incr.mean(axis=(-2, -1), keepdims=True)
    a = jnp.cumsum(incr, axis=1)   # (n_trials, NSTEP, N, M)

    field = a[0] if mode == "single" else a.mean(axis=0)   # (NSTEP, N, M)
    field_np = np.asarray(field)                           # (NSTEP, N, M)

    step_idx = np.linspace(0, NSTEP - 1, n_frames).astype(int)
    sampled = field_np[step_idx]                           # (n_frames, N, M)
    sampled_ds = sampled[:, ::grid_stride, ::grid_stride]  # downsample grid

    out = {
        "frames": sampled_ds.astype(np.float32).tolist(),
        "steps": step_idx.tolist(),
        "threshold": float(params["cr"]),
        "n": int(sampled_ds.shape[1]),
        "m": int(sampled_ds.shape[2]),
        "nstep": int(NSTEP),
    }

    # Single-trial only: trace the winning region's trajectory (the argmax
    # location of the accumulator at each sampled frame) and the commitment
    # frame (first frame whose field max exceeds the threshold). Coordinates
    # are in DOWNSAMPLED grid units so they line up with the rendered surface.
    if mode == "single":
        cr = float(params["cr"])
        trajectory = []
        crossing_frame = None
        for fi, frame in enumerate(sampled):              # full-res frame (N, M)
            flat = int(np.argmax(frame))
            row = flat // M
            col = flat % M
            val = float(frame[row, col])
            trajectory.append([row // grid_stride, col // grid_stride, val])
            if crossing_frame is None and val > cr:
                crossing_frame = fi
        out["trajectory"] = trajectory
        out["crossing_frame"] = crossing_frame

    return out


def phase_diagram(params: dict,
                  x_param: str = "cr",
                  x_range: tuple[float, float] = (4.0, 18.0),
                  y_param: str = "av1",
                  y_range: tuple[float, float] = (4.0, 24.0),
                  grid: int = 12,
                  nsim: int = 200,
                  metric: str = "accuracy",
                  key_seed: int = 0) -> dict:
    """Sweep a 2D parameter grid and return a heatmap of accuracy or mean RT.

    x_param/y_param : keys of the single-condition param dict (cr, av1, sis, ...).
    metric : "accuracy" (proportion of trials in category 1, the target region)
             or "rt" (mean decision time in ms).

    Because cr/av*/sis/etc. are traced (non-static) JIT args, simulate_b compiles
    once and every grid cell reuses it — the sweep is grid*grid fast calls.
    Returns {x_values, y_values, z (grid x grid), x_param, y_param, metric}.
    """
    if metric not in ("accuracy", "rt"):
        raise ValueError(f"metric must be 'accuracy' or 'rt', got {metric!r}")
    valid = ("ter", "st", "cr", "crsd", "sis", "sig", "av1", "av2", "av3")
    if x_param not in valid or y_param not in valid:
        raise ValueError(f"x_param/y_param must be one of {valid}")

    defaults = _get_device_defaults()
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)
    z = np.zeros((grid, grid), dtype=np.float64)

    base = dict(params)
    for yi, yv in enumerate(ys):
        for xi, xv in enumerate(xs):
            p = dict(base)
            p[x_param] = float(xv)
            p[y_param] = float(yv)
            key = jax.random.key(key_seed + yi * grid + xi)
            rt, cat = sim_b.simulate_b(
                key,
                ter=p["ter"], st=p["st"], cr=p["cr"], crsd=p["crsd"],
                av1=p["av1"], av2=p["av2"], av3=p["av3"],
                sis=p["sis"], sig=p["sig"], si=6.0,
                nsim=nsim, chunk_size=defaults["chunk_size"],
                use_kl=defaults["use_kl"],
            )
            cat_np = np.asarray(cat)
            if metric == "accuracy":
                z[yi, xi] = float((cat_np == 1).mean())
            else:
                z[yi, xi] = float(np.asarray(rt).mean())

    return {
        "x_values": xs.tolist(),
        "y_values": ys.tolist(),
        "z": z.tolist(),
        "x_param": x_param,
        "y_param": y_param,
        "metric": metric,
    }

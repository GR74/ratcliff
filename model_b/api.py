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
from model_b import simulate as sim_b
from model_b.objective import COND_MAP_B, clamp_b


def _get_device_defaults(backend: str | None = None) -> dict:
    """Return sensible nsim/chunk_size/use_kl based on JAX device.

    backend : "cpu", "gpu", or None to auto-detect via jax.default_backend().
    """
    if backend is None:
        backend = jax.default_backend()
    if backend == "gpu":
        return {"use_kl": True, "nsim": 9000, "chunk_size": 64}
    return {"use_kl": False, "nsim": 512, "chunk_size": 4}


def forward_sim_preview(params: dict, key_seed: int = 0) -> dict:
    """Fast small-nsim simulation for slider previews.

    params : dict with keys ter, st, cr, crsd, sis, sig, av1, av2, av3.
             A single condition's worth of drift bumps.
    Returns: {"rt": list[float], "cat": list[int]}, JSON-friendly.
    """
    defaults = _get_device_defaults()
    nsim_preview = min(defaults["nsim"], 256)
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
    return {
        "rt": [float(x) for x in np.asarray(rt)],
        "cat": [int(c) for c in np.asarray(cat)],
    }


def forward_sim_full(params: dict, nsim: int = 9000,
                     chunk_size: int | None = None,
                     key_seed: int = 0) -> dict:
    """Production-scale single-condition simulation."""
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

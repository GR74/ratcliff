"""Smoke tests for model_b/api.py wrapper layer (Stage 7.A)."""
import jax
import numpy as np
import pytest

from model_b import api as model_api


DEFAULT_PARAMS = {
    "ter": 200.0, "st": 50.0, "cr": 10.0, "crsd": 2.0,
    "sis": 12.0, "sig": 10.0,
    "av1": 15.0, "av2": 10.0, "av3": 8.0,
}


# ---- 7.A.1: device defaults ---------------------------------------------

def test_get_device_defaults_cpu():
    """On CPU, should pick FFT path + small nsim."""
    defaults = model_api._get_device_defaults(backend="cpu")
    assert defaults["use_kl"] is False
    assert defaults["nsim"] == 512
    assert defaults["chunk_size"] == 4


def test_get_device_defaults_gpu():
    """On GPU, should pick K-L path + production nsim."""
    defaults = model_api._get_device_defaults(backend="gpu")
    assert defaults["use_kl"] is True
    assert defaults["nsim"] == 9000
    assert defaults["chunk_size"] == 64


def test_get_device_defaults_auto_uses_jax_backend():
    """Default behavior: auto-detect from jax.default_backend()."""
    defaults = model_api._get_device_defaults()
    expected_use_kl = jax.default_backend() == "gpu"
    assert defaults["use_kl"] == expected_use_kl


# ---- 7.A.2: forward_sim_preview -----------------------------------------

def test_forward_sim_preview_returns_rt_and_cat():
    """Preview should run a small simulation and return RTs + cat lists."""
    out = model_api.forward_sim_preview(DEFAULT_PARAMS, key_seed=0)
    assert "rt" in out and "cat" in out
    assert len(out["rt"]) >= 64, "preview should run at least 64 trials"
    assert len(out["rt"]) == len(out["cat"])
    assert all(np.isfinite(out["rt"]))
    assert set(out["cat"]).issubset({1, 2, 3, 4, 5})


def test_forward_sim_preview_deterministic_for_same_seed():
    """Same seed -> same output."""
    a = model_api.forward_sim_preview(DEFAULT_PARAMS, key_seed=42)
    b = model_api.forward_sim_preview(DEFAULT_PARAMS, key_seed=42)
    assert a["rt"] == b["rt"]
    assert a["cat"] == b["cat"]


def test_forward_sim_preview_output_is_json_serializable():
    """RT/cat should be plain Python lists (not JAX arrays)."""
    import json
    out = model_api.forward_sim_preview(DEFAULT_PARAMS, key_seed=0)
    json.dumps(out)  # raises if non-JSON types present


# ---- param validation (robustness) --------------------------------------

def test_validate_rejects_sig_above_pd_ceiling():
    """sig past the positive-definite ceiling raises a clear error, not silent
    garbage."""
    bad = dict(DEFAULT_PARAMS, sig=25.0)
    with pytest.raises(ValueError, match="sig"):
        model_api.forward_sim_preview(bad)


def test_validate_rejects_nonpositive_sis():
    bad = dict(DEFAULT_PARAMS, sis=0.0)
    with pytest.raises(ValueError, match="sis"):
        model_api.forward_sim_preview(bad)


def test_validate_rejects_low_cr():
    bad = dict(DEFAULT_PARAMS, cr=0.5)
    with pytest.raises(ValueError, match="cr"):
        model_api.forward_sim_preview(bad)


def test_validate_accepts_valid_params():
    """Boundary sig values are accepted."""
    model_api._validate_params(dict(DEFAULT_PARAMS, sig=17.0))
    model_api._validate_params(dict(DEFAULT_PARAMS, sig=0.2))


# ---- 7.A.3: forward_sim_full --------------------------------------------

def test_forward_sim_full_respects_nsim():
    """Full should produce exactly nsim trials."""
    out = model_api.forward_sim_full(DEFAULT_PARAMS, nsim=128, chunk_size=8, key_seed=0)
    assert len(out["rt"]) == 128
    assert len(out["cat"]) == 128


# ---- 7.A.4: fit_simplex_b on_update callback ----------------------------

def test_fit_simplex_b_invokes_callback():
    """fit_simplex_b should call on_update(eval_n, loss, x) every eval."""
    import jax.numpy as jnp
    from model_b import simulate as sim_b
    from model_b import fit as fit_b
    from model_b.objective import COND_MAP_B, clamp_b
    from shared import prng

    TRUE = jnp.array([200.0, 50.0, 10.0, 2.0, 12.0, 10.0, 0.5,
                      15.0, 10.0, 8.0, 14.0, 11.0, 9.0])
    p = clamp_b(TRUE)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    props, counts, quants = [], [], []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(jax.random.key(0), ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
            sis, sig, si, nsim=32, chunk_size=8,
        )
        cat_np = np.asarray(cat); rt_np = np.asarray(rt)
        props.append(jnp.asarray([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)]))
        counts.append(jnp.asarray([(cat_np == c).sum() for c in (1, 2, 3, 4, 5)],
                                   dtype=jnp.int64))
        q = np.zeros((5, 5))
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat_np == c
            if mask.sum() >= 5:
                q[:, ki] = np.quantile(rt_np[mask], qs)
        quants.append(jnp.asarray(q))
    data = {"prop": jnp.stack(props), "count": jnp.stack(counts),
            "quant": jnp.stack(quants)}

    np.random.seed(0)
    x0 = TRUE * jnp.asarray(np.random.uniform(0.98, 1.02, size=13))

    calls = []
    def cb(eval_n, loss, x):
        calls.append((eval_n, float(loss), x.tolist()))

    fit_b.fit_simplex_b(
        data, jax.random.key(1), x0,
        nsim=32, maxiter=3, chunk_size=8, on_update=cb,
    )
    assert len(calls) >= 3, f"expected callback calls, got {len(calls)}"
    assert all(isinstance(c[0], int) for c in calls)
    assert all(np.isfinite(c[1]) for c in calls)
    assert all(len(c[2]) == 13 for c in calls)


# ---- 7.A.5: fit_model ---------------------------------------------------

@pytest.mark.slow
def test_fit_model_returns_expected_shape():
    """fit_model should return JSON-shaped result with all 4 keys."""
    import jax.numpy as jnp
    from model_b import simulate as sim_b
    from model_b.objective import COND_MAP_B, clamp_b
    from shared import prng

    TRUE = [200.0, 50.0, 10.0, 2.0, 12.0, 10.0, 0.5,
            15.0, 10.0, 8.0, 14.0, 11.0, 9.0]
    p = clamp_b(jnp.asarray(TRUE))
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    props, counts, quants = [], [], []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(jax.random.key(0), ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
            sis, sig, si, nsim=32, chunk_size=8,
        )
        cat_np = np.asarray(cat); rt_np = np.asarray(rt)
        props.append([(cat_np == c).mean().item() for c in (1, 2, 3, 4, 5)])
        counts.append([(cat_np == c).sum().item() for c in (1, 2, 3, 4, 5)])
        q = np.zeros((5, 5))
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat_np == c
            if mask.sum() >= 5:
                q[:, ki] = np.quantile(rt_np[mask], qs)
        quants.append(q.tolist())
    data = {"prop": props, "count": counts, "quant": quants}

    np.random.seed(0)
    x0 = (np.array(TRUE) * np.random.uniform(0.98, 1.02, size=13)).tolist()

    result = model_api.fit_model(data, x0, nsim=32, chunk_size=8, maxiter=3)
    assert set(result.keys()) == {"params", "loss", "n_iters", "converged"}
    assert len(result["params"]) == 13
    assert all(isinstance(p, float) for p in result["params"])
    assert isinstance(result["loss"], float)
    assert isinstance(result["n_iters"], int)
    assert isinstance(result["converged"], bool)


# ---- 7.A.6: predict_from_params -----------------------------------------

def test_predict_from_params_returns_per_condition():
    """predict_from_params should return n_conditions worth of predictions."""
    TRUE = [200.0, 50.0, 10.0, 2.0, 12.0, 10.0, 0.5,
            15.0, 10.0, 8.0, 14.0, 11.0, 9.0]
    out = model_api.predict_from_params(TRUE, n_conditions=2, nsim=64, key_seed=0)
    assert "by_condition" in out
    assert len(out["by_condition"]) == 2
    for cond in out["by_condition"]:
        assert "rt" in cond and "cat" in cond and "props" in cond
        assert len(cond["props"]) == 5
        assert abs(sum(cond["props"]) - 1.0) < 1e-9


# ---- 8.2: field_snapshots -----------------------------------------------

def test_field_snapshots_single_shape():
    """Single-trial field returns n_frames downsampled grids."""
    out = model_api.field_snapshots(DEFAULT_PARAMS, mode="single",
                                    n_frames=12, grid_stride=2, key_seed=0)
    assert len(out["frames"]) == 12
    assert len(out["steps"]) == 12
    # 100x160 grid at stride 2 → 50x80
    assert out["n"] == 50 and out["m"] == 80
    assert len(out["frames"][0]) == 50
    assert len(out["frames"][0][0]) == 80
    assert out["nstep"] == 400
    assert out["threshold"] == DEFAULT_PARAMS["cr"]


def test_field_snapshots_mean_shape():
    """Mean-field over a few trials returns the same shape as single."""
    out = model_api.field_snapshots(DEFAULT_PARAMS, mode="mean",
                                    n_frames=8, n_trials_mean=4,
                                    grid_stride=4, key_seed=0)
    assert len(out["frames"]) == 8
    # 100x160 at stride 4 → 25x40
    assert out["n"] == 25 and out["m"] == 40


def test_field_snapshots_accumulates_over_time():
    """Field magnitude should grow over timesteps (cumulative accumulation)."""
    import numpy as np
    out = model_api.field_snapshots(DEFAULT_PARAMS, mode="single",
                                    n_frames=10, grid_stride=2, key_seed=0)
    first = np.abs(np.array(out["frames"][0])).max()
    last = np.abs(np.array(out["frames"][-1])).max()
    assert last > first, "accumulator field should grow over time"


def test_field_snapshots_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError, match="mode must be"):
        model_api.field_snapshots(DEFAULT_PARAMS, mode="bogus")


# ---- 9.A: trajectory trace ----------------------------------------------

def test_field_snapshots_single_has_trajectory():
    """Single mode returns a per-frame argmax trajectory + crossing frame."""
    out = model_api.field_snapshots(DEFAULT_PARAMS, mode="single",
                                    n_frames=12, grid_stride=2, key_seed=0)
    assert "trajectory" in out and "crossing_frame" in out
    assert len(out["trajectory"]) == 12
    for pt in out["trajectory"]:
        assert len(pt) == 3  # [row_ds, col_ds, value]
        assert 0 <= pt[0] < out["n"]
        assert 0 <= pt[1] < out["m"]
    # crossing_frame is an int index or None
    assert out["crossing_frame"] is None or isinstance(out["crossing_frame"], int)


def test_field_snapshots_mean_has_no_trajectory():
    """Mean mode omits trajectory (no single crossing)."""
    out = model_api.field_snapshots(DEFAULT_PARAMS, mode="mean",
                                    n_frames=6, n_trials_mean=4, grid_stride=4)
    assert "trajectory" not in out


# ---- 9.B: phase diagram -------------------------------------------------

def test_phase_diagram_accuracy_shape():
    out = model_api.phase_diagram(DEFAULT_PARAMS, grid=4, nsim=64, metric="accuracy")
    assert len(out["z"]) == 4 and len(out["z"][0]) == 4
    assert len(out["x_values"]) == 4 and len(out["y_values"]) == 4
    for row in out["z"]:
        for v in row:
            assert 0.0 <= v <= 1.0  # accuracy is a proportion


def test_phase_diagram_rt_metric():
    out = model_api.phase_diagram(DEFAULT_PARAMS, grid=3, nsim=64, metric="rt")
    for row in out["z"]:
        for v in row:
            assert v > 0  # mean RT positive


def test_phase_diagram_rejects_bad_metric():
    import pytest
    with pytest.raises(ValueError, match="metric must be"):
        model_api.phase_diagram(DEFAULT_PARAMS, grid=2, metric="bogus")


def test_phase_diagram_rejects_bad_param():
    import pytest
    with pytest.raises(ValueError, match="x_param/y_param"):
        model_api.phase_diagram(DEFAULT_PARAMS, grid=2, x_param="nonsense")

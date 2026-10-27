"""Policy selection and dispatch tests; full-grid GPU checks live in the screen."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import grf_kl, simulate


@pytest.mark.parametrize("sig,generator,required", [(5., "fft", 7399), (10., "kl", 1850), (15., "kl", 821)])
def test_guarded_policy_reports_achieved_variance(sig, generator, required):
    result = grf_kl.guarded_kl_diagnostics(sig)
    assert result["generator"] == generator
    assert result["required_modes"] == required
    assert result["achieved_variance"] >= .999
    assert result["cap_hit"] == (generator == "fft")
    if generator == "fft":
        assert result["variance_at_cap"] < .8


@pytest.mark.parametrize("sig", [0, -1, float("nan"), float("inf")])
def test_guarded_rejects_invalid_sig(sig):
    with pytest.raises(ValueError, match="sig"):
        grf_kl.guarded_kl_diagnostics(sig)


def test_fallback_does_not_allocate_dense_basis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fallback must not construct a KL basis")
    monkeypatch.setattr(grf_kl, "calc_kl_basis", forbidden)
    sentinel = object()
    monkeypatch.setattr(simulate, "_simulate_b_fft", lambda *a: sentinel)
    result = simulate.simulate_b(jax.random.key(1), 200., 50., 10., 2., 15., 10., 8.,
                                 12., 5., 6., 2, use_kl=True, guarded_kl=True)
    assert result is sentinel


def test_eligible_dispatch_requests_strict_target(monkeypatch):
    seen = {}
    def basis(**kwargs):
        seen.update(kwargs)
        return np.zeros((1, 1)), 1850, .999
    monkeypatch.setattr(grf_kl, "calc_kl_basis", basis)
    monkeypatch.setattr(simulate, "_simulate_b_kl_inner", lambda *a: "kl")
    result = simulate.simulate_b(jax.random.key(1), 200., 50., 10., 2., 15., 10., 8.,
                                 12., 10., 6., 2, use_kl=True, guarded_kl=True)
    assert result == "kl"
    assert seen["variance_threshold"] == .999
    assert seen["pad_to_k_max"] is True


def test_diagnostics_are_not_mutable_cached_state():
    result = grf_kl.guarded_kl_diagnostics(10.)
    result["generator"] = "fft"
    assert grf_kl.guarded_kl_diagnostics(10.)["generator"] == "kl"


def test_guarded_dispatch_on_both_sides_of_capacity_boundary():
    low, high = 8., 10.
    for _ in range(16):
        middle = (low + high) / 2
        if grf_kl.guarded_kl_diagnostics(middle)["generator"] == "fft":
            low = middle
        else:
            high = middle
    below = grf_kl.guarded_kl_diagnostics(low)
    above = grf_kl.guarded_kl_diagnostics(high)
    assert below["required_modes"] > 2000
    assert below["generator"] == "fft" and below["variance_at_cap"] < .999
    assert above["required_modes"] <= 2000
    assert above["generator"] == "kl" and above["achieved_variance"] >= .999


def test_guard_requires_explicit_kl_opt_in():
    with pytest.raises(ValueError, match="requires"):
        simulate.simulate_b(jax.random.key(1), 200., 50., 10., 2., 15., 10., 8.,
                            12., 10., 6., 2, guarded_kl=True)


@pytest.mark.parametrize("option", ["kl_bucketed", "cumulative_kl"])
def test_guard_does_not_silently_combine_other_experiments(option):
    with pytest.raises(ValueError, match="no other"):
        simulate.simulate_b_blocked(jax.random.key(1), 200., 50., 10., 2., 15., 10., 8.,
                                    12., 10., 6., 2, use_kl=True, guarded_kl=True,
                                    **{option: True})


@pytest.mark.parametrize("block", [0, 16])
def test_objective_forwards_guard_to_each_condition(monkeypatch, block):
    from model_b import objective
    seen = []
    def fake_simulate(*args):
        seen.append(args[-2])
        return jnp.full((2,), 300.), jnp.ones(2, dtype=jnp.int32)
    monkeypatch.setattr(simulate, "simulate_b_blocked" if block else "simulate_b", fake_simulate)
    params = jnp.array([200., 50., 10., 2., 12., 10., .5, 15., 10., 8., 14., 11., 9.])
    data = {"prop": jnp.full((2, 5), .2), "count": jnp.full((2, 5), 100),
            "quant": jnp.full((2, 5, 5), 300.)}
    result = objective.fofs_b_new(params, data, jax.random.key(1), nsim=2,
                                  use_kl=True, guarded_kl=True, time_block_size=block)
    assert np.isfinite(result)
    assert seen and all(seen)


def test_fit_forwards_guarded_policy(monkeypatch):
    from model_b import fit
    seen = []
    def loss(p, *args, **kwargs):
        seen.append(kwargs)
        return float(jnp.sum(p*p))
    monkeypatch.setattr(fit.obj_b, "fofs_b_new", loss)
    fit.fit_simplex_b({}, jax.random.key(1), np.ones(13), maxiter=1,
                      use_kl=True, guarded_kl=True)
    assert seen and all(x["guarded_kl"] and x["use_kl"] for x in seen)

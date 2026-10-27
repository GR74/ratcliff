import jax
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.model_a_view import ModelARequest, history, router
from model_a import simulate
from shared import prng


def test_history_matches_existing_simulator_and_first_crossing():
    req = ModelARequest()
    result = history(req)
    rt, cat = simulate.simulate(prng.root_key(req.seed), req.ter, req.st, req.cr, req.crsd, req.si, req.sig, req.av, 1, 1)
    assert result["hit"]
    assert result["response_time"] == pytest.approx(float(rt[0]))
    pos = result["winner"] + 1
    expected_cat = 1 if 30 < pos < 42 else 3 if pos <= 20 or pos >= 52 else 2
    assert int(cat[0]) == expected_cat
    a = np.array(result["evidence"])
    assert a.shape[1] == 72
    assert np.all(a[:-1].max(axis=1) <= result["threshold"])
    assert a[-1].max() > result["threshold"]
    assert np.max(np.abs(a.mean(axis=1))) < 1e-10
    assert a.min() < 0
    assert result["angles"] == list(range(5, 361, 5))
    assert result["times"][-1] == result["response_time"]
    assert result == history(req)


def test_shared_paths_match_original_equations():
    req = ModelARequest()
    key = prng.trial_keys(prng.root_key(req.seed), 1)[0]
    ku, kz = jax.random.split(key)
    u = np.asarray(jax.random.uniform(ku, (1, 10)))[0]
    z = np.asarray(jax.random.normal(kz, (1, 400, 72)))[0]
    drift = np.asarray(simulate.drift_profile(req.av, req.si))
    noise = z @ np.asarray(simulate.chol_factor(req.sig)).T
    increments = drift + 5 * noise
    expected = np.cumsum(increments - increments.mean(axis=1, keepdims=True), axis=0)
    a, threshold, ndt = simulate.trial_history(prng.root_key(req.seed), req.ter, req.st, req.cr, req.crsd, req.si, req.sig, req.av)
    np.testing.assert_allclose(a, expected, atol=1e-9)
    assert float(threshold) == pytest.approx(req.cr + req.crsd * (u[4] - 0.5))
    assert float(ndt) == pytest.approx(req.ter + req.st * (0.5 - u[9]))


def test_no_crossing_and_nondecision_offset():
    req = ModelARequest(cr=10000, crsd=0, av=0, st=0)
    a = history(req)
    b = history(req.model_copy(update={"ter": 400}))
    assert not a["hit"]
    assert a["winner"] is None and a["response_time"] is None and a["decision_time"] is None
    assert len(a["evidence"]) == 401
    np.testing.assert_array_equal(a["evidence"], b["evidence"])
    np.testing.assert_allclose(np.array(b["times"]) - a["times"], 200)


def test_endpoint_and_parameter_ranges():
    app = FastAPI(); app.include_router(router)
    client = TestClient(app)
    for invalid in [{"sig": 0}, {"cr": 5, "crsd": 10}, {"ter": 10, "st": 50}]:
        assert client.post("/api/model-a/history", json=invalid).status_code == 422
    response = client.post("/api/model-a/history", json={})
    assert response.status_code == 200
    assert len(response.json()["angles"]) == 72

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.circular import CircularRequest, router, simulate_circular


def test_deterministic_drift_hits_correct_angle_time_and_radius():
    result = simulate_circular(CircularRequest(noise=0, drift=2, direction=90, threshold=1, nondecision=0.3))
    assert result["hit"]
    assert result["decision_time"] == pytest.approx(0.5)
    assert result["response_time"] == pytest.approx(0.8)
    assert result["angle"] == pytest.approx(90)
    assert result["path"][-1] == pytest.approx([0, 0.8, 1])


def test_seed_reproducibility_and_nondecision_only_offsets_time():
    req = CircularRequest()
    a = simulate_circular(req)
    assert a == simulate_circular(req)
    b = simulate_circular(req.model_copy(update={"nondecision": 0.7}))
    pa, pb = np.array(a["path"]), np.array(b["path"])
    assert np.allclose(pa[:, [0, 2]], pb[:, [0, 2]])
    assert np.allclose(pb[:, 1] - pa[:, 1], 0.5)
    radii = np.linalg.norm(pa[:, [0, 2]], axis=1)
    assert np.all(radii[:-1] < req.threshold)
    assert radii[-1] == pytest.approx(req.threshold)


def test_timeout_has_no_fabricated_response():
    result = simulate_circular(CircularRequest(noise=0, drift=0, max_time=0.1))
    assert not result["hit"]
    assert result["response_time"] is None and result["angle"] is None
    assert result["path"][-1] == pytest.approx([0, 0.3, 0])


def test_noise_variance_scales_with_elapsed_time():
    endpoints = [simulate_circular(CircularRequest(drift=0, noise=0.5, threshold=5, max_time=0.1, seed=i))["path"][-1][::2] for i in range(500)]
    # Each coordinate has variance sigma² * t = 0.025, not sigma² * step count.
    assert np.allclose(np.var(endpoints, axis=0), 0.025, rtol=0.2)


def test_endpoint_validates_and_returns_result():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.post("/api/circular", json={"threshold": 0}).status_code == 422
    response = client.post("/api/circular", json={"noise": 0, "drift": 2})
    assert response.status_code == 200
    assert response.json()["decision_time"] == pytest.approx(0.5)

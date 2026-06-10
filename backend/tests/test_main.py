"""Tests for backend/main.py endpoints (non-slow paths only)."""
import io

import pytest
from fastapi.testclient import TestClient

from backend import jobs
from backend.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_jobs():
    jobs.reset_for_tests()


# ---- /api/health -------------------------------------------------------

def test_health_returns_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "warm" in body  # warm-up flag present


# ---- /api/simulate -----------------------------------------------------

DEFAULT_PARAMS = {
    "ter": 200.0, "st": 50.0, "cr": 10.0, "crsd": 2.0,
    "sis": 12.0, "sig": 10.0,
    "av1": 15.0, "av2": 10.0, "av3": 8.0,
}


def test_simulate_preview_returns_lists():
    r = client.post("/api/simulate", json={"params": DEFAULT_PARAMS})
    assert r.status_code == 200
    body = r.json()
    assert "rt" in body and "cat" in body
    assert len(body["rt"]) >= 64
    assert len(body["rt"]) == len(body["cat"])
    assert set(body["cat"]).issubset({1, 2, 3, 4, 5})


def test_simulate_missing_param_returns_400():
    bad = {k: v for k, v in DEFAULT_PARAMS.items() if k != "ter"}
    r = client.post("/api/simulate", json={"params": bad})
    assert r.status_code == 400


# ---- /api/upload -------------------------------------------------------

def test_upload_csv_succeeds():
    rows = ["rt,cat,condition"]
    for cond in (1, 2):
        for cat in (1, 2, 3, 4, 5):
            for rt in (300, 350, 400, 450, 500):
                rows.append(f"{rt},{cat},{cond}")
    body = "\n".join(rows).encode("utf-8")
    r = client.post(
        "/api/upload",
        files={"file": ("data.csv", io.BytesIO(body), "text/csv")},
    )
    assert r.status_code == 200
    parsed = r.json()
    assert parsed["n_subjects"] == 1
    assert len(parsed["prop"]) == 2


def test_upload_invalid_returns_400():
    body = b"definitely not a valid data file"
    r = client.post(
        "/api/upload",
        files={"file": ("garbage.txt", io.BytesIO(body), "text/plain")},
    )
    assert r.status_code == 400


# ---- /api/fit/* --------------------------------------------------------

def test_fit_status_404_for_unknown_id():
    r = client.get("/api/fit/status/some-bogus-uuid")
    assert r.status_code == 404


def test_fit_result_404_for_unknown_id():
    r = client.get("/api/fit/result/some-bogus-uuid")
    assert r.status_code == 404


# ---- /api/predict ------------------------------------------------------

def test_predict_with_13_params():
    params = [200.0, 50.0, 10.0, 2.0, 12.0, 10.0, 0.5,
              15.0, 10.0, 8.0, 14.0, 11.0, 9.0]
    r = client.post("/api/predict", json={"params_full": params, "n_conditions": 2, "nsim": 64})
    assert r.status_code == 200
    body = r.json()
    assert "by_condition" in body
    assert len(body["by_condition"]) == 2
    for cond in body["by_condition"]:
        assert "rt" in cond and "cat" in cond and "props" in cond
        assert len(cond["props"]) == 5


def test_predict_rejects_wrong_param_count():
    r = client.post("/api/predict", json={"params_full": [1.0, 2.0], "n_conditions": 2})
    assert r.status_code == 400


def test_predict_rejects_bad_n_conditions():
    params = [200.0] * 13
    r = client.post("/api/predict", json={"params_full": params, "n_conditions": 5})
    assert r.status_code == 400


# ---- /api/field --------------------------------------------------------

def test_field_single_returns_frames():
    r = client.post(
        "/api/field",
        json={"params": DEFAULT_PARAMS, "mode": "single", "n_frames": 8, "grid_stride": 4},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["frames"]) == 8
    assert body["n"] == 25 and body["m"] == 40   # 100x160 stride 4
    assert body["nstep"] == 400
    assert body["threshold"] == DEFAULT_PARAMS["cr"]


def test_field_rejects_bad_mode():
    r = client.post("/api/field", json={"params": DEFAULT_PARAMS, "mode": "nope"})
    assert r.status_code == 400


def test_field_rejects_bad_n_frames():
    r = client.post(
        "/api/field",
        json={"params": DEFAULT_PARAMS, "mode": "single", "n_frames": 999},
    )
    assert r.status_code == 400


def test_field_missing_param_returns_400():
    bad = {k: v for k, v in DEFAULT_PARAMS.items() if k != "sig"}
    r = client.post("/api/field", json={"params": bad, "mode": "single", "n_frames": 4})
    assert r.status_code == 400


# ---- /api/phase --------------------------------------------------------

def test_phase_accuracy_returns_grid():
    r = client.post(
        "/api/phase",
        json={"params": DEFAULT_PARAMS, "grid": 3, "nsim": 64, "metric": "accuracy"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["z"]) == 3 and len(body["z"][0]) == 3
    assert body["metric"] == "accuracy"


def test_phase_rejects_bad_metric():
    r = client.post(
        "/api/phase",
        json={"params": DEFAULT_PARAMS, "grid": 2, "metric": "bogus"},
    )
    assert r.status_code == 400


def test_phase_rejects_bad_grid():
    r = client.post(
        "/api/phase",
        json={"params": DEFAULT_PARAMS, "grid": 99},
    )
    assert r.status_code == 400

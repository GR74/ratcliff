# Stage 7 Implementation Plan — CPU mode + React/FastAPI UI

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a CPU-friendly default simulator path + a public React/FastAPI web app for forward simulation, fitting, prediction, and comparison of the 2D GRF diffusion model.

**Architecture:** Add `model_b/api.py` as a thin wrapper layer above the existing JAX simulator. Build FastAPI backend (6 endpoints, BackgroundTask for fits, in-memory job dict) and React/Vite frontend (4 tabs, 13 param sliders, Plotly plots, localStorage for named configs). Deploy as a single HF Space via Docker.

**Tech Stack:** Python 3.12 + JAX (existing), FastAPI + Uvicorn, React 18 + Vite + TypeScript, TailwindCSS, Plotly.js, TanStack Query, Zustand.

**Design reference:** `docs/plans/2026-06-05-stage-7-design.md`

---

## Phase 7.A — CPU mode + API wrapper layer

Six tasks. Pure Python, no UI. Touches `model_b/api.py` (new), `model_b/fit.py` (one-line callback addition).

### Task 7.A.1: Device defaults helper

**Files:**
- Create: `model_b/api.py`
- Create: `model_b/tests/test_api_device_defaults.py`

**Step 1: Write failing test**

```python
"""Tests for model_b/api.py — device-aware defaults."""
import jax

from model_b import api as model_api


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
    assert defaults["use_kl"] == (jax.default_backend() == "gpu")
```

**Step 2: Run test, expect fail**

```powershell
.venv/Scripts/python.exe -m pytest model_b/tests/test_api_device_defaults.py -v
```

Expected: ModuleNotFoundError.

**Step 3: Implement**

```python
"""
model_b/api.py — high-level wrapper functions used by the FastAPI backend.

Three responsibilities:
1. Device-aware defaults: CPU gets FFT + small nsim; GPU gets K-L + 9000.
2. Stable JSON-shaped inputs/outputs: backend doesn't see JAX types.
3. A fit callback so the UI can render live progress.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _get_device_defaults(backend: str | None = None) -> dict:
    """Return sensible nsim/chunk_size/use_kl based on JAX device."""
    if backend is None:
        backend = jax.default_backend()
    if backend == "gpu":
        return {"use_kl": True, "nsim": 9000, "chunk_size": 64}
    return {"use_kl": False, "nsim": 512, "chunk_size": 4}
```

**Step 4: Run, expect pass.**

**Step 5: Commit**

```powershell
git add model_b/api.py model_b/tests/test_api_device_defaults.py
git commit -m "feat(api): device-aware defaults helper for CPU vs GPU"
```

---

### Task 7.A.2: forward_sim_preview

**Files:**
- Modify: `model_b/api.py` (append)
- Create: `model_b/tests/test_api_forward_sim.py`

**Step 1: Write failing test**

```python
"""Tests for forward_sim_preview / forward_sim_full."""
import numpy as np

from model_b import api as model_api


DEFAULT_PARAMS = {
    "ter": 200.0, "st": 50.0, "cr": 10.0, "crsd": 2.0,
    "sis": 12.0, "sig": 10.0,
    "av1": 15.0, "av2": 10.0, "av3": 8.0,
}


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
```

**Step 2: Run test, expect fail.**

**Step 3: Append to `model_b/api.py`**

```python
from model_b import simulate as sim_b


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
```

**Step 4: Run test, expect pass.**

**Step 5: Commit**

```powershell
git add model_b/api.py model_b/tests/test_api_forward_sim.py
git commit -m "feat(api): forward_sim_preview for slider live updates"
```

---

### Task 7.A.3: forward_sim_full

Append to `model_b/api.py`:

```python
def forward_sim_full(params: dict, nsim: int = 9000,
                     chunk_size: int | None = None,
                     key_seed: int = 0) -> dict:
    """Production-scale simulation. Same JSON return shape as preview."""
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
```

Add test that calls it at small nsim and confirms it produces more trials than preview.

Commit: `feat(api): forward_sim_full for production-scale single-cond simulation`.

---

### Task 7.A.4: fit_simplex_b callback hook

**Files:**
- Modify: `model_b/fit.py` — add `on_update` callback parameter

**Step 1: Add test in `model_b/tests/test_fit_b_callback.py`**

```python
"""Test that fit_simplex_b can call a per-eval callback."""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import simulate as sim_b
from model_b import fit as fit_b
from model_b.objective import COND_MAP_B, clamp_b
from shared import prng


TRUE = jnp.array([200.0, 50.0, 10.0, 2.0, 12.0, 10.0, 0.5,
                  15.0, 10.0, 8.0, 14.0, 11.0, 9.0])


def _tiny_synth(key, nsim=32, chunk_size=8):
    p = clamp_b(TRUE)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    props, counts, quants = [], [], []
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(key, ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
            sis, sig, si, nsim=nsim, chunk_size=chunk_size,
        )
        cat_np = np.asarray(cat); rt_np = np.asarray(rt)
        props.append(jnp.asarray([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)]))
        counts.append(jnp.asarray([(cat_np == c).sum() for c in (1, 2, 3, 4, 5)], dtype=jnp.int64))
        q = np.zeros((5, 5))
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat_np == c
            if mask.sum() >= 5: q[:, ki] = np.quantile(rt_np[mask], qs)
        quants.append(jnp.asarray(q))
    return {"prop": jnp.stack(props), "count": jnp.stack(counts), "quant": jnp.stack(quants)}


def test_fit_simplex_b_invokes_callback():
    """fit_simplex_b should call on_update(eval_n, loss, x) every eval."""
    data = _tiny_synth(jax.random.key(0))
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
```

**Step 2: Run, expect fail (kwarg unknown).**

**Step 3: Add `on_update` to `fit_simplex_b` in `model_b/fit.py`**

```python
def fit_simplex_b(data, key, x0, nsim: int = 256, maxiter: int = 2000,
                  tol: float = 1e-7, chunk_size: int = 4,
                  use_kl: bool = False,
                  on_update=None):
    """..."""
    from scipy.optimize import minimize

    n_evals = [0]

    def loss_numpy(p_np):
        n_evals[0] += 1
        p = jnp.asarray(p_np)
        val = obj_b.fofs_b_new(p, data, key, nsim=nsim,
                                chunk_size=chunk_size, use_kl=use_kl)
        val_f = float(val)
        if on_update is not None:
            on_update(n_evals[0], val_f, p_np.copy())
        return val_f

    res = minimize(
        loss_numpy, np.asarray(x0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": tol, "fatol": tol, "disp": False},
    )
    return FitResult(
        params=jnp.asarray(res.x),
        loss=float(res.fun),
        n_iters=int(res.nit),
        converged=bool(res.success),
    )
```

**Step 4: Run test, expect pass.**

**Step 5: Commit** `feat(fit): on_update callback for live per-eval progress`.

---

### Task 7.A.5: fit_model wrapper

Append to `model_b/api.py`:

```python
from model_b import fit as fit_b
from model_b.objective import COND_MAP_B, clamp_b


def fit_model(data: dict, x0: list[float],
              nsim: int | None = None,
              chunk_size: int | None = None,
              maxiter: int = 100,
              on_update=None) -> dict:
    """Run fit_simplex_b with optional per-eval callback.

    data : {"prop": (2,5), "count": (2,5), "quant": (2,5,5)} — lists not arrays
    x0   : 13-element list of floats
    on_update(eval_n, loss, x_list) callable, optional
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
        "loss": res.loss,
        "n_iters": res.n_iters,
        "converged": res.converged,
    }
```

Test that it returns the expected shape with a tiny synthetic-recovery setup. Commit.

---

### Task 7.A.6: predict_from_params

Append to `model_b/api.py`:

```python
def predict_from_params(params_full: list[float], n_conditions: int = 2,
                        nsim: int = 1024, key_seed: int = 0) -> dict:
    """Generate per-condition RT distributions from fitted params.

    params_full : 13-element list (ter, st, cr, crsd, sis, sig, sv, av1c1, ..., av3c2)
    Returns: {"by_condition": [{"rt": [...], "cat": [...], "props": [...]}, ...]}
    """
    p = clamp_b(jnp.asarray(params_full))
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
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
        props = [float((cat_arr == c).mean()) for c in (1, 2, 3, 4, 5)]
        sub["props"] = props
        out.append(sub)
    return {"by_condition": out}
```

Test and commit.

---

## Phase 7.B.1 — FastAPI backend

Eight tasks. New `backend/` directory at repo root.

### Task 7.B.1.1: FastAPI skeleton

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/main.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_main.py`
- Modify: `pyproject.toml` (add `[project.optional-dependencies] backend = [...]`)

**Step 1: Failing test**

```python
from fastapi.testclient import TestClient
from backend.main import app


def test_root_returns_html():
    """Root path serves the React bundle (or a placeholder)."""
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

**Step 2: Implement `backend/main.py`**

```python
"""FastAPI app for Stage 7 Ratcliff DDM web interface."""
from fastapi import FastAPI

app = FastAPI(title="Ratcliff DDM API", version="0.7.0")


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

Add `[project.optional-dependencies] backend = ["fastapi>=0.115", "uvicorn>=0.30", "python-multipart>=0.0.9"]` to `pyproject.toml`.

**Step 3: Install + test**

```powershell
.venv/Scripts/pip.exe install fastapi uvicorn python-multipart
.venv/Scripts/python.exe -m pytest backend/tests/test_main.py -v
```

**Step 4: Commit** `feat(backend): FastAPI skeleton with health endpoint`.

---

### Task 7.B.1.2: /simulate endpoint

Add Pydantic request/response models and the endpoint:

```python
from pydantic import BaseModel
from model_b import api as model_api


class SimulateRequest(BaseModel):
    params: dict
    key_seed: int = 0
    full: bool = False
    nsim: int | None = None


class SimulateResponse(BaseModel):
    rt: list[float]
    cat: list[int]


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    if req.full:
        out = model_api.forward_sim_full(req.params, nsim=req.nsim or 9000, key_seed=req.key_seed)
    else:
        out = model_api.forward_sim_preview(req.params, key_seed=req.key_seed)
    return out
```

Test that posting default params returns valid RTs. Commit.

---

### Task 7.B.1.3: /upload + data parsers

**File:** `backend/parsers.py`

Implement two parsers:
- `parse_twod3datanew(text)` — reproduce the format from `shared/data_io.py::load_twod3datanew`
- `parse_csv(text)` — generic CSV with columns RT, response_cat, condition, subject_id (optional)

Returns `{"prop": [[...]], "count": [[...]], "quant": [[[...]]]}` JSON shape.

`/api/upload` endpoint takes a `UploadFile`, tries each parser, returns parsed data or 400 error.

Tests cover both formats + the failure case. Commit.

---

### Task 7.B.1.4: /fit/start + BackgroundTask + jobs dict

**File:** `backend/jobs.py`

```python
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class JobStatus:
    job_id: str
    status: str = "pending"   # pending | running | done | error
    progress: list[dict] = field(default_factory=list)   # [{eval_n, loss}]
    result: dict | None = None
    error: str | None = None


_jobs: dict[str, JobStatus] = {}
_lock = Lock()


def new_job() -> JobStatus:
    job = JobStatus(job_id=str(uuid.uuid4()))
    with _lock:
        _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)


def update_progress(job_id: str, eval_n: int, loss: float):
    job = _jobs.get(job_id)
    if job:
        job.progress.append({"eval_n": eval_n, "loss": loss})


def set_result(job_id: str, result: dict):
    job = _jobs.get(job_id)
    if job:
        job.result = result
        job.status = "done"
```

In `main.py`:

```python
from fastapi import BackgroundTasks
from backend import jobs


class FitStartRequest(BaseModel):
    data: dict
    x0: list[float]
    maxiter: int = 100


@app.post("/api/fit/start")
def fit_start(req: FitStartRequest, bg: BackgroundTasks):
    job = jobs.new_job()
    bg.add_task(_run_fit, job.job_id, req.data, req.x0, req.maxiter)
    return {"job_id": job.job_id}


def _run_fit(job_id, data, x0, maxiter):
    try:
        jobs._jobs[job_id].status = "running"
        result = model_api.fit_model(
            data, x0, maxiter=maxiter,
            on_update=lambda n, loss, x: jobs.update_progress(job_id, n, loss),
        )
        jobs.set_result(job_id, result)
    except Exception as e:
        jobs._jobs[job_id].status = "error"
        jobs._jobs[job_id].error = str(e)


@app.get("/api/fit/status/{job_id}")
def fit_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(404, "job not found")
    return {"status": job.status, "progress": job.progress, "error": job.error}


@app.get("/api/fit/result/{job_id}")
def fit_result(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(404, "job not found")
    if job.status != "done":
        return {"status": job.status}
    return {"status": "done", "result": job.result}
```

Integration test: start a tiny fit, poll until done, check result shape. Mark slow. Commit.

---

### Task 7.B.1.5: /predict endpoint

Wraps `model_api.predict_from_params`. Pydantic request takes `params: list[float]` and `n_conditions: int = 2`. Response wraps the API output. Test and commit.

---

### Task 7.B.1.6: Static file serving

Once `frontend/dist` exists (after 7.B.2), add:

```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
```

This serves the React SPA at the root.

---

### Task 7.B.1.7: Run + smoke

```powershell
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000/api/health` in browser, expect `{"status":"ok"}`. Commit a short `README_backend.md` documenting this.

---

### Task 7.B.1.8: BYO-GPU stub

`backend/byo_gpu.py`:

```python
import httpx


def forward_to_remote(endpoint: str, path: str, body: dict) -> dict:
    """Forward a request to a remote backend running on a GPU."""
    with httpx.Client(timeout=600.0) as client:
        r = client.post(f"{endpoint}{path}", json=body)
        r.raise_for_status()
        return r.json()
```

Add an optional `gpu_endpoint: str | None = None` field to `FitStartRequest`. When present, the backend instead POSTs to `{gpu_endpoint}/api/fit/start` and proxies polling. Test with a mock httpx client. Commit.

---

## Phase 7.B.2 — React frontend (Forward Sim tab only, v1)

Seven tasks. The Fit/Predict/Compare tabs follow the same patterns once Forward Sim works.

### Task 7.B.2.1: Vite + TS + Tailwind scaffold

```powershell
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install @tanstack/react-query zustand plotly.js react-plotly.js
```

Configure `tailwind.config.js` content paths. Add basic `index.css` imports. Test `npm run dev` opens at localhost:5173. Commit.

### Task 7.B.2.2: TypeScript types + API client

`frontend/src/lib/types.ts`:

```typescript
export interface ParamSet {
  ter: number; st: number; cr: number; crsd: number;
  sis: number; sig: number;
  av1: number; av2: number; av3: number;
}

export const DEFAULT_PARAMS: ParamSet = {
  ter: 200, st: 50, cr: 10, crsd: 2,
  sis: 12, sig: 10,
  av1: 15, av2: 10, av3: 8,
};

export interface SimulateResponse {
  rt: number[];
  cat: number[];
}
```

`frontend/src/lib/api.ts`:

```typescript
import { ParamSet, SimulateResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE || "/api";

export async function postSimulate(params: ParamSet, keySeed = 0,
                                    full = false): Promise<SimulateResponse> {
  const r = await fetch(`${BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params, key_seed: keySeed, full }),
  });
  if (!r.ok) throw new Error(`simulate failed: ${r.statusText}`);
  return r.json();
}
```

Configure Vite dev proxy: `vite.config.ts` adds `server.proxy: { "/api": "http://localhost:8000" }`.

### Task 7.B.2.3: ParamSliders component

`frontend/src/components/ParamSliders.tsx` — 13 sliders (use 9 for forward-sim, the additional drifts for the second condition show on the Compare tab). Each slider has a label, range, step, numeric input, and a small "?" tooltip describing the parameter.

Uses Zustand for state so the value is shared across tabs.

### Task 7.B.2.4: RTHistogram component

```typescript
import Plot from "react-plotly.js";

export function RTHistogram({ rt, cat }: { rt: number[]; cat: number[] }) {
  const byCategory = [1, 2, 3, 4, 5].map((c) => ({
    name: `Cat ${c}`,
    x: rt.filter((_, i) => cat[i] === c),
    type: "histogram" as const,
    opacity: 0.6,
    nbinsx: 30,
  }));
  return (
    <Plot
      data={byCategory}
      layout={{
        barmode: "overlay",
        xaxis: { title: "RT (ms)" },
        yaxis: { title: "Count" },
        title: "Reaction time distribution by category",
        height: 400,
      }}
      style={{ width: "100%" }}
    />
  );
}
```

### Task 7.B.2.5: ForwardSimTab — assembly + debounced API calls

Combine `ParamSliders` + `RTHistogram` + a debounced effect that calls `postSimulate` 200ms after the last slider change. Show a loading indicator during the API call.

### Task 7.B.2.6: ConfigSidebar (localStorage save/load)

`frontend/src/lib/configs.ts`:

```typescript
const KEY = "ratcliff_configs";

export interface SavedConfig {
  name: string;
  timestamp: number;
  params: ParamSet;
  notes?: string;
}

export function loadAll(): SavedConfig[] {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]"); }
  catch { return []; }
}

export function save(cfg: SavedConfig) {
  const all = loadAll().filter(c => c.name !== cfg.name);
  all.push(cfg);
  localStorage.setItem(KEY, JSON.stringify(all));
}

export function remove(name: string) {
  const all = loadAll().filter(c => c.name !== name);
  localStorage.setItem(KEY, JSON.stringify(all));
}

export function exportAll(): string {
  return JSON.stringify(loadAll(), null, 2);
}

export function importJson(json: string) {
  const arr = JSON.parse(json) as SavedConfig[];
  localStorage.setItem(KEY, JSON.stringify(arr));
}
```

Sidebar component lists configs, has Save/Load/Export/Import buttons. Persists across tabs and page reloads.

### Task 7.B.2.7: Smoke + commit

```powershell
cd frontend && npm run build
```

Confirm `frontend/dist/` exists. Backend now serves it. Visit `http://localhost:8000/` and verify the Forward Sim tab loads + sliders update plot.

Commit the entire frontend.

---

## Phase 7.B.3 — Remaining tabs (skeleton outline)

Same patterns as 7.B.2. Each tab is one new component + corresponding API calls.

- **FitTab**: DataUpload (drag-drop, calls `/api/upload`), FitConfig (init params, maxiter), StartFitButton, FitProgress (polls `/api/fit/status/{id}` every 2s, plots loss curve), RecoveryTable (shown after `/api/fit/result/{id}` returns).
- **PredictTab**: load a SavedConfig, choose number of conditions, click "Predict", show per-condition RT distributions.
- **CompareTab**: pick configs A + B from saved list, run both simulations server-side, overlay histograms with two-color hues.

Each task: ~1-2 days. TDD where possible; for plotting, eyeball smoke tests are fine.

---

## Phase 7.C — Deployment

### Task 7.C.1: Dockerfile

```dockerfile
# Stage 1: build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY model_a/ model_a/
COPY model_b/ model_b/
COPY shared/ shared/
COPY backend/ backend/
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist
RUN pip install --no-cache-dir -e ".[fit,backend]"
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Task 7.C.2: HF Space config

Create `Spacefile` or use `README.md` frontmatter:

```yaml
---
title: Ratcliff DDM
emoji: 🧠
sdk: docker
app_port: 7860
---
```

Push to a new HF Space, wait for build, smoke-test the public URL.

### Task 7.C.3: README badge + cite block

Add a "Try it live" badge linking to the HF Space. Add a citation snippet to the repo README.

---

## Phase 7.D — BYO-GPU (v1: documented pattern)

Already stubbed in Task 7.B.1.8. Polish:

- Frontend Fit tab gets a `BYOGPUSettings` panel: text input for endpoint URL, "Test connection" button.
- Backend `/api/fit/start` forwards to remote when `gpu_endpoint` is set.
- Docs section: "Bringing your own GPU" — how to clone, run the same Docker container on a GPU box, plug the URL in.

---

## Success criteria recap

- [ ] All Phase 7.A tasks complete, tests green
- [ ] All Phase 7.B.1 tasks complete, tests green
- [ ] Phase 7.B.2 ForwardSimTab works end-to-end against local backend
- [ ] HF Space deployed and accessible at a public URL
- [ ] README updated with citation + live demo badge
- [ ] BYO-GPU docs + working with a separate GPU instance

---

## Execution options

**1. Subagent-driven (this session):** I dispatch fresh subagent per task, review between tasks. Stay in this session. Recommended for autonomous execution since user said "full go ahead no permissions."

**2. Parallel session:** Open new session with executing-plans, batch execution with checkpoints. Not what user requested.

**Picking 1 by default per user instruction.**

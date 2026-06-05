# Stage 7 Design — CPU-Friendly Mode + React/FastAPI Web Interface

**Date:** 2026-06-05
**Status:** APPROVED (autonomous-mode go-ahead from user)
**Scope:** Two deliverables, in order: (1) a CPU-friendly default path for the simulator + an API wrapper layer, and (2) a React + FastAPI web app for interactive forward simulation, fitting, prediction, and comparison. Both intended for hosted public use on HF Spaces with optional BYO-GPU power mode.

---

## 1. Motivation

Ratcliff's feedback after seeing Stage 6 results:
- "Make a separate version that can be CPU-friendly" — must be *runnable* on a laptop without a GPU. He said "a couple of hours is fine" for a full fit, so we don't need new optimization, just sensible defaults.
- "Make a digital interface to do fits and parameters and generate predictions" — a UI where users can play with sliders, upload data, run fits, see predictions, save/load configurations.

He also estimated a methods paper is doable in a month, so Stage 7 needs to ship in time to be the **deployable artifact cited from the paper**. The codebase becomes a real tool the community can use, not just a benchmarking record.

Research precedents that confirm the path:
- [PyDDM](https://pyddm.readthedocs.io/) has an interactive GUI for the standard 1D DDM. Same UX, different model.
- [dockerHDDM (Pan et al. 2025)](https://journals.sagepub.com/doi/10.1177/25152459241298700) is a recent paper specifically about making HDDM user-friendly — proves the field rewards usability work.
- [HF Spaces FastAPI+React template](https://huggingface.co/spaces/SpacesExamples/fastapi-react-app) gives us a working deployment scaffold.

## 2. Goals

1. **CPU mode**: `simulate_b(...)` runs on laptop CPU with sensible defaults (`nsim=512`, `chunk_size=4`, `use_kl=False`) and completes a forward sim in ~1-3 sec, a full fit in 1-2 hours.
2. **API layer**: `model_b/api.py` exposes 4 wrapper functions (`forward_sim_preview`, `forward_sim_full`, `fit_model`, `predict_from_params`) that the FastAPI backend calls. Decoupled from the JAX internals.
3. **Backend (FastAPI)**: 6 endpoints (`/simulate`, `/fit/start`, `/fit/status/{id}`, `/fit/result/{id}`, `/predict`, `/upload`). Stateless — no DB. Background tasks for long fits. Polling for status. Works on HF Spaces free tier.
4. **Frontend (React)**: 4-tab single-page app (Forward Sim, Fit, Predict, Compare) with 13 parameter sliders, drag-drop data upload, live plot updates, save/load named configs in `localStorage`.
5. **Deployment**: one HF Space, Docker-based, FastAPI serving both API and the built React bundle. Public URL, citable from the methods paper.
6. **BYO-GPU mode**: optional UI field where a user pastes a GPU endpoint URL (e.g., their own HF Jobs token, RunPod endpoint, or local GPU server). Backend routes fits to that endpoint instead of running locally on the Space's CPU.

## 3. Non-goals

- No user accounts / auth — save/load is browser-local via `localStorage`. Cross-user sharing is via JSON export/import.
- No multi-tenant queue. HF Space CPU is single-process; concurrent fits queue naturally.
- No new optimization work on the simulator itself. CPU mode reuses Stage 5 FFT path.
- No hierarchical Bayesian fitting in the UI — that's a research project for later.
- No real-time neural data integration — single-modal RT + category data only.
- No model variants other than Model B 2D GRF. Model A (1D) gets the same wrapper but isn't UI-exposed in v1.

## 4. Architecture

### 4.1 Component diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                       HF Space (Docker, CPU)                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  FastAPI backend  (Uvicorn on port 7860 — HF Space default)     │ │
│  │                                                                 │ │
│  │  POST /simulate          → returns RTs + category counts        │ │
│  │  POST /fit/start         → returns job_id, starts BackgroundTask│ │
│  │  GET  /fit/status/{id}   → returns progress + per-eval loss     │ │
│  │  GET  /fit/result/{id}   → returns final params + recovery info │ │
│  │  POST /predict           → predictions from named param set     │ │
│  │  POST /upload            → parses uploaded data, returns trials │ │
│  │  GET  /                  → serves React static bundle           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│           │                                                          │
│           │ imports                                                  │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  model_b/api.py  (Stage 7.A — wrapper layer)                    │ │
│  │                                                                 │ │
│  │  forward_sim_preview(params)         → small nsim, fast         │ │
│  │  forward_sim_full(params, nsim)       → user-specified nsim     │ │
│  │  fit_model(data, x0, on_update)       → fit with callback       │ │
│  │  predict_from_params(params, conds)   → per-condition forecasts │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│           │                                                          │
│           │ uses                                                     │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Existing JAX runtime                                           │ │
│  │  - simulate.py, objective.py, fit.py, grf.py, grf_kl.py         │ │
│  │  - Device auto-detect picks FFT vs K-L path                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                       Optional BYO-GPU compute
                                       (RunPod, HF Jobs, local box)
```

### 4.2 Data flow — forward simulation (live preview)

1. User drags a slider in the React Forward Sim tab.
2. Debounced 200ms → `POST /simulate` with full 13-param JSON body.
3. FastAPI calls `api.forward_sim_preview(params)` which runs `simulate_b(..., nsim=256, chunk_size=4, use_kl=False)`.
4. Result: `{ rt: [...], cat: [...] }` returned to frontend.
5. Frontend re-renders Plotly histogram. Total wall-clock: ~1-3 sec on HF Space CPU.

### 4.3 Data flow — fit (background job)

1. User uploads data file on Fit tab. `POST /upload` returns parsed `{prop, count, quant}` JSON.
2. User clicks "Start Fit". `POST /fit/start` with data + init params + chosen backend (local CPU vs BYO-GPU URL).
3. Backend assigns `job_id = uuid4()`, starts a FastAPI `BackgroundTask` running `api.fit_model(data, x0, on_update)`.
4. `on_update` callback writes `(eval_n, loss, current_params)` to an in-memory dict keyed by `job_id`. (HF Space `/tmp` is writable so we *could* persist, but in-memory is simpler.)
5. Frontend polls `GET /fit/status/{job_id}` every 2 sec → renders live loss curve + ETA.
6. When fit completes, status switches to `done` and `GET /fit/result/{job_id}` returns the final params + recovery table.

### 4.4 Save/load named configs

- Browser `localStorage` keyed by `"ratcliff_configs"` → JSON array of `{ name, timestamp, params: { ter, st, cr, ... }, notes }`.
- Sidebar component lists all saved configs.
- "Save current" button: prompts for a name, stores current sliders.
- "Load" button: populates sliders from the named config.
- "Export" button: downloads JSON file for sharing.
- "Import" button: upload JSON, merges into localStorage.

No backend DB. Survives page refresh. Lost only if user clears browser data.

## 5. Component breakdown

### 5.1 Backend (FastAPI)

**Files (new):**

```
backend/
├── main.py              # FastAPI app, routes, static React bundle
├── jobs.py              # in-memory job dict + helper functions
├── parsers.py           # twod3datanew + CSV → internal dict
├── byo_gpu.py           # routes fit to external GPU endpoint if URL given
├── requirements.txt     # fastapi, uvicorn, python-multipart, plus project deps
└── tests/
    ├── test_simulate_endpoint.py
    ├── test_fit_endpoint.py
    └── test_parsers.py
```

**Key choices:**
- FastAPI's `BackgroundTasks` for fits (no Celery/Redis — keep it simple)
- In-memory job dict (`dict[str, JobStatus]`). On Space restart, in-progress fits lost. Acceptable for v1.
- CORS enabled for development; production serves React from same origin so CORS is moot.
- Static React bundle served from `/` route via `StaticFiles`.

### 5.2 Frontend (React + Vite)

**Files (new):**

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── src/
│   ├── main.tsx               # entry, routes
│   ├── App.tsx                # tab navigation
│   ├── tabs/
│   │   ├── ForwardSimTab.tsx
│   │   ├── FitTab.tsx
│   │   ├── PredictTab.tsx
│   │   └── CompareTab.tsx
│   ├── components/
│   │   ├── ParamSliders.tsx   # 13 sliders, debounced onChange
│   │   ├── RTHistogram.tsx    # Plotly histogram per category
│   │   ├── ConfigSidebar.tsx  # localStorage save/load list
│   │   ├── DataUpload.tsx     # drag-drop, file type detection
│   │   ├── FitProgress.tsx    # live loss curve + ETA
│   │   └── RecoveryTable.tsx  # final-fit param recovery
│   ├── lib/
│   │   ├── api.ts             # TanStack Query hooks
│   │   ├── configs.ts         # localStorage helpers
│   │   └── types.ts           # shared types
│   └── styles.css
└── public/
```

**Stack:**
- React 18 + Vite + TypeScript
- TailwindCSS
- Plotly.js (interactive scientific plots)
- TanStack Query (API state)
- Zustand (client state, e.g., current sliders)

### 5.3 Backend wrapper layer (Stage 7.A — CPU mode)

**File: `model_b/api.py`**

```python
def forward_sim_preview(params: dict, key_seed: int = 0) -> dict:
    """Fast small-nsim simulation for slider previews. CPU-friendly."""

def forward_sim_full(params: dict, nsim: int = 9000, key_seed: int = 0) -> dict:
    """Production-scale simulation. Auto-picks FFT vs K-L by device."""

def fit_model(data: dict, x0: list[float], maxiter: int = 100,
              on_update: Callable | None = None) -> dict:
    """Run fit_simplex_b with optional per-eval callback for live progress."""

def predict_from_params(params: dict, conditions: list[dict]) -> dict:
    """Generate per-condition predicted RT distributions from fitted params."""

def _get_device_defaults() -> dict:
    """Auto-detect CPU vs GPU and return appropriate nsim/chunk/use_kl."""
```

Plus a small change to `fit_simplex_b` to accept an `on_update(eval_n, loss, x)` callback (one line).

### 5.4 BYO-GPU routing

**File: `backend/byo_gpu.py`**

When the user provides a GPU endpoint URL in the Fit tab:
- Frontend includes `gpu_endpoint` in the `POST /fit/start` body.
- Backend, instead of running `fit_model` locally, sends the same args to `{gpu_endpoint}/fit` over HTTP.
- Polls the remote endpoint for status the same way the frontend polls locally.

The "endpoint" is assumed to be another instance of the same backend running on a GPU (could be a separate HF Space with GPU, a RunPod container, or a local server). Documented pattern: clone this repo, run `python backend/main.py` on a GPU machine, give the URL to other users.

For v1 we ship this as a *documented* pattern with one example deployment. Full UX (e.g., HF Jobs token UI) is v2.

## 6. Deployment

### 6.1 HF Space layout

A single Space, Docker-based, builds React + serves via FastAPI:

```
Dockerfile
├── Stage 1: Node 20 → install frontend deps, run `npm run build`, output static bundle
├── Stage 2: Python 3.12 + JAX (CPU) + project deps + copy built frontend
└── CMD: uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

The Space URL becomes `https://huggingface.co/spaces/<your-username>/ratcliff-ddm`. Citable in the methods paper as the reference implementation.

### 6.2 Local development

```bash
# Backend
pip install -e ".[dev,fit,backend]"
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # port 5173, proxies API to 8000
```

### 6.3 BYO-GPU mode

User runs the same Docker container on their GPU box:
```bash
docker run -p 8000:7860 ratcliff-ddm
```
Then enters `http://their.gpu.box:8000` into the BYO-GPU field on the public Space.

## 7. Error handling

- **Invalid params**: backend validates against `clamp_b` bounds, returns 400 with explanation.
- **Bad data upload**: parser tries `twod3datanew` first, falls back to CSV, returns 400 with "couldn't parse — expected columns: ..." if neither works.
- **Fit crashes**: `BackgroundTask` catches exceptions, sets `JobStatus.error = str(e)`. Frontend renders the error in the FitProgress component.
- **HF Space cold start**: first request after idle takes ~30 sec for JAX to warm up. Frontend shows a "Spinning up the backend..." overlay on first call.
- **BYO-GPU endpoint unreachable**: backend returns 503, frontend shows an actionable message ("Couldn't reach your GPU at <url>. Is the server running?").

## 8. Testing

- **Backend unit tests**: each endpoint has a test using FastAPI's `TestClient`. Mock the JAX layer where slow.
- **Backend integration tests**: one end-to-end test that hits `/simulate` then `/fit/start` then polls to completion. Skipped unless `RUN_SLOW=1`.
- **Frontend smoke tests**: Vitest + React Testing Library for component rendering. Playwright for one end-to-end (drag a slider, see plot update).
- **API contract tests**: shared TypeScript types match backend Pydantic models. Generated via `openapi-typescript` from FastAPI's OpenAPI spec.

## 9. Success criteria

1. Public HF Space deploys and serves the React app at a URL.
2. Forward sim slider drag → plot updates in < 3 sec on HF Space CPU.
3. Full fit (nsim=512, maxiter=100) completes on HF Space CPU in < 2 hours.
4. BYO-GPU mode: with a separate GPU instance running, a fit completes in < 5 min using the K-L path.
5. Save/load configs survive page refresh.
6. Data upload accepts both `twod3datanew` format and generic CSV.
7. The Space URL is suitable to cite in the methods paper.
8. All backend endpoints have tests; all React components have at least a render smoke test.

## 10. Implementation order

1. **Stage 7.A**: `model_b/api.py` wrapper layer + tests. Pure Python, no UI. ~3-5 days.
2. **Stage 7.B.1**: FastAPI backend skeleton + 6 endpoints + tests. Local dev only. ~5-7 days.
3. **Stage 7.B.2**: React frontend — Forward Sim tab + ParamSliders + RTHistogram + ConfigSidebar. Local dev only. ~5-7 days.
4. **Stage 7.B.3**: Fit tab + Predict tab + Compare tab + DataUpload + FitProgress. ~5-7 days.
5. **Stage 7.C**: Dockerfile + HF Space deployment + smoke test on the deployed URL. ~2-3 days.
6. **Stage 7.D**: BYO-GPU routing + docs. ~2-3 days.

Total estimated effort: **~3-4 weeks** of focused work. Fits inside Ratcliff's 1-month paper timeline if you start now.

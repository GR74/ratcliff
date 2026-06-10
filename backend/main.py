"""FastAPI app for the Ratcliff DDM web interface.

Endpoints (all under /api):
  GET  /api/health                  health probe
  POST /api/simulate                forward simulation (preview or full)
  POST /api/upload                  parse uploaded data file
  POST /api/fit/start               kick off a background fit, returns job_id
  GET  /api/fit/status/{job_id}     poll for live progress
  GET  /api/fit/result/{job_id}     fetch final result (when status == done)
  POST /api/predict                 generate predictions from a parameter set

Static React bundle is served from /  when it exists at frontend/dist.
"""
# CRITICAL: enable x64 BEFORE any JAX import. The simulator code explicitly
# requests float64 / complex128 dtypes; if x64 isn't enabled, JAX silently
# truncates to fp32 / complex64 and the simulator produces numerically wrong
# output. Must come before any `from model_b import ...` line.
import jax
jax.config.update("jax_enable_x64", True)

import threading
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import jobs, parsers, byo_gpu
from model_b import api as model_api


app = FastAPI(title="Ratcliff DDM API", version="0.8.0")

# Warm-up state: the first simulate call JIT-compiles for minutes on CPU. We
# kick that off in a background thread at startup so the first real user request
# hits a warm cache instead of waiting.
_WARM = {"ready": False, "error": None}


@app.on_event("startup")
def _prewarm() -> None:
    def _run():
        try:
            model_api.warmup()
            _WARM["ready"] = True
        except Exception as e:  # pragma: no cover - defensive
            _WARM["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()

# CORS for local dev (Vite at 5173 talking to backend at 8000). In production
# the same FastAPI process serves both the API and the static bundle, so this
# is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- /api/health -------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "warm": _WARM["ready"], "warm_error": _WARM["error"]}


# ---- /api/simulate -----------------------------------------------------

class SimulateRequest(BaseModel):
    params: dict
    key_seed: int = 0
    full: bool = False
    nsim: Optional[int] = None
    chunk_size: Optional[int] = None


class SimulateResponse(BaseModel):
    rt: list[float]
    cat: list[int]


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    try:
        if req.full:
            out = model_api.forward_sim_full(
                req.params,
                nsim=req.nsim or 9000,
                chunk_size=req.chunk_size,
                key_seed=req.key_seed,
            )
        else:
            out = model_api.forward_sim_preview(req.params, key_seed=req.key_seed)
    except KeyError as e:
        raise HTTPException(400, f"missing required param: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"simulate failed: {e}") from e
    return SimulateResponse(**out)


# ---- /api/upload -------------------------------------------------------

class UploadResponse(BaseModel):
    prop: list[list[float]]
    count: list[list[int]]
    quant: list[list[list[float]]]
    n_subjects: int


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(400, f"file must be UTF-8 text: {e}") from e
    try:
        data = parsers.parse_auto(text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return UploadResponse(**data)


# ---- /api/fit/* --------------------------------------------------------

class FitStartRequest(BaseModel):
    data: dict
    x0: list[float]
    maxiter: int = 100
    nsim: Optional[int] = None
    chunk_size: Optional[int] = None
    gpu_endpoint: Optional[str] = None  # if set, proxy to remote BYO-GPU


@app.post("/api/fit/start")
def fit_start(req: FitStartRequest, bg: BackgroundTasks) -> dict:
    # BYO-GPU: proxy the entire request to the remote endpoint and return
    # whatever it gives us (which should include a job_id).
    if req.gpu_endpoint:
        try:
            return byo_gpu.forward_post(
                req.gpu_endpoint, "/api/fit/start", req.model_dump()
            )
        except Exception as e:
            raise HTTPException(503, f"BYO-GPU endpoint unreachable: {e}") from e

    job = jobs.new_job()
    bg.add_task(
        _run_fit_local,
        job.job_id,
        req.data,
        req.x0,
        req.maxiter,
        req.nsim,
        req.chunk_size,
    )
    return {"job_id": job.job_id}


def _run_fit_local(
    job_id: str,
    data: dict,
    x0: list[float],
    maxiter: int,
    nsim: Optional[int],
    chunk_size: Optional[int],
) -> None:
    """The actual local fit. Runs inside FastAPI's BackgroundTask thread."""
    try:
        jobs.set_status(job_id, "running")
        result = model_api.fit_model(
            data,
            x0,
            nsim=nsim,
            chunk_size=chunk_size,
            maxiter=maxiter,
            on_update=lambda n, loss, _x: jobs.update_progress(job_id, n, loss),
        )
        jobs.set_result(job_id, result)
    except Exception as e:
        jobs.set_error(job_id, str(e))


@app.get("/api/fit/status/{job_id}")
def fit_status(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job not found: {job_id}")
    return {
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
    }


@app.get("/api/fit/result/{job_id}")
def fit_result(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job not found: {job_id}")
    if job.status == "error":
        return {"status": "error", "error": job.error}
    if job.status != "done":
        return {"status": job.status}
    return {"status": "done", "result": job.result}


# ---- /api/predict ------------------------------------------------------

class PredictRequest(BaseModel):
    params_full: list[float]
    n_conditions: int = 2
    nsim: int = 1024
    key_seed: int = 0


class PredictResponse(BaseModel):
    by_condition: list[dict]


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if len(req.params_full) != 13:
        raise HTTPException(400, f"params_full must have 13 entries, got {len(req.params_full)}")
    if req.n_conditions not in (1, 2):
        raise HTTPException(400, "n_conditions must be 1 or 2")
    try:
        out = model_api.predict_from_params(
            req.params_full,
            n_conditions=req.n_conditions,
            nsim=req.nsim,
            key_seed=req.key_seed,
        )
    except Exception as e:
        raise HTTPException(500, f"predict failed: {e}") from e
    return PredictResponse(**out)


# ---- /api/field --------------------------------------------------------

class FieldRequest(BaseModel):
    params: dict
    mode: str = "single"        # "single" or "mean"
    n_frames: int = 48
    n_trials_mean: int = 64
    grid_stride: int = 2
    key_seed: int = 0


@app.post("/api/field")
def field(req: FieldRequest) -> dict:
    if req.mode not in ("single", "mean"):
        raise HTTPException(400, "mode must be 'single' or 'mean'")
    if not (1 <= req.n_frames <= 120):
        raise HTTPException(400, "n_frames must be in 1..120")
    try:
        return model_api.field_snapshots(
            req.params,
            mode=req.mode,
            n_frames=req.n_frames,
            n_trials_mean=req.n_trials_mean,
            grid_stride=req.grid_stride,
            key_seed=req.key_seed,
        )
    except KeyError as e:
        raise HTTPException(400, f"missing required param: {e}") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"field failed: {e}") from e


# ---- Static React bundle (mounted last so it doesn't shadow /api/*) ----

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

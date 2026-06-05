# Ratcliff DDM (JAX)

JAX port of Roger Ratcliff's spatially-extended diffusion decision models, with a Karhunen-Loève low-rank GRF generator and a web UI for interactive forward simulation, fitting, prediction, and comparison.

- **Model A** — 1D, `N=72` accumulators, Cholesky-correlated noise (`reference/gpgsq5deg3twod24.f`)
- **Model B** — 2D, `100×160` accumulator field, FFT-based circulant-embedding GRF (`reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`), with Stage 6 K-L low-rank alternative

## Headline benchmark (single rented H100, 2026-06-05)

| Metric | Original Fortran 6-node MPI | This codebase (Stage 6 K-L) |
|---|---|---|
| Per-call simulator at nsim=9000 | 11 sec | **4.7 sec** (~5×) |
| Per-call simulator (1-node Fortran) | 36 sec | 4.7 sec (~16×) |
| Full parameter recovery fit | "hours and hours" | **20.1 min** (~10-30×) |
| Cost per fit | cluster time + queue | ~$0.80 of rented GPU |
| All 12 active params within 7% of truth | yes | yes (avg 2.98% error) |

Original Fortran ran on a 6-node MPI cluster with hand-tuned legacy code. The JAX port runs on a single rented GPU on the public cloud at ~$2/hr.

## Interactive web interface

The `frontend/` + `backend/` directories implement a React + FastAPI app for interactive use:

- **Forward Sim** — drag 9 parameter sliders, see RT distributions update in real time
- **Fit** — upload data (twod3datanew or generic CSV), watch the simplex loss descend live
- **Predict** — generate per-condition RT distributions from fitted params
- **Compare** — overlay two saved configurations

Deploy as a single HF Space Docker image — see `HF_SPACE_README.md` for the publishing front-matter.

## Quick start (local development)

Requires Python 3.11+ and Node 20+ (for the frontend).

### Backend + JAX

```bash
python -m venv .venv
source .venv/bin/activate    # (or .venv/Scripts/Activate.ps1 on Windows PowerShell)
pip install -e ".[dev,fit,backend]"

# verify
pytest model_b/tests/test_api_smoke.py -v -m "not slow"
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000/api/health` to confirm the backend is up.

### Frontend (separate terminal)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` to the backend at `:8000`.

### Build for production (FastAPI serves the static bundle)

```bash
cd frontend && npm run build
# Then uvicorn backend.main:app picks up frontend/dist automatically.
```

### Docker (everything in one container)

```bash
docker build -t ratcliff-ddm .
docker run -p 7860:7860 ratcliff-ddm
```

## Stage status

- [x] Stage 0 — repo init, env setup
- [x] Stage 1 — validation infrastructure, smoke tests pass on laptop CPU
- [x] Stage 2 — single-GEMM + cumsum rewrite of `model_a/simulate.py`
- [x] Stage 3 — vectorized `fofs_new`, vmap over conditions, simplex fit driver
- [x] Stage 3.5 — smooth-objective surrogate for L-BFGS gradient fits (Model A)
- [x] Stage 4 — Model B GRF + accumulator port (parity-tested against Fortran archive outputs)
- [x] Stage 5 — H100 benchmark, ~5x per-call vs 6-node MPI, full fit in 30 min
- [x] Stage 6 — Karhunen-Loève low-rank GRF, ~2.25x per-eval over Stage 5, full fit in 20 min
- [x] Stage 7.A — `model_b/api.py` CPU-friendly wrapper layer
- [x] Stage 7.B — FastAPI backend + React frontend
- [x] Stage 7.C — Dockerfile + HF Space configuration
- [ ] Stage 7 deployment — push to HF Space and verify public URL (user action)
- [ ] Methods paper — manuscript draft

## Tags

- `v0.6.0-stage6-kl` — H100-validated K-L low-rank GRF release
- `v0.6.0-stage6-kl-laptop` — local-only, laptop validation milestone

## Documentation

- `docs/plans/` — design + implementation plans for every stage
- `docs/notes/` — completion summaries and session handoffs
- `docs/notes/2026-06-05-stage-6-completion.md` — Stage 6 H100 numbers + interpretation
- `docs/notes/2026-06-05-session-handoff.md` — complete record of the 2026-06-05 session

## License

MIT. See repo for the LICENSE file (if absent, all rights reserved pending license decision).

## Citation

```bibtex
@software{ratcliff_ddm_jax_2026,
  author = {GR74 and Ratcliff, Roger},
  title  = {JAX port of the spatially-extended diffusion decision model with Karhunen-Loève low-rank GRF},
  year   = {2026},
  url    = {https://github.com/GR74/ratcliff},
  note   = {v0.6.0-stage6-kl}
}
```

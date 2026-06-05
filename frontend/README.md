# Ratcliff DDM frontend

React + Vite + TypeScript + Tailwind + Plotly + Zustand.

## Dev

```bash
# in repo root, ensure the backend is running on :8000
cd ../  # repo root
uvicorn backend.main:app --reload --port 8000 &

# then in this directory
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api/*` to the backend at :8000.

## Build (for deployment via FastAPI static serving)

```bash
npm run build
```

Output goes to `frontend/dist/`. FastAPI serves it at `/` when present.

## Tabs

- **Forward Sim** — drag any of 9 single-condition sliders, plot updates ~200ms after the last change. Save named configs (browser localStorage). "Run full" button does nsim=9000 instead of nsim=256 preview.
- **Fit** — upload data (twod3datanew or CSV with rt/cat/condition columns), choose maxiter, optionally paste a BYO-GPU endpoint URL, click Start fit. Live loss plot updates as the simplex iterates. Recovery table shown when done.
- **Predict** — set 13 parameters by hand (or load from Fit), generate per-condition RT distributions at any nsim.
- **Compare** — pick any two saved configs, overlay their RT histograms.

## Browser storage

All saved configs live in `localStorage` under the key `ratcliff_configs_v1`. Use the Export / Import buttons in the sidebar to share between browsers.

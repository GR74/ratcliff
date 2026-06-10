# Stage 8 Design — three.js Field Visualization + Speed + Code Quality

**Date:** 2026-06-05
**Status:** APPROVED (autonomous-mode go-ahead)
**Scope:** Add a three.js 3D evidence-field visualization, fix the real perceived-speed bottlenecks (JIT pre-warm + caching + smaller preview), and do a code-quality pass over `grf_kl.py` and the frontend.

---

## 1. Motivation

User asked to "improve the entire code," "finish the UI using three.js," and "make it actually work well and fast-ish."

Honest constraints:
- three.js does **not** speed up the simulation. The bottleneck is JAX-on-CPU: ~2-5 min first-call JIT compile, 1-3 sec per call after. This is a backend problem; no frontend library fixes it.
- three.js for RT histograms would be a gimmick — 2D Plotly is more readable. three.js earns its place rendering the **2D evidence field** (the 100×160 accumulator grid) as an animated 3D surface, which is the genuinely novel "watch the model diffuse" visual.

So Stage 8 = (a) real speed fixes in the backend, (b) a three.js Field tab fed by a new endpoint, (c) code-quality cleanup.

## 2. Goals

1. **Pre-warm JIT** at server startup so the first user request is fast.
2. **Cache + smaller preview**: preview `nsim` 256→128, LRU cache on rounded params.
3. **`/api/field` endpoint** returning evidence-field snapshots (single trial or mean), downsampled.
4. **three.js Field tab** animating the surface with a threshold plane, orbit controls, play/scrub, single/mean toggle.
5. **AbortController** request cancellation in the Forward Sim tab.
6. **grf_kl.py cleanup**: vectorize the basis-construction loop, add a positive-variance guard.
7. No regressions: all existing tests still pass; new endpoint + field function have tests.

## 3. Non-goals

- Not replacing Plotly for the RT/quantile plots.
- Not making CPU simulation fundamentally faster (that's Stage 7-style GPU work, already done for the K-L path).
- Not live-updating the 3D field on every slider drag (mean-field is too heavy on CPU; use an on-demand Generate button).
- Not WebGL custom shaders beyond a basic height-colormap.

## 4. Backend

### 4.1 JIT pre-warm

`backend/main.py` gets a FastAPI startup hook that spawns a daemon thread which calls `model_api.forward_sim_preview(DEFAULT_PARAMS)` once. This triggers compilation so the first real request hits a warm cache. Runs in background so server accepts connections immediately; a `WARMED` flag + `/api/health` reports warm state.

### 4.2 Preview tuning + cache

- `forward_sim_preview` default preview nsim drops to 128.
- Module-level LRU cache (maxsize 128) in `model_b/api.py` keyed on `(round(each param, 4), key_seed)`. Repeated states (slider returns to a prior value) return instantly.

### 4.3 `/api/field` endpoint + `field_snapshots`

New `model_b/api.py::field_snapshots(params, mode, n_frames, n_trials_mean, grid_stride, key_seed)`:
- Builds `LAM`, drift bumps; generates the GRF path via the FFT F1/F2 trick for `n_trials` (1 for single, N for mean).
- Computes `a = cumsum(demeaned increment)` → `(n_trials, NSTEP, N, M)`.
- `single` → trial 0; `mean` → average over trials.
- Samples `n_frames` timesteps (linspace over NSTEP) and downsamples the grid by `grid_stride`.
- Returns `{frames, steps, threshold, n, m, nstep}` JSON. At `n_frames=48`, `grid_stride=2` (50×80): ~1.5 MB.

Endpoint `POST /api/field` with body `{params, mode, n_frames, key_seed}`. Validates params, returns the field payload.

## 5. Frontend

### 5.1 three.js Field tab

- Deps: `three` + `three/examples/jsm/controls/OrbitControls`.
- `FieldView` component:
  - One `THREE.PlaneGeometry(m, n, m-1, n-1)` with per-vertex z set from the current frame's field values.
  - A semi-transparent threshold plane at z = threshold.
  - Height-based colormap (viridis-ish) on the surface.
  - `OrbitControls` for rotate/zoom/pan.
  - Play/pause button + a frame scrubber slider.
  - Single/Mean toggle + "Generate field" button (fetches `/api/field`).
- Animation: a `requestAnimationFrame` loop advances the frame index; updating the geometry's position attribute z-values per frame is cheap on GPU (16k verts).

### 5.2 Request cancellation

`postSimulate` accepts an `AbortSignal`. ForwardSimTab uses an `AbortController` per debounced call and aborts the previous in-flight request when params change again.

## 6. Code quality

### 6.1 grf_kl.py

- Vectorize `_calc_kl_basis_impl`'s per-mode `for k in range(K)` loop into broadcasted numpy (build all K phase grids at once). Real wall-clock win at K≈1325-2000.
- Add `if sorted_eigvals.sum() <= 0: raise ValueError(...)` guard before the cumsum division.

### 6.2 Frontend cleanup

- Replace the `tick`-state debounce in ForwardSimTab with a cleaner AbortController + timeout pattern.
- Better loading and error display.

## 7. Testing

- `backend/tests/test_main.py`: add `/api/field` happy-path (single + mean) + validation-error tests.
- `model_b/tests/test_api_smoke.py`: add `field_snapshots` shape + determinism tests (single mode, small frames).
- `grf_kl` parity tests already exist and must still pass after the loop vectorization (proves the refactor is behavior-preserving).
- Frontend: manual smoke (three.js rendering is eyeball-verified).

## 8. Success criteria

1. Server pre-warms; first slider after startup responds in ~1-3 sec, not minutes.
2. `/api/field` returns valid snapshots for both modes; Field tab renders an animated 3D surface with threshold plane and orbit controls.
3. grf_kl vectorization passes the existing variance + ACF parity tests unchanged.
4. All backend + model_b tests pass.
5. No regression in the existing tabs.

## 9. Implementation order

1. grf_kl vectorization + guard (+ confirm parity tests pass).
2. `field_snapshots` in `model_b/api.py` + tests.
3. Backend: pre-warm hook, preview cache, `/api/field` endpoint + tests.
4. Frontend: AbortController cancellation; three.js Field tab.
5. Rebuild frontend, restart backend, smoke-test live.

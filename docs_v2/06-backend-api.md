# 06 — Backend and API

## What the backend does

\`backend/main.py\` is a FastAPI application. It translates HTTP requests into calls to \`model_b/api.py\`, validates request shapes with Pydantic, converts failures into HTTP responses, tracks long fits, and optionally serves the built React application.

~~~mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant Wrapper as model_b/api.py
    participant JAX as Model B / JAX
    Browser->>FastAPI: POST /api/simulate (JSON)
    FastAPI->>Wrapper: forward_sim_preview(params)
    Wrapper->>JAX: simulate_b(key, params)
    JAX-->>Wrapper: JAX arrays (rt, cat)
    Wrapper-->>FastAPI: Python lists
    FastAPI-->>Browser: JSON response
~~~

## Endpoint reference

| Method and path | Purpose | Main request |
|---|---|---|
| \`GET /api/health\` | process health and warm-up status | none |
| \`POST /api/simulate\` | preview or full single-condition simulation | parameter object, seed, full flag |
| \`POST /api/upload\` | parse legacy text or trial CSV | multipart file |
| \`POST /api/fit/start\` | create a background fit | aggregate data, 13-value \`x0\`, settings |
| \`GET /api/fit/status/{id}\` | poll loss history/status | job ID |
| \`GET /api/fit/result/{id}\` | retrieve final fit | job ID |
| \`POST /api/predict\` | simulate one or two fitted conditions | 13-value vector |
| \`POST /api/field\` | generate accumulator snapshots | params, single/mean mode |
| \`POST /api/phase\` | sweep two parameters | ranges, grid, metric |

FastAPI’s generated OpenAPI interface is normally available at \`http://localhost:8000/docs\`.

## Startup warm-up

The first JAX call compiles an XLA program and can take a long time on CPU. A startup hook launches \`model_api.warmup()\` on a daemon thread. The health response reports \`warm\` and \`warm_error\`, allowing a client to distinguish a live process from a compiled simulator.

## Background fit lifecycle

~~~mermaid
sequenceDiagram
    participant UI
    participant API
    participant Jobs as in-memory jobs
    participant Fit as SciPy/JAX fit
    UI->>API: POST /api/fit/start
    API->>Jobs: new_job() → UUID
    API-->>UI: job_id
    API->>Fit: run BackgroundTask
    loop Every objective evaluation
        Fit->>Jobs: append eval_n and loss
        UI->>API: GET /fit/status/id
        API-->>UI: status + progress
    end
    Fit->>Jobs: store result or error
    UI->>API: GET /fit/result/id
    API-->>UI: fitted parameters
~~~

\`backend/jobs.py\` stores jobs in a process-local dictionary guarded by a lock for creation/reset. Jobs disappear when the server restarts, are not shared across multiple worker processes, and have no expiration cleanup. That is suitable for a single-process demo but not a durable production queue.

## Upload parsing

\`backend/parsers.py\` first tries the 45-field legacy format, then CSV. It returns only subject 0’s data plus \`n_subjects\`; the current UI has no subject picker. CSV’s optional \`subject_id\` is not used to separate subjects—the rows are aggregated together.

## BYO-GPU routing

\`backend/byo_gpu.py\` can forward an HTTP request to another instance of the same backend. \`fit_start\` uses it when \`gpu_endpoint\` is supplied. However, the current frontend later polls the local \`/fit/status\` and \`/fit/result\` routes, and the backend’s \`forward_get\` helper is not wired into those routes. Therefore remote start forwarding exists, but end-to-end remote polling is incomplete unless an external setup mirrors job state or routing.

## Static production serving

After \`npm run build\`, \`frontend/dist/\` exists. The backend mounts that folder at \`/\` after registering API routes, so the same Uvicorn process serves JSON and the single-page app. During development, Vite serves the frontend separately and proxies \`/api\` to port 8000.

## Error and security notes

- Parameter dictionaries are only partly expressed as typed Pydantic fields; detailed validation occurs in \`model_b/api.py\`.
- CORS is limited to local Vite origins.
- Uploaded files are read fully into memory; there is no explicit size limit.
- A user-provided GPU URL makes the server perform outbound HTTP. A production version should validate allowed schemes/hosts and defend against server-side request forgery.
- There is no authentication, rate limiting, persistent storage, or multi-tenant isolation.
- Simulation is compute-heavy, so public deployment needs queueing and resource limits.

# 10 — Development, Testing, Benchmarking, and Deployment

## Local prerequisites

- Python 3.11 or newer;
- Node.js 20 or newer;
- enough RAM for JAX compilation and Model B chunks;
- optional CUDA-capable GPU for production-scale Model B work.

## First setup on Windows

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,fit,backend]"
npm ci --prefix frontend
~~~

Optional Python groups are:

- \`dev\`: pytest and parallel test support;
- \`fit\`: JAXopt and Optax;
- \`viz\`: Matplotlib and Jupyter;
- \`backend\`: FastAPI, Uvicorn, uploads, HTTP client.

## Run the application

Terminal 1:

~~~powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --port 8000
~~~

Terminal 2:

~~~powershell
npm run dev --prefix frontend
~~~

Open \`http://localhost:5173\`. Vite proxies \`/api/*\` to port 8000. The first simulation may wait for JAX compilation; \`/api/health\` reports warm-up state.

## Testing workflow

~~~mermaid
flowchart LR
    E["Edit code"] --> F["Focused package test"]
    F --> S["Regular suite<br/>not slow"]
    S --> B["Frontend production build"]
    B --> P{"Scientific path changed?"}
    P -- "Yes" --> V["Parity/recovery test"]
    P -- "No" --> R["Ready for review"]
    V --> R
~~~

Useful commands:

~~~powershell
pytest backend/tests/test_main.py -v
pytest model_b/tests/test_grf_kl_parity.py -v
pytest -m "not slow"
npm run build --prefix frontend
.\scripts\smoke.ps1
~~~

JAX tests enable 64-bit mode in \`conftest.py\`. Running modules outside pytest may require importing a path that enables x64 before model imports, as \`backend/main.py\` does.

## JAX performance concepts

- **JIT compilation:** the first call builds machine code; later calls with compatible shapes reuse it.
- **Static shapes:** changing \`nsim\`, \`chunk_size\`, or an unpadded basis can trigger recompilation.
- **Chunking:** larger chunks improve throughput but increase memory.
- **Asynchronous execution:** accurate benchmarks must block until JAX results are ready.
- **CPU/GPU differences:** a change that helps H100 throughput may hurt laptop CPU behavior.

Do not use preview timings as benchmark claims. The H100 scripts print devices, warm-up, chunk sweeps, objective timing, and recovery.

## Benchmark scripts

| Script | Role |
|---|---|
| \`h100_setup_and_run.sh\` | provision a rented Linux GPU environment |
| \`h100_smoke_timing.py\` | short optimized-simulator check |
| \`h100_model_b_benchmark.py\` | Stage 5 FFT benchmark |
| \`h100_section4_verbose.py\` | verbose FFT fitting progress |
| \`h100_stage6_kl_benchmark.py\` | KL fitting benchmark |
| \`_diag_kl_spectrum.py\` | one-off retained-variance diagnostic |
| \`sbi_mvp.py\` | amortized posterior training and validation |

The setup script still contains placeholder repository URLs and assumes a disposable Linux GPU machine. Review environment variables before running it.

## Production build

~~~powershell
npm run build --prefix frontend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
~~~

When \`frontend/dist/\` exists, FastAPI serves it. For a container:

~~~powershell
docker build -t ratcliff-ddm .
docker run --rm -p 7860:7860 ratcliff-ddm
~~~

The multi-stage Docker build:

1. installs locked Node dependencies and builds Vite;
2. installs the editable Python package with fit/backend extras;
3. copies the frontend bundle;
4. starts Uvicorn on port 7860;
5. health-checks \`/api/health\`.

## Operational limitations

The container is single-process research deployment infrastructure. Fit jobs are not durable, there is no database/authentication, and public compute requests are not controlled. For serious deployment, add a persistent job queue, result storage, request limits, upload limits, remote-endpoint allowlisting, observability, and explicit worker/GPU scheduling.

## A safe change checklist

1. Identify whether the change affects scientific behavior or only presentation.
2. Preserve explicit random-key handling.
3. Keep array shape/order documented at API boundaries.
4. Add a regression test beside the affected package.
5. For simulator changes, compare aggregate proportions and quantiles.
6. Run the frontend build after TypeScript/UI changes.
7. Report hardware and compile/warm state for performance claims.

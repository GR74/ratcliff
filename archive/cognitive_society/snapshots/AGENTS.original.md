# Repository Guidelines

## Project Structure & Module Organization

The Python code is split by responsibility: `model_a/` and `model_b/` contain the JAX diffusion models, `shared/` holds reusable validation, data, and PRNG utilities, and `backend/` provides the FastAPI service. `cognitive_society/` is a NumPy-based agent and collective-cognition layer. Each package keeps tests in its own `tests/` directory. The React/Vite application lives in `frontend/src/`, with reusable UI in `components/`, page-level views in `tabs/`, and API/types in `lib/`. Reference Fortran programs and archived parity outputs belong in `reference/`; experimental runners belong in `scripts/`; plans and results belong in `docs/`. Treat `data/` fixtures and reference outputs as reproducibility inputs, not generated scratch space.

## Build, Test, and Development Commands

- `python -m venv .venv` then `pip install -e ".[dev,fit,backend]"`: create a Python 3.11+ development environment.
- `pytest -m "not slow"`: run the regular Python suite across all packages.
- `pytest model_b/tests/test_api_smoke.py -v`: run a focused test while iterating.
- `uvicorn backend.main:app --reload --port 8000`: serve the API locally.
- `npm ci --prefix frontend`: install the locked frontend dependencies.
- `npm run dev --prefix frontend`: start Vite at `http://localhost:5173` with API proxying.
- `npm run build --prefix frontend`: type-check and produce `frontend/dist/`.
- `docker build -t ratcliff-ddm .`: build the combined deployment image.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for Python functions, modules, and tests; use `PascalCase` for classes. Preserve deterministic simulations by passing explicit JAX/NumPy random keys rather than using global randomness. TypeScript uses strict mode, two-space indentation, double quotes, trailing commas, `PascalCase` React components, and `camelCase` hooks or helpers. No repository-wide formatter or linter is configured, so match adjacent code and keep imports grouped clearly.

## Testing Guidelines

Pytest is the test framework. Name files `test_*.py` and tests `test_<behavior>`. Add regression tests beside the affected package and use fixed seeds for stochastic assertions. Mark expensive cases with `@pytest.mark.slow` or `@pytest.mark.perf`; there is no stated coverage threshold. The frontend has no automated test runner, so always run its production build after UI changes.

## Commit & Pull Request Guidelines

History follows concise Conventional Commit-style subjects such as `feat(cognitive_society): ...`, `fix: ...`, and `docs: ...`. Keep commits focused and describe the observed result, not only the implementation. Pull requests should summarize scope, list verification commands, link relevant issues or design notes, and include screenshots for visible UI changes. Call out benchmark hardware, seeds, and input data when reporting performance or scientific results; never commit credentials, local environments, generated frontend bundles, or large ad hoc outputs.

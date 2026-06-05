# Stage 1: build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime serving FastAPI + the built frontend
FROM python:3.12-slim

# System deps for scientific Python + healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project deps first so Docker caches the layer when only code changes.
COPY pyproject.toml ./
COPY model_a/ ./model_a/
COPY model_b/ ./model_b/
COPY shared/ ./shared/
COPY backend/ ./backend/
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -e ".[fit,backend]"

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

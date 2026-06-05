---
title: Ratcliff DDM
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 2D spatial diffusion decision model — interactive fits + predictions
---

# Ratcliff DDM — interactive 2D spatial diffusion model

Web interface for Roger Ratcliff's spatially-extended diffusion decision model (Smith & Ratcliff 2018+). Drag sliders to see how reaction-time distributions change with each parameter, upload your own data and fit the model, or compare named configurations.

The simulator under the hood is a JAX port of the original Fortran 2D GRF model, with a Karhunen-Loève low-rank GRF generator (Stage 6 K-L). On an H100 a full parameter recovery fit completes in ~20 min; on this Space's free CPU a fit takes ~1-2 hours. Drop in your own GPU endpoint URL in the Fit tab to route the fit to faster compute.

## Local development

```bash
# Backend
pip install -e ".[fit,backend]"
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

## Reproduce

Source code: <https://github.com/GR74/ratcliff>

Tag: `v0.6.0-stage6-kl`

## Citation

```bibtex
@software{ratcliff_ddm_jax,
  author = {GR74 and Ratcliff, Roger},
  title  = {JAX port of the spatially-extended diffusion decision model},
  year   = {2026},
  url    = {https://github.com/GR74/ratcliff},
  note   = {v0.6.0-stage6-kl — Karhunen-Loève low-rank GRF}
}
```

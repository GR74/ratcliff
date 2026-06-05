# Stage 6 Completion — K-L Low-Rank GRF

**Date:** [TODO when H100 benchmark lands]
**Status:** [TODO: passed / failed / partial]

---

## Headline result

[TODO one-sentence summary, e.g. "Full Model B fit at production nsim=9000 now runs in X min on a single H100, down from 30 min in Stage 5 and weeks on the 6-node MPI cluster."]

## Results vs Stage 5

| Metric | Stage 5 (FFT) | Stage 6 (K-L) | Improvement |
|---|---|---|---|
| Per-call simulator at nsim=9000 | 5.18 sec | [TODO] | [TODO]× |
| Per-objective (fofs_b_new) | 10.66 sec | [TODO] | [TODO]× |
| Full fit wall-clock | 30.3 min | [TODO] | [TODO]× |
| Parameters within 7% of truth | 12/12 | [TODO] | — |
| Average parameter error | 3.8% | [TODO] | — |
| chunk_size usable | 16 (vmap-of-2 cap) | [TODO] | — |

## K-L parameters used at runtime

- `sig_default` (test condition): 10.0
- `variance_threshold`: 0.99 (down from original design 0.999 — see Stage 6 design Section 1 for empirical correction)
- `k_max`: 2000
- K (modes retained at sig=10): 1325
- Variance captured at sig=10: ~99.0%

## Recovery table (synthetic, ±10% perturbed start)

```
[TODO paste recovery table from h100_stage6_kl.txt]
```

## Notes

- [TODO: any surprises during H100 run]
- [TODO: did chunk_size=64 work or did we have to drop it]
- [TODO: any regressions seen]
- [TODO: confirmed match to Stage 5 statistics within MC noise]

## Compared to Fortran

- Original Fortran 6-node MPI: 11 sec/call → Stage 6 [TODO] sec/call = [TODO]× faster
- Original Fortran "weeks per fit" calendar time → Stage 6 [TODO] min/fit on rented GPU = practically unlimited speedup once you include queue + restart overhead

## What this unlocks (per Stage 6 design)

- Per-individual DDM fitting at scale: each user/agent gets their own parameter fit in minutes
- Real-time inference for AI agents (Exo integration path)
- Parameter recovery studies on the 2D model (previously infeasible)
- Bootstrap confidence intervals for the 2D model
- Hierarchical Bayesian fits across subjects
- Behavioral neuropsych applications (longitudinal tracking)

## Files of record

- `model_b/grf_kl.py` — K-L basis builder + sampler
- `model_b/simulate.py` — `simulate_b(use_kl=True)` path
- `model_b/objective.py`, `model_b/fit.py` — `fofs_b_new(use_kl=True)`, `fit_simplex_b(use_kl=True)`
- `scripts/h100_stage6_kl_benchmark.py` — the benchmark script that produced these numbers
- `docs/plans/2026-06-05-model-b-stage-6-design.md` — design doc
- `docs/plans/2026-06-05-model-b-stage-6-implementation.md` — TDD implementation plan
- Tag: `v0.6.0-stage6-kl`

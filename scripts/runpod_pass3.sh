#!/usr/bin/env bash
# Pass 3 on a RunPod GPU pod (1x H100). Run from the repo root:
#   bash scripts/runpod_pass3.sh setup      # once: deps + test gate
#   bash scripts/runpod_pass3.sh speed      # clean dense vs separable timing
#   bash scripts/runpod_pass3.sh bakeoff    # NM vs CMA-ES, hard and soft G2
#   bash scripts/runpod_pass3.sh replicate  # fit subject 1 from the TACC start
# Fits use the Fortran-faithful model (fortran_variant "bench"); ENGINES picks
# the fit field: separable K-L at 99.9% (fast, ~+3 G2 bias) and/or exact FFT.
# Results land in results/pass3/; copy them off the pod before stopping it.
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUTF8=1
STARTS=${STARTS:-"0 1 2 3 4"}      # replication unit = start point
EVALS=${EVALS:-1500}
ENGINES=${ENGINES:-"separable_kl999_bench fft_bench"}

case "${1:-}" in
  setup)
    pip install -q "jax[cuda12]" scipy evosax pytest matplotlib
    python -c "import jax; print(jax.devices())"
    python -m pytest model_b/tests/test_grf_separable.py \
      model_b/tests/test_separable_simulation.py model_b/tests/test_soft_g2.py \
      shared/tests/test_cmaes_fit.py model_b/tests/test_fortran_fit_variant.py -q
    ;;
  speed)
    for n in 2048 9000; do
      python scripts/benchmark_separable_kl.py --nsim $n --evals 12 \
        --out results/pass3/separable_kl_h100_nsim$n.json
    done
    ;;
  bakeoff)
    for s in $STARTS; do
      for e in $ENGINES; do
        for tau in 0 2; do
          for o in nm cmaes; do
            python scripts/optimizer_bakeoff.py --optimizer $o --start-seed $s \
              --engine $e --soft-tau $tau --max-evals $EVALS
          done
        done
      done
    done
    ;;
  replicate)
    for e in $ENGINES; do
      for o in nm cmaes; do
        python scripts/replicate_ratcliff_subject1.py fit --optimizer $o \
          --fit-engine $e --max-evals $EVALS --soft-tau 2 --engines fft_bench
      done
    done
    ;;
  *) echo "usage: $0 setup|speed|bakeoff|replicate"; exit 1 ;;
esac

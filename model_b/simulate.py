"""
2D Gaussian-random-field diffusion simulator (Model B).

Mirrors `accum` from benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS. The simulator:
1. Builds 3 drift Gaussian bumps at fixed positions (uj1=80, uj2=30, uj3=130).
2. Builds 5-category zone array k(i,j) from bump positions.
3. Per timestep: generate one GRF via circulant_grf, accumulate, demean,
   check for crossing.

Note: the Fortran uses an F1/F2 caching trick to halve FFT count. We skip
this trick for simpler code (Stage 5 GPU benchmark will revisit if needed).
"""
from functools import partial

import jax
import jax.numpy as jnp

from model_b import grf
from shared import prng

# Model B field dimensions
N = 100
M = 160
NSTEP = 400
E = 10.0

# Drift bump positions (mirrors Fortran accum lines 432-450)
UI = 50.0       # row center, common to all 3 bumps
UJ1 = 80.0      # cat 1 bump column
UJ2 = 30.0      # cat 3 bump column
UJ3 = 130.0     # cat 4 bump column

# Cached coordinate grids
_I = jnp.arange(N, dtype=jnp.float64)
_J = jnp.arange(M, dtype=jnp.float64)
_I_GRID, _J_GRID = jnp.meshgrid(_I, _J, indexing="ij")  # both (N, M)


def drift_bumps(sis: float):
    """
    Three drift Gaussian bumps centered at (UI=50, UJ=80/30/130) with stddev `sis`.

    Returns (v1, v2, v3), each shape (N, M).
    """
    s3 = 2.0 * sis * sis
    s4 = sis * jnp.sqrt(2.0 * jnp.pi)
    def bump(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s3) / s4
    return bump(UJ1), bump(UJ2), bump(UJ3)


def zone_array(si: float = 6.0):
    """
    5-category zone array k(i,j) from the drift bump positions.

    Mirrors Fortran accum lines 432-450:
      - Initialize k = 5 (background).
      - Around UJ1: b > .0003 -> 2 (broader ring), b > .03 -> 1 (innermost).
      - Around UJ2: b > .0003 -> 3.
      - Around UJ3: b > .0003 -> 4.
    """
    s1 = 2.0 * si * si
    s2 = si * jnp.sqrt(2.0 * jnp.pi)
    def b_field(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s1) / s2

    k = jnp.full((N, M), 5, dtype=jnp.int32)
    # Around UJ1 (cat 1 innermost, cat 2 outer ring)
    b1 = b_field(UJ1)
    k = jnp.where(b1 > 0.0003, jnp.int32(2), k)
    k = jnp.where(b1 > 0.03,   jnp.int32(1), k)
    # Around UJ2 -> cat 3
    b2 = b_field(UJ2)
    k = jnp.where(b2 > 0.0003, jnp.int32(3), k)
    # Around UJ3 -> cat 4
    b3 = b_field(UJ3)
    k = jnp.where(b3 > 0.0003, jnp.int32(4), k)
    return k

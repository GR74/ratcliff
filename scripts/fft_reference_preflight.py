"""Independent source-derived checks for the isolated bench reference."""
import json
import hashlib
import os
from pathlib import Path
import sys


def run(source):
    if not (Path(source) / 'reference-manifest.json').exists():
        raise ValueError('Source preparation did not complete')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    sys.path.insert(0, str(source))
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from model_b import simulate, objective, grf
    from shared import blocked
    from run_fft_reference_recovery import load
    root = Path(__file__).resolve().parents[1]
    candidate = load(root / 'model_b/grf.py', 'reference_even_audit')
    checks = []
    # Independently transcribe Fortran's i=1..100, j=1..160 loops.
    y, x = np.meshgrid(np.arange(1, 101), np.arange(1, 161), indexing='ij')
    zones = np.full((100, 160), 5)
    for col, category in [(80, 2), (30, 3), (130, 4)]:
        field = np.exp(-((x-col)**2+(y-50)**2)/72)/(6*np.sqrt(2*np.pi))
        zones[field > .0003] = category
        if col == 80:
            zones[field > .03] = 1
    np.testing.assert_array_equal(simulate.zone_array(), zones)
    for col, actual in zip((80, 30, 130), simulate.drift_bumps(12.)):
        expected = np.exp(-((x-col)**2+(y-50)**2)/288)/(12*np.sqrt(2*np.pi))
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    checks.append('one-based geometry and bench region cutoffs')
    # qant1: omit bin 1, normalize through bin 400, interpolate, floor .01.
    rt = np.array([10, 20, 30, 300, 301, 349, 350, 3999, 4000, 4010, 4500.])
    r = np.zeros(401)
    for value in rt:
        index = int(value/10)
        if 1 <= index <= 400:
            r[index] += 1
    rr = np.zeros(401)
    for i in range(2, 401):
        rr[i] = rr[i-1] + r[i]
    rr /= rr[400]
    for q in (10., 20., 25., 300., 347.5, 4000.):
        art = q/10
        irt = min(max(int(art), 1), 399)
        expected = max(rr[irt]+(art-irt)*(rr[irt+1]-rr[irt]), .01)
        actual = objective._cdf_fortran_binned(jnp.asarray(rt), jnp.ones(len(rt), bool), q)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not np.isfinite(objective._cdf_fortran_binned(jnp.array([4500.]), jnp.array([True]), 300.))
    checks.append('qant1 histogram/interpolation, bin boundaries, overflow and empty-histogram failure')
    # Explicit deterministic crossing: strict >, largest evidence, active
    # compaction, short final block, and noncrosser final location.
    for early in (False, True):
        for compact in (False, True):
            result = blocked.first_passage(jax.random.split(jax.random.key(5), 3),
                jnp.array([2., 2., 100.]), jnp.array([1., -1., 0.]),
                jnp.ones(3, bool), lambda keys: jnp.zeros((len(keys), 2, 3)),
                n_steps=5, block_size=2, stop_early=early, compact=compact,
                drift_scale=jnp.array([1., 2., 1.]))
            np.testing.assert_array_equal(result.steps, [3, 2, 5])
            np.testing.assert_array_equal(result.crossed, [True, True, False])
            np.testing.assert_array_equal(result.positions[:2], [0, 0])
            np.testing.assert_array_equal(result.final_positions, [0, 0, 0])
    tied = np.zeros((1, 16000)); tied[0, 1] = tied[0, 160] = 1
    assert int(blocked._argmax_fortran(jnp.asarray(tied))[0]) == 160
    checks.append('strict crossing, scaled drift, blocking/compaction, horizon, column-major ties')
    max_error = 0.
    for sig in np.unique(np.r_[np.linspace(.2, 17, 101), 5., 10., 15.]):
        grf.assert_pd_embedding(s1=float(sig), s2=float(sig))
        eigen = np.asarray(candidate.even_spectrum(s1=sig, s2=sig))
        assert eigen.min() >= -1e-14
        a = np.asarray(grf.calc_LAM(s1=sig, s2=sig))
        b = np.asarray(candidate.calc_LAM_even(s1=sig, s2=sig))
        ca, cb = np.fft.fft2(a*a).real, np.fft.fft2(b*b).real
        iy, ix = np.arange(-99, 100), np.arange(-159, 160)
        err = np.max(np.abs(ca[iy[:, None] % 199, ix % 319]-cb[iy[:, None] % 200, ix % 320]))
        max_error = max(max_error, float(err))
    assert max_error < 1e-12
    checks.append('FFT199/FFT200 PSD and all required lag covariances over supported sig sweep')
    # Public simulator plumbing: sv affects results; inactive control differs;
    # no-crossers preserve their final region and step 401.
    common = dict(key=jax.random.key(7), ter=200., st=50., cr=10., crsd=2.,
                  av1=15., av2=10., av3=8., sis=12., sig=10., si=6., nsim=32,
                  chunk_size=8, use_kl=False, time_block_size=32)
    zero = simulate.simulate_b_blocked(**common, sv=0.)
    varied = simulate.simulate_b_blocked(**common, sv=.9)
    assert np.any(np.asarray(zero[0]) != np.asarray(varied[0]))
    no = simulate.simulate_b_blocked(**dict(common, cr=1e6), sv=.5)
    assert np.asarray(no[0]).min() >= 4010+175
    assert np.any(np.asarray(no[1]) != 5)
    checks.append('sv public wiring and noncrossing RT/category rule')
    result = dict(passed=True, checks=checks, max_covariance_error=max_error,
                  source_manifest_sha256=hashlib.sha256((Path(source) / 'reference-manifest.json').read_bytes()).hexdigest(),
                  fft200_sha256=hashlib.sha256((root / 'model_b/grf.py').read_bytes()).hexdigest(),
                  scope='Fortran-informed JAX bench reference; direct executable qualification pending',
                  unresolved=['Original MKL/MPI executable not run',
                              'Float32 accumulation versus Fortran double precision',
                              'JAX RNG differs from MKL; no path identity claim',
                              'Archived output provenance not sufficient for an executable equivalence certificate'])
    return result


if __name__ == '__main__':
    result = run(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

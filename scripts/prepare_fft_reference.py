"""Build an isolated, audited bench-semantics source from the frozen baseline.

No production files or historical benchmark sources are modified. Fail closed
when source patterns change. The output is a JAX port, not a Fortran executable.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.benchmark-results/2026-10-21-rtx4060-baseline'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old[:100]!r}')
    return text.replace(old, new)


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    frozen = BASE / 'source-322bd1f'
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((frozen / name.replace('\\', '/')).read_bytes()).hexdigest() == expected
    shutil.copytree(frozen, destination, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    # Only the shared kernel correction is taken from the existing correction
    # commit. Its optional arguments preserve other users; no optimizer changes.
    blocked = subprocess.check_output(
        ['git', 'show', '7c05436:shared/blocked.py'], cwd=ROOT, text=True)
    # Fortran MAXLOC visits the first array index fastest. In this isolated
    # Model B source we preserve that rule, even for exact ties.
    blocked = blocked.replace('jnp.argmax(\n                paths[jnp.arange(batch_size), first], axis=-1,\n            )',
                              '_argmax_fortran(paths[jnp.arange(batch_size), first])')
    blocked = replace_once(blocked, 'jnp.argmax(result.accumulator, axis=-1)',
                           '_argmax_fortran(result.accumulator)')
    blocked += '''\n\ndef _argmax_fortran(values):
    if values.shape[-1] != 16000:
        return jnp.argmax(values, axis=-1)
    transposed = values.reshape(values.shape[:-1] + (100, 160)).swapaxes(-1, -2)
    index = jnp.argmax(transposed.reshape(values.shape), axis=-1)
    return (index % 100) * 160 + index // 100
'''
    (destination / 'shared/blocked.py').write_text(blocked)
    path = destination / 'model_b/simulate.py'
    text = path.read_text()
    text = replace_once(text, 'jnp.arange(N, dtype=jnp.float64)', 'jnp.arange(1, N + 1, dtype=jnp.float64)')
    text = replace_once(text, 'jnp.arange(M, dtype=jnp.float64)', 'jnp.arange(1, M + 1, dtype=jnp.float64)')
    text = replace_once(text, '                             mode_ids=None):', '                             mode_ids=None, sv=0.0):')
    text = replace_once(text, '            noise_is_path=cumulative_kl,',
                        '            noise_is_path=cumulative_kl,\n            drift_scale=1.0 + sv * (u[:, 7] - 0.5),')
    text = replace_once(text, '        cat = jnp.where(result.crossed, k_zone[result.positions], 5)\n        rt = (result.steps.astype(jnp.float64) + ndt) * E',
                        '        positions = jnp.where(result.crossed, result.positions, result.final_positions)\n        cat = k_zone[positions]\n        steps = jnp.where(result.crossed, result.steps, NSTEP + 1)\n        rt = (steps.astype(jnp.float64) + ndt) * E')
    text = replace_once(text, '                                 stable_kl_noise=False):',
                        '                                 stable_kl_noise=False, sv=0.0):')
    text = replace_once(text, '                       kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False):',
                        '                       kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False, sv=0.0):')
    text = replace_once(text, '        cumulative_kl, mode_ids,', '        cumulative_kl, mode_ids, sv,')
    text = replace_once(text, '        kl_bucketed, cumulative_kl, stable_kl_noise,\n    )',
                        '        kl_bucketed, cumulative_kl, stable_kl_noise, sv,\n    )')
    path.write_text(text)
    path = destination / 'model_b/objective.py'
    text = path.read_text()
    correction = subprocess.check_output(['git', 'show', '7c05436:model_b/objective.py'], cwd=ROOT, text=True)
    cdf = correction[correction.index('E_MS ='):correction.index('def condition_g2_b')]
    # Empty in-range histograms are a declared failed objective, not invented
    # CDF values. This agrees with the reference's undefined zero division.
    cdf = cdf.replace('jnp.maximum(rr[-1], 1e-12)', 'rr[-1]')
    text = replace_once(text, 'def condition_g2_b(', cdf + 'def condition_g2_b(')
    text = replace_once(text, 'qc = jnp.array([(rt_i <= obs_quant[j, i]).sum() / denom for j in range(NQ)])',
                        'qc = jnp.array([_cdf_fortran_binned(rt, in_cat, obs_quant[j, i]) for j in range(NQ)])')
    text = replace_once(text, '                kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False):',
                        '                kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False):')
    text = replace_once(text, '    # Local imports to avoid hard top-level dependency cycle risk',
                        '    if not time_block_size:\n        raise ValueError("Reference study requires the corrected blocked simulator")\n    # Local imports to avoid hard top-level dependency cycle risk')
    text = replace_once(text, '            kl_bucketed, cumulative_kl, stable_kl_noise,',
                        '            kl_bucketed, cumulative_kl, stable_kl_noise, p[6],')
    # Match the historical lumped-category denominator; failures remain explicit.
    text = text.replace('(pxy + 1e-12)', 'pxy')
    path.write_text(text)
    hashes = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('model_b', 'shared') for p in (destination / package).rglob('*.py')}
    (destination / 'reference-manifest.json').write_text(json.dumps(dict(
        base_commit=manifest['commit'], files=hashes,
        scope='Fortran-informed JAX bench reference; direct executable qualification pending',
        corrections=['sv', 'one-based spatial coordinates', 'Fortran MAXLOC ties',
                     'noncrossing step 401 and final location', 'binned qant1 CDF']), indent=2))
    return destination


if __name__ == '__main__':
    import sys
    prepare(sys.argv[1])

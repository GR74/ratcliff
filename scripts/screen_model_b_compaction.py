"""Simulation-only compaction screen against the immutable laptop baseline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.benchmark-results/2026-10-21-rtx4060-baseline'
SOURCE = BASE / 'source-322bd1f'
OUT = ROOT / '.benchmark-results/2026-10-22-rtx4060-compaction-screen'
REPORT = ROOT / 'docs/results/2026-10-22-rtx4060-compaction-screen'


def main():
    active = subprocess.check_output([
        '/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True)
    if active.strip():
        raise RuntimeError('GPU is occupied: ' + active)
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name.replace('\\', '/')).read_bytes()).hexdigest() == digest
    OUT.mkdir(exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    sys.path.insert(0, str(SOURCE))
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'default')
    import jax.numpy as jnp
    import numpy as np
    from model_b import simulate, objective, grf_kl
    from shared import prng
    from model_b_baseline_observer import load_observer
    assert '4060' in jax.devices()[0].device_kind
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    observer = load_observer(SOURCE)
    truth = json.loads((BASE / 'archived-case/protocol.json').read_text())['truth']
    high = list(truth)
    high[2] *= 1.5
    evidence = dict(status='running', device=str(jax.devices()[0]),
                    source_manifest=manifest, seed=0, nsim=9000, chunk=8,
                    block=32, kl_bucketed=False, stable_kl_noise=True,
                    warmups=3, timing_pairs=6, cases={},
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.perf_counter()

    def save():
        evidence['elapsed_seconds'] = time.perf_counter() - started
        text = json.dumps(evidence, indent=2)
        (OUT / 'evidence.json').write_text(text)
        (REPORT / 'evidence.json').write_text(text)

    def run(module, params, compact, nsim):
        if time.perf_counter() - started > 1200:
            raise TimeoutError('Twenty-minute screen cap')
        p = objective.clamp_b(jnp.asarray(params))
        outputs = [module.simulate_b_blocked_with_work(
            prng.split_for_condition(jax.random.key(0), ci),
            p[0], p[1], p[2], p[3], p[a], p[b], p[c], p[4], float(p[5]), 6.,
            nsim, 8, True, 32, True, compact, stable_kl_noise=True,
        ) for ci, (a, b, c) in enumerate(objective.COND_MAP_B)]
        return jax.block_until_ready(outputs)

    try:
        for name, params in [('truth', truth), ('higher_boundary', high)]:
            case = dict(params=params, pairs=[])
            evidence['cases'][name] = case
            save()
            # Observer exposes first-passage steps, locations and crossing flags.
            observed = [run(observer, params, flag, 257) for flag in (True, False)]
            plain = [run(simulate, params, flag, 257) for flag in (True, False)]
            assert all(np.array_equal(np.asarray(observed[f][c][i]), np.asarray(plain[f][c][i]))
                       for f in range(2) for c in range(2) for i in (0, 1))
            assert all(np.array_equal(np.asarray(observed[0][c][i]), np.asarray(observed[1][c][i]))
                       for c in range(2) for i in (0, 1, 3, 4, 5))
            case['path_parity_514_trials'] = True
            for _ in range(3):
                for flag in (True, False):
                    run(simulate, params, flag, 9000)
            for index in range(6):
                pair, outputs = {}, {}
                for flag in ((True, False) if index % 2 == 0 else (False, True)):
                    begin = time.perf_counter()
                    outputs[flag] = run(simulate, params, flag, 9000)
                    pair['on' if flag else 'off'] = time.perf_counter() - begin
                assert all(np.array_equal(np.asarray(outputs[True][c][i]), np.asarray(outputs[False][c][i]))
                           for c in range(2) for i in (0, 1))
                case['pairs'].append(pair)
                save()
                print(name, index + 1, pair, flush=True)
            case['production_rt_category_exact'] = True
            case['on_median_seconds'] = float(np.median([p['on'] for p in case['pairs']]))
            case['off_median_seconds'] = float(np.median([p['off'] for p in case['pairs']]))
            case['off_over_on'] = case['off_median_seconds'] / case['on_median_seconds']
        evidence['status'] = 'completed'
    except Exception as exc:
        evidence['status'] = 'failed'
        evidence['error'] = repr(exc)
        raise
    finally:
        save()
        lines = ['# RTX 4060 compaction simulation screen', '',
                 '**Status: ' + evidence['status'] + '.**', '',
                 'Only compaction changes. Early stopping remains enabled; padding reduction is disabled.', '',
                 '| Case | On (s) | Off (s) | Off/on |', '| --- | --- | --- | --- |']
        for name, case in evidence['cases'].items():
            if 'off_over_on' in case:
                lines.append(f"| {name} | {case['on_median_seconds']:.3f} | {case['off_median_seconds']:.3f} | {case['off_over_on']:.3f} |")
        lines += ['', 'Three warm-ups per variant and six alternating timing pairs per case. '
                  'This is a small simulation-only screen, not a full objective or fitting benchmark. '
                  'Compilation is excluded from timed calls. Path checks use 257 trials per condition; '
                  'production RT/category checks use 9,000 per condition. Seed 0; chunk 8; block 32. '
                  'Exact engineering agreement does not independently establish scientific fidelity.', '',
                  '[Raw timings and settings](evidence.json)', '']
        (REPORT / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

"""Matched three-arm recovery on corrected JAX FFT199 observations.

Run only after independent preflight succeeds. Never silently fall back to the
uncorrected port. All results are a case study, not Fortran qualification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = '2026-10-27-fft-reference-recovery'
METHODS = ('fft199', 'fft200', 'repaired_kl')
TRUTH = [200, 50, 10, 2, 12, 10, .5, 15, 10, 8, 14, 11, 9]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def recovery(params, truth):
    import numpy as np
    p, t = np.asarray(params), np.asarray(truth)
    relative = np.abs((p-t)/t)
    return dict(signed_errors=(p-t).tolist(), relative_errors=relative.tolist(),
                mean_relative_error=float(relative.mean()), worst_relative_error=float(relative.max()),
                sv_active=True)


def fit_one(method, protocol, out, loss_call, clamp, status, clock=time.perf_counter,
            minimize=None):
    import numpy as np
    if minimize is None:
        from scipy.optimize import minimize
    status('fit_warmup', method=method)
    start = clock()
    value = float(loss_call(protocol['x0']))
    if not np.isfinite(value):
        raise ValueError('Nonfinite warmup objective')
    warmup = clock()-start
    start = clock()
    best = dict(params=protocol['x0'], loss=value, attained_seconds=0.)
    checkpoints, evaluations, pending = [], 0, list(protocol['checkpoint_seconds'])

    class Cap(Exception):
        pass

    def loss(p):
        nonlocal evaluations
        if clock()-start >= protocol['fit_cap_seconds']:
            raise Cap()
        before = clock()
        effective = np.asarray(clamp(p))
        value = float(loss_call(p))
        elapsed = clock()-start
        if not np.isfinite(value) or not np.isfinite(effective).all():
            raise ValueError('Invalid objective/proposal; stopping this study explicitly')
        while pending and elapsed >= pending[0]:
            checkpoints.append(dict(seconds=pending.pop(0), **best))
        evaluations += 1
        row = dict(evaluation=evaluations, seconds=elapsed, evaluation_seconds=clock()-before,
                   loss=value, raw_params=np.asarray(p).tolist(), params=effective.tolist())
        with (out / f'{method}-trace.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if value < best['loss']:
            best.update(params=effective.tolist(), loss=value, attained_seconds=elapsed)
        write(out / f'{method}-checkpoints.json', checkpoints)
        write(out / f'{method}-best.json', best)
        status('fit', method=method, evaluations=evaluations, training_seconds=elapsed, best_loss=best['loss'])
        return value

    try:
        result = minimize(loss, protocol['x0'], method='Nelder-Mead', options={
            'maxiter': protocol['maxiter'], 'xatol': 1e-5, 'fatol': 1e-5})
        termination = dict(success=bool(result.success), capped=False, message=str(result.message))
    except Cap:
        termination = dict(success=False, capped=True, message='135-minute soft training cap')
    result = dict(method=method, best=dict(best), checkpoints=checkpoints,
                  training_seconds=clock()-start, warmup_seconds=warmup, evaluations=evaluations,
                  termination=termination, recovery=recovery(best['params'], protocol['truth']))
    write(out / f'{method}-fit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    out = ROOT / '.benchmark-results' / NAME
    public = ROOT / 'docs/results' / NAME
    preflight = json.loads((out / 'preflight.json').read_text())
    manifest = json.loads((args.source / 'reference-manifest.json').read_text())
    if not preflight['passed'] or preflight['source_manifest_sha256'] != digest(args.source / 'reference-manifest.json'):
        raise ValueError('Missing or stale validated reference source')
    for name, sha in manifest['files'].items():
        if digest(args.source / name) != sha:
            raise ValueError(f'Source changed after validation: {name}')
    if (out / 'protocol.json').exists():
        raise FileExistsError('Study already started; no implicit overwrite or optimizer restart')
    sys.path.insert(0, str(args.source))
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    import scipy
    from model_b import simulate, objective, grf, grf_kl
    from shared import prng
    import fft200_comparison_metrics as metrics
    from model_b_baseline_observer import load_observer
    assert jax.default_backend() == 'gpu' and '4060' in jax.devices()[0].device_kind
    smi = '/usr/lib/wsl/lib/nvidia-smi'
    active = subprocess.check_output([smi, '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    if any(line.strip() and not line.startswith(str(os.getpid())+',') for line in active.splitlines()):
        raise RuntimeError(f'GPU occupied: {active}')
    public.mkdir(parents=True, exist_ok=True)
    frozen_tools = out / ('run-tools-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    frozen_tools.mkdir()
    for name in ('run_fft_reference_recovery.py', 'fft_reference_preflight.py',
                 'prepare_fft_reference.py',
                 'report_fft_reference_recovery.py',
                 'fft200_comparison_metrics.py', 'model_b_baseline_observer.py'):
        shutil.copyfile(ROOT / 'scripts' / name, frozen_tools / name)
    shutil.copyfile(ROOT / 'model_b/grf.py', frozen_tools / 'grf.py')
    candidate = load(frozen_tools / 'grf.py', 'reference_fft200')
    if digest(frozen_tools / 'grf.py') != preflight['fft200_sha256']:
        raise ValueError('FFT200 changed since preflight')
    # Instrument a separate copy, returning actual timeout positions and steps.
    observed_dir = out / 'observed-source'
    (observed_dir / 'model_b').mkdir(parents=True, exist_ok=True)
    text = (args.source / 'model_b/simulate.py').read_text()
    text = text.replace('        return rt, cat, result.computed_trial_steps',
                        '        result = result._replace(steps=steps, positions=positions)\n'
                        '        return rt, cat, result.computed_trial_steps')
    (observed_dir / 'model_b/simulate.py').write_text(text)
    observer = load_observer(observed_dir)
    # Observer exposes first-passage steps/positions; noncrossers are separately
    # identifiable and retained in their historical category by the simulator.
    truth = np.asarray(TRUTH, float)
    protocol = dict(version=1, methods=METHODS, truth=TRUTH,
        x0=np.asarray(objective.clamp_b(jnp.asarray(truth*np.random.default_rng(1).uniform(.8, 1.2, 13)))).tolist(),
        data_seed=2026102701, training_seed=321, reference_seed=2026102790,
        validation_seeds=list(range(2026102800, 2026102820)), nsim=9000,
        reference_nsim=180000, chunk=8, block=32, maxiter=1000,
        fit_cap_seconds=8100, checkpoint_seconds=[1800, 3600, 5400, 8100],
        launch_utc=datetime.now(timezone.utc).isoformat(),
        scope=manifest['scope'], sv_active=True,
        source_manifest_sha256=digest(args.source / 'reference-manifest.json'),
        tools={p.name: digest(p) for p in frozen_tools.iterdir() if p.is_file()},
        reporting='One dataset/start per arm. No recovery confidence intervals; prediction seeds are not repeated fits.',
        timing='Same 135-minute cap for every arm; later completion accepted rather than reducing fidelity or verification.')
    write(out / 'protocol.json', protocol)
    write(public / 'protocol.json', protocol)
    write(public / 'preflight.json', preflight)
    write(out / 'environment.json', dict(python=sys.version, numpy=np.__version__, jax=jax.__version__,
          scipy=scipy.__version__, device=str(jax.devices()), gpu=subprocess.check_output([smi], text=True)))
    fits, scores = [], []
    start = time.perf_counter()
    state = dict(status='running', pid=os.getpid())
    def status(phase, **detail):
        state.update(phase=phase, detail=detail, elapsed_seconds=time.perf_counter()-start,
                     updated_utc=datetime.now(timezone.utc).isoformat(), completed_fits=len(fits))
        write(out / 'status.json', state)
        write(public / 'status.json', state)
        write(public / 'results.json', dict(fits=fits, scores=scores))
        rows = '\n'.join(f"| {f['method']} | {f['training_seconds']/60:.2f} | {f['evaluations']} | {100*f['recovery']['mean_relative_error']:.2f}% | {f['termination']['message']} |" for f in fits)
        (public / 'README.md').write_text(f'''# FFT-reference recovery comparison

Status: **{state['status']}**. Phase: **{phase}**. Completed fits: {len(fits)}/3.
Updated UTC: {state['updated_utc']}. Detail: `{json.dumps(detail)}`.

Reference: **corrected, Fortran-informed JAX bench model**, not the original
Fortran executable. sv, one-based geometry, MAXLOC ties, timeout handling and
binned qant1 scoring are corrected identically in all arms. Direct executable
qualification remains pending; float32 accumulation and different RNG remain
limitations. No silent fallback to the old port is allowed.

All arms fit the identical saved FFT199 dataset, seed 2026102701, 9,000 trials
per condition. Training seed 321, shared initial point, simplex and 135-minute
cap. All 13 parameters are active. Caps/failures remain visible. The older
KL-generated-data fits are not controls for this experiment.

| Method | Training minutes | Evaluations | Mean parameter error | Termination |
| --- | --- | --- | --- | --- |
{rows}

Final estimates are independently scored first, with 20 seeds using both native
and common FFT199 predictions, then time checkpoints. Generating truth and the
fitted FFT199 are separate references. Raw samples, traces, hashes, resource
telemetry and diagnostic samples are retained under `.benchmark-results/{NAME}`.
This is one dataset and one start, not a general recovery or equivalence claim.
Prediction intervals describe Monte Carlo uncertainty conditional on estimates;
they are not recovery uncertainty. Engineering tolerances are not Fortran-calibrated.
''')
        print(json.dumps(dict(phase=phase, **detail)), flush=True)
    stop = threading.Event()
    def telemetry():
        with (out / 'gpu-telemetry.csv').open('a', buffering=1) as stream:
            while not stop.is_set():
                result = subprocess.run([smi, '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if result.returncode == 0:
                    stream.write(result.stdout)
                stop.wait(5)
    threading.Thread(target=telemetry, daemon=True).start()
    original = grf.calc_LAM
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    def select(method):
        grf.calc_LAM = candidate.calc_LAM_even if method == 'fft200' else original
        return dict(use_kl=method == 'repaired_kl', stable_kl_noise=method == 'repaired_kl')
    def sample(method, params, seed, n, observed=True):
        options = select(method)
        module = observer if observed else simulate
        p = np.asarray(objective.clamp_b(jnp.asarray(params)))
        values = [module.simulate_b_blocked_with_work(prng.split_for_condition(jax.random.key(seed), ci),
                  *p[:4], *p[list(indices)], p[4], float(p[5]), 6., n, 8,
                  time_block_size=32, sv=p[6], **options)
                  for ci, indices in enumerate(objective.COND_MAP_B)]
        jax.block_until_ready(values)
        names = ('rt','cat','work','steps','positions','crossed') if observed else ('rt','cat','work')
        return {name: np.stack([np.asarray(v[i]) for v in values]) for i, name in enumerate(names)}
    def record_kl(p, context):
        # Called after KL sampling/objectives: this reuses their cached basis.
        basis, modes, retained, _ = grf_kl.calc_kl_basis(sig=float(p[5]), n=100, m=160,
                                                       pad_to_k_max=True, return_mode_ids=True)
        row = dict(context=context, sig=float(p[5]), modes=int(modes), achieved_variance=float(retained),
                   requested_variance=.99, capacity=2000, target_missed=bool(retained < .99))
        with (out / 'kl-retention.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    def saved(label, method, params, seed, n):
        status('sample', label=label, method=method, seed=seed)
        before = time.perf_counter()
        values = sample(method, params, seed, n)
        if method == 'repaired_kl':
            record_kl(params, label)
        path = out / 'samples' / f'{label}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **values)
        write(path.with_suffix('.json'), dict(method=method, params=np.asarray(params).tolist(), seed=seed,
              nsim=n, seconds=time.perf_counter()-before, sha256=digest(path)))
        return values
    try:
        status('integration_preflight')
        for method in METHODS:
            a, b = sample(method, truth, 2026102700, 16), sample(method, truth, 2026102700, 16, False)
            for key in ('rt','cat','work'):
                np.testing.assert_array_equal(a[key], b[key])
        write(out / 'observer-parity.json', dict(passed=True, methods=METHODS))
        for sig, seed in zip((5., 10., 15.), range(2026102710, 2026102713)):
            p = truth.copy(); p[5] = sig
            for method in METHODS:
                saved(f'diagnostic-sig{sig:g}-{method}', method, p, seed, 4096)
        values = saved('training-observations', 'fft199', truth, protocol['data_seed'], 9000)
        count = np.array([[(cat == k).sum() for k in range(1,6)] for cat in values['cat']])
        if (count < 10).any():
            raise ValueError('Sparse observed category; do not silently omit or regenerate')
        quant = np.array([[np.quantile(values['rt'][ci][values['cat'][ci] == k], metrics.QUANTILES)
                          for k in range(1,6)] for ci in range(2)]).transpose(0,2,1)
        path = out / 'training-data.npz'
        np.savez_compressed(path, count=count, prop=count/9000, quant=quant)
        data_hash = digest(path)
        write(public / 'data-provenance.json', dict(generator='corrected JAX FFT199', seed=protocol['data_seed'],
              sha256=data_hash, raw_sha256=digest(out / 'samples/training-observations.npz')))
        data = dict(count=jnp.asarray(count), prop=jnp.asarray(count/9000), quant=jnp.asarray(quant))
        for method in METHODS:
            assert digest(path) == data_hash
            jax.clear_caches(); grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()
            def loss(p):
                result = float(objective.fofs_b_new(jnp.asarray(p), data, jax.random.key(321),
                    nsim=9000, chunk_size=8, time_block_size=32, **select(method)).block_until_ready())
                if method == 'repaired_kl':
                    record_kl(np.asarray(objective.clamp_b(jnp.asarray(p))), 'fit_evaluation')
                return result
            fits.append(fit_one(method, protocol, out, loss, lambda p: objective.clamp_b(jnp.asarray(p)), status))
            status('fit_completed', method=method)
            from report_fft_reference_recovery import report
            report(public)
        reference = saved('independent-truth', 'fft199', truth, protocol['reference_seed'], 180000)
        geometry = metrics.projection_geometry()
        masks = []
        for indices in objective.COND_MAP_B:
            drift = np.asarray(sum(truth[i]*b for i,b in zip(indices, simulate.drift_bumps(truth[4])))).ravel()
            masks.append([drift <= cutoff*drift.max() for cutoff in (.05,.10,.20)])
        def summarize(values):
            result = []
            for ci in range(2):
                cat, rt, crossed = values['cat'][ci], values['rt'][ci], values['crossed'][ci]
                result.append(dict(probabilities=[float(np.mean(cat == k)) for k in range(1,6)],
                    conditional_crossing_counts=[int(np.sum((cat == k) & crossed)) for k in range(1,6)],
                    rt_quantiles=[np.quantile(rt[(cat == k) & crossed], metrics.QUANTILES).tolist()
                                  if np.sum((cat == k) & crossed) >= 10 else [None]*5 for k in range(1,6)],
                    noncrossing=float(np.mean(~crossed)),
                    low_support=[float(np.mean(crossed & mask[values['positions'][ci]])) for mask in masks[ci]]))
            return result
        truth_summary = summarize(reference)
        write(public / 'truth-predictions.json', truth_summary)
        tasks = [(fit, 'final', fit['best'], generator) for fit in fits
                 for generator in dict.fromkeys((fit['method'], 'fft199'))]
        tasks += [(fit, str(cp['seconds']), cp, generator) for fit in fits for cp in fit['checkpoints']
                  for generator in dict.fromkeys((fit['method'], 'fft199'))]
        cache = {}
        fitted_reference = []
        for fit, label, estimate, generator in tasks:
            signature = (generator, tuple(estimate['params']))
            before = time.perf_counter()
            if signature in cache:
                predictions = cache[signature]
                reused = True
            else:
                predictions = []
                for seed in protocol['validation_seeds']:
                    values = saved(f"{fit['method']}-{label}-{generator}-{seed}", generator, estimate['params'], seed, 9000)
                    spatial = [float(metrics.spatial_distance({k:v[ci] for k,v in reference.items() if k!='work'},
                              {k:v[ci] for k,v in values.items() if k!='work'}, geometry)) for ci in range(2)]
                    predictions.append(dict(seed=seed, summaries=summarize(values), spatial_to_truth=spatial))
                    if fit['method'] == 'fft199' and label == 'final':
                        fitted_reference.append(values)
                    elif fitted_reference:
                        baseline = fitted_reference[len(predictions)-1]
                        predictions[-1]['spatial_to_fitted_fft199'] = [float(metrics.spatial_distance(
                            {k:v[ci] for k,v in baseline.items() if k!='work'},
                            {k:v[ci] for k,v in values.items() if k!='work'}, geometry)) for ci in range(2)]
                cache[signature] = predictions
                reused = False
            score = dict(method=fit['method'], checkpoint=label, generator=generator,
                         params=estimate['params'], recovery=recovery(estimate['params'], truth),
                         signed_parameter_difference_from_fft199=(np.asarray(estimate['params'])-np.asarray(fits[0]['best']['params'])).tolist(),
                         verification_seconds=time.perf_counter()-before, reused_predictions=reused, predictions=predictions)
            scores.append(score)
            write(out / f"score-{fit['method']}-{label}-{generator}.json", score)
            status('scored', method=fit['method'], checkpoint=label, generator=generator)
        state['status'] = 'complete_case_study'
        status('finished')
        report(public)
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        status('stopped', error=repr(error))
        raise
    finally:
        stop.set()
        grf.calc_LAM = original


if __name__ == '__main__':
    main()

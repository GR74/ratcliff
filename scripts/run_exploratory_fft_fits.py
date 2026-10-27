"""User-authorized fitting screen; behavioral qualification remains unresolved."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.benchmark-results/2026-10-21-rtx4060-baseline'
FROZEN = BASE / 'source-322bd1f'
NAME = '2026-10-26-fft200-exploratory-fit-comparison'
METHODS = ('fft200', 'fft199', 'repaired_kl', 'bucketed_kl')


def recovery(params, truth):
    import numpy as np
    p, t = np.asarray(params), np.asarray(truth)
    active = np.arange(len(t)) != 6
    relative = np.abs((p[active]-t[active])/t[active])
    return dict(signed_errors=(p-t).tolist(), active_relative_errors=relative.tolist(),
                mean_relative_error=float(relative.mean()), worst_relative_error=float(relative.max()),
                recovery_pass=bool(relative.mean() <= .05 and relative.max() <= .10),
                sv_note='Index 6 is inert in this frozen port; this is a known scientific limitation, not an irrelevant parameter.')


def fit_one(method, protocol, directory, loss_call, clamp, minimize, write, status,
            clock=time.perf_counter):
    """Save every proposal; do not restart an interrupted simplex silently."""
    import numpy as np
    destination = directory / f'{method}.json'
    trace = directory / f'{method}-trace.jsonl'
    if destination.exists():
        return json.loads(destination.read_text())
    if trace.exists():
        raise RuntimeError(f'Interrupted fit retained: {trace}; use a new study directory')
    status('fit_warmup', method=method)
    before = clock()
    warm_loss = loss_call(protocol['x0'])
    if not np.isfinite(warm_loss):
        raise ValueError('Nonfinite warmup loss')
    warmup = clock()-before
    started = clock()
    best = dict(loss=float('inf'), params=protocol['x0'], attained_seconds=None)
    checkpoints, evaluations = [], 0
    next_checkpoint = protocol['checkpoint_seconds']

    class FitCap(Exception):
        pass

    def loss(p):
        nonlocal evaluations, next_checkpoint
        if clock()-started >= protocol['fit_cap_seconds']:
            raise FitCap()
        effective = np.asarray(clamp(p))
        if not np.isfinite(p).all() or not np.isfinite(effective).all():
            raise ValueError('Nonfinite optimizer proposal')
        value = float(loss_call(p))
        elapsed = clock()-started
        evaluations += 1
        # A result available after a time boundary cannot count at that boundary.
        while elapsed >= next_checkpoint:
            checkpoints.append(dict(seconds=next_checkpoint, **best))
            next_checkpoint += protocol['checkpoint_seconds']
        row = dict(evaluation=evaluations, seconds=elapsed, loss=value,
                   raw_params=np.asarray(p).tolist(), params=effective.tolist(),
                   clamped=bool(np.any(np.asarray(p) != effective)))
        with trace.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        if not np.isfinite(value):
            raise ValueError('Nonfinite objective; fit is invalid')
        if value < best['loss']:
            best.update(loss=value, params=effective.tolist(), attained_seconds=elapsed)
        write(directory / f'{method}-checkpoints.json', checkpoints)
        write(directory / f'{method}-best.json', dict(evaluations=evaluations, **best))
        status('fit', method=method, evaluations=evaluations, training_seconds=elapsed,
               best_loss=best['loss'], recovery=recovery(best['params'], protocol['truth']))
        return value

    try:
        result = minimize(loss, protocol['x0'], method='Nelder-Mead', options={
            'maxiter': protocol['maxiter'], 'xatol': 1e-5, 'fatol': 1e-5})
        termination = dict(success=bool(result.success), message=str(result.message), capped=False)
    except FitCap:
        termination = dict(success=False, message='135-minute training cap', capped=True)
    elapsed = clock()-started
    checkpoints.append(dict(seconds=elapsed, **best))
    result = dict(method=method, warmup_seconds=warmup, training_seconds=elapsed,
                  evaluations=evaluations, best=best, checkpoints=checkpoints,
                  termination=termination, recovery=recovery(best['params'], protocol['truth']))
    write(destination, result)
    return result


def main():
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    sys.path.insert(0, str(FROZEN))
    import run_fft200_comparison as helper
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'default')
    import jax.numpy as jnp
    import scipy
    from scipy.optimize import minimize
    from model_b import grf, grf_kl, simulate, objective
    from shared import prng
    write, digest, load = helper.write, helper.digest, helper.load
    assert jax.default_backend() == 'gpu' and '4060' in jax.devices()[0].device_kind
    active = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    if any(s.strip() and not s.startswith(str(os.getpid())+',') for s in active.splitlines()):
        raise RuntimeError(f'GPU already occupied: {active}')
    out = ROOT / '.benchmark-results' / NAME
    public = ROOT / 'docs/results' / NAME
    out.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Study already exists; refusing implicit restart/overwrite')
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    for relative, sha in manifest['files'].items():
        assert digest(FROZEN / relative.replace('\\', '/')) == sha, relative
    frozen = out / 'run-tools'
    frozen.mkdir(exist_ok=True)
    hashes = {}
    for source in [Path(__file__), ROOT / 'scripts/run_fft200_comparison.py',
                   ROOT / 'scripts/fft200_comparison_metrics.py',
                   ROOT / 'scripts/model_b_baseline_observer.py', ROOT / 'model_b/grf.py',
                   ROOT / 'scripts/start_exploratory_fft_fits.ps1']:
        shutil.copyfile(source, frozen / source.name)
        hashes[source.name] = digest(source)
    screen = json.loads((ROOT / 'docs/results/2026-10-25-fft200-screen/screen.json').read_text())
    assert hashes['grf.py'] == screen['candidate_sha256']
    candidate = load(frozen / 'grf.py', 'exploratory_fft200')
    observer = load(frozen / 'model_b_baseline_observer.py', 'fit_observer').load_observer(FROZEN)
    metrics = load(frozen / 'fft200_comparison_metrics.py', 'fit_metrics')
    truth = np.array(json.loads((BASE / 'archived-case/protocol.json').read_text())['truth'])
    protocol = dict(version=1, purpose='Exploratory fitting screen explicitly authorized after stopping behavioral study; no qualification gate pass.',
        methods=METHODS, truth=truth.tolist(), x0=(truth*(1+np.random.default_rng(1).uniform(-.2,.2,13))).tolist(),
        data_seed=202610280, training_seed=321, reference_seed=202610500,
        calibration_seeds=list(range(202610400,202610420)), validation_seeds=list(range(202610300,202610320)),
        nsim=9000, reference_nsim=180000, chunk=8, block=32, maxiter=1000,
        fit_cap_seconds=8100, checkpoint_seconds=300, scoring_cap_seconds=8*3600,
        source_commit=manifest['commit'], source_hashes=hashes,
        limitations='One truth at sig=10, one dataset/start per arm; not recovery qualification or a statistically established fitting speedup. sv inert in frozen port.',
        scoring='All fits first. Final predictions scored first, native generator and common FFT199; then common FFT199 checkpoint predictions, subject to cap. 20 fresh seeds each; raw samples retained.',
        acceptance='Exploratory only: active recovery mean <=5%, worst <=10%; per-metric 95th percentile prediction error <= maximum from 20 truth calibration replicates. Not a population guarantee.',
        timing='Cold warmup, training, validation and total wall costs separate. Checkpoints use best available by boundary. No certified time-to-quality claim from this screen.')
    write(out / 'protocol.json', protocol)
    write(public / 'protocol.json', protocol)
    write(out / 'frozen-source-manifest.json', manifest)
    write(out / 'environment.json', dict(python=sys.version, jax=jax.__version__, numpy=np.__version__,
          scipy=scipy.__version__, device=jax.devices()[0].device_kind, x64=True,
          gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit','--format=csv,noheader'], text=True)))
    fits, scores = [], []
    state = dict(status='running', pid=os.getpid(), completed_fits=0)
    start = time.perf_counter()
    def status(phase, **detail):
        state.update(phase=phase, detail=detail, elapsed_seconds=time.perf_counter()-start,
                     updated_utc=datetime.now(timezone.utc).isoformat(), completed_fits=len(fits))
        write(out / 'status.json', state)
        write(public / 'status.json', state)
        write(public / 'fit-summary.json', dict(fits=fits, scores=scores))
        rows = '\n'.join(f"| {f['method']} | {f['evaluations']} | {f['training_seconds']/60:.1f} | {100*f['recovery']['mean_relative_error']:.2f}% | {f['termination']['message']} |" for f in fits)
        (public / 'README.md').write_text(f'''# Exploratory FFT/KL fitting comparison

Status: **{state['status']}**; phase: **{phase}**. Updated: {state['updated_utc']}.
Completed fits: {len(fits)}/4. Current detail: `{json.dumps(helper.clean(detail))}`.

The user stopped behavioral sampling after 12 complete rounds and authorized
fitting before behavioral qualification. The last formal behavioral look is 10;
the earlier inconclusive results remain unchanged. This study is not qualification.

All four methods fit the same fresh FFT199-generated dataset (9,000 trials per
condition), with the same start, fixed training seed and Nelder-Mead settings.
Each gets a 135-minute training cap plus separately recorded cold warmup.
All fits precede independent validation, which has an eight-hour cap. Results
at a cap remain censored/unfinished. Serial arm order is fixed, not randomized.

| Method | Evaluations | Training minutes | Mean active recovery error | Termination |
| --- | --- | --- | --- | --- |
{rows}

Final predictions use 20 fresh seeds in the fitted method and the common FFT199
scorer; saved checkpoints then use the common scorer. Errors cover region
probabilities, conditional RT quantiles, non-crossing, low-support wins and spatial
distance. Raw samples and proposal traces are saved in `{out.as_posix()}`.
Truth-based recovery never controls optimization. Parameter sv is inert in the
frozen port, a scientific limitation; its error is reported but excluded from
active recovery. One dataset/start cannot establish recovery rates, scientific
equivalence or a statistically conclusive time-to-quality improvement.
''', encoding='utf-8')
        print(json.dumps(helper.clean(dict(phase=phase, **detail))), flush=True)
    stop = threading.Event()
    def telemetry():
        with (out / 'gpu-telemetry.csv').open('a', buffering=1) as handle:
            while not stop.is_set():
                result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if result.returncode == 0:
                    handle.write(result.stdout)
                stop.wait(5)
    threading.Thread(target=telemetry, daemon=True).start()
    original = grf.calc_LAM
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    def select(method):
        grf.calc_LAM = candidate.calc_LAM_even if method == 'fft200' else original
        return dict(use_kl=True, stable_kl_noise=True, kl_bucketed=method=='bucketed_kl') if method.endswith('kl') else dict(use_kl=False)
    def sample(method, p, seed, n, observed=True):
        options = select(method)
        module = observer if observed else simulate
        p = np.asarray(p)
        outputs = [module.simulate_b_blocked_with_work(prng.split_for_condition(jax.random.key(seed), ci),
            *p[:4], *p[list(indices)], p[4], float(p[5]), 6., n, 8,
            time_block_size=32, stop_early=True, compact=True, **options)
            for ci, indices in enumerate(objective.COND_MAP_B)]
        jax.block_until_ready(outputs)
        names = ('rt','cat','work','steps','positions','crossed') if observed else ('rt','cat','work')
        values = {name: np.stack([np.asarray(o[i]) for o in outputs]) for i,name in enumerate(names)}
        if not np.isfinite(values['rt']).all():
            raise ValueError('Nonfinite simulation RT')
        return values
    def saved_sample(label, method, p, seed, n):
        path = out / 'samples' / f'{label}.npz'
        path.parent.mkdir(exist_ok=True)
        status('sample', label=label, method=method, seed=seed)
        before = time.perf_counter()
        values = sample(method, p, seed, n)
        temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **values)
        temp.replace(path)
        write(path.with_suffix('.json'), dict(method=method, params=np.asarray(p).tolist(), seed=seed,
              nsim=n, seconds=time.perf_counter()-before, sha256=digest(path)))
        return values
    try:
        status('observer_parity')
        for method in METHODS:
            a, b = sample(method, truth, 202610279, 16), sample(method, truth, 202610279, 16, False)
            for name in ('rt','cat','work'):
                np.testing.assert_array_equal(a[name], b[name])
        write(out / 'observer-parity.json', dict(passed=True, methods=METHODS))
        values = saved_sample('training-observations', 'fft199', truth, protocol['data_seed'], protocol['nsim'])
        count = np.array([[(c==k).sum() for k in range(1,6)] for c in values['cat']])
        if (count < 10).any():
            raise ValueError('Sparse training region; do not silently omit it')
        quant = np.array([[np.quantile(values['rt'][ci][values['cat'][ci]==k], metrics.QUANTILES)
                          for k in range(1,6)] for ci in range(2)]).transpose(0,2,1)
        np.savez_compressed(out / 'training-data.npz', count=count, prop=count/protocol['nsim'], quant=quant)
        data = dict(count=jnp.asarray(count), prop=jnp.asarray(count/protocol['nsim']), quant=jnp.asarray(quant))
        for method in METHODS:
            jax.clear_caches()
            grf_kl._basis_cache.clear()
            grf_kl._basis_cache_order.clear()
            def loss(p):
                return float(objective.fofs_b_new(jnp.asarray(p), data, jax.random.key(protocol['training_seed']),
                    nsim=protocol['nsim'], chunk_size=8, time_block_size=32, **select(method)).block_until_ready())
            fits.append(fit_one(method, protocol, out, loss, lambda p: objective.clamp_b(jnp.asarray(p)),
                                minimize, write, status))
            status('fit_completed', method=method)
        scoring_start = time.perf_counter()
        reference = saved_sample('prediction-reference', 'fft199', truth, protocol['reference_seed'], protocol['reference_nsim'])
        geometry = metrics.projection_geometry()
        masks = []
        for indices in objective.COND_MAP_B:
            drift = np.asarray(sum(truth[i]*b for i,b in zip(indices, simulate.drift_bumps(truth[4])))).ravel()
            masks.append([drift <= cutoff*drift.max() for cutoff in (.05,.10,.20)])
        def predictive(values):
            probabilities, rt_errors, spatial, low = [], [], [], []
            for ci in range(2):
                for k in range(1,6):
                    a, b = reference['cat'][ci]==k, values['cat'][ci]==k
                    probabilities.append(float(b.mean()-a.mean()))
                    ar = reference['rt'][ci][a & reference['crossed'][ci]]
                    br = values['rt'][ci][b & values['crossed'][ci]]
                    rt_errors.extend((np.quantile(br,metrics.QUANTILES)-np.quantile(ar,metrics.QUANTILES)).tolist() if min(len(ar),len(br))>=10 else [float('inf')]*5)
                probabilities.append(float(values['crossed'][ci].mean()-reference['crossed'][ci].mean()))
                for mask in masks[ci]:
                    low.append(float(np.mean(values['crossed'][ci] & mask[values['positions'][ci]])-np.mean(reference['crossed'][ci] & mask[reference['positions'][ci]])))
                spatial.append(metrics.spatial_distance({k:v[ci] for k,v in reference.items() if k!='work'},
                                                       {k:v[ci] for k,v in values.items() if k!='work'}, geometry))
            return dict(max_errors=[max(map(abs,probabilities)), max(map(abs,rt_errors)), max(spatial), max(map(abs,low))],
                        signed_probability_errors=probabilities, signed_rt_errors=rt_errors, spatial=spatial, signed_low_support_errors=low)
        calibration = [predictive(saved_sample(f'calibration-{i}', 'fft199', truth, seed, protocol['nsim']))
                       for i,seed in enumerate(protocol['calibration_seeds'])]
        thresholds = np.max([r['max_errors'] for r in calibration], axis=0)
        write(out / 'calibration.json', dict(thresholds=thresholds, samples=calibration))
        if not np.isfinite(thresholds).all():
            raise ValueError('Sparse predictive calibration; no predictive pass can be assigned')
        tasks = [(f, len(f['checkpoints'])-1, generator) for f in fits
                 for generator in dict.fromkeys((f['method'],'fft199'))]
        tasks += [(f,i,'fft199') for f in fits for i in range(len(f['checkpoints'])-1)
                  if f['checkpoints'][i]['attained_seconds'] is not None]
        for fit, index, generator in tasks:
            if time.perf_counter()-scoring_start >= protocol['scoring_cap_seconds']:
                state['status'] = 'validation_capped'
                status('finished', reason='All fits saved; remaining validation censored')
                return
            checkpoint = fit['checkpoints'][index]
            before = time.perf_counter()
            predictions = []
            for i,seed in enumerate(protocol['validation_seeds']):
                predictions.append(predictive(saved_sample(f"{fit['method']}-checkpoint-{index}-{generator}-seed-{i}",
                    generator, checkpoint['params'], seed, protocol['nsim'])))
            array = np.array([p['max_errors'] for p in predictions])
            rec = recovery(checkpoint['params'], truth)
            prediction_pass = bool(np.isfinite(array).all() and np.all(np.quantile(array,.95,axis=0)<=thresholds))
            score = dict(method=fit['method'], checkpoint=index, generator=generator,
                training_seconds=checkpoint['seconds'], verification_seconds=time.perf_counter()-before,
                recovery=rec, prediction_pass=prediction_pass, joint_pass=prediction_pass and rec['recovery_pass'],
                predictions=predictions)
            scores.append(score)
            write(out / f"{fit['method']}-checkpoint-{index}-{generator}-score.json", score)
            status('checkpoint_scored', method=fit['method'], checkpoint=index, generator=generator)
        state['status'] = 'complete_exploratory'
        status('finished')
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        status('stopped', error=repr(error))
        raise
    finally:
        grf.calc_LAM = original
        stop.set()


if __name__ == '__main__':
    main()

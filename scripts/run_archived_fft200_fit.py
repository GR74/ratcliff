"""Fit only FFT200 against the exact archived RTX 4060 fitting inputs."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.benchmark-results/2026-10-21-rtx4060-baseline'
FROZEN = BASE / 'source-322bd1f'
NAME = '2026-10-26-fft200-archived-fit-comparison'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_inputs():
    import numpy as np
    improved_path = BASE / 'archived-case/fit-baseline-1-early_stop.json'
    legacy_path = BASE / 'legacy-case/fit-baseline-1-legacy.json'
    improved, legacy = [json.loads(p.read_text()) for p in (improved_path, legacy_path)]
    protocol = json.loads((BASE / 'archived-case/protocol.json').read_text())
    legacy_protocol = json.loads((BASE / 'legacy-case/protocol.json').read_text())
    inputs = [improved_path, legacy_path, BASE / 'archived-case/protocol.json', BASE / 'legacy-case/protocol.json']
    for relative in ('historical/data-baseline.npz', 'archived-case/data-baseline.npz', 'legacy-case/data-baseline.npz'):
        path = BASE / relative
        assert sha(path) == protocol['data_sha256'] == legacy_protocol['data_sha256'], relative
        inputs.append(path)
    np.testing.assert_array_equal(improved['x0'], legacy['x0'])
    np.testing.assert_array_equal(improved['true_params'], legacy['true_params'])
    np.testing.assert_array_equal(improved['true_params'], protocol['truth'])
    assert improved['data_sha256'] == legacy['data_sha256'] == protocol['data_sha256']
    assert protocol['training_seed'] == legacy_protocol['training_seed'] == 321
    assert protocol['score_seeds'] == legacy_protocol['score_seeds']
    fields = ('nsim', 'chunk', 'start_seed', 'start_spread', 'maxiter',
              'fit_seconds_limit', 'checkpoint_every', 'basis_cache_size', 'stable_kl_noise')
    for field in fields:
        assert improved['settings'][field] == legacy['settings'][field], field
    versions = improved['metadata']['packages']
    assert versions == legacy['metadata']['packages']
    for package, version in versions.items():
        assert importlib.metadata.version(package) == version, package
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert sha(FROZEN / relative.replace('\\','/')) == expected, relative
    assert improved['metadata']['commit'] == legacy['metadata']['commit'] == manifest['commit']
    expected_start = np.array(protocol['truth'])*np.random.default_rng(protocol['start_seed']).uniform(
        1-protocol['start_spread'], 1+protocol['start_spread'], 13)
    np.testing.assert_array_equal(expected_start, improved['x0'])
    return dict(passed=True, source_commit=manifest['commit'], packages=versions,
                data_sha256=protocol['data_sha256'], truth=protocol['truth'], x0=improved['x0'],
                settings={field: improved['settings'][field] for field in fields},
                training_seed=protocol['training_seed'], score_seeds=protocol['score_seeds'],
                input_hashes={str(p.relative_to(BASE)):sha(p) for p in inputs},
                comparator_correction='Both prior fits used stable KL: legacy full horizon and improved blocked early stopping. Neither was an FFT199 or Fortran fit.'), improved, legacy


def fft_objective(original):
    """Only replace the noise implementation; retain the archived objective."""
    def call(*args, **kwargs):
        kwargs.update(use_kl=False, stable_kl_noise=False, kl_bucketed=False, cumulative_kl=False)
        return original(*args, **kwargs)
    return call


def main():
    audit, improved, legacy = audit_inputs()
    if '--audit-only' in sys.argv:
        print(json.dumps(audit, indent=2))
        return
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.5')
    sys.path.insert(0, str(FROZEN))
    import subprocess
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'default')
    import jax.numpy as jnp
    from model_b import objective, grf, grf_kl
    from scripts import validate_early_stopping as historical
    import run_fft200_comparison as helper
    from baseline_model_b_laptop import observable_metrics, recovery
    from model_b_baseline_observer import load_observer, sample_observed
    assert jax.default_backend() == 'gpu' and jax.devices()[0].device_kind == improved['metadata']['device_kind']
    active = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'], text=True)
    if any(s.strip() and not s.startswith(str(os.getpid())+',') for s in active.splitlines()):
        raise RuntimeError(f'GPU occupied: {active}')
    out, public = ROOT / '.benchmark-results' / NAME, ROOT / 'docs/results' / NAME
    out.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Existing study preserved; refusing implicit restart')
    snapshots = out / 'run-tools'
    snapshots.mkdir(exist_ok=True)
    source_paths = [Path(__file__), ROOT/'model_b/grf.py', ROOT/'scripts/start_archived_fft200_fit.ps1',
                    ROOT/'scripts/baseline_model_b_laptop.py', ROOT/'scripts/model_b_baseline_observer.py',
                    ROOT/'scripts/run_fft200_comparison.py']
    hashes = {}
    for path in source_paths:
        shutil.copyfile(path, snapshots/path.name)
        hashes[path.name] = sha(path)
    screen = json.loads((ROOT/'docs/results/2026-10-25-fft200-screen/screen.json').read_text())
    assert hashes['grf.py'] == screen['candidate_sha256']
    candidate = helper.load(snapshots/'grf.py','archived_fft200_builder')
    shutil.copyfile(BASE/'archived-case/data-baseline.npz', out/'data-baseline.npz')
    assert sha(out/'data-baseline.npz') == audit['data_sha256']
    for label, record in [('legacy_full_horizon_kl',legacy), ('improved_kl',improved)]:
        helper.write(out/f'reused-{label}.json', record)
    args = SimpleNamespace(**improved['settings'])
    args.output = str(out)
    args.result_file = str(out/'fit-fft200.json')
    protocol = dict(audit=audit, source_hashes=hashes, fit_settings=vars(args),
        new_fits=['fft200'], reused_fits=['legacy_full_horizon_kl','improved_kl'],
        optimizer='Exact frozen historical worker and fit_simplex_b; Nelder-Mead, tol 1e-5, all 13 coordinates, unchanged default simplex.',
        change='Same blocked early-stop pipeline as improved KL; replace noise by audited full-spectrum FFT200.',
        data_origin='Byte-identical archived legacy-generated data, not new FFT-generated data.',
        stopping='Same 8100-second soft cap checked AFTER evaluation, first/every-25-evaluation checkpoints, final simplex result preserved.',
        scoring='Reuse saved KL scores/samples at keys 6001..6020. Fill missing legacy scores using same common early-stop KL scorer. Score FFT200 native and common KL on those keys. No old optimizer reruns.',
        limits='Single archived dataset/start. Legacy capped at 84 evaluations, improved KL converged at 563; no completed-fit speed ratio for capped legacy. No full qualification; sv inert in frozen port.',
        supersedes='Fresh-data run stopped after about 162 seconds; retained separately, excluded from archived comparison.')
    helper.write(out/'protocol.json',protocol)
    helper.write(public/'protocol.json',protocol)
    helper.write(public/'input-audit.json',audit)
    state = dict(status='running', phase='fit', pid=os.getpid())
    start = time.perf_counter()
    fits = dict(legacy_full_horizon_kl=legacy, improved_kl=improved)
    def status(phase, **detail):
        state.update(phase=phase, detail=detail, updated_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-start)
        helper.write(out/'status.json',state)
        helper.write(public/'status.json',state)
        rows = '\n'.join(f"| {label} | {fit['status']} | {fit['n_evals']} | {fit['fit_seconds']/60:.2f} | {100*fit['mean_active_effective_parameter_error']:.2f}% |" for label,fit in fits.items())
        (public/'README.md').write_text(f'''# FFT200 matched to the two archived RTX 4060 fits

Status: **{state['status']}**, phase **{phase}**. Updated: {state['updated_utc']}.
Detail: `{json.dumps(helper.clean(detail))}`.

Only FFT200 is being fitted. Both previous fits are reused. Dataset SHA-256,
truth, exact start, fixed key 321, 9,000 trials per condition, chunk 8, all 13
parameters, Nelder-Mead/tolerances, 1,000 iterations, 135-minute soft cap,
checkpoint rules, frozen source and package versions have been checked.
FFT200 uses block 32 and early stopping like the improved KL comparator.

| Method | Status | Evaluations | Training minutes | Mean active recovery error |
| --- | --- | --- | --- | --- |
{rows}

The prior "original" fit was full-horizon JAX **KL**, not FFT199 or Fortran.
Its cap means it is not a completed-fit timing reference. The improved KL fit
converged but its worst active error was 16.64%, above the 10% recovery screen.
The earlier fresh-data FFT200 run is interrupted and excluded from this comparison.

Final independent scores use the original keys 6001–6020. Existing KL samples
and scores are reused; only missing legacy scores and FFT200 predictions are
computed. FFT200 is scored natively and through the common KL scorer used for
the old fits. Report native and common results separately. Training warmup,
training, scoring and total wall time stay separate. This is an archived-case
screen, not behavioral equivalence or broad recovery qualification. Parameter
sv remains inert in this port, a known scientific limitation.
''',encoding='utf-8')
        print(json.dumps(helper.clean(dict(phase=phase,**detail))),flush=True)
    stop = threading.Event()
    def monitor():
        with (out/'gpu-telemetry.csv').open('a', buffering=1) as handle:
            while not stop.wait(10):
                result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if result.returncode == 0:
                    handle.write(result.stdout)
                if state['phase'] == 'fit':
                    progress_path = Path(args.result_file).with_suffix('.progress.json')
                    if progress_path.exists():
                        try:
                            progress = json.loads(progress_path.read_text())
                            status('fit', latest_eval=progress['latest_eval'], training_seconds=progress['elapsed_seconds'])
                        except json.JSONDecodeError:
                            pass
    original_objective, original_lam = objective.fofs_b_new, grf.calc_LAM
    try:
        status('fit', latest_eval=0, detail='Cold compilation/warmup')
        threading.Thread(target=monitor,daemon=True).start()
        grf.calc_LAM = candidate.calc_LAM_even
        objective.fofs_b_new = fft_objective(original_objective)
        result = historical.worker(args)
        np.testing.assert_array_equal(result['x0'], audit['x0'])
        assert result['data_sha256'] == audit['data_sha256']
        result['noise_override'] = dict(use_kl=False, fft_workspace=[200,320], wrapper_sha256=hashes[Path(__file__).name])
        result['parameter_recovery'] = recovery(np.asarray(objective.clamp_b(jnp.asarray(result['params']))),np.asarray(audit['truth']))
        helper.write(out/'fit-fft200.json',result)
        fits['fft200'] = result
        helper.write(public/'fit-comparison.json',fits)
        status('independent_scoring')
        objective.fofs_b_new, grf.calc_LAM = original_objective, original_lam
        observer = load_observer(FROZEN)
        data_np = dict(np.load(out/'data-baseline.npz'))
        data = {k:jnp.asarray(v) for k,v in data_np.items()}
        records = []
        for label, fit in fits.items():
            for scorer in (('native_fft200','common_kl') if label=='fft200' else ('common_kl',)):
                for seed in audit['score_seeds']:
                    source = BASE/('archived-case' if label=='improved_kl' else 'legacy-case')
                    old_json, old_npz = source/f'score-fit-{seed}.json', source/f'score-fit-{seed}.npz'
                    status('independent_scoring', method=label, scorer=scorer, seed=seed)
                    if label!='fft200' and old_json.exists() and old_npz.exists():
                        record = dict(json.loads(old_json.read_text()), method=label, scorer=scorer,
                                      reused=True, source=str(old_json), source_sha256=sha(old_json), samples_sha256=sha(old_npz))
                    else:
                        before = time.perf_counter()
                        p = np.asarray(objective.clamp_b(jnp.asarray(fit['params'])))
                        grf.calc_LAM = candidate.calc_LAM_even if scorer=='native_fft200' else original_lam
                        if scorer=='common_kl':
                            values = sample_observed(observer,p,seed)
                        else:
                            from shared import prng
                            outputs = [observer.simulate_b_blocked_with_work(prng.split_for_condition(jax.random.key(seed),ci),
                                *p[:4], *p[list(indices)], p[4],float(p[5]),6.,9000,8,
                                use_kl=False,time_block_size=32,stop_early=True,compact=True)
                                for ci,indices in enumerate(objective.COND_MAP_B)]
                            jax.block_until_ready(outputs)
                            values = {name:np.stack([np.asarray(o[i]) for o in outputs]) for i,name in enumerate(('rt','cat','work','steps','positions','crossed'))}
                            positions = values['positions']
                            xy = 2*np.stack((positions%160,positions//160),axis=-1)
                            values['xy_pixels'] = np.where(values['crossed'][...,None],xy,np.nan)
                        score = original_objective(jnp.asarray(p),data,jax.random.key(seed),nsim=9000,chunk_size=8,
                            time_block_size=32,use_kl=scorer=='common_kl',stable_kl_noise=scorer=='common_kl')
                        target = out/f'score-{label}-{scorer}-{seed}.npz'
                        np.savez_compressed(target,**values)
                        record = dict(method=label,scorer=scorer,seed=seed,reused=False,g2=float(score),
                            seconds=time.perf_counter()-before,no_crossing_fraction=(~values['crossed']).mean(axis=1).tolist(),
                            metrics=observable_metrics(values['rt'],values['cat'],data_np),samples_sha256=sha(target))
                    records.append(record)
                    helper.write(out/f'score-{label}-{scorer}-{seed}.json',record)
                    helper.write(public/'independent-scores.json',records)
        state['status']='complete_archived_screen'
        status('finished')
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        status('stopped',error=repr(error))
        raise
    finally:
        stop.set()
        objective.fofs_b_new,grf.calc_LAM=original_objective,original_lam


if __name__ == '__main__':
    main()

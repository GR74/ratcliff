"""Run one isolated BADS screen after the existing laptop baseline finishes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import subprocess

import numpy as np

from model_b_bads_support import (
    FitDeadline, RecordedObjective, checkpoint_candidates, classify,
    paired_analysis, select_candidate, serializable, write_json,
)
from baseline_model_b_laptop import recovery, observable_metrics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import fcntl
    root, out, report_dir = map(lambda p: Path(p).resolve(), (args.baseline, args.output, args.report))
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    lock = (root / 'bads-screen.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'status.json').exists():
        raise RuntimeError('Output directory already has a run; preserve it instead of restarting.')
    previous = root / 'archived-case'
    source = root / 'source-322bd1f'
    control_fit = read(previous / 'fit-baseline-1-early_stop.json')
    protocol = read(previous / 'protocol.json')
    source_manifest = read(root / 'source-manifest.json')
    primary_seconds = control_fit['fit_seconds']
    cutoffs = [1200, 2400, 3600, 4800, primary_seconds, 6600, 8100]
    launched = time.perf_counter()
    state = {'stage': 'preparing', 'pid': os.getpid()}
    summary = {}
    reporting_lock = threading.Lock()
    stop_monitor = threading.Event()

    def publish(stage=None, **extra):
        with reporting_lock:
            if stage:
                state['stage'] = stage
            state.update(extra, utc=datetime.now(timezone.utc).isoformat(),
                         session_elapsed_seconds=time.perf_counter()-launched)
            write_json(out / 'status.json', state)
            lines = ['# RTX 4060 Model B BADS screen', '',
                f"**Status: {state['stage']}**. Updated {state['utc']}.", '',
                'One optimizer change against the frozen improved simulator. The original-version baseline runs first.', '',
                '| Measurement | Value |', '| --- | --- |',
                f'| Primary matched training budget | {primary_seconds/60:.3f} minutes |',
                '| Maximum training budget | 135 minutes, soft cap |']
            for name in ['evaluations', 'training_seconds', 'best_training_loss', 'waiting_for', 'error']:
                if name in state:
                    lines.append(f'| {name} | {state[name]} |')
            if 'primary' in summary:
                rec = summary['primary']['recovery']
                lines += [f"| Primary mean parameter error | {100*rec['mean_absolute_relative_error']:.3f}% |",
                          f"| Primary worst parameter error | {100*rec['worst_absolute_relative_error']:.3f}% |"]
            if 'decision' in summary:
                lines.append(f"| Primary screen | {summary['decision']['classification']} |")
            lines += ['', 'Independent scoring is performed after training. A promising result is a single-case screening result, not scientific validation or a time-to-quality claim. The 27-fit suite remains deferred.', '',
                'Raw trace, manifests, checkpoints, telemetry and scores are retained in the ignored benchmark output directory. [Evidence](evidence.json)', '']
            path = report_dir / 'README.md'
            temp = path.with_suffix('.md.tmp'); temp.write_text('\n'.join(lines)); temp.replace(path)
            write_json(report_dir / 'evidence.json', {'status': state, 'results': summary})
        if stage:
            print(stage, extra, flush=True)

    def verify_inputs():
        for name, expected in source_manifest['files'].items():
            if digest(source / name.replace('\\', '/')) != expected:
                raise ValueError(f'Frozen scientific source changed: {name}')
        if digest(previous / 'data-baseline.npz') != protocol['data_sha256']:
            raise ValueError('Dataset hash mismatch')
        for line in (root / 'requirements-historical.txt').read_text().splitlines():
            if '==' in line:
                name, version = line.strip().split('==')
                if importlib.metadata.version(name) != version:
                    raise ValueError(f'Frozen runtime changed: {name}')
        if importlib.metadata.version('pybads') != '1.5.1':
            raise ValueError('This runner is verified against PyBADS 1.5.1')

    def baseline_finished():
        status = read(root / 'legacy-case/status.json')
        if status['stage'] == 'error':
            raise RuntimeError(f'Baseline failed; review before BADS: {status.get("error")}')
        pid = status.get('pid')
        cmd_path = Path(f'/proc/{pid}/cmdline')
        alive = cmd_path.exists() and b'baseline_model_b_legacy_laptop.py' in cmd_path.read_bytes()
        accepted = status['stage'] == 'legacy_comparison_completed'
        if args.allow_stopped_baseline and status['stage'] == 'stopped_by_user_after_fit':
            fit = read(root / 'legacy-case/fit-baseline-1-legacy.json')
            timing = read(root / 'legacy-case/simulation-timings.json')
            if fit['status'] not in ['completed', 'fit_time_limit'] or len(timing['pairs']) != 30:
                raise RuntimeError('Stopped baseline lacks completed fit and simulation measurements')
            accepted = True
        if not accepted and not alive:
            raise RuntimeError('Baseline ended without a completed status; do not silently proceed.')
        return accepted and not alive, status['stage']

    try:
        verify_inputs()
        x0 = np.asarray(control_fit['x0'])
        if x0.shape != (13,) or not np.isfinite(x0).all() or not (x0 > 0).all():
            raise ValueError('Expected the archived positive 13-coordinate starting vector')
        user_options = dict(random_seed=1701, uncertainty_handling=False,
            specify_target_noise=False, nonlinear_scaling=False, max_fun_evals=6500,
            max_iter=2600, display='iter', show_tips=False)
        input_paths = [root / 'source-manifest.json', previous / 'protocol.json',
                       previous / 'fit-baseline-1-early_stop.json', previous / 'trace-baseline-1-early_stop.jsonl',
                       previous / 'data-baseline.npz', previous / 'spatial-projections.json']
        input_paths += sorted(previous.glob('score-*.json')) + sorted(previous.glob('score-*.npz'))
        input_paths += [previous / 'spatial-scores.json', previous / 'historical-scores.json']
        environment = {}
        for dist in importlib.metadata.distributions():
            record = dist.read_text('RECORD')
            environment[dist.metadata['Name']] = dict(version=dist.version,
                installed_record_sha256=hashlib.sha256(record.encode()).hexdigest() if record else None)
            if dist.metadata['Name'].lower() in ['pybads', 'gpyreg']:
                environment[dist.metadata['Name']]['installed_file_hashes'] = {
                    str(p): digest(dist.locate_file(p)) for p in dist.files or []
                    if '__pycache__' not in str(p) and dist.locate_file(p).is_file()}
        manifest = dict(source_commit=source_manifest['commit'],
            input_hashes={str(p.relative_to(root)): digest(p) for p in input_paths},
            tool_hashes={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
            baseline_protocol=protocol, x0=x0.tolist(), plausible_lower=(.5*x0).tolist(),
            plausible_upper=(1.5*x0).tolist(), hard_bounds=None, options=user_options,
            primary_seconds=primary_seconds, cap_seconds=8100, checkpoints=cutoffs,
            environment=environment, seed=321, optimizer_seed=1701, analysis_seed=1702,
            allow_stopped_baseline=args.allow_stopped_baseline,
            scoring_seeds=list(range(6001, 6021)), historical_seeds=[999, 5001, 5002],
            full_scorecard_complete=False)
        write_json(out / 'manifest.json', manifest)
        publish('waiting_for_baseline')
        while True:
            done, phase = baseline_finished()
            if done:
                break
            publish(waiting_for=phase)
            time.sleep(30)
        # Review and retain the predecessor's terminal measurements before GPU use.
        summary['preceding_baseline'] = read(root / 'legacy-case/summary.json')
        write_json(out / 'preceding-baseline-summary.json', summary['preceding_baseline'])
        verify_inputs()
        for name, expected in manifest['input_hashes'].items():
            if digest(root / name) != expected:
                raise ValueError(f'Frozen scoring/control input changed while waiting: {name}')
        # Refuse another active process on this GPU. This process has not yet imported JAX.
        query = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name',
                                '--format=csv,noheader'], capture_output=True, text=True, check=True)
        if query.stdout.strip():
            raise RuntimeError('Another GPU compute process is present: ' + query.stdout.strip())
        state.pop('waiting_for', None)
        gpu_started = time.perf_counter()
        publish('gpu_preflight')
        os.environ.setdefault('JAX_PLATFORMS', 'cuda')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.5')
        sys.path.insert(0, str(source))
        import jax
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_default_matmul_precision', 'default')
        import jax.numpy as jnp
        from model_b import objective, grf_kl
        from scripts import benchmark_speedups as bench
        from model_b_baseline_observer import load_observer, sample_observed, sliced_wasserstein
        from pybads import BADS
        assert jax.default_backend() == 'gpu' and '4060' in jax.devices()[0].device_kind
        grf_kl._BASIS_CACHE_MAXSIZE = 2
        data_np = dict(np.load(previous / 'data-baseline.npz'))
        data = {k: jnp.asarray(v) for k, v in data_np.items()}
        truth = np.asarray(control_fit['true_params'])
        options = dict(nsim=9000, chunk_size=8, use_kl=True, time_block_size=32,
                       stop_early=True, compact=True, stable_kl_noise=True)
        effective = lambda p: np.asarray(objective.clamp_b(jnp.asarray(p)))

        def loss(p):
            return float(jax.block_until_ready(objective.fofs_b_new(
                jnp.asarray(p), data, jax.random.key(321), **options)))

        def monitor():
            with (out / 'gpu-telemetry.csv').open('a', buffering=1) as handle:
                while not stop_monitor.is_set():
                    result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
                        '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
                    if result.returncode == 0:
                        handle.write(result.stdout)
                    stop_monitor.wait(5)
        monitor_thread = threading.Thread(target=monitor, daemon=True); monitor_thread.start()
        # Test wrapper parity and clamping with a small trial budget, never a new objective.
        preflight = []
        for vector in [x0, x0.copy()]:
            if preflight:
                vector[0], vector[1], vector[2], vector[3], vector[5] = 100., 500., .5, 20., 18.
            small = dict(options, nsim=8)
            a = float(objective.fofs_b_new(jnp.asarray(vector), data, jax.random.key(321), **small))
            b = float(objective.fofs_b_new(jnp.asarray(effective(vector)), data, jax.random.key(321), **small))
            if not np.isfinite(a) or a != b:
                raise AssertionError('Clamped objective preflight mismatch')
            preflight.append(dict(raw=vector.tolist(), effective=effective(vector).tolist(), loss=a))
        write_json(out / 'preflight.json', preflight)
        jax.clear_caches(); grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()
        publish('warming_objective')
        warm_start = time.perf_counter(); initial_loss = loss(x0)
        warm_seconds = time.perf_counter()-warm_start
        initial_control = json.loads((previous / 'trace-baseline-1-early_stop.jsonl').read_text().splitlines()[0])
        np.testing.assert_allclose(initial_loss, initial_control['loss'], rtol=1e-12, atol=1e-9)
        write_json(out / 'warmup.json', dict(seconds=warm_seconds, loss=initial_loss,
            expected_control_loss=initial_control['loss']))

        def update(row):
            valid = [r for r in recorded.rows if r['valid']]
            write_json(out / 'checkpoints.json', checkpoint_candidates(valid, cutoffs))
            publish(evaluations=len(recorded.rows), training_seconds=row['elapsed'],
                    best_training_loss=min((r['loss'] for r in valid), default=None))

        publish('fitting_bads')
        recorded = RecordedObjective(loss, effective, out / 'trace.jsonl', on_update=update)
        callbacks = []
        def callback(p, optim_state, flag):
            callbacks.append(dict(elapsed=recorded.elapsed(), flag=flag,
                actual_objective_calls=recorded.requests,
                bads_func_count=optimizer.function_logger.func_count,
                bads_cache_count=optimizer.function_logger.cache_count,
                state=serializable(optim_state)))
            write_json(out / 'optimizer-callbacks.json', callbacks)
            return recorded.elapsed() >= recorded.budget
        user_options['output_fcn'] = callback
        native = None
        fit_status = 'completed'
        try:
            optimizer = BADS(recorded, x0=x0, lower_bounds=None, upper_bounds=None,
                plausible_lower_bounds=.5*x0, plausible_upper_bounds=1.5*x0, options=user_options)
            write_json(out / 'resolved-options.json', dict(optimizer.options))
            native = optimizer.optimize()
            if recorded.elapsed() >= 8100:
                fit_status = 'fit_time_limit'
        except FitDeadline:
            fit_status = 'fit_time_limit'
        fit_seconds = recorded.elapsed()
        rows = [r for r in recorded.rows if r['valid']]
        primary = select_candidate(rows, primary_seconds)
        final = select_candidate(rows, min(fit_seconds, 8100))
        if primary is None or final is None:
            raise RuntimeError('No valid candidate completed within the declared budget')
        control_rows = [json.loads(line) for line in (previous / 'trace-baseline-1-early_stop.jsonl').read_text().splitlines()]
        control = select_candidate(control_rows, primary_seconds)
        fit_record = dict(status=fit_status, fit_seconds=fit_seconds, evaluations=len(rows),
            warmup_seconds=warm_seconds, primary=primary, final=final,
            native_result=serializable(native), selected_control=control,
            objective_seconds=sum(r['objective_seconds'] for r in rows),
            optimizer_and_logging_seconds=fit_seconds-sum(r['objective_seconds'] for r in rows))
        fit_record['bads_counters'] = dict(func_count=optimizer.function_logger.func_count,
            cache_count=optimizer.function_logger.cache_count,
            note='Native counters; actual synchronized calls are recorded separately. A cap-crossing call can precede native logger bookkeeping.')
        write_json(out / 'fit.json', fit_record)
        summary['fit'] = fit_record
        for label, row in [('primary', primary), ('final', final), ('control', control)]:
            summary[label] = {'params': row['params'], 'recovery': recovery(effective(row['params']), truth)}
        publish('fit_completed', training_seconds=fit_seconds)
        jax.clear_caches(); grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()
        observer = load_observer(source)
        directions = np.asarray(read(previous / 'spatial-projections.json'))
        spatial_control = {r['seed']: r['sw1_pixels_by_condition'] for r in read(previous / 'spatial-scores.json')['records']}
        settings = type('Settings', (), dict(chunk_a=256, chunk_b=8, block_size=32,
                    kl_bucketed=False, cumulative_kl=False, stable_kl_noise=True))()
        scores_by_label = {}
        score_cache = {}
        for label in ['control', 'primary', 'final']:
            p = summary[label]['params']
            identity = tuple(effective(p).tolist())
            if identity in score_cache:
                scores_by_label[label] = score_cache[identity]
                continue
            records = []
            reuse = label == 'control' and np.array_equal(effective(p), effective(control_fit['params']))
            for seed in range(6001, 6021):
                publish('independent_scoring', estimate=label, seed=seed)
                if reuse:
                    record = dict(read(previous / f'score-fit-{seed}.json'),
                        sw1_pixels_by_condition=spatial_control[seed], reused_from_control=True)
                else:
                    t = time.perf_counter()
                    captured = sample_observed(observer, p, seed)
                    if seed == 6001:
                        rt, cat, _ = bench.simulate_conditions(jax, 'b', p, 9000, seed, 'early_stop', settings)
                        np.testing.assert_array_equal(captured['rt'], rt)
                        np.testing.assert_array_equal(captured['cat'], cat)
                    np.savez_compressed(out / f'score-{label}-{seed}.npz', **captured)
                    g2 = sum(float(objective.condition_g2_b(jnp.asarray(captured['rt'][c]),
                        jnp.asarray(captured['cat'][c]), data['prop'][c], data['count'][c], data['quant'][c])) for c in range(2))
                    with np.load(previous / f'score-truth-{seed}.npz') as archive:
                        truth_xy = archive['xy_pixels']
                    record = dict(seed=seed, g2=g2, estimate=label,
                        metrics=observable_metrics(captured['rt'], captured['cat'], data_np),
                        no_crossing_fraction=(~captured['crossed']).mean(axis=1).tolist(),
                        sw1_pixels_by_condition=[sliced_wasserstein(truth_xy[c], captured['xy_pixels'][c], directions) for c in range(2)],
                        seconds=time.perf_counter()-t)
                write_json(out / f'score-{label}-{seed}.json', record); records.append(record)
            score_cache[identity] = records
            scores_by_label[label] = records
        write_json(out / 'independent-scores.json', scores_by_label)
        analysis = paired_analysis(scores_by_label['primary'], scores_by_label['control'])
        summary['analysis'] = analysis
        summary['decision'] = classify(summary['primary']['recovery'], analysis)
        # Retain cellwise changes in error, including regressions hidden by maxima.
        for name in ['probability_absolute_errors_pp', 'quantile_absolute_errors_ms']:
            a = np.asarray([r['metrics'][name] for r in scores_by_label['primary']], dtype=float)
            b = np.asarray([r['metrics'][name] for r in scores_by_label['control']], dtype=float)
            complete = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
            delta = np.mean(a-b, axis=0)
            summary.setdefault('cellwise_error_changes', {})[name] = dict(
                mean_candidate_minus_control=np.where(complete, delta, None).tolist(),
                worsened_cells=np.argwhere(complete & (delta > 0)).tolist())
        write_json(out / 'analysis.json', summary)
        jax.clear_caches(); grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()
        historical_cache = {}
        historical_results = {}
        for label in ['primary', 'final']:
            p = summary[label]['params']; identity = tuple(effective(p).tolist())
            if identity not in historical_cache:
                values = []
                for seed in [999, 5001, 5002]:
                    publish('historical_scoring', estimate=label, seed=seed)
                    t = time.perf_counter()
                    value = float(objective.fofs_b_new(jnp.asarray(p), data, jax.random.key(seed),
                        nsim=9000, chunk_size=8, use_kl=True, stable_kl_noise=True))
                    record = dict(seed=seed, g2=value, seconds=time.perf_counter()-t)
                    write_json(out / f'historical-{label}-{seed}.json', record); values.append(record)
                historical_cache[identity] = values
            historical_results[label] = historical_cache[identity]
        summary['historical_scores'] = historical_results
        verify_inputs()
        summary['total_seconds_after_queue'] = time.perf_counter()-gpu_started
        summary['full_scorecard_complete'] = False
        write_json(out / 'summary.json', summary)
        publish('completed')
    except BaseException as exc:
        publish('error', error=repr(exc))
        raise
    finally:
        stop_monitor.set()
        if 'monitor_thread' in locals():
            monitor_thread.join(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--allow-stopped-baseline', action='store_true',
        help='Explicit user authorization to proceed after stopping optional baseline measurements after its fit.')
    main(parser.parse_args())

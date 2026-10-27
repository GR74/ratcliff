"""Measure the archived legacy comparator without changing scientific code.

Run only after the improved archived-case screen has completed. Timed simulator
and objective pairs alternate order; fitting uses the unchanged historical worker.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from baseline_model_b_laptop import (
    median_interval, observable_metrics, recovery, write_json,
)


def main(args):
    root = Path(args.root).resolve()
    source = root / 'source-322bd1f'
    previous = root / 'archived-case'
    if json.loads((previous / 'status.json').read_text())['stage'] != 'archived_case_screen_completed':
        raise RuntimeError('The improved run must finish before this GPU study starts.')
    manifest = json.loads((root / 'source-manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        if hashlib.sha256((source / relative.replace('\\', '/')).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    out = root / 'legacy-case'
    out.mkdir(exist_ok=True)
    destination = Path(args.report).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    original_protocol = json.loads((previous / 'protocol.json').read_text())
    protocol = dict(original_protocol,
        variant='legacy', time_block_size=0, stop_early=False, compact=False,
        comparator='Archived full-horizon JAX/K-L comparator with stable mode-indexed noise; not a new Fortran run.',
        fit_seconds_limit=8100, timing_order='Alternate legacy/early_stop and early_stop/legacy by pair.',
        timing_cache='Clear basis cache before each objective call for symmetric cache misses.',
        timing_warmups_per_variant=3, timing_pairs=30,
        phase_order=['paired_simulation', 'legacy_fit', 'paired_objective', 'fresh_scoring', 'historical_scoring'],
        shared_truth_scores='Reuse archived-case truth scores/samples from this laptop, with hashes recorded.',
        scoring_simulator='Same early-stop observable scorer as improved fit; separate full-horizon historical scorer.',
        scope='User authorized original comparator simulation and fit on RTX 4060; 27-fit suite remains deferred.')
    protocol['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol['helper_sha256'] = hashlib.sha256(Path(__file__).with_name('baseline_model_b_laptop.py').read_bytes()).hexdigest()
    protocol['observer_sha256'] = hashlib.sha256(Path(__file__).with_name('model_b_baseline_observer.py').read_bytes()).hexdigest()
    inputs = [previous / 'fit-baseline-1-early_stop.json', previous / 'protocol.json']
    for seed in protocol['score_seeds']:
        inputs.extend([previous / f'score-truth-{seed}.json', previous / f'score-truth-{seed}.npz'])
    protocol['input_hashes'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; preserve this run and use a new output directory.')
    write_json(path, protocol)
    data_path = out / 'data-baseline.npz'
    if not data_path.exists():
        shutil.copyfile(root / 'historical/data-baseline.npz', data_path)
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == protocol['data_sha256']
    if args.prepare_only:
        print('Frozen source, input hashes, protocol and data verified; no GPU work started.')
        return
    if (out / 'status.json').exists():
        raise RuntimeError('Existing execution status: inspect it before starting a replacement run.')

    sys.path.insert(0, str(source))
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.5')
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'default')
    import jax.numpy as jnp
    import numpy as np
    from scripts import benchmark_speedups as bench
    from scripts import validate_early_stopping as historical
    from model_b import grf_kl, objective
    from model_b_baseline_observer import load_observer, sample_observed, sliced_wasserstein
    assert jax.default_backend() == 'gpu' and '4060' in jax.devices()[0].device_kind
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    truth = np.asarray(protocol['truth'])
    data_np = dict(np.load(data_path))
    data = {k: jnp.asarray(v) for k, v in data_np.items()}
    settings = SimpleNamespace(chunk_a=256, chunk_b=8, block_size=32,
        kl_bucketed=False, cumulative_kl=False, stable_kl_noise=True)
    started = time.perf_counter()
    stop = threading.Event()
    report_lock = threading.Lock()

    def report():
        with report_lock:
            current = json.loads((out / 'status.json').read_text())
            lines = ['# RTX 4060 original versus improved Model B', '',
                f"Status: **{current['stage']}**. Updated {datetime.now(timezone.utc).isoformat()}.", '',
                'Original means the archived full-horizon JAX comparator at `322bd1f`, with the same noise repair as the improved version. No scientific or optimizer code is changed.', '',
                '| Measurement | Original | Improved |', '| --- | --- | --- |']
            snapshot = {'status': current, 'protocol': protocol}
            for phase in ['simulation', 'objective']:
                path = out / f'{phase}-timings.json'
                if path.exists():
                    timing = json.loads(path.read_text()); snapshot[phase] = timing
                    lines.append(f"| Paired {phase}, median seconds | {timing['legacy']['median_seconds']:.3f} | {timing['early_stop']['median_seconds']:.3f} |")
                    lines.append(f"| {phase} speed ratio (original/improved) | {timing['ratio_of_medians']:.3f}x | Same-session 30 pairs |")
            fit_path = out / 'fit-baseline-1-legacy.json'
            if fit_path.exists():
                fit_result = json.loads(fit_path.read_text()); snapshot['fit'] = fit_result
                improved = json.loads((previous / 'fit-baseline-1-early_stop.json').read_text())
                for title, left, right in [
                    ('Fit status', fit_result['status'], improved['status']),
                    ('Converged', fit_result['converged'], improved['converged']),
                    ('Training minutes', f"{fit_result['fit_seconds']/60:.2f}", f"{improved['fit_seconds']/60:.2f}"),
                    ('Evaluations', fit_result['n_evals'], improved['n_evals']),
                    ('Mean active error', f"{100*fit_result['parameter_recovery']['mean_absolute_relative_error']:.3f}%", f"{100*improved['parameter_recovery']['mean_absolute_relative_error']:.3f}%"),
                    ('Worst active error', f"{100*fit_result['parameter_recovery']['worst_absolute_relative_error']:.3f}%", f"{100*improved['parameter_recovery']['worst_absolute_relative_error']:.3f}%")]:
                    lines.append(f'| {title} | {left} | {right} |')
            else:
                trace = out / 'trace-baseline-1-legacy.jsonl'
                if trace.exists():
                    rows = trace.read_text().splitlines()
                    # Ignore a partially written final line while the worker appends.
                    for line in reversed(rows):
                        try:
                            latest = json.loads(line); break
                        except json.JSONDecodeError:
                            continue
                    else:
                        latest = None
                    if latest:
                        snapshot['latest_fit_evaluation'] = latest
                        lines.append(f"| Fit progress | Evaluation {latest['eval']}; {latest['elapsed']/60:.2f} min | Completed |")
            for name in ['independent-scores', 'historical-scores', 'spatial-scores', 'summary']:
                path = out / f'{name}.json'
                if path.exists():
                    snapshot[name] = json.loads(path.read_text())
            if 'historical-scores' in snapshot:
                values = [r['g2'] for r in snapshot['historical-scores']]
                old_scores = json.loads((previous / 'historical-scores.json').read_text())
                improved_values = [r['g2'] for r in old_scores if r['estimate'] == 'fit']
                lines.append(f'| Historical fitted median G2 | {np.median(values):.3f} | {np.median(improved_values):.3f} |')
            lines += ['', 'Training uses the historical 135-minute soft cap, checked after each evaluation. A capped fit is not a completed fit; no time-to-quality speedup is established. Warm-up and scoring are reported separately in the evidence.', '',
                'Timing intervals describe this session only. Objective pairs clear the basis cache symmetrically; these differ from the earlier unpaired timing suite. Common fresh-seed scoring reuses the improved run\'s saved truth samples. Independent scientific validation, prediction-threshold calibration and the 27-fit suite remain outstanding.', '',
                '[Improved-run report](README.md) · [Comparison evidence](legacy-comparison-evidence.json)', '']
            target = destination / 'legacy-comparison.md'
            temp = target.with_suffix('.md.tmp'); temp.write_text('\n'.join(lines)); temp.replace(target)
            write_json(destination / 'legacy-comparison-evidence.json', snapshot)

    def status(stage, **extra):
        write_json(out / 'status.json', dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), session_elapsed_seconds=time.perf_counter()-started, **extra))
        report()
        print(stage, extra, flush=True)

    def monitor():
        with (out / 'gpu-telemetry.csv').open('a', buffering=1) as handle:
            while not stop.is_set():
                result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
                    '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if result.returncode == 0:
                    handle.write(result.stdout)
                try:
                    report()
                except (OSError, ValueError) as exc:
                    print('Report refresh warning:', repr(exc), flush=True)
                stop.wait(10)

    def metadata():
        value = bench.metadata(jax)
        value['enclosing_checkout_commit'] = value.get('commit')
        value['commit'] = manifest['commit']
        value['verified_source_files'] = len(manifest['files'])
        return value

    def clear_basis():
        grf_kl._basis_cache.clear(); grf_kl._basis_cache_order.clear()

    def sim(p, variant, seed=0):
        return bench.simulate_conditions(jax, 'b', p, 9000, seed, variant, settings)

    def loss(p, variant, seed=321):
        cfg = {} if variant == 'legacy' else dict(time_block_size=32, stop_early=True, compact=True)
        return float(jax.block_until_ready(objective.fofs_b_new(jnp.asarray(p), data,
            jax.random.key(seed), nsim=9000, chunk_size=8, use_kl=True, stable_kl_noise=True, **cfg)))

    def paired_timing(phase):
        status(f'warming_{phase}')
        jax.clear_caches(); clear_basis()
        warm = {'legacy': [], 'early_stop': []}
        times = {'legacy': [], 'early_stop': []}
        scores = {'legacy': [], 'early_stop': []}
        pairs = []
        def call(p, variant):
            if phase == 'objective':
                clear_basis()
            t = time.perf_counter()
            result = sim(p, variant) if phase == 'simulation' else loss(p, variant)
            elapsed = time.perf_counter()-t
            if phase == 'simulation':
                rt, cat, _ = result
                if not np.isfinite(rt).all() or not np.isin(cat, [1, 2, 3, 4, 5]).all():
                    raise ValueError('Invalid simulation output')
                np.savez_compressed(out / f'timing-samples-{variant}.npz', rt=rt, cat=cat)
            elif not np.isfinite(result):
                raise ValueError('Nonfinite objective')
            return elapsed, result
        for i in range(3):
            for variant in (['legacy', 'early_stop'] if i % 2 == 0 else ['early_stop', 'legacy']):
                elapsed, _ = call(truth, variant); warm[variant].append(elapsed)
                status(f'warming_{phase}', warmups={k: len(v) for k,v in warm.items()})
        for i in range(30):
            p = truth if phase == 'simulation' else protocol['timing_vectors'][i]
            order = ['legacy', 'early_stop'] if i % 2 == 0 else ['early_stop', 'legacy']
            for variant in order:
                elapsed, result = call(p, variant); times[variant].append(elapsed)
                if phase == 'objective':
                    scores[variant].append(result)
            pairs.append({'index': i, 'order': order, 'params': list(p),
                          'seconds': {k: v[-1] for k, v in times.items()}})
            write_json(out / f'{phase}-progress.json', dict(warmups=warm, pairs=pairs, losses=scores))
            status(f'timing_{phase}', pairs_completed=i+1, pairs_total=30)
        a, b = np.asarray(times['legacy']), np.asarray(times['early_stop'])
        rng = np.random.default_rng(702)
        indices = rng.integers(0, 30, size=(10000, 30))
        ratios = np.median(a[indices], axis=1) / np.median(b[indices], axis=1)
        differences = np.median((a-b)[indices], axis=1)
        write_json(out / f'{phase}-timings.json', dict(
            legacy=median_interval(a), early_stop=median_interval(b), warmup_seconds=warm,
            pairs=pairs, losses=scores, invalid_evaluations=0,
            ratio_of_medians=float(np.median(a)/np.median(b)),
            ratio_95_percent_paired_bootstrap_interval=np.quantile(ratios, [.025,.975]).tolist(),
            median_paired_difference_seconds=float(np.median(a-b)),
            difference_95_percent_paired_bootstrap_interval=np.quantile(differences, [.025,.975]).tolist(),
            metadata=metadata(), interval_note='Descriptive paired bootstrap within one session, not across-day uncertainty.'))

    status('starting_legacy_comparison')
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        paired_timing('simulation')
        jax.clear_caches(); clear_basis()
        status('fitting_legacy')
        fit_path = out / 'fit-baseline-1-legacy.json'
        if (out / 'trace-baseline-1-legacy.jsonl').exists():
            raise RuntimeError('Preserve existing fit trace; do not silently append a replacement fit.')
        fit_args = historical.parser().parse_args([
            '--worker', '--action', 'fit', '--variant', 'legacy', '--stable-kl-noise',
            '--gpu-kind', '4060', '--nsim', '9000', '--chunk', '8', '--start-seed', '1',
            '--start-spread', '.2', '--maxiter', '1000', '--checkpoint-every', '25',
            '--fit-seconds-limit', '8100', '--basis-cache-size', '2',
            '--output', str(out), '--result-file', str(fit_path)])
        fitted = historical.worker(fit_args)
        fitted['metadata'] = metadata()
        fitted['parameter_recovery'] = recovery(np.asarray(objective.clamp_b(jnp.asarray(fitted['params']))), truth)
        fitted['x0_recovery'] = recovery(fitted['x0'], truth)
        trace = [json.loads(line) for line in (out / 'trace-baseline-1-legacy.jsonl').read_text().splitlines()]
        best, last = float('inf'), 0.
        for row in trace:
            if row['loss'] < best:
                best, last = row['loss'], row['elapsed']
        fitted['seconds_after_last_strict_improvement'] = fitted['fit_seconds']-last
        write_json(fit_path, fitted)
        status('legacy_fit_finished', fit_status=fitted['status'])
        paired_timing('objective')
        jax.clear_caches(); clear_basis()
        observer = load_observer(source)
        directions = np.asarray(json.loads((previous / 'spatial-projections.json').read_text()))
        scores, spatial = [], []
        for seed in protocol['score_seeds']:
            status('fresh_scoring_legacy_fit', seed=seed)
            t = time.perf_counter()
            captured = sample_observed(observer, fitted['params'], seed)
            if seed == protocol['score_seeds'][0]:
                rt, cat, _ = sim(fitted['params'], 'early_stop', seed)
                np.testing.assert_array_equal(captured['rt'], rt)
                np.testing.assert_array_equal(captured['cat'], cat)
            np.savez_compressed(out / f'score-fit-{seed}.npz', **captured)
            g2 = sum(float(objective.condition_g2_b(jnp.asarray(captured['rt'][c]),
                jnp.asarray(captured['cat'][c]), data['prop'][c], data['count'][c], data['quant'][c])) for c in range(2))
            with np.load(previous / f'score-truth-{seed}.npz') as archive:
                reference_xy = archive['xy_pixels']
            spatial.append(dict(seed=seed, sw1_pixels_by_condition=[sliced_wasserstein(
                reference_xy[c], captured['xy_pixels'][c], directions) for c in range(2)]))
            record = dict(estimate='legacy_fit', seed=seed, g2=g2,
                no_crossing_fraction=(~captured['crossed']).mean(axis=1).tolist(),
                metrics=observable_metrics(captured['rt'], captured['cat'], data_np),
                seconds=time.perf_counter()-t)
            write_json(out / f'score-fit-{seed}.json', record); scores.append(record)
        write_json(out / 'independent-scores.json', dict(records=scores,
            truth_source='../archived-case/score-truth-*.json', prediction_acceptance='Uncalibrated'))
        write_json(out / 'spatial-scores.json', dict(records=spatial, directions=directions.tolist(),
            note='Conditional on crossing; missing locations are excluded and no-crossing mass is reported separately.'))
        jax.clear_caches(); clear_basis()
        old_scores = []
        for seed in protocol['historical_score_seeds']:
            status('historical_scoring_legacy_fit', seed=seed)
            t = time.perf_counter(); value = loss(fitted['params'], 'legacy', seed)
            record = dict(estimate='legacy_fit', seed=seed, g2=value, seconds=time.perf_counter()-t)
            write_json(out / f'historical-score-fit-{seed}.json', record); old_scores.append(record)
        write_json(out / 'historical-scores.json', old_scores)
        write_json(out / 'summary.json', dict(status='legacy_comparison_completed', metadata=metadata(),
            fit_status=fitted['status'], training_seconds=fitted['fit_seconds'],
            session_elapsed_seconds=time.perf_counter()-started, full_scorecard_complete=False,
            no_time_to_quality_claim=True, outstanding=['Prediction acceptance calibration',
            'Independent scientific-reference validation', '27-fit qualification remains deferred']))
        status('legacy_comparison_completed')
    except BaseException as exc:
        status('error', error=repr(exc)); raise
    finally:
        stop.set(); thread.join(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args())

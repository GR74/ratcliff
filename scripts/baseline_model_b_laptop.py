"""Record an archived-case baseline against an immutable Model B source snapshot.

This measurement driver does not change simulator or optimizer implementations.
Run inside the pinned Linux GPU environment with --root pointing to the prepared
directory containing source-322bd1f, source-manifest.json and historical/.
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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def median_interval(values, seed=701):
    import numpy as np
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    bootstrap = np.median(rng.choice(values, (10000, len(values))), axis=1)
    return {"median_seconds": float(np.median(values)),
            "median_95_percent_bootstrap_ci_seconds": np.quantile(bootstrap, [.025, .975]).tolist(),
            "raw_seconds": values.tolist(),
            "ci_note": "Descriptive within-session IID bootstrap; not across-day uncertainty."}


def recovery(vector, truth):
    import numpy as np
    active = np.asarray([i for i in range(13) if i != 6])
    signed = np.asarray(vector) - np.asarray(truth)
    relative = np.abs(signed[active]) / np.abs(np.asarray(truth)[active])
    return {"signed_error": signed.tolist(), "absolute_error": np.abs(signed).tolist(),
            "active_indices": active.tolist(), "absolute_relative_error": relative.tolist(),
            "mean_absolute_relative_error": float(relative.mean()),
            "worst_absolute_relative_error": float(relative.max()),
            "recovery_screen_pass": bool(relative.mean() <= .05 and relative.max() <= .10),
            "inactive_sv_note": "Historical sv is inert, excluded from recovery acceptance."}


def observable_metrics(rt, cat, data):
    import numpy as np
    counts = np.asarray([[(c == k).sum() for k in range(1, 6)] for c in cat])
    proportions = counts / rt.shape[1]
    quantiles = np.full((2, 5, 5), np.nan)
    for ci in range(2):
        for k in range(1, 6):
            selected = rt[ci][cat[ci] == k]
            if len(selected) >= 10:
                quantiles[ci, :, k-1] = np.quantile(selected, [.1, .3, .5, .7, .9])
    probability_errors = np.abs(proportions - data['prop']) * 100
    errors = np.abs(quantiles - data['quant'])
    valid = np.isfinite(errors)
    return {
        'counts': counts.tolist(), 'proportions': proportions.tolist(),
        'conditional_quantiles_ms': np.where(np.isfinite(quantiles), quantiles, None).tolist(),
        'probability_absolute_errors_pp': probability_errors.tolist(),
        'probability_mean_absolute_error_pp': float(probability_errors.mean()),
        'probability_max_absolute_error_pp': float(probability_errors.max()),
        'quantile_absolute_errors_ms': np.where(valid, errors, None).tolist(),
        'quantile_mean_absolute_error_ms': float(errors[valid].mean()) if valid.any() else None,
        'quantile_max_absolute_error_ms': float(errors[valid].max()) if valid.any() else None,
        'missing_quantiles': int((~valid).sum()),
        'note': 'Errors against the fixed observed synthetic data; not simulator-equivalence tests.',
    }


def main(args):
    root = Path(args.root).resolve()
    source = root / 'source-322bd1f'
    manifest = json.loads((root / 'source-manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        if hashlib.sha256((source / relative.replace('\\', '/')).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    sys.path.insert(0, str(source))
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.5')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'default')
    import jax.numpy as jnp
    import numpy as np
    from model_b_baseline_observer import load_observer, sample_observed, sliced_wasserstein
    from scripts import benchmark_speedups as bench
    from scripts import validate_early_stopping as original
    from model_b import grf_kl, objective
    from types import SimpleNamespace
    assert jax.default_backend() == 'gpu'
    assert '4060' in jax.devices()[0].device_kind
    grf_kl._BASIS_CACHE_MAXSIZE = 2
    truth = np.asarray(bench.TRUE['b'])
    out = root / 'archived-case'
    out.mkdir(exist_ok=True)
    source_data = root / 'historical/data-baseline.npz'
    data_path = out / 'data-baseline.npz'
    if not data_path.exists():
        shutil.copyfile(source_data, data_path)
    assert data_path.read_bytes() == source_data.read_bytes()
    data_np = dict(np.load(data_path))
    data = {k: jnp.asarray(v) for k, v in data_np.items()}
    settings = SimpleNamespace(chunk_a=256, chunk_b=8, block_size=32,
                               kl_bucketed=False, cumulative_kl=False, stable_kl_noise=True)
    options = dict(nsim=9000, chunk_size=8, use_kl=True, time_block_size=32,
                   stop_early=True, compact=True, stable_kl_noise=True)
    # Thirty archived vectors deliberately force changing-width cache misses.
    vectors = []
    for index in range(30):
        p = truth.copy()
        p[5] = 8. + 4. * index / 29.
        vectors.append(p.tolist())
    protocol = dict(source_commit=manifest['commit'], truth=truth.tolist(), nsim=9000,
        data_sha256=hashlib.sha256(source_data.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observer_sha256=hashlib.sha256(Path(__file__).with_name('model_b_baseline_observer.py').read_bytes()).hexdigest(),
        warmups=3, timed_calls=30, chunk=8, block=32, basis_cache_size=2,
        training_seed=321, start_seed=1, start_spread=.2, maxiter=1000,
        fit_seconds_limit=8100, checkpoint_every=25,
        score_seeds=list(range(6001, 6021)), historical_score_seeds=[999, 5001, 5002],
        timing_vectors=vectors, precision='fp32 fields, fp64 RT, default matmul',
        scope='Archived-case screen only; 27-fit qualification awaits measured runtime review.',
        scientific_fidelity='Not independently validated against the scientific reference.',
        prediction_acceptance='Uncalibrated: report observables and truth MC distribution; no joint-quality pass claim.')
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Protocol changed; use a new output directory')
    else:
        write_json(protocol_path, protocol)
    started = time.perf_counter()
    stop_telemetry = threading.Event()

    def telemetry():
        with (out / 'gpu-telemetry.csv').open('a', buffering=1) as handle:
            while not stop_telemetry.is_set():
                result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
                    '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if result.returncode == 0:
                    handle.write(result.stdout)
                stop_telemetry.wait(5)

    monitor = threading.Thread(target=telemetry, daemon=True)
    monitor.start()

    def status(stage, **extra):
        record = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                      pid=os.getpid(), session_elapsed_seconds=time.perf_counter()-started, **extra)
        write_json(out / 'status.json', record)
        print(json.dumps(record), flush=True)

    def metadata():
        result = bench.metadata(jax)
        # git rev-parse from an unpacked snapshot otherwise finds the enclosing repo.
        result['enclosing_checkout_commit'] = result.get('commit')
        result['commit'] = manifest['commit']
        result['source_snapshot'] = {'commit': manifest['commit'], 'verified_files': len(manifest['files'])}
        return result

    def run_sim(p, seed=0, variant='early_stop', n=9000):
        return bench.simulate_conditions(jax, 'b', p, n, seed, variant, settings)

    def loss(p):
        return float(jax.block_until_ready(objective.fofs_b_new(
            jnp.asarray(p), data, jax.random.key(321), **options)))

    try:
        if not (out / 'timings.json').exists():
            status('timing_simulator')
            warm = []
            for _ in range(3):
                t = time.perf_counter(); run_sim(truth); warm.append(time.perf_counter()-t)
            sim_times = []
            for i in range(30):
                t = time.perf_counter(); rt, cat, work = run_sim(truth)
                sim_times.append(time.perf_counter()-t)
                write_json(out / 'timing-progress.json', dict(simulation_seconds=sim_times))
                if (i+1) % 5 == 0:
                    status('timing_simulator', completed=i+1, total=30)
            np.savez_compressed(out / 'baseline-seed-0.npz', rt=rt, cat=cat)
            status('paired_control')
            t = time.perf_counter()
            control = run_sim(truth, variant='blocked_full')
            paired = dict(seconds_including_first_compile=time.perf_counter()-t,
                          exact_rt_fraction=float(np.mean(control[0] == rt)),
                          exact_category_fraction=float(np.mean(control[1] == cat)))
            if paired['exact_rt_fraction'] != 1 or paired['exact_category_fraction'] != 1:
                raise AssertionError('Production early-stop/control mismatch')
            status('timing_objective')
            objective_warmups = []
            for _ in range(3):
                t = time.perf_counter(); loss(truth); objective_warmups.append(time.perf_counter()-t)
            obj_times, losses, modes = [], [], []
            for i, p in enumerate(vectors):
                t = time.perf_counter(); losses.append(loss(p)); obj_times.append(time.perf_counter()-t)
                basis = grf_kl.calc_kl_basis(p[5], pad_to_k_max=True, return_mode_ids=True)
                modes.append(dict(sig=p[5], retained_modes=basis[1], retained_variance=basis[2]))
                write_json(out / 'timing-progress.json', dict(simulation_seconds=sim_times,
                    objective_seconds=obj_times, loss=losses, modes=modes))
                if (i+1) % 5 == 0:
                    status('timing_objective', completed=i+1, total=30)
            write_json(out / 'timings.json', dict(simulation=median_interval(sim_times),
                simulator_warmup_seconds=warm, objective=median_interval(obj_times),
                objective_warmup_seconds=objective_warmups, losses=losses, modes=modes,
                invalid_evaluations=sum(not np.isfinite(v) for v in losses),
                paired_control=paired, computed_trial_steps=work, metadata=metadata()))

        # Only observation code exports extra values; timings and fits above/below
        # continue to call the original frozen functions.
        observed = load_observer(source)
        if not (out / 'observation-parity.json').exists():
            status('validating_observation_capture')
            captured = sample_observed(observed, truth, 0)
            plain = np.load(out / 'baseline-seed-0.npz')
            np.testing.assert_array_equal(captured['rt'], plain['rt'])
            np.testing.assert_array_equal(captured['cat'], plain['cat'])
            control = sample_observed(observed, truth, 0, stop_early=False)
            for name in ['rt', 'cat', 'steps', 'positions', 'crossed']:
                np.testing.assert_array_equal(captured[name], control[name])
            np.savez_compressed(out / 'observed-baseline-seed-0.npz', **captured)
            write_json(out / 'observation-parity.json', dict(status='passed', trials=18000,
                rt_category_exact_with_frozen=True, steps_positions_crossed_exact_with_blocked_control=True,
                no_crossing_fraction=(~captured['crossed']).mean(axis=1).tolist(),
                spatial_units='Display pixels, 2 pixels per cell; manuscript Experiment 9 p.38.',
                scientific_reference_independent=False))

        fit_path = out / 'fit-baseline-1-early_stop.json'
        if not fit_path.exists():
            if (out / 'trace-baseline-1-early_stop.jsonl').exists():
                raise RuntimeError('An interrupted fit trace exists; preserve it and choose a new fit run, not silent append.')
            # Discard observation/timing executables before the fresh fit warmup.
            jax.clear_caches()
            status('fitting_archived_case')
            fit_args = original.parser().parse_args([
                '--worker', '--action', 'fit', '--variant', 'early_stop', '--stable-kl-noise',
                '--gpu-kind', '4060', '--nsim', '9000', '--chunk', '8', '--start-seed', '1',
                '--start-spread', '.2', '--maxiter', '1000', '--checkpoint-every', '25',
                '--fit-seconds-limit', '8100', '--output', str(out), '--result-file', str(fit_path)])
            result = original.worker(fit_args)
            result['metadata'] = metadata()
            effective = np.asarray(objective.clamp_b(jnp.asarray(result['params'])))
            result['parameter_recovery'] = recovery(effective, truth)
            result['x0_recovery'] = recovery(result['x0'], truth)
            trace = [json.loads(line) for line in (out/'trace-baseline-1-early_stop.jsonl').read_text().splitlines()]
            best, last = float('inf'), 0.
            for row in trace:
                if row['loss'] < best:
                    best, last = row['loss'], row['elapsed']
            result['seconds_after_last_strict_improvement'] = result['fit_seconds'] - last
            write_json(fit_path, result)
        fit_result = json.loads(fit_path.read_text())

        # Re-score truth and returned fit at fresh keys. Each sample archive is
        # saved before computing summaries so interrupted scoring can resume.
        score_rows = []
        for label, p in [('truth', truth), ('fit', fit_result['params'])]:
            for seed in protocol['score_seeds']:
                record_path = out / f'score-{label}-{seed}.json'
                if record_path.exists():
                    score_rows.append(json.loads(record_path.read_text()))
                    continue
                status('independent_scoring', estimate=label, seed=seed)
                t = time.perf_counter()
                captured = sample_observed(observed, p, seed)
                rts, cats = captured['rt'], captured['cat']
                if seed == protocol['score_seeds'][0]:
                    plain_rt, plain_cat, _ = run_sim(p, seed=seed)
                    np.testing.assert_array_equal(rts, plain_rt)
                    np.testing.assert_array_equal(cats, plain_cat)
                np.savez_compressed(out / f'score-{label}-{seed}.npz', **captured)
                g2 = sum(float(objective.condition_g2_b(jnp.asarray(rts[c]), jnp.asarray(cats[c]),
                    data['prop'][c], data['count'][c], data['quant'][c])) for c in range(2))
                record = dict(estimate=label, seed=seed, seconds=time.perf_counter()-t,
                              no_crossing_fraction=(~captured['crossed']).mean(axis=1).tolist(),
                              g2=g2, metrics=observable_metrics(rts, cats, data_np))
                write_json(record_path, record); score_rows.append(record)
        write_json(out / 'independent-scores.json', dict(records=score_rows,
            simulator='Frozen repaired early-stop with fresh RNG; not an independent scientific implementation.',
            acceptance='Not calibrated; no joint-quality success asserted.'))
        angles = np.arange(64) * np.pi / 64
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        write_json(out / 'spatial-projections.json', directions.tolist())
        spatial = []
        for seed in protocol['score_seeds']:
            a = np.load(out / f'score-truth-{seed}.npz')['xy_pixels']
            b = np.load(out / f'score-fit-{seed}.npz')['xy_pixels']
            spatial.append(dict(seed=seed, sw1_pixels_by_condition=[
                sliced_wasserstein(a[c], b[c], directions) for c in range(2)]))
        write_json(out / 'spatial-scores.json', dict(records=spatial,
            note='Conditional on crossing. No-crossing mass is reported separately. Same-seed truth vs fitted predictions; not an equivalence gate.'))

        # Historical scores use the original full-horizon objective, including
        # its original vectorization and loss. Do not mix these with new scores.
        jax.clear_caches()
        grf_kl._basis_cache.clear()
        grf_kl._basis_cache_order.clear()
        historical = []
        for label, p in [('truth', truth), ('fit', fit_result['params'])]:
            for seed in protocol['historical_score_seeds']:
                path = out / f'historical-score-{label}-{seed}.json'
                if not path.exists():
                    status('historical_scoring', estimate=label, seed=seed)
                    t = time.perf_counter()
                    g2 = float(objective.fofs_b_new(jnp.asarray(p), data, jax.random.key(seed),
                        nsim=9000, chunk_size=8, use_kl=True, stable_kl_noise=True))
                    write_json(path, dict(estimate=label, seed=seed, g2=g2, seconds=time.perf_counter()-t))
                historical.append(json.loads(path.read_text()))
        write_json(out / 'historical-scores.json', historical)
        write_json(out / 'screen-summary.json', dict(
            status='archived_case_screen_completed', timings=json.loads((out/'timings.json').read_text()),
            parameter_recovery=fit_result['parameter_recovery'], fit_seconds=fit_result['fit_seconds'],
            fit_status=fit_result['status'], fit_evaluations=fit_result['n_evals'],
            independent_g2={label: [r['g2'] for r in score_rows if r['estimate']==label] for label in ['truth','fit']},
            historical_scores=historical, metadata=metadata(),
            outstanding=['Prediction acceptance calibration and qualifying checkpoint scoring',
                '27-fit qualification (user requested runtime report first)',
                'Independent scientific reference fidelity'],
            full_scorecard_complete=False))
        status('archived_case_screen_completed', full_scorecard_complete=False)
    except BaseException as exc:
        status('error', error=repr(exc))
        raise
    finally:
        stop_telemetry.set()
        monitor.join(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    main(parser.parse_args())

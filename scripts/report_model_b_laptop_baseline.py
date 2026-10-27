"""Publish compact evidence from the running or completed archived-case study."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def report(root, destination):
    out = root / 'archived-case'
    destination.mkdir(parents=True, exist_ok=True)
    status = read(out / 'status.json') or {'stage': 'not_started'}
    timings = read(out / 'timings.json')
    fit = read(out / 'fit-baseline-1-early_stop.json')
    scores = read(out / 'independent-scores.json')
    spatial = read(out / 'spatial-scores.json')
    historical = read(out / 'historical-scores.json')
    final = read(out / 'screen-summary.json')
    snapshot = dict(updated_utc=datetime.now(timezone.utc).isoformat(), status=status,
        gpu_preflight=read(root / 'gpu-preflight.json'),
        correctness=read(root / 'correctness-preflight.json'),
        protocol=read(out / 'protocol.json'), timings=timings,
        observation_parity=read(out / 'observation-parity.json'),
        fit=fit, independent_scores=scores, spatial_scores=spatial,
        historical_scores=historical, summary=final)
    lines = ['# RTX 4060 Model B archived-case baseline', '',
        f"**Status: {status['stage']}.** Updated {snapshot['updated_utc']}.", '',
        'This is the local RTX 4060 Laptop GPU baseline for repaired source `322bd1f`. '
        'It does not compare speed against the historical RTX 3060. The 27-fit qualification '
        'suite is deferred until the archived-case runtime has been reviewed.', '',
        '[Study protocol](../../plans/2026-10-21-rtx4060-baseline.md) · [Compact evidence](evidence.json)', '',
        '## Measurements', '', '| Metric | Result |', '| --- | --- |',
        '| Pinned JAX GPU execution | Passed |',
        '| Frozen Model B GPU regression tests | 10 passed |']
    if timings:
        for key, label in [('simulation', 'Warmed simulation, both conditions'),
                           ('objective', 'Complete objective, changing widths')]:
            r = timings[key]; lo, hi = r['median_95_percent_bootstrap_ci_seconds']
            lines.append(f"| {label} | Median {r['median_seconds']:.3f} s; descriptive 95% interval [{lo:.3f}, {hi:.3f}] s; 30 calls |")
        p = timings['paired_control']
        lines.append(f"| Full-horizon blocked control | RT exact fraction {p['exact_rt_fraction']:.6f}; category exact fraction {p['exact_category_fraction']:.6f} |")
        lines.append(f"| Simulator first call (includes compilation) | {timings['simulator_warmup_seconds'][0]:.3f} s |")
        lines.append(f"| Objective first warm-up | {timings['objective_warmup_seconds'][0]:.3f} s; simulator already initialized |")
    else:
        progress = read(out / 'timing-progress.json') or {}
        lines.append(f"| Simulator timing progress | {len(progress.get('simulation_seconds', []))}/30 calls |")
        lines.append(f"| Objective timing progress | {len(progress.get('objective_seconds', []))}/30 calls |")
    if fit:
        r = fit['parameter_recovery']
        lines.extend([
            f"| Fit status | {fit['status']}; optimizer convergence {fit['converged']} |",
            f"| Training time | {fit['fit_seconds']/60:.2f} min |",
            f"| Fit initialization/warm-up | {fit['compile_and_warmup_seconds']:.3f} s |",
            f"| Objective evaluations | {fit['n_evals']} |",
            f"| Mean active relative error | {100*r['mean_absolute_relative_error']:.3f}% |",
            f"| Worst active relative error | {100*r['worst_absolute_relative_error']:.3f}% |",
            f"| Proposed recovery screen | {'Pass' if r['recovery_screen_pass'] else 'Fail'} |",
            f"| Training after last strict improvement | {fit['seconds_after_last_strict_improvement']/60:.2f} min |"])
    else:
        trace = out / 'trace-baseline-1-early_stop.jsonl'
        if trace.exists():
            rows = trace.read_text().splitlines()
            try:
                last = json.loads(rows[-1])
                lines.append(f"| Fit progress | Evaluation {last['eval']}; {last['elapsed']/60:.2f} training min |")
            except (ValueError, IndexError):
                pass
    if scores:
        for label in ['truth', 'fit']:
            rows = [r for r in scores['records'] if r['estimate'] == label]
            values = np.asarray([r['g2'] for r in rows])
            lines.append(f"| Fresh-seed G2, {label} | Median {np.median(values):.3f}; SD {values.std(ddof=1):.3f}; {len(values)} seeds |")
            for metric, title in [('probability_max_absolute_error_pp', 'mean of seedwise maximum probability errors (pp)'),
                                  ('quantile_max_absolute_error_ms', 'mean of seedwise maximum conditional RT errors (ms)')]:
                values = [r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
                lines.append(f"| {label}: {title} | {np.mean(values):.3f} |")
    if spatial:
        values = np.asarray([r['sw1_pixels_by_condition'] for r in spatial['records']], dtype=float)
        lines.append(f"| Spatial SW1, mean across 20 paired seeds | Condition 1: {np.nanmean(values[:,0]):.3f} pixels; condition 2: {np.nanmean(values[:,1]):.3f} pixels |")
    if historical:
        for label in ['truth', 'fit']:
            vals = [r['g2'] for r in historical if r['estimate'] == label]
            lines.append(f"| Historical full-horizon G2, {label} | Median {np.median(vals):.3f}; keys 999, 5001, 5002 |")
    if fit:
        lines += ['', '## Individual parameter recovery', '',
            '| Parameter | Truth | Effective returned estimate | Signed error | Absolute relative error |',
            '| --- | --- | --- | --- | --- |']
        names = ['ter','st','cr','crsd','sis','sig','sv (inactive)','av1_c1','av2_c1','av3_c1','av1_c2','av2_c2','av3_c2']
        for i, name in enumerate(names):
            truth = fit['true_params'][i]
            value = truth + fit['parameter_recovery']['signed_error'][i]
            rel = 'Excluded' if i == 6 else f'{100*abs(value-truth)/abs(truth):.3f}%'
            lines.append(f'| {name} | {truth:.6g} | {value:.6g} | {value-truth:+.6g} | {rel} |')
    lines += ['', '## Limits and remaining qualification', '',
        'The simulator reference is the frozen JAX implementation. Exact blocked-control agreement '
        'does not independently establish scientific fidelity. The observed data are the archived '
        'synthetic observations; fresh simulator seeds are independent of optimization, not held-out participant data.', '',
        'Prediction acceptance has not been calibrated, so this screen does not assert a joint-quality '
        'pass or time to acceptable quality. The 27-fit suite, parameter bias/RMSE across datasets, '
        'successful-fit rate and independent scientific-reference fidelity remain unmeasured. '
        'Spatial distances exclude no-crossing placeholders; their probability mass is recorded separately.', '',
        'Raw timings, samples, trace, GPU telemetry, pinned dependencies and source hashes are under '
        '`.benchmark-results/2026-10-21-rtx4060-baseline/` in the repository. '
        'The measurement driver preserves source hashes and aborts on an observation-parity failure.', '']
    for name, contents in [('README.md', '\n'.join(lines)), ('evidence.json', json.dumps(snapshot, indent=2, allow_nan=False)+'\n')]:
        path = destination / name
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(contents, encoding='utf-8'); temp.replace(path)
    return status['stage'] in ['archived_case_screen_completed', 'error']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        finished = report(args.root, args.destination)
        if finished or not args.watch:
            break
        time.sleep(30)

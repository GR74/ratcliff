"""Pure analysis and budget controls for the frozen Model B BADS screen."""
import json
import math
from pathlib import Path
import time

import numpy as np


class FitDeadline(Exception):
    """The predeclared soft wall-time budget was reached."""


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if callable(value):
        return f'{value.__module__}.{value.__qualname__}'
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def select_candidate(rows, cutoff):
    eligible = [r for r in rows if r.get('valid', True) and r['elapsed'] <= cutoff
                and math.isfinite(r['loss'])]
    return min(eligible, key=lambda r: (r['loss'], r['elapsed'], r['eval'])) if eligible else None


def checkpoint_candidates(rows, cutoffs):
    return [{'cutoff_seconds': t, 'candidate': select_candidate(rows, t)} for t in cutoffs]


class RecordedObjective:
    def __init__(self, function, effective, path, *, budget=8100, clock=time.perf_counter,
                 on_update=None):
        self.function = function
        self.effective = effective
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError('Refusing to append a replacement fit to an existing trace')
        self.budget = budget
        self.clock = clock
        self.started = clock()
        self.rows = []
        self.requests = 0
        self.on_update = on_update

    def elapsed(self):
        return self.clock() - self.started

    def check_deadline(self):
        if self.elapsed() >= self.budget:
            raise FitDeadline('Training wall-time cap reached')

    def __call__(self, p):
        self.check_deadline()
        self.requests += 1
        begun = self.elapsed()
        record = {'eval': self.requests, 'params': np.asarray(p).tolist(),
                  'started_seconds': begun, 'valid': False}
        try:
            record['effective_params'] = np.asarray(self.effective(p)).tolist()
            value = float(self.function(p))
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite objective: {value}')
            record.update(valid=True, loss=value)
        except Exception as exc:
            record['error'] = repr(exc)
            raise
        finally:
            record['elapsed'] = self.elapsed()
            record['objective_seconds'] = record['elapsed'] - begun
            self.rows.append(record)
            with self.path.open('a') as handle:
                handle.write(json.dumps(serializable(record), allow_nan=False) + '\n')
            if self.on_update:
                self.on_update(record)
        # Preserve a completed over-budget evaluation, but exclude it from any
        # checkpoint whose deadline preceded its completion.
        self.check_deadline()
        return value


def paired_analysis(candidate, control):
    if len(candidate) != 20 or len(control) != 20:
        raise ValueError('Exactly 20 paired scoring seeds are required')
    c = {r['seed']: r for r in candidate}
    b = {r['seed']: r for r in control}
    seeds = list(range(6001, 6021))
    if sorted(c) != seeds or sorted(b) != seeds:
        raise ValueError('Scoring keys must be unique and match the frozen seed list')
    names = ['g2', 'maximum_probability_error_pp', 'maximum_quantile_error_ms',
             'spatial_sw1_condition_1', 'spatial_sw1_condition_2']
    def vector(row):
        m = row['metrics']
        return [row['g2'], m['probability_max_absolute_error_pp'],
                m['quantile_max_absolute_error_ms'], *row['sw1_pixels_by_condition']]
    candidate_values = np.asarray([vector(c[s]) for s in seeds], dtype=float)
    control_values = np.asarray([vector(b[s]) for s in seeds], dtype=float)
    delta = candidate_values - control_values
    indices = np.random.default_rng(1702).integers(0, 20, (10000, 20))
    complete = all(c[s]['metrics']['missing_quantiles'] == 0
                   and b[s]['metrics']['missing_quantiles'] == 0 for s in seeds)
    results = {}
    for i, name in enumerate(names):
        if not np.isfinite(delta[:, i]).all():
            complete = False
            results[name] = {'mean_difference': None, 'interval_99': None}
        else:
            boot = np.mean(delta[indices, i], axis=1)
            results[name] = {'mean_difference': float(delta[:, i].mean()),
                'interval_99': np.quantile(boot, [.005, .995]).tolist(),
                'paired_differences': delta[:, i].tolist(),
                'candidate_median': float(np.median(candidate_values[:, i])),
                'control_median': float(np.median(control_values[:, i]))}
    return {'metrics': results, 'complete': complete, 'seeds': seeds,
            'note': 'Descriptive paired seed bootstrap, five 99% intervals; not across datasets or optimizer runs.'}


def classify(recovery_result, analysis):
    intervals = [r['interval_99'] for r in analysis['metrics'].values()]
    flags = []
    if not recovery_result['recovery_screen_pass']:
        flags.append('recovery_target_missed')
    if any(ci is not None and ci[0] > 0 for ci in intervals):
        flags.append('predictive_regression')
    if flags:
        return {'classification': flags[0], 'flags': flags}
    g2 = analysis['metrics']['g2']['interval_99']
    if analysis['complete'] and g2 is not None and g2[1] < 0 and all(
            ci is not None and ci[1] <= 0 for ci in intervals):
        return {'classification': 'promising_for_replication', 'flags': []}
    return {'classification': 'inconclusive', 'flags': []}

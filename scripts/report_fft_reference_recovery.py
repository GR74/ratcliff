"""Readable case-study plots and descriptive Monte Carlo intervals."""
import json
from pathlib import Path


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import t
    directory = Path(directory)
    data = json.loads((directory / 'results.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    truth = np.asarray(protocol['truth'])
    names = ['ter', 'st', 'cr', 'crsd', 'sis', 'sig', 'sv', 'av1_c1', 'av2_c1', 'av3_c1', 'av1_c2', 'av2_c2', 'av3_c2']
    if data['fits']:
        fig, ax = plt.subplots(figsize=(12, 5))
        for offset, fit in enumerate(data['fits']):
            errors = 100*(np.asarray(fit['best']['params'])-truth)/truth
            ax.plot(np.arange(13)+(offset-1)*.13, errors, 'o', label=fit['method'])
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(xticks=np.arange(13), xticklabels=names, ylabel='Signed error from generating truth (%)',
               title='One synthetic dataset and one start per method; all 13 parameters active')
        ax.legend(); fig.tight_layout(); fig.savefig(directory / 'parameter-errors.png', dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for fit in data['fits']:
            points = fit['checkpoints'] + [dict(seconds=fit['training_seconds'], **fit['best'])]
            ax.plot([p['seconds']/60 for p in points],
                    [100*np.mean(np.abs((np.asarray(p['params'])-truth)/truth)) for p in points],
                    'o-', label=fit['method'])
        ax.set(xlabel='Training minutes', ylabel='Mean absolute parameter error (%)',
               title='Diagnostic recovery at predeclared checkpoints; truth never guides fitting')
        ax.legend(); fig.tight_layout(); fig.savefig(directory / 'cost-recovery.png', dpi=160); plt.close(fig)

    def vector(summary):
        result = []
        for ci, condition in enumerate(summary):
            for key in ('probabilities', 'low_support', 'rt_quantiles'):
                for i, value in enumerate(np.asarray(condition[key], dtype=float).ravel()):
                    result.append((f'condition{ci+1}:{key}:{i}', value))
            result.append((f'condition{ci+1}:noncrossing', condition['noncrossing']))
        return result

    truth_path = directory / 'truth-predictions.json'
    if not truth_path.exists():
        return
    truth_values = np.array([x[1] for x in vector(json.loads(truth_path.read_text()))])
    baseline = next((s for s in data['scores'] if s['method']=='fft199' and s['checkpoint']=='final'), None)
    rows = []
    for score in data['scores']:
        values = np.array([[x[1] for x in vector(p['summaries'])] for p in score['predictions']])
        labels = [x[0] for x in vector(score['predictions'][0]['summaries'])]
        comparisons = [('truth_reference', values-truth_values)]
        if baseline:
            control = np.array([[x[1] for x in vector(p['summaries'])] for p in baseline['predictions']])
            comparisons.append(('fitted_fft199', values-control))
        for reference, differences in comparisons:
            for label, column in zip(labels, differences.T):
                valid = column[np.isfinite(column)]
                # Never silently omit rare-region seeds to claim precision.
                complete = len(valid) == len(column) and len(valid) > 1
                mean = float(valid.mean()) if len(valid) else None
                width = float(t.ppf(.975, len(valid)-1)*valid.std(ddof=1)/np.sqrt(len(valid))) if complete else None
                rows.append(dict(method=score['method'], checkpoint=score['checkpoint'], generator=score['generator'],
                    reference=reference, endpoint=label, mean_signed_difference=mean,
                    interval95=[mean-width, mean+width] if complete else None,
                    complete_seeds=len(valid), requested_seeds=len(column),
                    interpretation='Descriptive pointwise Monte Carlo interval conditional on estimates and finite reference; not equivalence or recovery CI'))
    (directory / 'prediction-differences.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (directory / 'figures.md').write_text('''# Recovery and prediction diagnostics

![Parameter errors](parameter-errors.png)

![Training cost versus recovery](cost-recovery.png)

`prediction-differences.json` contains signed region, RT, low-support and
noncrossing differences against truth predictions and the fitted FFT199.
Intervals describe simulation uncertainty conditional on fixed estimates and
the finite reference sample. They exclude dataset and optimizer variability,
are pointwise, and do not certify equivalence. Spatial distances and raw
per-seed summaries are in `results.json`. Missing rare-region values remain
unassessed. All fit failures/caps must be read alongside these plots.
''')


if __name__ == '__main__':
    import sys
    report(sys.argv[1])

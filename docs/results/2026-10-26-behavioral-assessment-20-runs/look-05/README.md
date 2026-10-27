# Behavioral look after 5 independent runs

Up to 45,000 trials per condition and method in each case. Earlier-stopped
candidates retain their frozen reference prefix and original look intervals;
sample counts are explicit. Core intervals adjust for all registered endpoints
and all four scheduled looks. Stress panels are separately labeled diagnostics.
No parameter recovery or fitting is part of this study.

| Method | Case | Condition | Runs | Low-support difference and interval (pp) | Core cell status |
| --- | --- | --- | --- | --- | --- |
| fft200 | sig5_ordinary | 1 | 5 | -0.411 [-2.343, +1.521] | inconclusive |
| fft200 | sig5_ordinary | 2 | 5 | +0.271 [-1.634, +2.176] | inconclusive |
| fft200 | sig10_ordinary | 1 | 5 | -0.064 [-2.063, +1.934] | inconclusive |
| fft200 | sig10_ordinary | 2 | 5 | +0.122 [-1.861, +2.105] | inconclusive |
| fft200 | sig15_ordinary | 1 | 5 | +0.262 [-1.784, +2.308] | inconclusive |
| fft200 | sig15_ordinary | 2 | 5 | -0.444 [-2.468, +1.579] | inconclusive |
| repaired_kl | sig5_ordinary | 1 | 5 | -9.847 [-11.613, -8.073] | material_mismatch |
| repaired_kl | sig5_ordinary | 2 | 5 | -8.811 [-10.554, -7.062] | material_mismatch |
| repaired_kl | sig10_ordinary | 1 | 5 | -0.416 [-2.410, +1.579] | inconclusive |
| repaired_kl | sig10_ordinary | 2 | 5 | -0.502 [-2.478, +1.473] | inconclusive |
| repaired_kl | sig15_ordinary | 1 | 5 | -0.493 [-2.532, +1.545] | inconclusive |
| repaired_kl | sig15_ordinary | 2 | 5 | -0.480 [-2.503, +1.544] | inconclusive |
| bucketed_kl | sig5_ordinary | 1 | 5 | -9.962 [-11.726, -8.191] | material_mismatch |
| bucketed_kl | sig5_ordinary | 2 | 5 | -9.369 [-11.099, -7.632] | material_mismatch |
| bucketed_kl | sig10_ordinary | 1 | 5 | -0.011 [-2.010, +1.988] | inconclusive |
| bucketed_kl | sig10_ordinary | 2 | 5 | -0.296 [-2.273, +1.683] | inconclusive |
| bucketed_kl | sig15_ordinary | 1 | 5 | -0.553 [-2.591, +1.485] | inconclusive |
| bucketed_kl | sig15_ordinary | 2 | 5 | -0.684 [-2.705, +1.337] | inconclusive |

![Core effects](core/effects.png)

![Stress diagnostics](stress/effects.png)

All endpoint values, conditional sample counts and intervals are in [endpoints.csv](endpoints.csv). Maps, region probabilities, conditional RT quantiles and RT CDFs are in `figures/`. Sparse/unbounded intervals remain inconclusive. Maps and point-estimate comparisons alone are not tests of superiority.

![Low-support precision across independent runs](precision.png)

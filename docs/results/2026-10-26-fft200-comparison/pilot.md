# Behavioral pilot

4,096 trials per condition, case and method. This separate pilot is not reused
for qualification. Differences below are candidate minus original FFT, in
percentage points. Intervals include the predeclared family adjustment.
For zero drift, the low-support mask is the frozen ordinary-drift geographic
mask; it is not interpreted as a zero-drift support classification.

Precision plan feasible: **False**.
Requested sample size: **2,970,353 trials per condition**;
rounded design: 20 seeds x 148,520.
Resource ceiling: 20 x 45,000 and eight hours of qualifying simulation.
Unresolved sparse-tail case/condition entries: 6.
The precision calculation ignores observed differences and targets precision at
equality. See `pilot-summary.json` for each limiting endpoint, conditional RT
interval, spatial bound and diagnostic. No inconclusive endpoint is a pass.

| Case | Condition | Method | Low-support difference (pp) | Simultaneous interval (pp) |
| --- | --- | --- | --- | --- |
| sig5_ordinary | 1 | fft200 | +0.073 | [-6.232, +6.378] |
| sig5_ordinary | 1 | repaired_kl | -9.863 | [-15.597, -4.058] |
| sig5_ordinary | 1 | bucketed_kl | -11.011 | [-16.658, -5.284] |
| sig5_ordinary | 2 | fft200 | -1.831 | [-8.016, +4.367] |
| sig5_ordinary | 2 | repaired_kl | -10.107 | [-15.782, -4.360] |
| sig5_ordinary | 2 | bucketed_kl | -10.498 | [-16.143, -4.777] |
| sig5_quarter_drift | 1 | fft200 | -0.122 | [-7.286, +7.043] |
| sig5_quarter_drift | 1 | repaired_kl | -4.175 | [-11.358, +3.038] |
| sig5_quarter_drift | 1 | bucketed_kl | -3.076 | [-10.257, +4.126] |
| sig5_quarter_drift | 2 | fft200 | -1.221 | [-8.437, +6.004] |
| sig5_quarter_drift | 2 | repaired_kl | -3.784 | [-11.002, +3.461] |
| sig5_quarter_drift | 2 | bucketed_kl | -2.393 | [-9.611, +4.843] |
| sig5_zero_drift | 1 | fft200 | +1.196 | [-5.704, +8.088] |
| sig5_zero_drift | 1 | repaired_kl | +1.489 | [-5.405, +8.373] |
| sig5_zero_drift | 1 | bucketed_kl | +1.050 | [-5.854, +7.946] |
| sig5_zero_drift | 2 | fft200 | +0.464 | [-6.513, +7.438] |
| sig5_zero_drift | 2 | repaired_kl | -0.635 | [-7.630, +6.365] |
| sig5_zero_drift | 2 | bucketed_kl | +1.562 | [-5.395, +8.509] |
| sig5_boundary_5 | 1 | fft200 | -0.049 | [-7.241, +7.144] |
| sig5_boundary_5 | 1 | repaired_kl | -5.103 | [-12.209, +2.041] |
| sig5_boundary_5 | 1 | bucketed_kl | -5.127 | [-12.233, +2.016] |
| sig5_boundary_5 | 2 | fft200 | -0.830 | [-8.009, +6.354] |
| sig5_boundary_5 | 2 | repaired_kl | -6.030 | [-13.113, +1.096] |
| sig5_boundary_5 | 2 | bucketed_kl | -6.470 | [-13.542, +0.649] |
| sig5_boundary_20 | 1 | fft200 | +0.049 | [-2.605, +2.702] |
| sig5_boundary_20 | 1 | repaired_kl | -2.637 | [-4.590, -0.662] |
| sig5_boundary_20 | 1 | bucketed_kl | -2.197 | [-4.311, -0.066] |
| sig5_boundary_20 | 2 | fft200 | -0.171 | [-2.846, +2.505] |
| sig5_boundary_20 | 2 | repaired_kl | -2.588 | [-4.656, -0.498] |
| sig5_boundary_20 | 2 | bucketed_kl | -2.539 | [-4.625, -0.432] |
| sig10_ordinary | 1 | fft200 | +0.000 | [-6.512, +6.512] |
| sig10_ordinary | 1 | repaired_kl | -1.270 | [-7.730, +5.200] |
| sig10_ordinary | 1 | bucketed_kl | -1.855 | [-8.291, +4.594] |
| sig10_ordinary | 2 | fft200 | +0.903 | [-5.549, +7.349] |
| sig10_ordinary | 2 | repaired_kl | +1.074 | [-5.384, +7.525] |
| sig10_ordinary | 2 | bucketed_kl | +0.293 | [-6.134, +6.718] |
| sig10_quarter_drift | 1 | fft200 | -0.342 | [-7.506, +6.825] |
| sig10_quarter_drift | 1 | repaired_kl | +0.562 | [-6.595, +7.714] |
| sig10_quarter_drift | 1 | bucketed_kl | +1.367 | [-5.782, +8.506] |
| sig10_quarter_drift | 2 | fft200 | -0.464 | [-7.650, +6.726] |
| sig10_quarter_drift | 2 | repaired_kl | -1.367 | [-8.558, +5.834] |
| sig10_quarter_drift | 2 | bucketed_kl | -0.537 | [-7.724, +6.654] |
| sig10_zero_drift | 1 | fft200 | -0.049 | [-6.877, +6.780] |
| sig10_zero_drift | 1 | repaired_kl | -0.903 | [-7.750, +5.950] |
| sig10_zero_drift | 1 | bucketed_kl | -0.464 | [-7.302, +6.377] |
| sig10_zero_drift | 2 | fft200 | -0.854 | [-7.776, +6.073] |
| sig10_zero_drift | 2 | repaired_kl | -0.757 | [-7.677, +6.169] |
| sig10_zero_drift | 2 | bucketed_kl | +0.024 | [-6.881, +6.929] |
| sig10_boundary_5 | 1 | fft200 | +0.684 | [-6.549, +7.912] |
| sig10_boundary_5 | 1 | repaired_kl | +0.586 | [-6.646, +7.814] |
| sig10_boundary_5 | 1 | bucketed_kl | +1.685 | [-5.556, +8.913] |
| sig10_boundary_5 | 2 | fft200 | +0.244 | [-6.974, +7.461] |
| sig10_boundary_5 | 2 | repaired_kl | +0.195 | [-7.023, +7.412] |
| sig10_boundary_5 | 2 | bucketed_kl | +0.195 | [-7.023, +7.412] |
| sig10_boundary_20 | 1 | fft200 | -0.415 | [-3.694, +2.867] |
| sig10_boundary_20 | 1 | repaired_kl | -0.708 | [-3.943, +2.532] |
| sig10_boundary_20 | 1 | bucketed_kl | +0.244 | [-3.131, +3.617] |
| sig10_boundary_20 | 2 | fft200 | +0.049 | [-3.326, +3.423] |
| sig10_boundary_20 | 2 | repaired_kl | -0.635 | [-3.909, +2.644] |
| sig10_boundary_20 | 2 | bucketed_kl | -1.123 | [-4.321, +2.083] |
| sig15_ordinary | 1 | fft200 | -1.221 | [-7.904, +5.472] |
| sig15_ordinary | 1 | repaired_kl | -1.929 | [-8.587, +4.744] |
| sig15_ordinary | 1 | bucketed_kl | -2.368 | [-9.010, +4.291] |
| sig15_ordinary | 2 | fft200 | -1.196 | [-7.699, +5.315] |
| sig15_ordinary | 2 | repaired_kl | -0.049 | [-6.597, +6.499] |
| sig15_ordinary | 2 | bucketed_kl | -0.195 | [-6.737, +6.348] |
| sig15_quarter_drift | 1 | fft200 | +1.221 | [-5.884, +8.316] |
| sig15_quarter_drift | 1 | repaired_kl | +1.392 | [-5.711, +8.484] |
| sig15_quarter_drift | 1 | bucketed_kl | +0.391 | [-6.724, +7.502] |
| sig15_quarter_drift | 2 | fft200 | +0.928 | [-6.251, +8.100] |
| sig15_quarter_drift | 2 | repaired_kl | +1.318 | [-5.858, +8.485] |
| sig15_quarter_drift | 2 | bucketed_kl | +1.514 | [-5.661, +8.677] |
| sig15_zero_drift | 1 | fft200 | +2.295 | [-4.517, +9.090] |
| sig15_zero_drift | 1 | repaired_kl | +0.659 | [-6.191, +7.505] |
| sig15_zero_drift | 1 | bucketed_kl | +1.245 | [-5.592, +8.073] |
| sig15_zero_drift | 2 | fft200 | -1.099 | [-7.985, +5.796] |
| sig15_zero_drift | 2 | repaired_kl | -1.318 | [-8.209, +5.582] |
| sig15_zero_drift | 2 | bucketed_kl | +0.488 | [-6.366, +7.339] |
| sig15_boundary_5 | 1 | fft200 | +0.830 | [-6.425, +8.079] |
| sig15_boundary_5 | 1 | repaired_kl | -0.391 | [-7.639, +6.861] |
| sig15_boundary_5 | 1 | bucketed_kl | +0.269 | [-6.984, +7.519] |
| sig15_boundary_5 | 2 | fft200 | +0.439 | [-6.805, +7.681] |
| sig15_boundary_5 | 2 | repaired_kl | -1.001 | [-8.234, +6.239] |
| sig15_boundary_5 | 2 | bucketed_kl | -0.562 | [-7.798, +6.679] |
| sig15_boundary_20 | 1 | fft200 | +0.391 | [-3.154, +3.933] |
| sig15_boundary_20 | 1 | repaired_kl | -0.220 | [-3.683, +3.245] |
| sig15_boundary_20 | 1 | bucketed_kl | -0.220 | [-3.683, +3.245] |
| sig15_boundary_20 | 2 | fft200 | -0.244 | [-3.704, +3.217] |
| sig15_boundary_20 | 2 | repaired_kl | -0.073 | [-3.556, +3.410] |
| sig15_boundary_20 | 2 | bucketed_kl | -0.024 | [-3.514, +3.465] |
| archived_start | 1 | fft200 | -2.026 | [-9.037, +4.998] |
| archived_start | 1 | repaired_kl | -0.415 | [-7.462, +6.635] |
| archived_start | 1 | bucketed_kl | -2.686 | [-9.680, +4.328] |
| archived_start | 2 | fft200 | +0.415 | [-6.403, +7.230] |
| archived_start | 2 | repaired_kl | +0.098 | [-6.711, +6.906] |
| archived_start | 2 | bucketed_kl | -1.416 | [-8.177, +5.355] |
| archived_fit | 1 | fft200 | +1.807 | [-4.924, +8.524] |
| archived_fit | 1 | repaired_kl | +0.391 | [-6.294, +7.072] |
| archived_fit | 1 | bucketed_kl | +0.806 | [-5.892, +7.498] |
| archived_fit | 2 | fft200 | -0.781 | [-7.415, +5.858] |
| archived_fit | 2 | repaired_kl | -2.002 | [-8.590, +4.601] |
| archived_fit | 2 | bucketed_kl | -1.172 | [-7.792, +5.456] |

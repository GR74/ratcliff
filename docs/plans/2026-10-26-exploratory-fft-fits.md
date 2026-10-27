# Exploratory fitting comparison after user stop

The user explicitly stopped the 20-round behavioral study and requested fitting.
Twelve complete rounds and the partially collected thirteenth are retained.
The last formal analysis remains look 10. This authorization permits an
exploratory fitting study despite inconclusive behavioral qualification; it does
not change the earlier acceptance rules or qualify FFT200.

Run `scripts/start_exploratory_fft_fits.ps1` on the RTX 4060. The new runner uses
the previously validated frozen source 322bd1f and audited FFT200 covariance
builder. It never runs concurrently with another GPU workload. Production
defaults and the stopped behavioral study are unchanged.

Compare FFT200, FFT199, repaired KL and bucketed KL, serially in that order.
All fit one common fresh FFT199-generated dataset at the archived truth (sig=10),
9,000 trials per condition, the same 20%-perturbed start (seed 1), and the same
fixed training key 321. Use unchanged Nelder-Mead, maxiter 1000 and both tolerances
1e-5, with a 135-minute training cap per arm. Save raw and clamped proposals,
objective evaluations, every improvement and best-available five-minute
checkpoints. Report cold warmup separately. Caps are censored, not converged fits.
All four fits finish or hit their cap before independent prediction scoring.

Then generate a 180,000-trial-per-condition FFT199 truth reference. Predetermine
exploratory thresholds from maxima over 20 fresh truth replicates of 9,000 trials,
before inspecting validation outcomes. Score final fits with 20 fresh seeds using
both their native generator and a common FFT199 scorer. Then score all retained
time checkpoints with the common scorer, within an eight-hour validation cap;
unscored checkpoints remain explicitly unassessed. Retain raw samples, signed
region/non-crossing probability and conditional RT-quantile differences, spatial
distance and fixed truth-mask low-support differences at 5%, 10% and 20% cutoffs.
No-crossing indicators remain available; the training objective retains its
historical handling so this experiment isolates the noise generator.

Recovery reports all parameter errors; the known inert sv in this port is
excluded from active recovery acceptance (mean <=5%, worst <=10%). It remains a
scientific limitation, not an assumption that sv is irrelevant. Prediction
screening compares each metric's 95th percentile error across validation seeds
to the corresponding calibration maximum. This finite calibration is exploratory,
not a confidence guarantee. Saved protocol fixes seed lists, budgets and hashes.

Report training, warmup, independent validation, total wall time and GPU telemetry.
Do not claim certified time to acceptable quality or a statistically established
fit speedup from this single, fixed-order dataset/start comparison. Independent
checkpoint scoring will show whether faster fitting is accompanied by comparable
recovery and predictions. Multi-case recovery qualification remains separate.

# Match FFT200 to the two prior RTX 4060 fits

The user requested reusing the prior full-horizon and improved KL fittings,
rather than rerunning them. The fresh-observations four-arm study was interrupted
after about 162 seconds and retained separately. It is excluded from this
archived-data comparison. No control optimizer will be rerun.

The prior "old Ratcliff" control is the archived full-horizon JAX/KL implementation,
not an FFT199 fit or a Fortran run. It used stable mode-indexed noise and reached
the 135-minute cap at 84 evaluations. Improved KL converged at 563 evaluations
in 94.63 minutes. Both share the exact archived observations and start.

The new runner `scripts/run_archived_fft200_fit.py` verifies all three archived
dataset copies against SHA-256
`841f8ba76a41d7a90575563bf408ac586c3811452fa37fd568cd7c14ee032cd2`, verifies
the source manifest and installed package versions, and records input hashes.
It uses the same frozen historical fitting worker, not a reimplementation of
the optimization loop. Preserve exact truth and initial vector, training key
321, 9,000 trials per condition, chunk 8, block 32/early stopping as in improved
KL, all 13 coordinates, default Nelder-Mead simplex, maxiter 1000, x/f tolerances
1e-5, one cold objective warmup, the 8100-second post-evaluation soft cap,
first/every-25-evaluation checkpoints, and the returned final simplex estimate.
Only the noise calls change to full-spectrum FFT on a 200-by-320 workspace.

Compare recovery and training time against saved controls, explicitly retaining
the legacy cap and improved KL's worst-parameter recovery failure. Do not claim
an isolated FFT199-to-FFT200 fitting speedup from these KL controls. The separate
fixed-vector timing and behavioral studies remain the FFT199 comparisons.

For independent predictions, reuse existing raw samples and scores at keys
6001–6020. Fill missing legacy scoring with the same common early-stop KL scorer
used in the old comparison. Score FFT200 with its own generator and that common
KL scorer, labeling them separately. Missing scoring does not require refitting.
Report region and conditional RT-quantile errors, non-crossing fractions and G2;
retain spatial coordinates for further analysis. No new calibrated predictive
acceptance claim is made. Existing behavioral qualification remains unresolved.

Warmup, training, reused historical scoring cost and newly incurred validation
cost remain separate. Between-session timing is descriptive; this single
dataset/start is not broad recovery qualification. The inert sv in the frozen
port remains a scientific limitation and is excluded from active recovery
acceptance, not described as scientifically irrelevant.

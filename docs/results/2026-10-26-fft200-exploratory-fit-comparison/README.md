# Exploratory FFT/KL fitting comparison

Status: **interrupted**; phase: **stopped**. Updated: 2026-10-26T23:42:48.704999+00:00.
Completed fits: 0/4. Current detail: `{"error": "KeyboardInterrupt()"}`.

The user stopped behavioral sampling after 12 complete rounds and authorized
fitting before behavioral qualification. The last formal behavioral look is 10;
the earlier inconclusive results remain unchanged. This study is not qualification.

All four methods fit the same fresh FFT199-generated dataset (9,000 trials per
condition), with the same start, fixed training seed and Nelder-Mead settings.
Each gets a 135-minute training cap plus separately recorded cold warmup.
All fits precede independent validation, which has an eight-hour cap. Results
at a cap remain censored/unfinished. Serial arm order is fixed, not randomized.

| Method | Evaluations | Training minutes | Mean active recovery error | Termination |
| --- | --- | --- | --- | --- |


Final predictions use 20 fresh seeds in the fitted method and the common FFT199
scorer; saved checkpoints then use the common scorer. Errors cover region
probabilities, conditional RT quantiles, non-crossing, low-support wins and spatial
distance. Raw samples and proposal traces are saved in `/mnt/c/Users/Yash/repos/ratcliff/.benchmark-results/2026-10-26-fft200-exploratory-fit-comparison`.
Truth-based recovery never controls optimization. Parameter sv is inert in the
frozen port, a scientific limitation; its error is reported but excluded from
active recovery. One dataset/start cannot establish recovery rates, scientific
equivalence or a statistically conclusive time-to-quality improvement.

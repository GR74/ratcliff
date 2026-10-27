# RTX 4060 BADS experiment: completed, not adopted

BADS 1.5.1 was tested as the only optimizer change against the frozen improved Model B simulator. It failed the recovery screen and worsened the RT-error summary, despite improving independent G2, region probabilities and spatial predictions. Keep the improved simplex implementation as the engineering baseline; it also fails the worst-parameter recovery target.

## Comparison with laptop baselines

| Metric | Original full-horizon simplex | Improved simplex | BADS matched-time | BADS final |
| --- | --- | --- | --- | --- |
| Training minutes | 135.919 | 94.630 | 94.630 | 135.091 |
| Mean active parameter error (%) | 8.098 | 4.586 | 19.535 | 19.550 |
| Worst active parameter error (%) | 19.546 | 16.635 | 29.248 | 28.726 |
| Training loss | 142.000 | 51.405 | 28.331 | 27.180 |
| Outcome | Time cap | Converged | Recovery target missed | Time cap |

The selected BADS candidate completed evaluation at 5,363.123 seconds and was available before the fixed 5,677.777018686-second matched-time cutoff. BADS completed 808 evaluations; original and improved simplex completed 84 and 563. The original run's remaining objective benchmark and scoring were stopped at the user's request; unavailable original independent scores are not filled in from the RTX 3060.

## Independent scoring at the matched-time checkpoint

Twenty paired simulator seeds, disjoint from training, at 9,000 trials per condition. Negative differences favor BADS. Intervals are descriptive within-dataset 99% paired-bootstrap intervals, with five summaries screened together; they are not evidence across participants, datasets or optimizer starts.

| Metric | Simplex median | BADS median | Mean paired difference | 99% interval |
| --- | --- | --- | --- | --- |
| g2 | 71.623 | 58.550 | -13.203 | [-19.645, -5.597] |
| maximum_probability_error_pp | 1.722 | 0.883 | -0.847 | [-1.016, -0.665] |
| maximum_quantile_error_ms | 3.797 | 4.725 | +0.840 | [+0.525, +1.215] |
| spatial_sw1_condition_1 | 1.398 | 1.094 | -0.240 | [-0.453, -0.027] |
| spatial_sw1_condition_2 | 1.146 | 0.798 | -0.317 | [-0.498, -0.146] |

G2 and spatial/probability summaries improved, but maximum RT-quantile error increased by 0.840 ms on average (99% interval +0.525 to +1.215 ms). This is a regression under the predeclared screen, not a claim that this magnitude is scientifically unacceptable. The large parameter-recovery failure independently rules out adoption for the stated goal.

Final-checkpoint fresh-seed G2 median: 57.673.

## Individual recovery

| Parameter | Truth | Simplex estimate | BADS matched estimate | Simplex error (%) | BADS error (%) |
| --- | --- | --- | --- | --- | --- |
| ter | 200 | 207.017 | 221.321 | 3.509 | 10.660 |
| st | 50 | 54.0586 | 56.4313 | 8.117 | 12.863 |
| cr | 10 | 9.5873 | 8.38873 | 4.127 | 16.113 |
| crsd | 2 | 2.3327 | 2.56828 | 16.635 | 28.414 |
| sis | 12 | 11.3598 | 12.3991 | 5.335 | 3.326 |
| sig | 10 | 9.90233 | 11.7187 | 0.977 | 17.187 |
| sv (inactive) | 0.5 | 0.468379 | 0.276746 | Excluded | Excluded |
| av1_c1 | 15 | 14.8975 | 18.3991 | 0.683 | 22.661 |
| av2_c1 | 10 | 10.2655 | 12.9248 | 2.655 | 29.248 |
| av3_c1 | 8 | 7.60133 | 10.1469 | 4.983 | 26.836 |
| av1_c2 | 14 | 14.1672 | 17.1741 | 1.194 | 22.672 |
| av2_c2 | 11 | 11.0883 | 13.4399 | 0.803 | 22.181 |
| av3_c2 | 9 | 8.45826 | 11.0035 | 6.019 | 22.261 |

## Decision and limits

Do not adopt this BADS configuration or automatically broaden/tune it. Preserve it as a negative recovery experiment. Fresh-seed prediction improvements show that fixed-stream overfitting alone does not explain the outcome; parameter tradeoffs and the information retained by the objective remain hypotheses, not identified causes. The later 135-minute result did not rescue recovery (19.550% mean error).

No time-to-acceptable-quality ratio is established. `sv` remains inactive in the historical port; all scientific-preservation caveats in `CLAUDE.md` continue to apply.

[Full evidence](evidence.json) � [Predeclared plan](../../plans/2026-10-21-model-b-bads-single-change.md) � [Original/improved laptop comparison](../2026-10-21-rtx4060-baseline/legacy-comparison.md)

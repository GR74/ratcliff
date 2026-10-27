# RTX 4060 compaction simulation screen

**Status: completed.**

Only compaction changes. Early stopping remains enabled; padding reduction is disabled.

| Case | On (s) | Off (s) | Off/on |
| --- | --- | --- | --- |
| truth | 7.303 | 7.170 | 0.982 |
| higher_boundary | 7.306 | 7.244 | 0.991 |

Three warm-ups per variant and six alternating timing pairs per case. This is a small simulation-only screen, not a full objective or fitting benchmark. Compilation is excluded from timed calls. Path checks use 257 trials per condition; production RT/category checks use 9,000 per condition. Seed 0; chunk 8; block 32. Exact engineering agreement does not independently establish scientific fidelity.

[Raw timings and settings](evidence.json)

# Raw H100 experiment backup

These archives were downloaded before the pod stopped and are committed on
`feat/h100-blocked-staged-fitting` in our own `anik1617/Ratcliff` repository.
They preserve synthetic observations, simulated samples, run settings, fit
traces, worker logs, original results and profiler outputs. Source code is
versioned in the commits listed below. Local copies remain in `.benchmark-results/`.

| Archive | Size | Contents |
|---|---:|---|
| [pilot-h100.tar.gz](pilot-h100.tar.gz) | 3.93 MB | Pilot samples, observations, fit traces, logs and original results |
| [full-results-h100.tar.gz](full-results-h100.tar.gz) | 15.33 MB | Completed 9000-trial baseline pass and 90 additional held-out scores |
| [extensions-final.tar.gz](extensions-final.tar.gz) | 4.32 MB | Ideas 3-5 samples, observations, fit traces, logs, additional scores and trace recovery |
| [profiles-h100.tar.gz](profiles-h100.tar.gz) | 0.49 MB | Small-budget A/B cProfile and line_profiler binary/text outputs |
| [profile-h100-production.tar.gz](profile-h100-production.tar.gz) | 0.37 MB | 9000-trial A/B cProfile and line_profiler binary/text outputs |

## Integrity and provenance

The [manifest](manifest.json) records exact bytes, SHA-256 hashes, archive
roots, source commits and verified pod shutdown time. File hashes were checked
after local download/copy. The first four also match the recorded pod archive
hashes. The production profiler archive has a local integrity hash; all 14
cProfile and 14 line_profiler outputs were verified after extraction.

The smaller profiler archive contains 16 cProfile and 16 line_profiler outputs
across both models. Profiles retain JIT and include instrumentation/async wait
effects; use the [profiling guide](../../../plans/2026-10-13-python-profiling.md)
to interpret them.

Original timeout records remain intact. Extension start 0 for `active_only`
has recovered parameters/scoring in a separate JSON; its exact timing remains
unknown. The pilot simulator timer was superseded by the corrected larger
pass. Read the reports before interpreting the raw numbers.

To restore into the ignored local results directory, create `.benchmark-results/`
and extract the desired archive from the repository root:

```bash
mkdir -p .benchmark-results
tar -xzf docs/results/2026-10-13-h100/raw/extensions-final.tar.gz -C .benchmark-results
```

Repeat for the other archives. Paths are relative to their recorded root
folders. Credentials, SSH keys and local control binaries are excluded.

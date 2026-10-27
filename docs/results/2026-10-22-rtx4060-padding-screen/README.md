# RTX 4060: smaller K-L padding screen

**Status: completed.**

Only `kl_bucketed=True` changes. Retained modes, noise indexing and model definition are unchanged.

Decision: **promising_for_full_fit**. Elapsed: 14.70 minutes.

Basis checks: 14; path checks: 38; objective timing pairs: 30/30.

Control median: 11.624 s; candidate median: 9.879 s.
Candidate/control ratio: 0.850; descriptive 95% interval [0.7885276186669021, 0.9602019329209941].

No full fit is launched by this screen. An exact-check failure requires separate numerical/distributional investigation; it does not alone establish scientific harm.

[Evidence](evidence.json) · [Protocol](../../plans/2026-10-22-model-b-padding-screen.md)

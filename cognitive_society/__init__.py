"""Cognitive Society — lightweight diffusion-decision agents that decide, talk,
infer each other's styles, and adapt.

  agent.py        checkpoint 1 — DDM agents decide (1D, pure NumPy)
  comms.py        checkpoint 2 — DDM-coupled trust-weighted communication
  ez_diffusion.py checkpoint 3 — closed-form cognitive mapping (infer a peer's style)
  society.py      checkpoints 3+4 — the integrated society + uncertainty-gated adaptation
  rl.py           checkpoint 4 (RL) — a learned, adaptive deference policy
  engine.py       engine-agnostic interface (swap in the 2D JAX model)

Experiments: experiment.py (anchor robustness), rl_experiment.py (RL adaptation),
demo.py (cognitive mapping). See docs/plans/2026-06-05-OVERALL-PLAN-cognitive-society.md.
"""


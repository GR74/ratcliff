"""EZ-diffusion: closed-form recovery of 1D DDM parameters from behavior.

For the 1D diffusion model there is a closed-form inverse (Wagenmakers, van der
Maas & Grasman, 2007): from a batch of decisions (choices + reaction times) you
recover drift, boundary, and non-decision time *instantly* — no fitting, no
neural network, no GPU. This is the cognitive-mapping engine for the society:
an agent observes a peer's choices + timing and reads off the peer's cognitive
style in closed form.

(The 2D spatial model has no such closed form — that's where amortized SBI is
needed, Track B. For the lightweight 1D society agents, EZ is exact-enough and
free.)

Our agent model: bounds at +boundary and -boundary, start at 0 (unbiased), noise
sd = sigma. Standard-DDM boundary *separation* a = 2*boundary; we convert back.
"""
import numpy as np


def ez_recover(choices, rts, evidence, sigma: float = 1.0) -> dict:
    """Recover DDM parameters from observed (choices, rts) at a known evidence sign.

    choices  : int array {0,1}
    rts      : float array, seconds
    evidence : the signed evidence the decisions were made under (sign sets which
               response counts as "correct" = drift-favored)
    sigma    : within-trial noise sd (must match the generating model)

    Returns dict: drift, boundary, ndt, plus diagnostics (Pc, MRT, VRT).
    Raises ValueError when the batch is too small, has too few correct responses,
    or has ~zero correct-RT variance (degenerate / censored) — so a caller never
    consumes a silently-wrong recovery (recover_from_agent_observations skips it).
    """
    choices = np.asarray(choices)
    rts = np.asarray(rts, dtype=float)
    n = len(choices)
    if n < 10:
        raise ValueError("ez_recover needs >= 10 trials for a stable estimate")

    correct_choice = 1 if evidence > 0 else 0
    correct = choices == correct_choice
    Pc = correct.mean()

    # Edge corrections (Wagenmakers et al. 2007): avoid Pc in {0, 0.5, 1}.
    if Pc >= 1.0:
        Pc = 1.0 - 1.0 / (2 * n)
    elif Pc <= 0.0:
        Pc = 1.0 / (2 * n)
    if abs(Pc - 0.5) < 1e-6:
        Pc = 0.5 + 1.0 / (2 * n)

    # Standard EZ uses CORRECT-response RT moments. Too few correct responses
    # (near-chance / adversarial) or ~zero RT variance (e.g. censored RTs pinned
    # to the MAX_STEPS ceiling) make the recovery wildly unreliable — raise so the
    # pooling caller skips this batch rather than consuming a silently-bad estimate.
    n_correct = int(correct.sum())
    if n_correct < 10:
        raise ValueError(
            f"ez_recover: only {n_correct} correct responses — too few for a "
            "stable correct-RT estimate (near-chance / adversarial regime)"
        )
    rt_correct = rts[correct]
    MRT = float(rt_correct.mean())
    VRT = float(rt_correct.var())
    if VRT < 1e-5:
        raise ValueError(
            f"ez_recover: degenerate correct-RT variance ({VRT:.2e}); likely "
            "censored / pinned RTs — recovery would be wildly off"
        )

    s = sigma
    s2 = s * s
    L = np.log(Pc / (1.0 - Pc))  # logit

    # Drift (Wagenmakers eq.); the quartic term can go negative numerically at
    # near-chance — clamp to keep the root real.
    inner = L * (L * Pc * Pc - L * Pc + Pc - 0.5) / VRT
    inner = max(inner, 0.0)
    v = np.sign(Pc - 0.5) * s * inner ** 0.25
    if abs(v) < 1e-6:
        v = 1e-6 * (1 if Pc >= 0.5 else -1)

    a_sep = s2 * L / v  # boundary separation
    # Mean decision time -> non-decision time.
    y = -v * a_sep / s2
    # numerically stable tanh-like ratio
    mdt = (a_sep / (2.0 * v)) * (1.0 - np.exp(y)) / (1.0 + np.exp(y))
    ndt = MRT - mdt

    return {
        "drift": float(v),
        "boundary": float(a_sep / 2.0),   # convert separation -> our half-boundary
        "ndt": float(ndt),
        "Pc": float(Pc),
        "MRT": MRT,
        "VRT": VRT,
    }


def recover_from_agent_observations(observations, sigma: float = 1.0) -> dict:
    """Pool multiple (choices, rts, evidence) observation batches and recover.

    observations: list of (choices, rts, evidence) tuples — e.g. a peer observed
    across several evidence levels. We recover per-batch then average the params
    (a simple, robust aggregate; weighted by trial count).
    """
    recs, weights = [], []
    for choices, rts, ev in observations:
        if len(choices) < 10:
            continue
        try:
            recs.append(ez_recover(choices, rts, ev, sigma=sigma))
            weights.append(len(choices))
        except ValueError:
            continue
    if not recs:
        raise ValueError("no usable observation batches")
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    out = {}
    for key in ("drift", "boundary", "ndt"):
        out[key] = float(np.sum([r[key] * wi for r, wi in zip(recs, w)]))
    out["n_batches"] = len(recs)
    return out

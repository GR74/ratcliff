"""Expose the existing JAX Model A histories without changing model equations."""
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_a import simulate
from shared import prng

router = APIRouter()


class ModelARequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    ter: float = Field(200, ge=0, le=2000)
    st: float = Field(50, ge=0, le=2000)
    cr: float = Field(50, ge=1, le=10000)
    crsd: float = Field(10, ge=0, le=1000)
    av: float = Field(20, ge=0, le=200)
    si: float = Field(4, ge=0.2, le=30)
    sig: float = Field(5, ge=0.2, le=30)
    seed: int = Field(42, ge=0, le=4294967295)

    @model_validator(mode="after")
    def valid_ranges(self):
        if self.crsd >= 2 * self.cr:
            raise ValueError("Threshold range must be less than twice the mean threshold.")
        if self.st > 2 * self.ter:
            raise ValueError("Nondecision range must not exceed twice its mean.")
        return self


def history(req: ModelARequest) -> dict:
    a, threshold, ndt = simulate.trial_history(
        prng.root_key(req.seed), req.ter, req.st, req.cr, req.crsd, req.si, req.sig, req.av
    )
    a = np.asarray(a)
    threshold, ndt = float(threshold), float(ndt)
    if not np.isfinite(a).all() or not np.isfinite(threshold + ndt):
        raise ValueError("Model produced non-finite values; try a smaller spatial kernel width.")
    crossings = np.flatnonzero(a.max(axis=1) > threshold)
    hit = len(crossings) > 0
    steps = int(crossings[0]) + 1 if hit else simulate.NSTEP
    winner = int(a[steps - 1].argmax()) if hit else None
    evidence = np.vstack([np.zeros(simulate.N), a[:steps]])
    return {
        "params": req.model_dump(), "angles": (np.arange(1, simulate.N + 1) * 5).tolist(),
        "times": (ndt + np.arange(steps + 1) * simulate.E).tolist(),
        "evidence": evidence.tolist(), "threshold": threshold, "nondecision": ndt,
        "hit": hit, "winner": winner,
        "winning_angle": (winner + 1) * 5 if hit else None,
        "decision_time": steps * simulate.E if hit else None,
        "response_time": ndt + steps * simulate.E if hit else None,
        "dt": simulate.E, "max_decision_time": simulate.NSTEP * simulate.E,
        "drift_peak": float(simulate.drift_profile(req.av, req.si).max()),
    }


@router.post("/api/model-a/history")
def model_a_history(req: ModelARequest) -> dict:
    try:
        return history(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

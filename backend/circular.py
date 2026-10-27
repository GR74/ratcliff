"""Educational constant-drift circular diffusion (Smith 2016), in seconds.

dX = mu dt + sigma dW; two independent Wiener coordinates, absorbing disk.
Finite-step Euler paths with line/circle intersection at the first observed
exit. This is not an exact Brownian first-passage sampler or a fitted model.
Reference: https://doi.org/10.1007/s42113-023-00174-5, Equation 1.
"""
import math

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class CircularRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    nondecision: float = Field(default=0.2, ge=0, le=3)
    threshold: float = Field(default=1, ge=0.1, le=5)
    drift: float = Field(default=1, ge=0, le=10)
    direction: float = Field(default=45, ge=0, le=360)
    noise: float = Field(default=0.6, ge=0, le=3)
    seed: int = Field(default=42, ge=0, le=4294967295)
    max_time: float = Field(default=5, ge=0.1, le=20)


def simulate_circular(req: CircularRequest) -> dict:
    dt = 0.001
    rng = np.random.default_rng(req.seed)
    angle = math.radians(req.direction)
    mu = req.drift * np.array([math.cos(angle), math.sin(angle)])
    pos = np.zeros(2)
    path = [[0.0, req.nondecision, 0.0]]  # X, elapsed time (Y), Z
    t = 0.0
    hit = False
    while t < req.max_time:
        step = min(dt, req.max_time - t)
        delta = mu * step + req.noise * math.sqrt(step) * rng.standard_normal(2)
        nxt = pos + delta
        if np.dot(nxt, nxt) >= req.threshold ** 2:
            aa = float(np.dot(delta, delta))
            bb = 2 * float(np.dot(pos, delta))
            cc = float(np.dot(pos, pos)) - req.threshold ** 2
            fraction = np.clip((-bb + math.sqrt(max(0, bb * bb - 4 * aa * cc))) / (2 * aa), 0, 1)
            pos = pos + fraction * delta
            t += float(fraction) * step
            hit = True
        else:
            pos = nxt
            t = min(req.max_time, t + step)
        path.append([float(pos[0]), t + req.nondecision, float(pos[1])])
        if hit:
            break
    return {
        "path": path, "hit": hit, "decision_time": t if hit else None,
        "response_time": t + req.nondecision if hit else None,
        "angle": math.degrees(math.atan2(pos[1], pos[0])) % 360 if hit else None,
        "dt": dt, "params": req.model_dump(),
    }


@router.post("/api/circular")
def circular(req: CircularRequest) -> dict:
    return simulate_circular(req)

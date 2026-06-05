"""BYO-GPU routing: forward fit requests to a user-supplied remote backend.

When a user provides a `gpu_endpoint` URL in the fit-start request, the
local backend (running on free CPU) proxies the fit to the user's own
GPU-backed instance of this same FastAPI app. The user runs the same
Docker container on their GPU box, gives the URL to the public Space,
and gets fast fits without us paying GPU costs.
"""
import httpx


def forward_post(endpoint: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    """POST to {endpoint}{path} with JSON body and return the parsed response."""
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{endpoint.rstrip('/')}{path}", json=body)
        r.raise_for_status()
        return r.json()


def forward_get(endpoint: str, path: str, timeout: float = 30.0) -> dict:
    """GET {endpoint}{path} and return the parsed JSON response."""
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{endpoint.rstrip('/')}{path}")
        r.raise_for_status()
        return r.json()

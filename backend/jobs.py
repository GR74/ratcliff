"""In-memory job tracking for long-running fits.

Each fit gets a UUID job_id. The dict holds per-eval progress, the final
result, and any error. HF Spaces only allows /tmp writes and restarts on
idle, so in-memory is the right shape — survives until process exit only.
"""
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class JobStatus:
    job_id: str
    status: str = "pending"  # pending | running | done | error
    progress: list[dict] = field(default_factory=list)  # [{"eval_n", "loss"}]
    result: Optional[dict] = None
    error: Optional[str] = None


_jobs: dict[str, JobStatus] = {}
_lock = Lock()


def new_job() -> JobStatus:
    """Create a new job with a fresh UUID."""
    job = JobStatus(job_id=str(uuid.uuid4()))
    with _lock:
        _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobStatus]:
    """Look up a job by id. Returns None if unknown."""
    return _jobs.get(job_id)


def set_status(job_id: str, status: str) -> None:
    job = _jobs.get(job_id)
    if job:
        job.status = status


def update_progress(job_id: str, eval_n: int, loss: float) -> None:
    """Record one per-eval progress point."""
    job = _jobs.get(job_id)
    if job:
        job.progress.append({"eval_n": eval_n, "loss": loss})


def set_result(job_id: str, result: dict) -> None:
    """Mark a job complete and store its final result."""
    job = _jobs.get(job_id)
    if job:
        job.result = result
        job.status = "done"


def set_error(job_id: str, err: str) -> None:
    """Mark a job failed and record the error message."""
    job = _jobs.get(job_id)
    if job:
        job.error = err
        job.status = "error"


def reset_for_tests() -> None:
    """Clear all jobs. Used in tests; do not call from production paths."""
    with _lock:
        _jobs.clear()

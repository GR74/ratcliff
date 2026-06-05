"""Tests for backend/jobs.py — in-memory job tracking."""
import pytest

from backend import jobs


@pytest.fixture(autouse=True)
def reset():
    jobs.reset_for_tests()


def test_new_job_creates_unique_ids():
    a = jobs.new_job()
    b = jobs.new_job()
    assert a.job_id != b.job_id
    assert a.status == "pending"
    assert a.progress == []


def test_get_job_returns_existing_or_none():
    job = jobs.new_job()
    assert jobs.get_job(job.job_id) is job
    assert jobs.get_job("nonexistent-uuid") is None


def test_set_status_transitions():
    job = jobs.new_job()
    jobs.set_status(job.job_id, "running")
    assert jobs.get_job(job.job_id).status == "running"


def test_update_progress_appends():
    job = jobs.new_job()
    jobs.update_progress(job.job_id, 1, 100.5)
    jobs.update_progress(job.job_id, 2, 95.2)
    progress = jobs.get_job(job.job_id).progress
    assert len(progress) == 2
    assert progress[0] == {"eval_n": 1, "loss": 100.5}
    assert progress[1] == {"eval_n": 2, "loss": 95.2}


def test_set_result_completes_job():
    job = jobs.new_job()
    jobs.set_result(job.job_id, {"params": [1.0, 2.0], "loss": 50.0})
    j = jobs.get_job(job.job_id)
    assert j.status == "done"
    assert j.result["loss"] == 50.0


def test_set_error_marks_failed():
    job = jobs.new_job()
    jobs.set_error(job.job_id, "things broke")
    j = jobs.get_job(job.job_id)
    assert j.status == "error"
    assert j.error == "things broke"


def test_update_progress_on_unknown_job_is_silent():
    """Don't raise — late callback after eviction shouldn't crash."""
    jobs.update_progress("nonexistent", 1, 100.0)   # should not raise
    jobs.set_result("nonexistent", {})              # should not raise
    jobs.set_error("nonexistent", "oops")          # should not raise

"""Tests for the InProcessQueue (src/idp/queue/jobs.py).

The InProcessQueue is the dev/test queue. Production deployments
swap in ARQ/Celery/SQS but the same JobQueue Protocol is honored.
"""
from __future__ import annotations

import time

import pytest

from idp.queue.jobs import (
    InProcessQueue,
    Job,
    JobStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_runner(jobs_seen: list[Job], raise_on: set[str] | None = None):
    """Build a runner that records which jobs it processed.

    `raise_on` is an optional set of doc_paths whose runner call
    should raise (used to test failure handling).
    """
    raise_on = raise_on or set()

    def runner(job: Job) -> None:
        jobs_seen.append(job)
        if job.doc_path in raise_on:
            raise RuntimeError(f"intentional failure for {job.doc_path}")

    return runner


# ---------------------------------------------------------------------------
# Submit + status
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_submit_returns_job_with_id() -> None:
    jobs_seen: list[Job] = []
    q = InProcessQueue(runner=_make_runner(jobs_seen))
    job = await q.submit(doc_path="a.pdf", schema_name="Invoice", backend_name="mock")
    assert isinstance(job, Job)
    assert len(job.id) == 16
    assert job.doc_path == "a.pdf"
    assert job.schema_name == "Invoice"
    assert job.backend_name == "mock"
    assert job.status == JobStatus.PENDING  # status flips to RUNNING async


@pytest.mark.asyncio
async def test_status_returns_submitted_job() -> None:
    q = InProcessQueue(runner=_make_runner([]))
    submitted = await q.submit(doc_path="x.pdf", schema_name="Invoice")
    fetched = await q.status(submitted.id)
    assert fetched is not None
    assert fetched is submitted
    assert fetched.id == submitted.id


@pytest.mark.asyncio
async def test_status_unknown_returns_none() -> None:
    q = InProcessQueue(runner=_make_runner([]))
    assert await q.status("does-not-exist") is None


@pytest.mark.asyncio
async def test_list_returns_submitted_jobs_newest_first() -> None:
    q = InProcessQueue(runner=_make_runner([]))
    j1 = await q.submit("a.pdf", "Invoice")
    # Tiny sleep so created_at timestamps differ
    time.sleep(0.001)
    j2 = await q.submit("b.pdf", "Contract")
    jobs = await q.list()
    assert len(jobs) == 2
    # Newest first
    assert jobs[0].id == j2.id
    assert jobs[1].id == j1.id


# ---------------------------------------------------------------------------
# Worker behavior
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_worker_runs_submitted_job() -> None:
    """The runner actually executes after submit() returns."""
    jobs_seen: list[Job] = []
    q = InProcessQueue(runner=_make_runner(jobs_seen))
    await q.start()
    try:
        job = await q.submit("a.pdf", "Invoice")
        # Wait for worker to drain the queue
        await q._q.join()
        assert len(jobs_seen) == 1
        assert jobs_seen[0].id == job.id
        # Job transitioned to SUCCEEDED
        assert job.status == JobStatus.SUCCEEDED
        assert job.started_at is not None
        assert job.finished_at is not None
        assert job.started_at <= job.finished_at
    finally:
        await q.stop()


@pytest.mark.asyncio
async def test_worker_marks_failed_job() -> None:
    """Runner raises -> job.status == FAILED, error is recorded."""
    q = InProcessQueue(runner=_make_runner([], raise_on={"bad.pdf"}))
    await q.start()
    try:
        job = await q.submit("bad.pdf", "Invoice")
        await q._q.join()
        assert job.status == JobStatus.FAILED
        assert job.error is not None
        assert "intentional failure" in job.error
        assert job.finished_at is not None
    finally:
        await q.stop()


@pytest.mark.asyncio
async def test_worker_keeps_running_after_failure() -> None:
    """A failed job does NOT poison the worker — subsequent jobs still run."""
    jobs_seen: list[Job] = []
    q = InProcessQueue(runner=_make_runner(jobs_seen, raise_on={"bad.pdf"}))
    await q.start()
    try:
        bad = await q.submit("bad.pdf", "Invoice")
        good = await q.submit("good.pdf", "Invoice")
        await q._q.join()
        # Both ran (the failed one still went through the worker)
        assert {j.id for j in jobs_seen} == {bad.id, good.id}
        assert bad.status == JobStatus.FAILED
        assert good.status == JobStatus.SUCCEEDED
    finally:
        await q.stop()


@pytest.mark.asyncio
async def test_stop_cancels_worker() -> None:
    """stop() cancels the worker task cleanly."""
    q = InProcessQueue(runner=_make_runner([]))
    await q.start()
    assert q._worker_task is not None
    await q.stop()
    assert q._worker_task is None
    # Calling stop twice is a no-op (no error)
    await q.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    """start() called twice doesn't spawn two workers."""
    q = InProcessQueue(runner=_make_runner([]))
    await q.start()
    await q.start()  # second call should be a no-op
    assert q._worker_task is not None
    await q.stop()


# ---------------------------------------------------------------------------
# Multiple jobs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multiple_jobs_all_processed() -> None:
    jobs_seen: list[Job] = []
    q = InProcessQueue(runner=_make_runner(jobs_seen))
    await q.start()
    try:
        ids = []
        for i in range(5):
            j = await q.submit(f"doc_{i}.pdf", "Invoice")
            ids.append(j.id)
        await q._q.join()
        assert len(jobs_seen) == 5
        assert {j.id for j in jobs_seen} == set(ids)
        for j in jobs_seen:
            assert j.status == JobStatus.SUCCEEDED
    finally:
        await q.stop()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------
def test_in_process_queue_implements_protocol() -> None:
    """InProcessQueue satisfies the JobQueue Protocol structurally.

    JobQueue is a typing.Protocol without @runtime_checkable, so we
    can't use isinstance(). Instead we verify all the protocol methods
    are present and have the right signature.
    """
    q = InProcessQueue(runner=_make_runner([]))
    for method in ("submit", "status", "list"):
        assert hasattr(q, method), f"InProcessQueue missing {method!r}"
        assert callable(getattr(q, method))


# ---------------------------------------------------------------------------
# JobStatus enum
# ---------------------------------------------------------------------------
def test_job_status_enum_values() -> None:
    """Enum values are stable strings (depended on by external callers + JSON dumps)."""
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.SUCCEEDED.value == "succeeded"
    assert JobStatus.FAILED.value == "failed"
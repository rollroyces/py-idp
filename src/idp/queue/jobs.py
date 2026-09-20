# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Async job queue abstraction.

For production you'd swap in ARQ (Redis), Celery, or AWS SQS.
The InProcess queue ships for development + tests + single-node deploys.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    doc_path: str
    schema_name: str
    backend_name: str
    status: JobStatus = JobStatus.PENDING
    result_id: str | None = None
    error: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class JobQueue(Protocol):
    async def submit(self, doc_path: str, schema_name: str, backend_name: str = "auto") -> Job: ...
    async def status(self, job_id: str) -> Job | None: ...
    async def list(self) -> list[Job]: ...


class InProcessQueue(JobQueue):
    """Single-process asyncio queue. Suitable for dev + lightweight self-host.

    Concurrency is bounded by ``max_concurrent`` (default 4) via an
    ``asyncio.Semaphore``. The semaphore is acquired *inside* the
    worker loop, so callers see jobs in ``PENDING`` state while they
    wait their turn. This lets ``/jobs/{id}`` report an honest
    ``estimated_queue_position`` based on actual queue depth.

    ``max_concurrent=0`` disables the cap (use with care; a flood of
    submissions can exhaust the runner's thread pool).
    """

    def __init__(
        self,
        runner: Callable[[Job], None],
        max_concurrent: int = 4,
    ) -> None:
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._q: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        # Bounded concurrency. None means "no cap" (max_concurrent <= 0).
        self._max_concurrent = max_concurrent if max_concurrent > 0 else None
        self._sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )
        # Live counts for observability. Updated inside the worker.
        self._inflight = 0
        self._total_started = 0
        self._total_finished = 0

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    @property
    def inflight(self) -> int:
        """Number of jobs currently executing (inside the semaphore)."""
        return self._inflight

    @property
    def queue_depth(self) -> int:
        """Number of jobs waiting in the queue (not yet picked up)."""
        return self._q.qsize()

    def estimated_queue_position(self, job_id: str) -> int | None:
        """Return 1-indexed position of ``job_id`` in the queue.

        Returns:
          * 0 if the job is currently running
          * 1..N for jobs in PENDING state (1 = next to run)
          * None if the job is unknown or already finished
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status == JobStatus.RUNNING:
            return 0
        if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            return None
        # Walk the in-memory job list (small; not the asyncio.Queue).
        # Order: pending jobs in submission order, then running, then done.
        pending = [j for j in self._jobs.values() if j.status == JobStatus.PENDING]
        pending.sort(key=lambda j: j.created_at)
        for i, j in enumerate(pending, start=1):
            if j.id == job_id:
                return i
        return None

    async def _worker(self) -> None:
        while True:
            try:
                job = await self._q.get()
                # Acquire the concurrency cap before mutating the job.
                # If no semaphore is configured, run inline.
                if self._sem is not None:
                    acquire_start = asyncio.get_event_loop().time()
                    await self._sem.acquire()
                    wait = asyncio.get_event_loop().time() - acquire_start
                    if wait > 0.5:
                        # Heuristic: only warn on meaningful waits so we don't
                        # spam the log on bursty traffic.
                        log.warning(
                            "job %s waited %.2fs for a concurrency slot "
                            "(inflight=%d, max=%d)",
                            job.id, wait, self._inflight,
                            self._max_concurrent or 0,
                        )
                try:
                    job.status = JobStatus.RUNNING
                    job.started_at = asyncio.get_event_loop().time()
                    self._inflight += 1
                    self._total_started += 1
                    try:
                        self._runner(job)
                        job.status = JobStatus.SUCCEEDED
                    except Exception as e:  # noqa: BLE001
                        log.exception("job %s failed", job.id)
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                    finally:
                        self._inflight -= 1
                        self._total_finished += 1
                        job.finished_at = asyncio.get_event_loop().time()
                finally:
                    if self._sem is not None:
                        self._sem.release()
                    self._q.task_done()
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                log.exception("worker tick failed: %s", e)

    async def submit(self, doc_path: str, schema_name: str, backend_name: str = "auto") -> Job:
        job = Job(
            id=uuid.uuid4().hex[:16],
            doc_path=str(doc_path),
            schema_name=schema_name,
            backend_name=backend_name,
            created_at=time.time(),
        )
        self._jobs[job.id] = job
        await self.start()
        await self._q.put(job)
        return job

    async def status(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: -j.created_at)

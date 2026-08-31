# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
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
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
    """Single-process asyncio queue. Suitable for dev + lightweight self-host."""

    def __init__(
        self,
        runner: Callable[[Job], None],
    ) -> None:
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._q: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    async def _worker(self) -> None:
        while True:
            try:
                job = await self._q.get()
                job.status = JobStatus.RUNNING
                job.started_at = asyncio.get_event_loop().time()
                try:
                    self._runner(job)
                    job.status = JobStatus.SUCCEEDED
                except Exception as e:  # noqa: BLE001
                    log.exception("job %s failed", job.id)
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                finally:
                    job.finished_at = asyncio.get_event_loop().time()
                    self._q.task_done()
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                log.exception("worker tick failed: %s", e)

    async def submit(self, doc_path: str, schema_name: str, backend_name: str = "auto") -> Job:
        import time
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

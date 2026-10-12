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

"""Async wrapper around :class:`idp.pipeline.pipeline.Pipeline`.

The sync :meth:`Pipeline.run` is the workhorse; :class:`AsyncPipeline`
is a thin wrapper that lets you ``await`` a single extraction and lets
you run many extractions concurrently via :func:`asyncio.gather`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from idp.core.document import Document
from idp.pipeline.pipeline import Pipeline, PipelineResult


class AsyncPipeline:
    """Async-friendly wrapper around :class:`Pipeline`.

    Use this when:

    - You're already in an async context (FastAPI, Jupyter with
      ipykernel, asyncio scripts).
    - You want to process many documents concurrently via
      ``asyncio.gather([p.arun(d) for d in docs])`` — see
      :func:`idp.batch.process_batch_async`.

    Construction is identical to :class:`Pipeline` (``backend=...``,
    ``schema=...``). :meth:`arun` is the async equivalent of
    :meth:`Pipeline.run`. If ``doc`` is a string or path, it is
    auto-resolved to a :class:`Document` via
    :meth:`Document.from_path`, mirroring the convenience already
    used in :func:`idp.batch.process_batch`.
    """

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        schema: str | type[BaseModel] = "Invoice",
        **kwargs: Any,
    ) -> AsyncPipeline:
        return cls(Pipeline(backend=backend, schema=schema, **kwargs))

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    async def arun(self, doc: Document | str | Path) -> PipelineResult:
        """Async equivalent of :meth:`Pipeline.run`.

        Runs the sync :meth:`Pipeline.run` in a worker thread via
        :meth:`asyncio.AbstractEventLoop.run_in_executor` so
        event-loop-blocking backend HTTP calls don't freeze the loop.

        If ``doc`` is a string or :class:`Path`, it is resolved to a
        :class:`Document` via :meth:`Document.from_path` first.
        """
        if isinstance(doc, str | Path):
            doc = Document.from_path(doc)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._pipeline.run, doc)

    async def arun_many(
        self, docs: list[Document | str | Path]
    ) -> list[PipelineResult]:
        """Run many docs concurrently. Returns results in input order."""
        return await asyncio.gather(*[self.arun(d) for d in docs])

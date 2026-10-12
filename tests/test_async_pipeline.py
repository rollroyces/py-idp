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

"""Async pipeline tests."""
from __future__ import annotations

import asyncio
import time

import pytest

from idp.batch import BatchItemResult, process_batch, process_batch_async
from idp.core.document import Document
from idp.llm.backend import Backend, CompletionRequest, Message, MockBackend
from idp.pipeline.async_pipeline import AsyncPipeline
from idp.pipeline.pipeline import Pipeline


def test_async_pipeline_constructs_from_pipeline():
    p = Pipeline(backend=MockBackend(), schema="Invoice")
    ap = AsyncPipeline(p)
    assert ap.pipeline is p


def test_async_pipeline_from_backend():
    ap = AsyncPipeline.from_backend(backend=MockBackend(), schema="Invoice")
    assert isinstance(ap.pipeline, Pipeline)


@pytest.mark.asyncio
async def test_arun_returns_same_as_run(tmp_path):
    """AsyncPipeline.arun should match Pipeline.run type-for-type."""
    ap = AsyncPipeline.from_backend(backend=MockBackend(), schema="Invoice")
    # Document requires source_path/doc_id; use a real file so the
    # plain-text parser has something to read.
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("dummy invoice text")
    doc = Document.from_path(str(doc_path))
    sync_result = Pipeline(backend=MockBackend(), schema="Invoice").run(doc)
    async_result = await ap.arun(doc)
    # MockBackend's behavior should be deterministic
    assert type(async_result) is type(sync_result)
    # Both should have a populated extraction
    assert async_result.document.extraction is not None
    assert sync_result.document.extraction is not None


@pytest.mark.asyncio
async def test_arun_accepts_string_path(tmp_path):
    txt = tmp_path / "inv.txt"
    txt.write_text("dummy invoice text")
    ap = AsyncPipeline.from_backend(backend=MockBackend(), schema="Invoice")
    result = await ap.arun(str(txt))
    # Should not raise
    assert result is not None
    assert isinstance(result.document, Document)


@pytest.mark.asyncio
async def test_arun_accepts_path_object(tmp_path):
    txt = tmp_path / "inv.txt"
    txt.write_text("dummy invoice text")
    ap = AsyncPipeline.from_backend(backend=MockBackend(), schema="Invoice")
    result = await ap.arun(txt)  # pathlib.Path
    assert result is not None


@pytest.mark.asyncio
async def test_arun_many_returns_in_order(tmp_path):
    ap = AsyncPipeline.from_backend(backend=MockBackend(), schema="Invoice")
    paths = []
    for i in range(5):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"doc {i}")
        paths.append(p)
    results = await ap.arun_many(paths)
    assert len(results) == 5
    # order is preserved
    for i, r in enumerate(results):
        assert r is not None
        assert r.document.source_path == str(paths[i])


@pytest.mark.asyncio
async def test_process_batch_async_returns_results(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"inv{i}.txt"
        p.write_text(f"invoice {i}")
        paths.append(p)
    results = await process_batch_async(
        paths,
        backend=MockBackend(),
        schema="Invoice",
        concurrency=2,
    )
    assert len(results) == 3
    assert all(isinstance(r, BatchItemResult) for r in results)
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_process_batch_async_respects_concurrency(tmp_path):
    """With concurrency=1, the wall time should be at least N * per-item latency."""
    paths = []
    for i in range(3):
        p = tmp_path / f"inv{i}.txt"
        p.write_text(f"invoice {i}")
        paths.append(p)

    # Add a tiny per-call sleep to make timing measurable
    class SlowMock(MockBackend):
        def complete(self, req):
            time.sleep(0.05)
            return super().complete(req)

    t0 = time.perf_counter()
    results = await process_batch_async(
        paths,
        backend=SlowMock(),
        schema="Invoice",
        concurrency=1,
    )
    elapsed = time.perf_counter() - t0
    assert len(results) == 3
    # 3 * 0.05s = 0.15s minimum sequential. Add a small slack for
    # executor startup overhead.
    assert elapsed >= 0.15


@pytest.mark.asyncio
async def test_process_batch_async_progress_callback(tmp_path):
    paths = [(tmp_path / f"p{i}.txt") for i in range(3)]
    for p in paths:
        p.write_text("x")
    progress_calls: list[tuple[int, int]] = []
    await process_batch_async(
        paths,
        backend=MockBackend(),
        schema="Invoice",
        concurrency=2,
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )
    assert progress_calls  # not empty
    assert progress_calls[-1] == (3, 3)  # final call is done=total


@pytest.mark.asyncio
async def test_process_batch_async_preserves_order(tmp_path):
    """Results must come back in the same order as the input paths."""
    paths = []
    for i in range(7):
        p = tmp_path / f"o{i}.txt"
        p.write_text(f"order marker {i}")
        paths.append(p)
    results = await process_batch_async(
        paths,
        backend=MockBackend(),
        schema="Invoice",
        concurrency=4,
    )
    for i, (p, r) in enumerate(zip(paths, results, strict=False)):
        assert r.success
        assert r.path == str(p)
        # The file content is in doc.raw_text (parser sets it)
        assert f"order marker {i}" in r.result.document.raw_text


@pytest.mark.asyncio
async def test_process_batch_async_handles_errors(tmp_path):
    """A bad path must not crash the batch — it should be recorded as a failure."""
    good = tmp_path / "good.txt"
    good.write_text("ok")
    bad = tmp_path / "does_not_exist.txt"
    paths = [str(good), str(bad)]
    results = await process_batch_async(
        paths,
        backend=MockBackend(),
        schema="Invoice",
        concurrency=2,
    )
    assert len(results) == 2
    # Good one wins
    assert results[0].success
    assert results[0].result is not None
    # Bad one is captured
    assert not results[1].success
    assert results[1].error is not None
    assert results[1].result is None


def test_backend_acomplete_default_uses_thread():
    """The default acomplete should fall back to asyncio.to_thread."""
    async def go():
        b = MockBackend()
        req = CompletionRequest(messages=[Message(role="user", content="hi")])
        out = await b.acomplete(req)
        return out

    out = asyncio.run(go())
    assert isinstance(out, str)
    # MockBackend's default extract path returns "{}" or a stubbed schema;
    # either is a valid non-empty string here.
    assert out


def test_backend_acomplete_is_coroutine():
    """`acomplete` must be a coroutine function on the base Backend."""
    import inspect

    assert inspect.iscoroutinefunction(Backend.acomplete)
    assert inspect.iscoroutinefunction(MockBackend.acomplete)


def test_sync_process_batch_unchanged():
    """Sanity: the sync process_batch still yields the same dataclass."""
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        p = _P(td) / "x.txt"
        p.write_text("hello")
        results = list(
            process_batch([str(p)], Pipeline(backend=MockBackend(), schema="Invoice"))
        )
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, BatchItemResult)
        # Both old and new fields should be populated
        assert r.ok
        assert r.success
        assert r.elapsed_seconds >= 0
        assert r.latency_s == r.elapsed_seconds

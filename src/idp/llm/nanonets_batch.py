"""Batch extraction helper for NanonetsVLBackend.

Use case: process 100s-1000s of documents through the same loaded
model. Without this, callers would need to write the same loop pattern
each time:

    backend = NanonetsVLBackend(device="cuda")
    pipeline = Pipeline(backend=backend, schema="Invoice")
    for path in paths:
        doc = Document.from_path(path)
        result = pipeline.run(doc)   # model stays loaded across calls
        ...

`process_batch()` does the loop, adds per-document error handling, and
returns a structured result you can write to Delta Lake / a DataFrame
/ disk.

For Databricks: combine with a single-node GPU cluster + this helper
to process 1000 docs in one job run (5-15s/doc × 1000 = 1.5-4 hours).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from idp.checkpoint import CheckpointEntry, CheckpointStore
from idp.core.document import Document
from idp.pipeline.pipeline import Pipeline, PipelineResult

log = logging.getLogger(__name__)


@dataclass
class BatchItemResult:
    """Result for a single document in a batch.

    Either ``result`` is set (success) or ``error`` is set (failure).
    Exactly one of them is non-None.
    """
    path: str
    result: PipelineResult | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for DataFrame / Delta Lake rows."""
        d: dict[str, Any] = {
            "path": self.path,
            "ok": self.ok,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.ok and self.result is not None:
            d["doc_id"] = self.result.document.doc_id
            d["schema"] = self.result.schema_name
            d["backend"] = self.result.backend_name
            d["classification"] = self.result.classification
            d["extraction"] = self.result.document.extraction or {}
            d["confidence"] = self.result.confidence or {}
            d["validation_passed"] = self.result.validation_passed
            # Convenience: which fields the policy marks for review
            d["needs_review"] = any(
                c < 0.7 for c in (self.result.confidence or {}).values()
            )
        else:
            d["error"] = self.error
        return d


def process_batch(
    paths: list[str] | Iterator[str],
    pipeline: Pipeline,
    *,
    progress_every: int = 10,
    on_progress: Callable[[int, int, BatchItemResult], None] | None = None,
    checkpoint: CheckpointStore | str | None = None,
    archive_at_start: bool = False,
) -> Iterator[BatchItemResult]:
    """Run ``pipeline`` over many documents, yielding a result per item.

    Args:
        paths:        list or iterator of file paths (local or DBFS/S3
                      paths supported by ``Document.from_path``).
        pipeline:     a configured ``Pipeline``. The model (if any) is loaded
                      on the first call and reused for the rest of the batch.
        progress_every: log a progress line every N documents. 0 = silent.
        on_progress:  optional callback ``(i, total, result)`` for custom
                      reporting (e.g. Databricks widget updates).

    Yields:
        ``BatchItemResult`` for each document. Exactly one of ``result``
        or ``error`` is set.

    Errors are caught and recorded in ``BatchItemResult.error``; the
        iterator does NOT raise. The caller decides what to do (skip,
        retry, send to dead-letter queue).

    Args:
        paths:        list or iterator of file paths (local or DBFS/S3
                      paths supported by ``Document.from_path``).
        pipeline:     a configured ``Pipeline``. The model (if any) is loaded
                      on the first call and reused for the rest of the batch.
        progress_every: log a progress line every N documents. 0 = silent.
        on_progress:  optional callback ``(i, total, result)`` for custom
                      reporting (e.g. Databricks widget updates).
        checkpoint:   optional ``CheckpointStore`` (or path string) for
                      resume-after-crash. When set:
                      - On start, paths already in the checkpoint are
                        SKIPPED — the iterator does NOT yield them.
                      - After each doc, the result is recorded (success
                        OR failure), so a crash mid-batch leaves the
                        ledger consistent.
                      - Re-running the same batch with the same checkpoint
                        is IDEMPOTENT: same paths get skipped, only new
                        paths are processed.
                      Idempotent resume — no flag needed; just pass the
                      same checkpoint path on retry. Use ``CheckpointStore.clear()``
                      if you want to force reprocessing.
        archive_at_start: when True (default False), if the checkpoint
                      file already exists with entries, rename it to
                      ``<path>.<timestamp>.jsonl`` and start a fresh
                      empty ledger. Use this between production runs
                      so each run gets a clean checkpoint AND you keep
                      a history of past runs for debugging. Idempotent
                      resume DOES NOT use this — it's only for explicit
                      "fresh batch, new ledger" semantics.

    Example (Databricks):
        from idp import Pipeline
        from idp.llm.nanonets import NanonetsVLBackend
        from idp.llm.nanonets_batch import process_batch
        from idp.checkpoint import CheckpointStore

        pipeline = Pipeline(
            backend=NanonetsVLBackend(device="cuda"),
            schema="Invoice",
        )
        # Idempotent batch — safe to re-run if the job fails mid-way:
        results = process_batch(
            [p.path for p in dbutils.fs.ls("/mnt/invoices/inbox/")],
            pipeline,
            checkpoint="/dbfs/mnt/idp/checkpoints/inbox.jsonl",
        )
        for r in results:
            if r.ok:
                write_to_delta(r.to_dict())
            else:
                log.error(f"Failed: {r.path}: {r.error}")
    """
    # Resolve checkpoint store (lazy construction from string or Path)
    cp: CheckpointStore | None = None
    if checkpoint is not None:
        cp = checkpoint if isinstance(checkpoint, CheckpointStore) else CheckpointStore(checkpoint)
        if archive_at_start:
            archived = cp.archive()
            if archived is not None:
                log.info(
                    "checkpoint: archived previous run to %s; starting fresh",
                    archived,
                )

    if isinstance(paths, list):
        paths_iter: Iterator[str] = iter(paths)
        total = len(paths)
    else:
        paths_iter = paths
        total = None  # unknown for iterators

    for i, path in enumerate(paths_iter, start=1):
        # Idempotent resume: skip paths already in the checkpoint.
        if cp is not None and path in cp:
            log.info("checkpoint: skipping already-done path %s", path)
            continue
        t0 = time.perf_counter()
        try:
            doc = Document.from_path(path)
            result = pipeline.run(doc)
            item = BatchItemResult(
                path=str(path),
                result=result,
                elapsed_seconds=time.perf_counter() - t0,
            )
        except Exception as e:
            log.warning("Failed to process %s: %s", path, e)
            item = BatchItemResult(
                path=str(path),
                error=f"{type(e).__name__}: {e}",
                elapsed_seconds=time.perf_counter() - t0,
            )
        # Record the outcome BEFORE yielding so a crash between
        # process() and write_to_delta() doesn't lose this doc's state.
        if cp is not None:
            try:
                entry_result = (
                    item.result.document.extraction
                    if (item.result is not None and item.result.document.extraction)
                    else None
                )
                cp.record(CheckpointEntry(
                    path=item.path,
                    ok=item.ok,
                    error=item.error,
                    result=entry_result,
                    elapsed_seconds=item.elapsed_seconds,
                ))
            except Exception as e:  # noqa: BLE001
                # Don't let a checkpoint write failure kill the batch —
                # the user can always retry; the only cost is reprocessing
                # the docs that didn't get recorded.
                log.warning("checkpoint: failed to record %s: %s", item.path, e)
        if progress_every and i % progress_every == 0 and total:
            log.info("Batch progress: %d / %d (%.1f%%)",
                     i, total, 100 * i / total)
        if on_progress is not None:
            try:
                on_progress(i, total or -1, item)
            except Exception:
                log.exception("on_progress callback raised (ignored)")
        yield item


__all__ = ["BatchItemResult", "process_batch"]
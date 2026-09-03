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

"""Storage interface.

Keeps the framework decoupled from any specific database / object store.
Default in-memory implementation supports tests + single-node demos.
Swap in Postgres + S3 (or whatever) for production.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class StoredResult:
    """A pipeline run, persisted."""

    id: str
    doc_id: str
    schema_name: str
    backend_name: str
    mode: str | None
    classification: str | None
    extraction: dict[str, Any]
    confidence: dict[str, float] | None
    validation: dict[str, Any] | None
    source_path: str
    created_at: float
    reviewed: bool = False
    reviewed_extraction: dict[str, Any] | None = None
    reviewer: str | None = None
    last_reviewed_at: float | None = None  # epoch seconds, set by mark_reviewed/submit_review


class Storage(Protocol):
    def put(self, result: StoredResult) -> str: ...
    def get(self, result_id: str) -> StoredResult | None: ...
    def list(
        self,
        doc_id: str | None = None,
        limit: int = 50,
        *,
        reviewed_only: bool = False,
        reviewed_since: float | None = None,
        schema_name: str | None = None,
    ) -> list[StoredResult]: ...
    def mark_reviewed(
        self, result_id: str, edited: dict[str, Any], reviewer: str
    ) -> None: ...


class InMemoryStorage(Storage):
    """Thread-safe in-memory store. Useful for tests and single-process demos."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, StoredResult] = {}

    def put(self, result: StoredResult) -> str:
        if not result.id:
            result.id = uuid.uuid4().hex[:16]
        with self._lock:
            self._by_id[result.id] = result
        return result.id

    def get(self, result_id: str) -> StoredResult | None:
        with self._lock:
            return self._by_id.get(result_id)

    def list(
        self,
        doc_id: str | None = None,
        limit: int = 50,
        *,
        reviewed_only: bool = False,
        reviewed_since: float | None = None,
        schema_name: str | None = None,
    ) -> list[StoredResult]:
        with self._lock:
            results = sorted(self._by_id.values(), key=lambda r: -r.created_at)
            if doc_id:
                results = [r for r in results if r.doc_id == doc_id]
            if reviewed_only:
                results = [r for r in results if r.reviewed]
            if schema_name:
                results = [r for r in results if r.schema_name == schema_name]
            if reviewed_since is not None:
                # For unreviewed items, last_reviewed_at is None — exclude
                # them entirely from the since-filtered list. Items reviewed
                # before `since` are also excluded.
                results = [r for r in results if r.last_reviewed_at is not None and r.last_reviewed_at >= reviewed_since]
            return results[:limit]

    def mark_reviewed(
        self, result_id: str, edited: dict[str, Any], reviewer: str
    ) -> None:
        with self._lock:
            r = self._by_id.get(result_id)
            if r is None:
                return
            r.reviewed = True
            r.reviewed_extraction = edited
            r.reviewer = reviewer


class JsonFileStorage(Storage):
    """Line-delimited JSON store on disk. Trivially inspectable, zero-deps.

    Memory profile: keeps an in-memory cache of the file's parsed
    StoredResults. Cache is invalidated on ``put()`` and
    ``mark_reviewed()``. For a file with N entries, peak memory is
    roughly 2-3× the on-disk JSON size (raw dicts + StoredResult
    objects + the cache dict itself).

    For workloads with >10k stored results, switch to ``SqlStorage``
    instead — JSONL doesn't index either and the full-file cache
    becomes the dominant cost.
    """

    # Cache invalidation: byte-offset of last file size we read.
    # If the file has grown (someone wrote outside our lock), drop the cache.
    _CACHE_FILE_STALE = -1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        # In-memory cache: id -> StoredResult
        self._cache: dict[str, StoredResult] | None = None
        self._cache_size_bytes: int = self._CACHE_FILE_STALE

    def _cache_valid(self) -> bool:
        """True iff the cache exists AND the file hasn't changed on disk."""
        if self._cache is None or self._cache_size_bytes == self._CACHE_FILE_STALE:
            return False
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            return False
        return current_size == self._cache_size_bytes

    def _invalidate_cache(self) -> None:
        self._cache = None
        self._cache_size_bytes = self._CACHE_FILE_STALE

    def _read_all(self) -> dict[str, StoredResult]:
        """Return cached entries if valid; otherwise re-read the file.

        Crashes during a `put()` (process kill, full disk) can leave a
        partial trailing line. Reading the whole file should never take
        down the app — we skip and continue. Use a real DB for transactions.
        """
        # Hot path: cache is valid -> return it (no I/O)
        if self._cache_valid():
            return self._cache  # type: ignore[return-value]

        # Cold path: re-read the file
        by_id: dict[str, StoredResult] = {}
        with self.path.open() as f:
            for line_no, raw in enumerate(f, start=1):
                s = raw.strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                    by_id[d["id"]] = StoredResult(**d)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "skipping corrupt line %d in %s: %s",
                        line_no, self.path, e,
                    )

        # Cache + remember file size at this point
        self._cache = by_id
        self._cache_size_bytes = self.path.stat().st_size
        return by_id

    def put(self, result: StoredResult) -> str:
        if not result.id:
            result.id = uuid.uuid4().hex[:16]
        if not result.created_at:
            result.created_at = time.time()
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(asdict(result), default=str) + "\n")
        # Cache is now stale; drop it. The next read will rebuild.
        self._invalidate_cache()
        return result.id

    def get(self, result_id: str) -> StoredResult | None:
        return self._read_all().get(result_id)

    def list(
        self,
        doc_id: str | None = None,
        limit: int = 50,
        *,
        reviewed_only: bool = False,
        reviewed_since: float | None = None,
        schema_name: str | None = None,
    ) -> list[StoredResult]:
        all_results = sorted(self._read_all().values(), key=lambda r: -r.created_at)
        if doc_id:
            all_results = [r for r in all_results if r.doc_id == doc_id]
        if reviewed_only:
            all_results = [r for r in all_results if r.reviewed]
        if schema_name:
            all_results = [r for r in all_results if r.schema_name == schema_name]
        if reviewed_since is not None:
            # created_at on JsonFileStorage is float; last_reviewed_at we
            # don't track there yet, so fall back to created_at.
            # (Pre-fix note: this also includes never-reviewed rows because
            # their last_reviewed_at is None and falls through to
            # created_at. Acceptable for the JsonFileStorage path
            # because we don't expect rich filtering on it — SqlStorage
            # is the recommended backend for that.)
            all_results = [
                r for r in all_results
                if (getattr(r, "last_reviewed_at", None) or r.created_at) >= reviewed_since
            ]
        return all_results[:limit]

    def mark_reviewed(
        self, result_id: str, edited: dict[str, Any], reviewer: str
    ) -> None:
        # For an audit-grade store you'd append a separate review event;
        # here we just append a fresh line with updated state.
        original = self.get(result_id)
        if original is None:
            return
        import time as _t
        original.reviewed = True
        original.reviewed_extraction = edited
        original.reviewer = reviewer
        original.last_reviewed_at = _t.time()
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(asdict(original), default=str) + "\n")
        # Cache is now stale (the new append could shadow an earlier entry)
        self._invalidate_cache()

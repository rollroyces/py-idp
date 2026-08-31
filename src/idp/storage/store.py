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
from dataclasses import asdict, dataclass, field
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


class Storage(Protocol):
    def put(self, result: StoredResult) -> str: ...
    def get(self, result_id: str) -> StoredResult | None: ...
    def list(self, doc_id: str | None = None, limit: int = 50) -> list[StoredResult]: ...
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

    def list(self, doc_id: str | None = None, limit: int = 50) -> list[StoredResult]:
        with self._lock:
            results = sorted(self._by_id.values(), key=lambda r: -r.created_at)
            if doc_id:
                results = [r for r in results if r.doc_id == doc_id]
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
    """Line-delimited JSON store on disk. Trivially inspectable, zero-deps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def _read_all(self) -> dict[str, StoredResult]:
        """Re-read the entire file. Skips corrupt lines (logs a warning).

        Crashes during a `put()` (process kill, full disk) can leave a
        partial trailing line. Reading the whole file should never take
        down the app — we skip and continue. Use a real DB for transactions.
        """
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
        return by_id

    def put(self, result: StoredResult) -> str:
        if not result.id:
            result.id = uuid.uuid4().hex[:16]
        if not result.created_at:
            result.created_at = time.time()
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(result), default=str) + "\n")
        return result.id

    def get(self, result_id: str) -> StoredResult | None:
        return self._read_all().get(result_id)

    def list(self, doc_id: str | None = None, limit: int = 50) -> list[StoredResult]:
        all_results = sorted(self._read_all().values(), key=lambda r: -r.created_at)
        if doc_id:
            all_results = [r for r in all_results if r.doc_id == doc_id]
        return all_results[:limit]

    def mark_reviewed(
        self, result_id: str, edited: dict[str, Any], reviewer: str
    ) -> None:
        # For an audit-grade store you'd append a separate review event;
        # here we just append a fresh line with updated state.
        original = self.get(result_id)
        if original is None:
            return
        original.reviewed = True
        original.reviewed_extraction = edited
        original.reviewer = reviewer
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(original), default=str) + "\n")

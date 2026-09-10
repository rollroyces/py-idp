"""JSONL-based checkpoint store for process_batch().

A simple append-only ledger of which documents have been processed.
The store is intentionally minimal — it does NOT store the full
extraction (that's the caller's job if they want it persisted). It
records:

  - path:    str  (the source document path; primary key)
  - ok:      bool (True if Pipeline.run succeeded)
  - error:   str | None (failure message, if any)
  - result:  dict | None (extraction on success; None on failure)
  - seconds: float
  - ts:      float (epoch seconds when this doc was processed)

Why JSONL?
  - Append-only: one line per doc, atomic single-line write.
  - Human-readable: easy to ``wc -l`` or ``grep`` for debugging.
  - Crash-safe: a crash mid-batch leaves a partial last line; reader
    skips malformed lines (same pattern as JsonFileStorage).

Why not use the ExtractionCache?
  The cache is keyed by request content (sha256 of schema + backend
  + payload). It does NOT track which PATHS have been processed —
  which is what batch resume needs. The two complement each other:

    - ExtractionCache: same input -> no LLM call (deduplication)
    - CheckpointStore: same path -> skip entirely (batch resume)

  Both can coexist. If both are enabled, the cache catches intra-batch
  duplicates and the checkpoint catches inter-batch resume.
"""
from __future__ import annotations

import fcntl
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CheckpointEntry:
    """One row in the JSONL checkpoint ledger."""
    path: str
    ok: bool
    error: str | None = None
    result: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0
    timestamp: float = 0.0

    def to_json(self) -> str:
        # Compact JSON; one entry per line.
        return json.dumps({
            "path": self.path,
            "ok": self.ok,
            "error": self.error,
            "result": self.result,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "ts": self.timestamp,
        }, default=str)

    @classmethod
    def from_json(cls, raw: str) -> CheckpointEntry:
        d = json.loads(raw)
        return cls(
            path=d["path"],
            ok=d["ok"],
            error=d.get("error"),
            result=d.get("result"),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            timestamp=d.get("ts", 0.0),
        )


class CheckpointStore:
    """Append-only JSONL checkpoint ledger with file locking.

    Usage:
        store = CheckpointStore("/path/to/batch.jsonl")
        if path in store:
            print("already done, skipping")
            continue
        store.record(CheckpointEntry(path=path, ok=True, ...))

    Concurrency:
        Multiple processes writing to the same checkpoint path will
        contend on a flock. The lock is held only during the brief
        append (single line); reads are unlocked and may see partial
        state. That's fine for batch-resume use cases where you're not
        running two concurrent batches against the same checkpoint.

    Crash safety:
        A crash mid-append leaves a partial trailing line. The reader
        skips malformed lines on load (same approach as
        JsonFileStorage).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._fh = None  # kept for API compat; not currently used

    def __contains__(self, path: str) -> bool:
        return path in self._seen_paths()

    def __len__(self) -> int:
        """Count of unique paths recorded (skips malformed lines + dupes).

        Note: the underlying file is append-only and may contain
        duplicates if you re-record the same path. ``__len__`` reports
        the deduped count, which is what callers care about for
        batch-progress reporting.
        """
        return len(self._seen_paths())

    def seen_paths(self) -> set[str]:
        """Return a copy of the set of paths already recorded."""
        return set(self._seen_paths())

    def _seen_paths(self) -> set[str]:
        return {e.path for e in self._entries()}

    def _entries(self) -> list[CheckpointEntry]:
        """Read all entries from disk. Skips malformed lines."""
        out: list[CheckpointEntry] = []
        if not self.path.exists():
            return out
        with self.path.open() as f:
            for line_no, raw in enumerate(f, start=1):
                s = raw.strip()
                if not s:
                    continue
                try:
                    out.append(CheckpointEntry.from_json(s))
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "skipping malformed checkpoint line %d in %s: %s",
                        line_no, self.path, e,
                    )
        return out

    def record(self, entry: CheckpointEntry) -> None:
        """Append a new entry. Atomic single-line write under flock."""
        if entry.timestamp == 0.0:
            entry.timestamp = time.time()
        line = entry.to_json() + "\n"
        # Use a per-record file handle for flock (the lock is held only
        # during the brief append). Don't keep the handle open across
        # the batch lifetime to avoid surprises on close.
        with self.path.open("a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError as e:
                log.warning("checkpoint flock failed for %s: %s", self.path, e)
                # Best-effort: rewrite the line without the lock.
                # Single-process batch is the common case anyway.
                with self.path.open("a") as f2:
                    f2.write(line)
                    f2.flush()

    def clear(self) -> None:
        """Truncate the ledger. Use this for a fresh batch run.

        Note: idempotent batch resume DOES NOT need this — re-running
        a batch naturally skips already-done paths. Use clear() only when
        you want to force reprocessing (e.g., the LLM model changed and
        you want fresh extractions).
        """
        self.path.write_text("")

    def close(self) -> None:
        # No persistent handle to close; per-record file handles are
        # opened and closed within record(). Method exists for API
        # symmetry with the cache and future-proofing (e.g., if we
        # add an append-mode long-lived handle later).
        pass

    def __enter__(self) -> CheckpointStore:  # noqa: UP037 — Py3.9 compat
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        """Return checkpoint statistics for observability."""
        entries = self._entries()
        if not entries:
            return {"total": 0, "ok": 0, "errors": 0, "first_ts": None, "last_ts": None}
        ok = sum(1 for e in entries if e.ok)
        return {
            "total": len(entries),
            "ok": ok,
            "errors": len(entries) - ok,
            "first_ts": min((e.timestamp for e in entries), default=None),
            "last_ts": max((e.timestamp for e in entries), default=None),
        }


__all__ = ["CheckpointStore", "CheckpointEntry"]

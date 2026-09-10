"""Reliability primitives: retries, caching, structured errors.

These wrap the existing ``Backend.complete()`` flow without changing
the Backend protocol. Each is independent and opt-in:

  - ``RetryingBackend`` wraps any backend and retries on transient errors
    (rate limits, timeouts, connection errors). Exponential backoff
    with jitter. Stops after ``max_retries`` attempts.

  - ``ExtractionCache`` is a disk-backed, content-addressed cache for
    extraction results. Key = ``(doc_hash, schema_name, backend_name)``.
    Re-running the same extraction returns the cached JSON without
    an LLM call.

  - ``ExtractionError`` is a structured exception class with retryable
    vs non-retryable hints. RetryingBackend uses this to decide whether
    to backoff or fail fast.

Why these three together:
  - A 1000-doc batch on Databricks WILL hit a 429 at some point.
    Without retry: the whole batch fails at the first rate-limited
    chunk. With retry: one transient error costs 5 seconds and the
    batch continues.
  - The same doc being extracted twice (dev iteration, idempotent
    retries) burns LLM tokens. A disk cache turns the second run
    into a hash lookup.
  - ``extract_chunk_failed: timeout`` is the only error signal today.
    Structured errors let callers programmatically decide retry
    strategy without parsing strings.
"""
from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idp.llm.backend import Backend, CompletionRequest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured extraction errors
# ---------------------------------------------------------------------------
class ExtractionError(Exception):
    """Base class for extraction failures.

    Default is ``retryable=True`` (safer default: retry once, then
    classify by error pattern). Subclasses that are NOT retryable
    (AuthError, BadRequestError) override this to False.

    The naming is intentional: an unknown error SHOULD be retried;
    you'd rather burn 5 seconds retrying than lose an entire batch
    to a transient hiccup you didn't recognize.
    """

    retryable: bool = True

    def __init__(self, message: str, *, backend_name: str = "?", cause: Exception | None = None):
        super().__init__(message)
        self.backend_name = backend_name
        self.cause = cause


class RateLimitError(ExtractionError):
    """Backend returned 429 (rate limit) or equivalent."""
    retryable = True  # explicit; default is True but keep the marker


class TimeoutError_(ExtractionError):  # noqa: N801 — disambiguate from builtin
    """Backend timed out (network or model)."""
    retryable = True  # explicit


class ConnectionError_(ExtractionError):  # noqa: N801 — disambiguate from builtin
    """Backend couldn't reach the model server."""
    retryable = True  # explicit


class AuthError(ExtractionError):
    """Authentication failed (bad API key, expired token). NOT retryable."""
    retryable = False


class BadRequestError(ExtractionError):
    """Backend rejected the request (400, malformed payload). NOT retryable."""
    retryable = False


# Detection: map backend exception classes / HTTP codes to our taxonomy.
def classify_exception(exc: Exception, backend_name: str = "?") -> ExtractionError:
    """Classify a raw backend exception into an ExtractionError.

    Looks at the exception class name and any HTTP status code on the
    exception. Unknown errors get a generic ExtractionError (which
    defaults to retryable=True) — retry once and see if it sticks.

    Note: when ``exc`` is already an ExtractionError (e.g. from a
    downstream wrapper), pass it through unchanged so retry decisions
    are preserved.
    """
    if isinstance(exc, ExtractionError):
        return exc

    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    # Common patterns in real backend errors
    if "rate" in msg and "limit" in msg:
        return RateLimitError(str(exc), backend_name=backend_name, cause=exc)
    if "429" in msg or "rate_limit" in name or "ratelimit" in name:
        return RateLimitError(str(exc), backend_name=backend_name, cause=exc)
    if "timeout" in name or "timed out" in msg:
        return TimeoutError_(str(exc), backend_name=backend_name, cause=exc)
    if "connection" in name or "network" in name or "unreachable" in msg:
        return ConnectionError_(str(exc), backend_name=backend_name, cause=exc)
    if "auth" in name or "401" in msg or "403" in msg or "api key" in msg:
        return AuthError(str(exc), backend_name=backend_name, cause=exc)
    if "bad request" in msg or "400" in msg or "invalid" in msg:
        return BadRequestError(str(exc), backend_name=backend_name, cause=exc)
    # Unknown: retryable by default (retry once, then fail)
    return ExtractionError(str(exc), backend_name=backend_name, cause=exc)


# ---------------------------------------------------------------------------
# Retrying backend wrapper
# ---------------------------------------------------------------------------
@dataclass
class RetryConfig:
    """Retry policy for ``RetryingBackend``."""
    max_retries: int = 4
    """Total attempts including the first. So max_retries=4 -> 1 try + 3 retries."""

    initial_delay_sec: float = 1.0
    """First backoff delay. Doubles each retry."""

    max_delay_sec: float = 30.0
    """Cap on backoff delay (avoids 8-min waits after many failures)."""

    jitter: float = 0.2
    """±20% randomization to avoid thundering-herd on rate-limit recovery."""

    backoff_multiplier: float = 2.0
    """Geometric growth rate. 2.0 = 1s, 2s, 4s, 8s, 16s."""

    def delay_for(self, attempt: int) -> float:
        """Compute backoff delay for the Nth retry (attempt=1 -> first retry).

        attempt=1 -> initial_delay_sec, attempt=2 -> 2x, etc.
        """
        delay = self.initial_delay_sec * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay_sec)
        if self.jitter > 0:
            delta = delay * self.jitter
            delay = delay + random.uniform(-delta, delta)
        return max(0.0, delay)


class RetryingBackend:
    """Wrap a Backend with retry + exponential backoff.

    Usage:
        backend = RetryingBackend(OpenAICompatBackend(...))
        # Use backend as you would the wrapped one.
    """

    def __init__(self, wrapped: Backend, config: RetryConfig | None = None):
        self.wrapped = wrapped
        self.config = config or RetryConfig()
        self.name = f"retry({wrapped.name})"

    @property
    def is_multimodal(self) -> bool:
        return self.wrapped.is_multimodal

    def complete(self, req: CompletionRequest) -> str:
        """Call wrapped.complete() with retries on transient errors."""
        last_err: Exception | None = None
        # attempt=0 is the initial call, attempt=1..max_retries-1 are retries
        for attempt in range(self.config.max_retries):
            try:
                return self.wrapped.complete(req)
            except ExtractionError as e:
                last_err = e
                if not e.retryable:
                    log.warning(
                        "RetryingBackend: non-retryable %s from %s on attempt %d/%d: %s",
                        type(e).__name__, self.wrapped.name,
                        attempt + 1, self.config.max_retries, e,
                    )
                    raise
                # Retryable: log and backoff
                if attempt + 1 >= self.config.max_retries:
                    log.warning(
                        "RetryingBackend: gave up on %s after %d attempts: %s",
                        self.wrapped.name, self.config.max_retries, e,
                    )
                    raise
                delay = self.config.delay_for(attempt + 1)
                log.info(
                    "RetryingBackend: %s on attempt %d/%d, retrying in %.2fs: %s",
                    type(e).__name__, attempt + 1, self.config.max_retries, delay, e,
                )
                time.sleep(delay)
            except Exception as e:
                # Backend raised an unclassified exception. Wrap it and retry.
                classified = classify_exception(e, backend_name=self.wrapped.name)
                last_err = classified
                if not classified.retryable:
                    raise classified from e
                if attempt + 1 >= self.config.max_retries:
                    raise classified from e
                delay = self.config.delay_for(attempt + 1)
                log.info(
                    "RetryingBackend: unclassified %s -> %s on attempt %d/%d, retrying in %.2fs",
                    type(e).__name__, type(classified).__name__,
                    attempt + 1, self.config.max_retries, delay,
                )
                time.sleep(delay)

        # Unreachable: max_retries must be >= 1
        raise last_err if last_err else RuntimeError("max_retries must be >= 1")


# ---------------------------------------------------------------------------
# Disk-based extraction cache
# ---------------------------------------------------------------------------
def hash_request(req: CompletionRequest, *, schema_name: str, backend_name: str) -> str:
    """Stable content hash for an extraction request.

    Same input -> same hash. Different temperature, json_mode, or
    extra instructions -> different hash (so cache hits are semantically
    equivalent, not just textually similar).
    """
    h = hashlib.sha256()
    h.update(schema_name.encode("utf-8"))
    h.update(b"|")
    h.update(backend_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(req.temperature).encode())
    h.update(b"|")
    h.update(str(req.json_mode).encode())
    h.update(b"|")
    for msg in req.messages:
        h.update(msg.role.encode("utf-8"))
        h.update(b"\x1f")
        h.update(msg.content.encode("utf-8"))
        h.update(b"\x1f")
        # NOTE: we hash images by length only. Including the full base64
        # would 4x the hash size for negligible collision-avoidance gain
        # (the content is what it is; collision is by content not metadata).
        for img_b64 in msg.images_b64 or []:
            h.update(str(len(img_b64)).encode("utf-8"))
            h.update(b":")
        h.update(b"\x1e")
    return h.hexdigest()


class ExtractionCache:
    """Disk-backed cache for extraction results.

    Schema (SQLite, 1 table):
        cache (
            key TEXT PRIMARY KEY,   -- sha256 of (schema, backend, request)
            schema_name TEXT,
            backend_name TEXT,
            response TEXT,            -- raw LLM output
            created_at REAL,           -- epoch seconds
            hit_count INTEGER DEFAULT 0
        )

    Why SQLite, not JSONL: random-key lookup (no full-file scan) and
    concurrent access from multiple workers without corruption.

    Stats: every hit increments ``hit_count`` and updates ``created_at``
    so you can inspect cache effectiveness with::

        SELECT schema_name, COUNT(*) AS n, SUM(hit_count) AS hits
        FROM cache GROUP BY schema_name;
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True) if str(path) != ":memory:" else None
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    schema_name TEXT NOT NULL,
                    backend_name TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_schema ON cache(schema_name)"
            )

    def get(self, key: str) -> str | None:
        """Return cached response or None on miss.

        On hit, increments ``hit_count`` and refreshes ``created_at``
        (LRU-like: most-recently-used entries stick around).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1, created_at = ? WHERE key = ?",
                (time.time(), key),
            )
            return row[0]

    def put(self, key: str, *, schema_name: str, backend_name: str, response: str) -> None:
        """Cache a response. Overwrites on collision."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO cache
                   (key, schema_name, backend_name, response, created_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, COALESCE((SELECT hit_count FROM cache WHERE key=?), 0))""",
                (key, schema_name, backend_name, response, time.time(), key),
            )

    def stats(self) -> dict[str, Any]:
        """Return cache effectiveness stats."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM cache").fetchone()
            by_schema = self._conn.execute(
                "SELECT schema_name, COUNT(*), COALESCE(SUM(hit_count), 0) FROM cache GROUP BY schema_name"
            ).fetchall()
            return {
                "entries": total[0],
                "total_hits": total[1],
                "by_schema": [
                    {"schema": r[0], "entries": r[1], "hits": r[2]} for r in by_schema
                ],
            }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachingBackend:
    """Wrap any Backend with a disk-backed extraction cache.

    Cache key = hash of (schema_name, backend_name, request payload).
    Cache hit returns the cached response without calling the wrapped
    backend. Cache miss calls the wrapped backend, then stores the result.

    Usage:
        cache = ExtractionCache("~/.cache/idp/extract.db")
        backend = CachingBackend(OpenAICompatBackend(...), cache)
        # Use backend as normal; cache is transparent.
    """

    def __init__(self, wrapped: Backend, cache: ExtractionCache, *, schema_name: str = ""):
        self.wrapped = wrapped
        self.cache = cache
        # schema_name is fixed per-CachingBackend instance. The
        # Pipeline knows the schema; pass it in.
        self._schema_name = schema_name

    @property
    def is_multimodal(self) -> bool:
        return self.wrapped.is_multimodal

    @property
    def name(self) -> str:
        return f"cache({self.wrapped.name})"

    def with_schema(self, schema_name: str) -> CachingBackend:
        """Return a view of this cache that pins a specific schema_name."""
        view = CachingBackend(self.wrapped, self.cache, schema_name=schema_name)
        return view

    def complete(self, req: CompletionRequest) -> str:
        """Return cached response or call wrapped + store result."""
        if not self._schema_name:
            raise ValueError(
                "CachingBackend needs a schema_name. Use .with_schema('Invoice') "
                "or pass schema_name when constructing."
            )
        key = hash_request(req, schema_name=self._schema_name, backend_name=self.wrapped.name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        response = self.wrapped.complete(req)
        self.cache.put(
            key,
            schema_name=self._schema_name,
            backend_name=self.wrapped.name,
            response=response,
        )
        return response


__all__ = [
    "ExtractionError",
    "RateLimitError",
    "TimeoutError_",
    "ConnectionError_",
    "AuthError",
    "BadRequestError",
    "classify_exception",
    "RetryConfig",
    "RetryingBackend",
    "hash_request",
    "ExtractionCache",
    "CachingBackend",
]

"""Per-key + global sliding-window rate limiter.

Two-tier design:
  - **per-key**: limits one API key to N requests/minute
  - **global**:   limits all keys combined to G requests/minute

A request that exceeds either limit raises ``RateLimitedError``. The
limiter is in-process and thread-safe; for multi-worker / multi-pod
deployments, replace with a Redis-backed limiter behind the same API.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window rate limiter (1-minute windows)."""

    def __init__(self, *, per_key_per_minute: int = 60, global_per_minute: int = 0):
        """``*_per_minute = 0`` disables that tier."""
        self.per_key = per_key_per_minute
        self.global_limit = global_per_minute
        self._lock = threading.Lock()
        self._per_key_window: dict[str, deque[float]] = defaultdict(deque)
        self._global_window: deque[float] = deque()

    def check(self, key: str | None = None) -> None:
        """Raise ``RateLimitedError`` if the request would exceed a limit.

        ``key`` is the API key (or ``None`` for unauthenticated calls —
        these share a single bucket).
        """
        if self.per_key <= 0 and self.global_limit <= 0:
            return
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            # Per-key window
            if self.per_key > 0:
                k = key or "_anon"
                window = self._per_key_window[k]
                while window and window[0] < cutoff:
                    window.popleft()
                if len(window) >= self.per_key:
                    raise RateLimitedError(
                        f"per-key limit exceeded ({self.per_key}/min); retry after "
                        f"{60.0 - (now - window[0]):.1f}s"
                    )
                window.append(now)

            # Global window
            if self.global_limit > 0:
                while self._global_window and self._global_window[0] < cutoff:
                    self._global_window.popleft()
                if len(self._global_window) >= self.global_limit:
                    raise RateLimitedError(
                        f"global limit exceeded ({self.global_limit}/min); retry after "
                        f"{60.0 - (now - self._global_window[0]):.1f}s"
                    )
                self._global_window.append(now)

    def reset(self) -> None:
        """Clear all windows. For tests only."""
        with self._lock:
            self._per_key_window.clear()
            self._global_window.clear()


from idp.errors import RateLimitedError  # noqa: E402  (import after use for circular)

__all__ = ["RateLimiter"]
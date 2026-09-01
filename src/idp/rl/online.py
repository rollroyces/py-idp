"""Online (per-review) RL policy cache.

Every time `mark_reviewed()` or `submit_review()` is called on a storage
backend, the just-stored review is fed into this cache. The cache:
  - keeps the policy in memory (single source of truth)
  - debounces disk flushes (default 1s) so a burst of reviews doesn't
    cause O(N) re-aggregations
  - flushes atomically (write tmp, then `os.replace`) so a crash mid-write
    can't leave a half-written policy.json
  - guarantees every review within `min_reviews` of a field has been
    folded into the in-memory policy before the next pipeline run

Recommended usage:
    storage = SqlStorage("sqlite:///./idp.db")
    cache = PolicyCache(policy_path="policy.json", flush_interval_sec=1.0)
    storage.mark_reviewed = cache.wrap_mark_reviewed(storage.mark_reviewed)
    # OR: cache.attach_to_storage(storage)  for the common case

For multi-process deploys, only ONE process should hold the cache. Other
processes just read policy.json from disk (existing behaviour, no change).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from idp.rl.policy import PolicyConfig, update_policy
from idp.rl.reward import ReviewRewards, aggregate_rewards, derive_field_rewards
from idp.storage.store import StoredResult

log = logging.getLogger(__name__)


class PolicyCache:
    """In-memory policy cache that observes per-review updates."""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        flush_interval_sec: float = 1.0,
        min_reviews: int = 10,
        policy: PolicyConfig | None = None,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.flush_interval_sec = flush_interval_sec
        self.min_reviews = min_reviews

        self._lock = threading.Lock()
        # accumulated ReviewRewards since last flush (not yet folded into _stats)
        self._pending: list[ReviewRewards] = []
        # full accumulated stats (in-memory copy of what disk has)
        self._stats_per_field: dict[str, dict[str, int]] = {}
        # current policy in memory
        self._policy = policy or self._load_from_disk()

        # Background flusher
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True, name="PolicyCache")
        self._flusher.start()

    # ---- Public API ----
    def on_review(self, result: StoredResult) -> None:
        """Called by Storage.mark_reviewed/submit_review after persisting."""
        rewards = derive_field_rewards(result)
        if rewards is None:
            return
        with self._lock:
            self._pending.append(rewards)
        self._dirty.set()

    def attach_to_storage(self, storage: Any) -> None:
        """Monkey-patch storage.mark_reviewed so on_review fires automatically.

        Idempotent: subsequent calls are no-ops.
        """
        if getattr(storage, "_policy_cache_attached", False):
            return
        original = storage.mark_reviewed

        def wrapped(result_id, edited, reviewer):
            original(result_id, edited, reviewer)
            stored = storage.get(result_id)
            if stored is not None:
                self.on_review(stored)

        wrapped.__wrapped__ = original  # keep a handle for tests
        storage.mark_reviewed = wrapped
        storage._policy_cache_attached = True

    @property
    def policy(self) -> PolicyConfig:
        return self._policy

    def flush_now(self) -> None:
        """Synchronously fold pending reviews into policy + write to disk."""
        with self._lock:
            pending = self._pending
            self._pending = []
            self._merge_locked(pending)
            self._write_locked()

    def stop(self) -> None:
        self._stop.set()
        self._flusher.join(timeout=5)
        # final flush on shutdown
        try:
            self.flush_now()
        except Exception:  # noqa: BLE001
            pass

    # ---- Internals ----
    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            triggered = self._dirty.wait(timeout=self.flush_interval_sec)
            if self._stop.is_set():
                break
            if triggered:
                try:
                    self.flush_now()
                except Exception as e:  # noqa: BLE001
                    log.warning("policy flush failed: %s", e)
                self._dirty.clear()

    def _merge_locked(self, pending: list[ReviewRewards]) -> None:
        for r in pending:
            for fr in r.field_rewards:
                key = f"{fr.reward:+d}" if fr.reward != 0 else "0"
                d = self._stats_per_field.setdefault(fr.field, {})
                d[key] = d.get(key, 0) + 1
        # recompute policy from full stats
        stats = _stats_dict_to_policystats(self._stats_per_field, len(pending))
        self._policy = update_policy(stats, current=self._policy)

    def _write_locked(self) -> None:
        """Atomic write: tmp -> os.replace. Survives crashes mid-write."""
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.policy_path.with_suffix(self.policy_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._policy.to_dict(), indent=2))
        os.replace(tmp, self.policy_path)

    def _load_from_disk(self) -> PolicyConfig:
        if not self.policy_path.exists():
            return PolicyConfig(min_reviews=self.min_reviews)
        try:
            d = json.loads(self.policy_path.read_text())
            p = PolicyConfig(**d)
            # Don't overwrite min_reviews if it was persisted. The user
            # can override it on the new cache via the kwarg if they want.
            return p
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load policy from %s: %s; using defaults", self.policy_path, e)
            return PolicyConfig(min_reviews=self.min_reviews)


def _stats_dict_to_policystats(per_field: dict[str, dict[str, int]], n_reviews: int):
    """Construct a PolicyStats-shaped object from a flat per-field dict."""
    # Lazy import to avoid a cycle (reward.py imports from store; we
    # don't want online.py -> reward.py -> store.py -> online.py)
    from idp.rl.reward import PolicyStats

    return PolicyStats(n_reviews=n_reviews, n_fields=len(per_field), per_field=per_field)

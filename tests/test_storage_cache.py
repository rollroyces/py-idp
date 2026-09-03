"""Tests for JsonFileStorage in-memory cache (RAM optimization).

The cache makes repeat reads O(1) instead of O(N) bytes (re-reading the
file every time). It's invalidated automatically on put/mark_reviewed
and on file-size mismatch.
"""
from __future__ import annotations

import time
from unittest.mock import patch

from idp.storage.store import JsonFileStorage, StoredResult


def _make_result(idx: int, doc_id: str = "doc-1") -> StoredResult:
    return StoredResult(
        id=f"id-{idx}",
        doc_id=doc_id,
        source_path=f"/tmp/{idx}.pdf",
        schema_name="Invoice",
        backend_name="mock",
        mode="mock",
        classification="invoice",
        extraction={"vendor": f"V{idx}", "amount": float(idx)},
        confidence={"vendor": 0.95},
        validation={"vendor": True},
        created_at=time.time(),
    )


def test_first_read_populates_cache(tmp_path):
    """First read after invalidation populates the cache."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    for i in range(5):
        s.put(_make_result(i))
    s._invalidate_cache()
    assert s._cache is None
    s.get("id-2")
    assert s._cache is not None
    assert len(s._cache) == 5


def test_second_read_uses_cache(tmp_path):
    """Second consecutive read does NOT re-open the file."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    s.put(_make_result(0))
    s._invalidate_cache()
    s.get("id-0")  # cold read

    # Patch open() at the storage module to detect re-reads
    with patch("builtins.open", wraps=open) as mock_open:
        s.get("id-0")  # should use cache
        # No file open calls for the cache hit path
        assert mock_open.call_count == 0


def test_put_invalidates_cache(tmp_path):
    """put() drops the cache so the next read sees the new entry."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    s.put(_make_result(0))
    s.get("id-0")
    assert s._cache is not None

    s.put(_make_result(1))
    assert s._cache is None
    # Next read rebuilds cache with both entries
    s.get("id-0")
    assert "id-1" in s._cache


def test_mark_reviewed_invalidates_cache(tmp_path):
    """mark_reviewed() drops the cache (a new line is appended)."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    s.put(_make_result(0))
    s.get("id-0")
    assert s._cache is not None

    s.mark_reviewed("id-0", edited={"vendor": "NEW"}, reviewer="alice")
    assert s._cache is None


def test_cache_invalidates_when_file_grows_externally(tmp_path):
    """If something else writes to the file (size changed), cache drops."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    s.put(_make_result(0))
    s.get("id-0")
    cached_size = s._cache_size_bytes
    assert cached_size > 0

    # External write (any process) — file size changes. The external
    # line is intentionally incomplete (missing optional fields) so it
    # gets skipped by _read_all's corrupt-line handler — but the
    # important assertion is that the cache detected the size change
    # and re-read.
    with path.open("a") as f:
        f.write('{"id": "external", "incomplete": true}\n')

    s.get("id-0")  # Should invalidate cache and re-read
    assert s._cache_size_bytes > cached_size


def test_cache_invalidates_when_file_truncated(tmp_path):
    """If file size shrinks (truncation), cache drops."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    s.put(_make_result(0))
    s.put(_make_result(1))
    s.get("id-0")
    cached_size = s._cache_size_bytes
    assert cached_size > 0

    # Truncate to half the cached size
    with path.open("w") as f:
        f.write('{"truncated": true}')

    s.get("id-0")  # Should invalidate and re-read
    assert s._cache_size_bytes != cached_size


def test_cache_returns_same_object_on_subsequent_calls(tmp_path):
    """Repeated get() calls return from the cache (same dict object)."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    s.put(_make_result(0))

    r1 = s.get("id-0")
    r2 = s.get("id-0")
    r3 = s.get("id-0")

    # Same object — proves cache returned the same entry each time
    assert r1 is r2 is r3


def test_list_uses_cache(tmp_path):
    """list() also benefits from the cache (was the main hotspot)."""
    path = tmp_path / "r.jsonl"
    s = JsonFileStorage(path)
    for i in range(50):
        s.put(_make_result(i, doc_id=f"doc-{i % 5}"))

    # Cold read to populate cache
    s.list(limit=10)

    # Now 100 list() calls — should all hit cache
    with patch("builtins.open", wraps=open) as mock_open:
        for _ in range(100):
            results = s.list(limit=10)
            assert len(results) > 0
        # No file opens (everything served from cache)
        assert mock_open.call_count == 0
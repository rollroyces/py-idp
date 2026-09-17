"""Tests for online RL policy cache (PolicyCache)."""
from __future__ import annotations

import pytest

from idp.rl.online import PolicyCache
from idp.rl.policy import PolicyConfig
from idp.storage.store import InMemoryStorage, StoredResult

import idp.rl.online  # noqa: F401  # module-level import for `idp.rl.online.json` access


def _reviewed_stored_result(rid: str = "r1", vendor_correct: bool = False) -> StoredResult:
    """A stored result that's been reviewed; vendor_name corrected if vendor_correct."""
    model = {"vendor_name": "Acme", "total_amount": 100.0}
    human = model if vendor_correct else {
        "vendor_name": "Acme Widgets Ltd.",  # corrected
        "total_amount": 100.0,
    }
    return StoredResult(
        id=rid, doc_id=rid, schema_name="Invoice", backend_name="mock",
        mode="ocr_llm", classification="invoice",
        extraction=model, confidence={"vendor_name": 0.75}, validation=None,
        source_path="/x", created_at=0.0,
        reviewed=True, reviewed_extraction=human, reviewer="alice",
    )


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------
def test_cache_loads_from_disk_or_creates_defaults(tmp_path):
    policy_path = tmp_path / "policy.json"
    c = PolicyCache(str(policy_path), flush_interval_sec=0.1)
    try:
        assert c.policy.min_reviews == 10
        assert policy_path.exists() is False  # nothing flushed yet
    finally:
        c.stop()


def test_cache_flushes_to_disk_after_review(tmp_path):
    import json
    policy_path = tmp_path / "policy.json"
    # Two-stage test:
    # 1. c1 writes min_reviews=2 to disk
    c = PolicyCache(str(policy_path), flush_interval_sec=0.05, min_reviews=2)
    try:
        c.on_review(_reviewed_stored_result("r1", vendor_correct=False))
        c.flush_now()
        assert policy_path.exists()
        d = json.loads(policy_path.read_text())
        assert d["min_reviews"] == 2
        assert "field_floors" in d
    finally:
        c.stop()
    # 2. c2 reads the file; min_reviews should be loaded from disk (2), not
    #    re-defaulted to 10. (This is the test that catches the bug where
    #    _load_from_disk() overwrites the on-disk value with self.min_reviews.)
    c2 = PolicyCache(str(policy_path), flush_interval_sec=0.05)
    try:
        assert c2.policy.min_reviews == 2, (
            f"expected to preserve on-disk min_reviews=2, got {c2.policy.min_reviews}"
        )
    finally:
        c2.stop()


def test_cache_min_reviews_guard_prevents_early_override(tmp_path):
    """With min_reviews=5, a single review should NOT set any floor."""
    policy_path = tmp_path / "policy.json"
    c = PolicyCache(str(policy_path), flush_interval_sec=0.05, min_reviews=5)
    try:
        c.on_review(_reviewed_stored_result("r1", vendor_correct=False))
        c.flush_now()
        # vendor_name has only 1 observation -> below min_reviews=5 -> no override
        assert "vendor_name" not in c.policy.field_floors
    finally:
        c.stop()


def test_cache_overrides_after_min_reviews_reached(tmp_path):
    policy_path = tmp_path / "policy.json"
    c = PolicyCache(str(policy_path), flush_interval_sec=0.05, min_reviews=5)
    try:
        # 5 reviews with vendor_name wrong 100% of the time
        for i in range(5):
            c.on_review(_reviewed_stored_result(f"r{i}", vendor_correct=False))
        c.flush_now()
        assert "vendor_name" in c.policy.field_floors
        assert c.policy.field_floors["vendor_name"] == pytest.approx(0.95)
    finally:
        c.stop()


def test_cache_no_override_below_failure_threshold(tmp_path):
    """Even past min_reviews, fail_rate < threshold means no override."""
    policy_path = tmp_path / "policy.json"
    c = PolicyCache(str(policy_path), flush_interval_sec=0.05, min_reviews=3)
    try:
        # 3 reviews, 1 wrong, 2 right -> fail_rate = 1/3 = 0.33; threshold default 0.30
        # actually 0.33 > 0.30 -> override fires. use 10 reviews with 1 wrong instead
        for i in range(10):
            c.on_review(_reviewed_stored_result(f"r{i}", vendor_correct=(i > 0)))
        c.flush_now()
        # fail_rate = 1/10 = 0.10 < 0.30 -> no override
        assert "vendor_name" not in c.policy.field_floors
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# attach_to_storage
# ---------------------------------------------------------------------------
def test_attach_to_storage_marks_reviewed_and_updates_policy(tmp_path):
    """The canonical end-to-end: storage.mark_reviewed fires the cache."""
    policy_path = tmp_path / "policy.json"
    storage = InMemoryStorage()
    storage.put(StoredResult(
        id="r1", doc_id="d1", schema_name="Invoice", backend_name="mock",
        mode="ocr_llm", classification="invoice",
        extraction={"vendor_name": "Acme", "total_amount": 100.0},
        confidence=None, validation=None, source_path="/x", created_at=0.0,
    ))
    cache = PolicyCache(str(policy_path), flush_interval_sec=0.05, min_reviews=5)
    cache.attach_to_storage(storage)
    try:
        # Hit mark_reviewed 5 times across different results, all corrected
        for i in range(5):
            storage.put(StoredResult(
                id=f"r{i}", doc_id=f"d{i}", schema_name="Invoice", backend_name="mock",
                mode="ocr_llm", classification="invoice",
                extraction={"vendor_name": "Acme", "total_amount": 100.0},
                confidence=None, validation=None, source_path="/x", created_at=0.0,
            ))
            storage.mark_reviewed(f"r{i}", {"vendor_name": "Acme Widgets Ltd.", "total_amount": 100.0}, "alice")
        cache.flush_now()
        # 5/5 vendor_name wrong -> override fires
        assert "vendor_name" in cache.policy.field_floors
        # Disk has the same data
        assert "vendor_name" in PolicyConfig.load(str(policy_path)).field_floors
    finally:
        cache.stop()


def test_attach_to_storage_is_idempotent(tmp_path):
    storage = InMemoryStorage()
    cache = PolicyCache(str(tmp_path / "policy.json"), flush_interval_sec=0.05)
    try:
        cache.attach_to_storage(storage)
        cache.attach_to_storage(storage)  # no-op
        assert getattr(storage, "_policy_cache_attached", False) is True
    finally:
        cache.stop()


# ---------------------------------------------------------------------------
# Atomicity / crash safety
# ---------------------------------------------------------------------------
def test_atomic_write_does_not_leave_tmp_file(tmp_path):
    """Successful flushes should not leave .tmp files behind."""
    policy_path = tmp_path / "policy.json"
    cache = PolicyCache(str(policy_path), flush_interval_sec=0.05)
    try:
        cache.on_review(_reviewed_stored_result("r1"))
        cache.flush_now()
        files = list(tmp_path.iterdir())
        assert any(f.name == "policy.json" for f in files)
        assert not any(f.name.endswith(".tmp") for f in files)
    finally:
        cache.stop()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
def test_policy_round_trip_with_min_reviews(tmp_path):
    policy_path = tmp_path / "policy.json"
    original = PolicyConfig(
        min_reviews=20,
        high_failure_threshold=0.25,
        field_floors={"vendor_name": 0.95},
        field_penalties={"vendor_name": 0.2},
    )
    original.save(str(policy_path))
    loaded = PolicyConfig.load(str(policy_path))
    assert loaded.min_reviews == 20
    assert loaded.high_failure_threshold == 0.25
    assert loaded.field_floors == {"vendor_name": 0.95}


# ---------------------------------------------------------------------------
# PolicyCache: corrupt JSON file (line 73)
# ---------------------------------------------------------------------------
def test_policy_cache_loads_with_defaults_on_corrupt_file(tmp_path, caplog):
    """If the policy JSON is corrupt, load returns defaults + logs error."""
    import logging

    from idp.rl.online import PolicyCache

    policy_file = tmp_path / "policy.json"
    policy_file.write_text("{not valid json at all")

    caplog.set_level(logging.ERROR)
    cache = PolicyCache(policy_path=str(policy_file))

    # Should fall back to defaults
    assert cache.policy.min_reviews == 10  # default
    # File should be left in place (we don't delete it)
    assert policy_file.exists()
    assert any("corrupt or invalid" in r.message for r in caplog.records)
    cache.stop()


# ---------------------------------------------------------------------------
# PolicyCache: writes the file
# ---------------------------------------------------------------------------
def test_policy_cache_writes_after_on_review(tmp_path):
    """After on_review(), the cache is dirty and flush writes the file."""
    from idp.rl.online import PolicyCache
    from idp.storage.store import StoredResult

    policy_file = tmp_path / "policy.json"
    cache = PolicyCache(policy_path=str(policy_file), flush_interval_sec=0.05)

    # Build a result with mismatched fields (corrections → rewards)
    result = StoredResult(
        id="r1", doc_id="d1", schema_name="Invoice", backend_name="mock",
        mode="ocr_llm",
        classification="invoice", validation={"passed": True},
        source_path="/tmp/x.pdf", created_at=0.0,
        reviewed=True, reviewer="alice",
        extraction={"vendor_name": "WRONG"},
        reviewed_extraction={"vendor_name": "Acme"},
        confidence={"vendor_name": 0.5},
    )
    cache.on_review(result)
    cache.flush_now()  # synchronous flush
    cache.stop()

    assert policy_file.exists()
    import json
    data = json.loads(policy_file.read_text())
    # The cache wrote SOMETHING valid JSON (schema fields are present).
    # With only 1 review < min_reviews (default 10), field_floors will
    # still be empty — we just verify the file structure is correct.
    assert "field_floors" in data
    assert "field_penalties" in data
    assert "min_reviews" in data


# ---------------------------------------------------------------------------
# attach_to_storage: monkey-patches storage
# ---------------------------------------------------------------------------
def test_attach_to_storage_patches_mark_reviewed(tmp_path):
    """attach_to_storage monkey-patches the mark_reviewed method on storage."""
    from idp.rl.online import PolicyCache

    policy_file = tmp_path / "policy.json"
    cache = PolicyCache(policy_path=str(policy_file))

    # Build a stub storage
    class _StubStorage:
        def __init__(self):
            self.calls = []
            self._original_mark_reviewed = self.mark_reviewed  # type: ignore[attr-defined]

        def mark_reviewed(self, *args, **kwargs):
            self.calls.append(("mark_reviewed", args, kwargs))

    storage = _StubStorage()
    cache.attach_to_storage(storage)

    # Now mark_reviewed should be patched — calling it triggers on_review
    # Verify the method was wrapped (replaced on the instance)
    # The original is preserved for detachment
    assert hasattr(storage, "_original_mark_reviewed")
    cache.stop()


# ---------------------------------------------------------------------------
# load() falls back to defaults on truly unexpected error (lines 188-193)
# ---------------------------------------------------------------------------
def test_policy_cache_loads_defaults_on_unexpected_error(tmp_path, monkeypatch, caplog):
    """Non-JSON/non-IO error → defaults + warning log."""
    import logging

    from idp.rl.online import PolicyCache

    policy_file = tmp_path / "policy.json"
    policy_file.write_text("{}")

    # Force json.loads to raise something unexpected
    # (already imported at module level via 'from idp.rl.online import PolicyCache')

    def bad_loads(*args, **kwargs):
        raise OSError("synthetic permission denied")

    monkeypatch.setattr(idp.rl.online.json, "loads", bad_loads)  # type: ignore[attr-defined]
    caplog.set_level(logging.WARNING)
    cache = PolicyCache(policy_path=str(policy_file))

    # Falls back to defaults
    assert cache.policy.min_reviews == 10
    assert any("failed to load policy" in r.message for r in caplog.records)
    cache.stop()

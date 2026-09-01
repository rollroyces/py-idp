"""Tests for InMemoryStorage.list() filter kwargs.

The `reviewed_only`, `reviewed_since`, `schema_name` filters were added
in the SqlStorage refactor. The InMemoryStorage implementation also
supports them but was never tested directly.
"""
from __future__ import annotations

import time

from idp.storage.store import InMemoryStorage, StoredResult


def _make_result(rid: str = "r1", *, reviewed: bool = False,
                schema_name: str = "Invoice", doc_id: str = "doc-1") -> StoredResult:
    r = StoredResult(
        id=rid, doc_id=doc_id, schema_name=schema_name, backend_name="mock",
        mode="ocr_llm", classification="invoice",
        extraction={"vendor_name": "Acme"},
        confidence=None, validation=None, source_path="/x", created_at=0.0,
        reviewed=reviewed,
        reviewed_extraction={"vendor_name": "Acme Widgets"} if reviewed else None,
        reviewer="alice" if reviewed else None,
    )
    if reviewed:
        r.last_reviewed_at = time.time()
    return r


# ---------------------------------------------------------------------------
# reviewed_only
# ---------------------------------------------------------------------------
def test_inmemory_reviewed_only_returns_only_reviewed():
    s = InMemoryStorage()
    s.put(_make_result("r1", reviewed=False))
    s.put(_make_result("r2", reviewed=True))
    s.put(_make_result("r3", reviewed=True))
    reviewed = s.list(reviewed_only=True)
    assert len(reviewed) == 2
    assert {r.id for r in reviewed} == {"r2", "r3"}


def test_inmemory_reviewed_only_false_returns_all():
    s = InMemoryStorage()
    s.put(_make_result("r1", reviewed=False))
    s.put(_make_result("r2", reviewed=True))
    all_items = s.list(reviewed_only=False)
    assert len(all_items) == 2


# ---------------------------------------------------------------------------
# schema_name
# ---------------------------------------------------------------------------
def test_inmemory_schema_name_filter():
    s = InMemoryStorage()
    s.put(_make_result("r1", schema_name="Invoice"))
    s.put(_make_result("r2", schema_name="Contract"))
    s.put(_make_result("r3", schema_name="Contract"))
    contracts = s.list(schema_name="Contract")
    assert {r.id for r in contracts} == {"r2", "r3"}


def test_inmemory_schema_name_filter_no_match_returns_empty():
    s = InMemoryStorage()
    s.put(_make_result("r1", schema_name="Invoice"))
    assert s.list(schema_name="Receipt") == []


# ---------------------------------------------------------------------------
# reviewed_since
# ---------------------------------------------------------------------------
def test_inmemory_reviewed_since_filters_old_reviews():
    s = InMemoryStorage()
    now = time.time()
    # r1 reviewed long ago
    r_old = _make_result("r1", reviewed=True)
    r_old.last_reviewed_at = now - 3600  # 1 hour ago
    s.put(r_old)
    # r2 reviewed just now
    r_new = _make_result("r2", reviewed=True)
    r_new.last_reviewed_at = now
    s.put(r_new)
    # r3 never reviewed
    s.put(_make_result("r3", reviewed=False))

    since_60s_ago = now - 60
    recent = s.list(reviewed_since=since_60s_ago)
    # r1 (1h old) and r3 (never reviewed) filtered out; only r2 stays
    assert {r.id for r in recent} == {"r2"}


def test_inmemory_reviewed_since_epoch_zero_returns_all_reviewed():
    """since=0 returns everything that's been reviewed (epoch 0 is far past)."""
    s = InMemoryStorage()
    s.put(_make_result("r1", reviewed=True))
    s.put(_make_result("r2", reviewed=False))
    items = s.list(reviewed_since=0)
    assert {r.id for r in items} == {"r1"}


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------
def test_inmemory_combined_schema_name_and_reviewed_only():
    s = InMemoryStorage()
    s.put(_make_result("r1", schema_name="Invoice", reviewed=True))
    s.put(_make_result("r2", schema_name="Invoice", reviewed=False))
    s.put(_make_result("r3", schema_name="Contract", reviewed=True))
    items = s.list(schema_name="Invoice", reviewed_only=True)
    assert {r.id for r in items} == {"r1"}


def test_inmemory_list_returns_empty_when_no_storage():
    s = InMemoryStorage()
    assert s.list() == []


def test_inmemory_list_limit():
    s = InMemoryStorage()
    for i in range(10):
        s.put(_make_result(f"r{i}"))
    assert len(s.list(limit=5)) == 5


def test_inmemory_list_positional_doc_id_and_limit():
    """doc_id + limit as positional args still work (backward compat)."""
    s = InMemoryStorage()
    s.put(_make_result("r1", doc_id="doc-1"))
    s.put(_make_result("r2", doc_id="doc-2"))
    s.put(_make_result("r3", doc_id="doc-1"))
    out = s.list("doc-1", 10)
    assert {r.id for r in out} == {"r1", "r3"}

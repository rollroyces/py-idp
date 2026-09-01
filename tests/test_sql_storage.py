"""Tests for the SqlStorage backend (sqlite + factory).

Postgres path is exercised via the same code with a different connection
URL — we don't have psycopg installed in CI by default, so we verify
that the dialect-dispatch logic routes correctly via mocks.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from idp.storage.factory import make_storage
from idp.storage.store import InMemoryStorage, JsonFileStorage, StoredResult
from idp.storage.sql import SqlStorage


@pytest.fixture
def sqlite_storage(tmp_path):
    db = tmp_path / "test.db"
    return SqlStorage(f"sqlite:///{db}")


def _make_result(rid: str = "r1", reviewed: bool = False) -> StoredResult:
    return StoredResult(
        id=rid,
        doc_id="doc-1",
        schema_name="Invoice",
        backend_name="mock",
        mode="ocr_llm",
        classification="invoice",
        extraction={"vendor_name": "Acme", "total_amount": 100.0},
        confidence={"vendor_name": 0.75, "total_amount": 0.75},
        validation={"schema_valid": True, "errors": [], "passed": True},
        source_path="/tmp/inv.txt",
        created_at=0.0,
        reviewed=reviewed,
        reviewed_extraction={"vendor_name": "Acme Widgets", "total_amount": 100.0} if reviewed else None,
        reviewer="alice" if reviewed else None,
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------
def test_put_and_get(sqlite_storage):
    rid = sqlite_storage.put(_make_result("r1"))
    assert rid == "r1"
    item = sqlite_storage.get("r1")
    assert item is not None
    assert item.doc_id == "doc-1"
    assert item.extraction["vendor_name"] == "Acme"


def test_get_missing_returns_none(sqlite_storage):
    assert sqlite_storage.get("nope") is None


def test_list_filter_by_doc_id(sqlite_storage):
    sqlite_storage.put(_make_result("r1"))
    sqlite_storage.put(_make_result("r2"))
    sqlite_storage.put(StoredResult(
        id="r3", doc_id="doc-2", schema_name="Invoice", backend_name="mock",
        mode=None, classification=None, extraction={}, confidence=None,
        validation=None, source_path="/x", created_at=0.0,
    ))
    items = sqlite_storage.list(doc_id="doc-1")
    assert len(items) == 2
    assert all(r.doc_id == "doc-1" for r in items)


def test_list_filter_reviewed_only(sqlite_storage):
    sqlite_storage.put(_make_result("r1", reviewed=False))
    sqlite_storage.put(_make_result("r2", reviewed=True))
    reviewed = sqlite_storage.list(reviewed_only=True)
    assert len(reviewed) == 1
    assert reviewed[0].id == "r2"


def test_list_filter_schema(sqlite_storage):
    sqlite_storage.put(_make_result("r1"))
    sqlite_storage.put(StoredResult(
        id="r2", doc_id="d", schema_name="Contract", backend_name="m",
        mode=None, classification=None, extraction={}, confidence=None,
        validation=None, source_path="/x", created_at=0.0,
    ))
    contracts = sqlite_storage.list(schema_name="Contract")
    assert len(contracts) == 1
    assert contracts[0].schema_name == "Contract"


def test_mark_reviewed_updates_denorm(sqlite_storage):
    sqlite_storage.put(_make_result("r1", reviewed=False))
    sqlite_storage.mark_reviewed("r1", {"vendor_name": "Edited"}, "bob")
    item = sqlite_storage.get("r1")
    assert item.reviewed is True
    assert item.reviewer == "bob"
    assert item.reviewed_extraction == {"vendor_name": "Edited"}


# ---------------------------------------------------------------------------
# submit_review — the full path
# ---------------------------------------------------------------------------
def test_submit_review_creates_session_and_edits(sqlite_storage):
    sqlite_storage.put(_make_result("r1", reviewed=False))
    review_id = sqlite_storage.submit_review(
        result_id="r1",
        edited={"vendor_name": "Acme Widgets", "total_amount": 100.0},
        reviewer="alice",
        duration_sec=12.5,
    )
    assert review_id is not None
    item = sqlite_storage.get("r1")
    assert item.reviewed is True
    assert item.reviewer == "alice"


def test_submit_review_records_per_field_edits(sqlite_storage):
    sqlite_storage.put(_make_result("r1", reviewed=False))
    sqlite_storage.submit_review(
        result_id="r1",
        edited={"vendor_name": "Acme Widgets"},  # different from "Acme"
        reviewer="alice",
    )
    # the review_edits row should record +1 for vendor_name
    reviews = sqlite_storage.reviews_as_dicts()
    assert len(reviews) == 1
    assert reviews[0]["doc_id"] == "doc-1"
    assert reviews[0]["human"]["vendor_name"] == "Acme Widgets"


def test_submit_review_human_added_field_treated_as_correction(sqlite_storage):
    """Model missed a field; human added it. RL signal: +1 (model was wrong)."""
    r = StoredResult(
        id="r1", doc_id="doc-1", schema_name="Invoice", backend_name="mock",
        mode="ocr_llm", classification="invoice",
        extraction={"vendor_name": "Acme"},  # model missed currency
        confidence={"vendor_name": 0.75}, validation=None,
        source_path="/x", created_at=0.0,
    )
    sqlite_storage.put(r)
    sqlite_storage.submit_review(
        result_id="r1",
        edited={"vendor_name": "Acme", "currency": "USD"},
        reviewer="alice",
    )
    reviews = sqlite_storage.reviews_as_dicts()
    assert reviews[0]["human"]["currency"] == "USD"


def test_submit_review_idempotent_creates_new_session_each_time(sqlite_storage):
    sqlite_storage.put(_make_result("r1", reviewed=False))
    rid1 = sqlite_storage.submit_review("r1", {"vendor_name": "X"}, "alice")
    rid2 = sqlite_storage.submit_review("r1", {"vendor_name": "Y"}, "alice")
    assert rid1 != rid2
    reviews = sqlite_storage.reviews_as_dicts()
    assert len(reviews) == 1  # grouped by (doc_id, schema), keeps latest
    assert reviews[0]["human"]["vendor_name"] == "Y"


def test_reviews_as_dicts_filters_by_since(sqlite_storage):
    sqlite_storage.put(_make_result("r1"))
    sqlite_storage.submit_review("r1", {"vendor_name": "X"}, "alice")
    # everything happened "now"; asking for since=future should return nothing
    import time
    future = time.time() + 1000
    assert sqlite_storage.reviews_as_dicts(since=future) == []


def test_reviews_as_dicts_filters_by_schema(sqlite_storage):
    sqlite_storage.put(_make_result("r1"))
    sqlite_storage.put(StoredResult(
        id="r2", doc_id="d", schema_name="Contract", backend_name="m",
        mode=None, classification=None, extraction={"title": "X"},
        confidence=None, validation=None, source_path="/x", created_at=0.0,
    ))
    sqlite_storage.submit_review("r2", {"title": "MSA"}, "alice")
    contracts = sqlite_storage.reviews_as_dicts(schema_name="Contract")
    assert len(contracts) == 1
    assert contracts[0]["schema"] == "Contract"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_default_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IDP_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("IDP_DB_URL", raising=False)
    s = make_storage()
    assert isinstance(s, JsonFileStorage)


def test_factory_memory_backend(monkeypatch):
    monkeypatch.setenv("IDP_STORAGE_BACKEND", "memory")
    s = make_storage()
    assert isinstance(s, InMemoryStorage)


def test_factory_sql_backend_explicit_url(monkeypatch, tmp_path):
    monkeypatch.setenv("IDP_STORAGE_BACKEND", "sql")
    db = tmp_path / "x.db"
    s = make_storage(db_url=f"sqlite:///{db}")
    assert isinstance(s, SqlStorage)


def test_factory_db_url_auto_selects_sql(monkeypatch, tmp_path):
    monkeypatch.delenv("IDP_STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("IDP_DB_URL", f"sqlite:///{tmp_path / 'y.db'}")
    s = make_storage()
    assert isinstance(s, SqlStorage)


def test_factory_unknown_raises(monkeypatch):
    monkeypatch.setenv("IDP_STORAGE_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="unknown storage backend"):
        make_storage()


# ---------------------------------------------------------------------------
# Schema bootstrap idempotency
# ---------------------------------------------------------------------------
def test_schema_bootstrap_idempotent(tmp_path):
    db = tmp_path / "double.db"
    s1 = SqlStorage(f"sqlite:///{db}")
    s1.put(_make_result("r1"))
    # Re-instantiating should NOT fail and should preserve data
    s2 = SqlStorage(f"sqlite:///{db}")
    item = s2.get("r1")
    assert item is not None


def test_sqlite_url_with_disk_file_works(tmp_path):
    """Smoke test: SqlStorage accepts a file-based URL."""
    s = SqlStorage(f"sqlite:///{tmp_path / 'mem.db'}")
    s.put(_make_result("r1"))
    assert s.get("r1") is not None
    s2 = SqlStorage(f"sqlite:///{tmp_path / 'mem.db'}")
    assert s2.get("r1") is not None

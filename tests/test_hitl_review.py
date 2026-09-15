"""Tests for idp.hitl.review (the pure data-handling module).

These tests exercise the HITL data logic without spinning up
Streamlit. A3 in docs/ROADMAP.md: extract the data-handling logic
out of idp.hitl.app into a pure module, then add tests.
"""
from __future__ import annotations

import pytest

from idp.hitl.review import (
    coerce_field_value,
    count_corrections,
    format_result_label,
    save_review,
    sort_pending_newest_first,
)
from idp.storage.store import InMemoryStorage, JsonFileStorage, StoredResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_result(
    *,
    result_id: str = "r1",
    doc_id: str = "d1",
    schema: str = "Invoice",
    extraction: dict | None | object = None,
    reviewed_extraction: dict | None = None,
    created_at: float = 1000.0,
) -> StoredResult:
    # The default extraction is a non-empty dict so the test for
    # "all fields corrected" has something to compare. Use a sentinel
    # so callers can distinguish "didn't pass extraction" (use default)
    # from "passed extraction=None" (store None).
    _SENTINEL = object()
    if extraction is _SENTINEL:
        # Treat as "not passed" — but this is unreachable because the
        # default value is None, not the sentinel. Kept for clarity.
        actual_extraction = {"vendor_name": "Acme", "total_amount": 100.0}
    elif extraction is None:
        actual_extraction = None
    else:
        actual_extraction = extraction
    return StoredResult(
        id=result_id,
        doc_id=doc_id,
        schema_name=schema,
        backend_name="mock",
        mode="ocr_llm",
        classification=None,
        extraction=actual_extraction,
        confidence=None,
        validation={"passed": True},
        source_path=f"/tmp/{doc_id}.pdf",
        created_at=created_at,
        reviewed=reviewed_extraction is not None,
        reviewed_extraction=reviewed_extraction,
        reviewer="alice" if reviewed_extraction is not None else None,
    )


# ---------------------------------------------------------------------------
# save_review: dispatch logic
# ---------------------------------------------------------------------------
def test_save_review_dispatches_to_submit_review_when_available(tmp_path) -> None:
    """SqlStorage has submit_review; review.save_review calls it."""
    pytest.importorskip("idp.storage.sql")
    from idp.storage.sql import SqlStorage

    storage = SqlStorage(f"sqlite:///{tmp_path / 'test.db'}")
    storage.put(_make_result(extraction={"vendor_name": "X"}))

    save_review(
        storage,
        result_id="r1",
        edited={"vendor_name": "Acme"},
        reviewer="alice",
    )

    listed = storage.list()
    assert len(listed) == 1
    assert listed[0].reviewed_extraction == {"vendor_name": "Acme"}
    assert listed[0].reviewer == "alice"


def test_save_review_falls_back_to_mark_reviewed(tmp_path) -> None:
    """InMemoryStorage has no submit_review; falls back to mark_reviewed."""
    storage = InMemoryStorage()
    storage.put(_make_result(extraction={"vendor_name": "X"}))

    save_review(
        storage,
        result_id="r1",
        edited={"vendor_name": "Acme"},
        reviewer="alice",
    )

    listed = storage.list()
    assert listed[0].reviewed_extraction == {"vendor_name": "Acme"}


def test_save_review_falls_back_for_json_storage(tmp_path) -> None:
    """JsonFileStorage has no submit_review; falls back to mark_reviewed."""
    storage = JsonFileStorage(str(tmp_path / "results.jsonl"))
    storage.put(_make_result(extraction={"vendor_name": "X"}))

    save_review(
        storage,
        result_id="r1",
        edited={"vendor_name": "Acme"},
        reviewer="alice",
    )

    listed = storage.list()
    assert listed[0].reviewed_extraction == {"vendor_name": "Acme"}


# ---------------------------------------------------------------------------
# format_result_label
# ---------------------------------------------------------------------------
def test_format_result_label_contains_id_schema_and_path() -> None:
    r = _make_result(result_id="r-abc123", schema="Invoice", doc_id="inv-001")
    label = format_result_label(r)
    assert "r-abc123" in label
    assert "Invoice" in label
    assert "/tmp/inv-001.pdf" in label


def test_format_result_label_round_trip_recovers_id() -> None:
    """The label format must keep the id recoverable for the selectbox."""
    r = _make_result(result_id="r-xyz")
    label = format_result_label(r)
    assert r.id in label
    # prefix is id (first segment, before the em-dash separator)
    assert label.split("  —  ")[0] == r.id


# ---------------------------------------------------------------------------
# sort_pending_newest_first
# ---------------------------------------------------------------------------
def test_sort_pending_newest_first_orders_by_created_at_desc() -> None:
    r1 = _make_result(result_id="r1", created_at=100.0)
    r2 = _make_result(result_id="r2", created_at=300.0)
    r3 = _make_result(result_id="r3", created_at=200.0)
    sorted_list = sort_pending_newest_first([r1, r2, r3])
    assert [r.id for r in sorted_list] == ["r2", "r3", "r1"]


def test_sort_pending_newest_first_empty() -> None:
    assert sort_pending_newest_first([]) == []


def test_sort_pending_newest_first_does_not_mutate_input() -> None:
    original = [
        _make_result(result_id="r1", created_at=100.0),
        _make_result(result_id="r2", created_at=300.0),
    ]
    original_ids = [r.id for r in original]
    sort_pending_newest_first(original)
    # Caller's list should be unchanged
    assert [r.id for r in original] == original_ids


# ---------------------------------------------------------------------------
# count_corrections
# ---------------------------------------------------------------------------
def test_count_corrections_zero_when_no_review() -> None:
    r = _make_result(reviewed_extraction=None)
    assert count_corrections(r) == 0


def test_count_corrections_zero_when_all_match() -> None:
    ext = {"vendor_name": "Acme", "total_amount": 100.0}
    r = _make_result(extraction=ext, reviewed_extraction=dict(ext))
    assert count_corrections(r) == 0


def test_count_corrections_counts_differing_fields() -> None:
    r = _make_result(
        extraction={"vendor_name": "X", "total_amount": 100.0, "date": "2024-01-01"},
        reviewed_extraction={"vendor_name": "Acme", "total_amount": 100.0, "date": "2024-01-01"},
    )
    assert count_corrections(r) == 1  # only vendor_name changed


def test_count_corrections_all_differ() -> None:
    r = _make_result(
        extraction={"a": 1, "b": 2, "c": 3},
        reviewed_extraction={"a": 10, "b": 20, "c": 30},
    )
    assert count_corrections(r) == 3


def test_count_corrections_handles_missing_extraction() -> None:
    """Defensive: if extraction is None, count returns 0."""
    r = _make_result(extraction=None, reviewed_extraction={"x": 1})
    assert count_corrections(r) == 0


# ---------------------------------------------------------------------------
# coerce_field_value
# ---------------------------------------------------------------------------
def test_coerce_field_value_scalar_passes_through() -> None:
    """For non-collection values, return raw without parsing."""
    val, warning = coerce_field_value("hello", "original")
    assert val == "hello"
    assert warning is None


def test_coerce_field_value_dict_parses_json() -> None:
    val, warning = coerce_field_value('{"a": 1, "b": 2}', {"a": 0, "b": 0})
    assert val == {"a": 1, "b": 2}
    assert warning is None


def test_coerce_field_value_list_parses_json() -> None:
    val, warning = coerce_field_value("[1, 2, 3]", [0, 0, 0])
    assert val == [1, 2, 3]
    assert warning is None


def test_coerce_field_value_invalid_json_falls_back_to_raw() -> None:
    """Bad JSON for a dict/list field -> save as raw string + warning."""
    val, warning = coerce_field_value("{not valid json", {"x": 0})
    assert val == "{not valid json"
    assert warning is not None
    # Case-insensitive check: the warning should mention JSON (or json)
    assert "json" in warning.lower()


def test_coerce_field_value_none_original_returns_raw() -> None:
    """If the original is None, the field is treated as scalar."""
    val, warning = coerce_field_value("anything", None)
    assert val == "anything"
    assert warning is None
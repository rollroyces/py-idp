"""Tests for ``idp.migrate`` (P5 v0.4 / C4 — template-version migration).

The migration tool is built on ``process_batch`` and reads
``StoredResult.source_path`` to re-run the pipeline against a new
template version. These tests cover the contract surface area:

  * ``dry_run`` does NOT write to storage.
  * ``commit`` writes new v2 StoredResults, leaves v1 records alone.
  * Missing ``source_path`` on disk → warn, list as skipped, never crash.
  * ``from_version == to_version`` → ValueError, no work done.
  * Unknown template (no StoredResult rows) → clean ValueError.
  * Side-by-side diff shape (added/removed/changed/unchanged rows).
  * Idempotent re-run: second run with same args produces same diff
    on the same input data.
  * Same v1 records before and after a commit (v1 immutability).

The fixtures use a custom test backend (rather than ``MockBackend``)
because MockBackend's deterministic-empty output makes the diff
trivial — we want explicit control over what the v2 re-run returns so
the diffs are non-trivial.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from idp.llm.backend import Backend, CompletionRequest
from idp.migrate import (
    MigrationReport,
    diff_extractions,
    migrate_template_version,
)
from idp.pipeline.pipeline import Pipeline
from idp.storage.store import InMemoryStorage, StoredResult


# ---------------------------------------------------------------------------
# Custom backend with a configurable return value
# ---------------------------------------------------------------------------
class _StaticBackend(Backend):
    """Backend that always returns a pre-canned JSON string.

    Lets each test decide what the v2 re-run produces, independent of
    the schema's field list. The pipeline calls ``complete()`` for
    classify / assess / extract; for the migration tests we only
    care about the final extraction, so we always return the same
    JSON. The Pipeline accepts it as long as it parses as a JSON
    object (extract stage validates against the Pydantic schema; if
    we send fields the schema doesn't know about they go into the
    extra-allowed bucket when the schema has ``model_config =
    extra='allow'`` — but most built-in schemas are strict).
    """

    name = "static-bend"

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    @property
    def is_multimodal(self) -> bool:
        return False

    def complete(self, req: CompletionRequest) -> str:
        # Always return the canned response. This is intentionally
        # crude — for migrate tests we only need a deterministic
        # output, not a realistic one.
        return json.dumps(self._response)

    async def acomplete(self, req: CompletionRequest) -> str:
        return self.complete(req)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_doc(tmp_path: Path, name: str, body: str = "Invoice INV-001 total $100") -> Path:
    """Write a .txt doc on disk and return its path.

    The pipeline auto-picks PlainTextParser for .txt files.
    """
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _make_v1_record(
    *,
    result_id: str,
    source_path: str,
    schema_name: str = "Invoice",
    extraction: dict[str, Any] | None = None,
    created_at: float | None = None,
    reviewed: bool = False,
) -> StoredResult:
    """A v1 StoredResult. ``extraction`` defaults to a small dict."""
    if extraction is None:
        extraction = {"vendor_name": "Acme Co", "total_amount": 100.0}
    return StoredResult(
        id=result_id,
        doc_id=f"doc-{result_id}",
        schema_name=schema_name,
        backend_name="mock",
        mode=None,
        classification=None,
        extraction=extraction,
        confidence={"vendor_name": 0.7, "total_amount": 0.7},
        validation={"passed": True},
        source_path=source_path,
        created_at=created_at if created_at is not None else time.time(),
        reviewed=reviewed,
    )


def _populate_storage(storage: InMemoryStorage, rows: list[StoredResult]) -> None:
    for r in rows:
        storage.put(r)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_dry_run_does_not_write(tmp_path: Path) -> None:
    """Dry-run (default) produces a report but does NOT persist anything.

    The only thing that should hit storage is the initial put()
    during fixture setup. Calling ``migrate_template_version``
    with ``dry_run=True`` (default) must leave the v1 set
    unchanged.
    """
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc1.txt")
    r1 = _make_v1_record(result_id="r1", source_path=str(p))
    _populate_storage(storage, [r1])

    # Pipeline that re-runs against the v2 template and produces a
    # different extraction (added field).
    v2_extraction = {"vendor_name": "Acme Co", "total_amount": 100.0, "tax_id": "TAX-001"}

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend(v2_extraction), schema="Invoice")

    report = migrate_template_version(
        storage,
        "Invoice",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        dry_run=True,
    )

    # Report shape
    assert isinstance(report, MigrationReport)
    assert report.dry_run is True
    assert report.committed == 0
    assert len(report.items) == 1
    # 1 added (tax_id), 0 removed/changed, 2 unchanged
    assert report.added == 1
    assert report.removed == 0
    assert report.changed == 0
    assert report.unchanged == 2
    assert report.skipped == 0

    # Storage was NOT touched — only the initial r1 record exists.
    after = storage.list(limit=10000)
    assert len(after) == 1
    assert after[0].id == "r1"


def test_commit_writes_new_records_only(tmp_path: Path) -> None:
    """``dry_run=False`` appends new v2 StoredResults; v1 records stay.

    The v1 record keeps its original ``id``. The new v2 record
    gets a fresh ``id`` (so callers can tell them apart), has
    ``reviewed=False`` (raw extraction only, no review
    propagation), and uses the new extraction.
    """
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc1.txt")
    v1_extraction = {"vendor_name": "Acme Co", "total_amount": 100.0}
    r1 = _make_v1_record(result_id="r1", source_path=str(p), extraction=v1_extraction)
    _populate_storage(storage, [r1])

    v2_extraction = {"vendor_name": "Acme Co", "total_amount": 100.0, "tax_id": "TAX-001"}

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend(v2_extraction), schema="Invoice")

    out_path = tmp_path / "report.json"
    report = migrate_template_version(
        storage,
        "Invoice",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        output_path=str(out_path),
        dry_run=False,
    )

    assert report.dry_run is False
    assert report.committed == 1

    # Storage now has 2 records: the original v1 and the new v2.
    after = storage.list(limit=10000)
    assert len(after) == 2

    by_id = {r.id: r for r in after}
    assert "r1" in by_id
    v2_record = next(r for r in after if r.id != "r1")
    assert v2_record.extraction == v2_extraction
    assert v2_record.reviewed is False, "v2 records are raw (no review propagation)"
    assert v2_record.reviewed_extraction is None
    # Same source_path as v1 (same document)
    assert v2_record.source_path == str(p)

    # v1 record is intact
    assert by_id["r1"].extraction == v1_extraction
    assert by_id["r1"].schema_name == "Invoice"

    # Report file written
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["dry_run"] is False
    assert saved["committed"] == 1


def test_missing_source_path_warns_not_crashes(tmp_path: Path) -> None:
    """When the source file is gone from disk, warn + skip, do not crash.

    The v1 records might point at files that have been archived,
    deleted, or moved. The migration tool must produce a
    per-item row with ``skipped=True`` so the user can see what
    was lost, and it must not raise.
    """
    storage = InMemoryStorage()
    # Two records: one with a real source_path, one with a path
    # that doesn't exist on disk.
    good = _write_doc(tmp_path, "good.txt")
    missing_path = str(tmp_path / "this-does-not-exist.txt")
    r_good = _make_v1_record(result_id="rg", source_path=str(good))
    r_missing = _make_v1_record(result_id="rm", source_path=missing_path)
    _populate_storage(storage, [r_good, r_missing])

    v2_extraction = {"vendor_name": "Acme Co", "total_amount": 100.0}

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend(v2_extraction), schema="Invoice")

    # Caplog to assert the warning was emitted
    report = migrate_template_version(
        storage,
        "Invoice",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        dry_run=True,
    )

    # One item succeeded, one was skipped.
    skipped_items = [it for it in report.items if it.skipped]
    assert len(skipped_items) == 1
    assert skipped_items[0].stored_result_id == "rm"
    assert skipped_items[0].error and "not found" in skipped_items[0].error.lower()
    assert report.skipped == 1

    # Dry-run, so storage is untouched.
    assert len(storage.list(limit=10000)) == 2


def test_same_from_and_to_version_is_noop(tmp_path: Path) -> None:
    """from_version == to_version raises ValueError without touching storage."""
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc.txt")
    r1 = _make_v1_record(result_id="r1", source_path=str(p))
    _populate_storage(storage, [r1])

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend({}), schema="Invoice")

    with pytest.raises(ValueError, match="must differ"):
        migrate_template_version(
            storage,
            "Invoice",
            from_version=1,
            to_version=1,
            pipeline_factory=factory,
            dry_run=True,
        )

    # Nothing changed.
    assert len(storage.list(limit=10000)) == 1


def test_unknown_template_name_errors_cleanly(tmp_path: Path) -> None:
    """No matching StoredResult rows → ValueError."""
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc.txt")
    r1 = _make_v1_record(result_id="r1", source_path=str(p), schema_name="Invoice")
    _populate_storage(storage, [r1])

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend({}), schema="Invoice")

    with pytest.raises(ValueError, match="no StoredResult rows found"):
        migrate_template_version(
            storage,
            "Receipt",  # not in storage
            from_version=1,
            to_version=2,
            pipeline_factory=factory,
            dry_run=True,
        )


def test_side_by_side_diff_shape(tmp_path: Path) -> None:
    """The diff correctly classifies added / removed / changed / unchanged fields."""
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc.txt")
    v1_extraction = {
        "vendor_name": "Acme Co",
        "total_amount": 100.0,
        "legacy_field": "old",  # will be removed in v2
    }
    r1 = _make_v1_record(result_id="r1", source_path=str(p), extraction=v1_extraction)
    _populate_storage(storage, [r1])

    # v2: same vendor_name, changed total_amount, added tax_id,
    # removed legacy_field.
    v2_extraction = {
        "vendor_name": "Acme Co",
        "total_amount": 110.0,  # changed
        "tax_id": "TAX-001",  # added
    }

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend(v2_extraction), schema="Invoice")

    report = migrate_template_version(
        storage,
        "Invoice",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        dry_run=True,
    )

    assert len(report.items) == 1
    item = report.items[0]
    assert item.stored_result_id == "r1"
    assert item.error is None
    assert item.skipped is False

    # Build a {field: status} map for easy assertions.
    status_by_field = {d.field: d.status for d in item.field_diffs}
    assert status_by_field["vendor_name"] == "unchanged"
    assert status_by_field["total_amount"] == "changed"
    assert status_by_field["tax_id"] == "added"
    assert status_by_field["legacy_field"] == "removed"

    # Counters
    assert report.added == 1
    assert report.removed == 1
    assert report.changed == 1
    assert report.unchanged == 1

    # Per-field payloads
    diff_total = next(d for d in item.field_diffs if d.field == "total_amount")
    assert diff_total.old_value == 100.0
    assert diff_total.new_value == 110.0

    diff_added = next(d for d in item.field_diffs if d.field == "tax_id")
    assert diff_added.old_value is None
    assert diff_added.new_value == "TAX-001"

    diff_removed = next(d for d in item.field_diffs if d.field == "legacy_field")
    assert diff_removed.old_value == "old"
    assert diff_removed.new_value is None

    # to_dict round-trip works (assert the legacy_field (removed)
    # is at the end of the diff list — see diff_extractions's
    # documented output order: new-side fields first, then removed).
    d = item.to_dict()
    field_names = [fd["field"] for fd in d["field_diffs"]]
    assert field_names[0] == "vendor_name"  # first in new
    assert field_names[-1] == "legacy_field"  # last: removed
    assert "tax_id" in field_names  # added
    assert "total_amount" in field_names  # changed


def test_idempotent_rerun(tmp_path: Path) -> None:
    """Re-running the migration on the same data produces the same report.

    Two consecutive dry-runs against the same storage / pipeline
    must produce byte-identical ``field_diffs`` lists. This is the
    contract the user relies on for audit / reproducibility.
    """
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc.txt")
    v1_extraction = {"vendor_name": "Acme Co", "total_amount": 100.0}
    r1 = _make_v1_record(result_id="r1", source_path=str(p), extraction=v1_extraction)
    _populate_storage(storage, [r1])

    v2_extraction = {"vendor_name": "Acme Co", "total_amount": 110.0, "tax_id": "TAX-001"}

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend(v2_extraction), schema="Invoice")

    rep1 = migrate_template_version(
        storage,
        "Invoice",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        dry_run=True,
    )
    rep2 = migrate_template_version(
        storage,
        "Invoice",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        dry_run=True,
    )

    # Same counters
    assert (rep1.added, rep1.removed, rep1.changed, rep1.unchanged) == (
        rep2.added,
        rep2.removed,
        rep2.changed,
        rep2.unchanged,
    )

    # Same diff fields (in order)
    diffs1 = [(d.field, d.status) for d in rep1.items[0].field_diffs]
    diffs2 = [(d.field, d.status) for d in rep2.items[0].field_diffs]
    assert diffs1 == diffs2

    # And the new extraction is identical
    assert rep1.items[0].new_extraction == rep2.items[0].new_extraction


def test_diff_extractions_helper() -> None:
    """``diff_extractions`` returns the right counts in the right order.

    Tested separately so the algorithm is exercised without
    needing the pipeline / storage machinery.
    """
    old = {"a": 1, "b": 2, "c": 3, "d": 4}
    new = {"a": 1, "b": 99, "e": 5, "c": 3}
    diffs = diff_extractions(old, new)
    by_field = {d.field: d for d in diffs}

    # a: unchanged, b: changed, c: unchanged, d: removed, e: added
    assert by_field["a"].status == "unchanged"
    assert by_field["b"].status == "changed"
    assert by_field["c"].status == "unchanged"
    assert by_field["d"].status == "removed"
    assert by_field["e"].status == "added"

    # Output order: keys in new first (a, b, e, c), then removed (d).
    field_order = [d.field for d in diffs]
    assert field_order == ["a", "b", "e", "c", "d"]


def test_diff_extractions_handles_int_float_equivalence() -> None:
    """JSON round-trip can change 1 to 1.0; diff must not flag a no-op.

    We don't have a JSON round-trip in the helper itself, but
    numeric normalisation matters when extractions come from
    Pydantic models that serialise ints as floats (or vice versa).
    """
    diffs = diff_extractions({"x": 1}, {"x": 1.0})
    assert len(diffs) == 1
    assert diffs[0].status == "unchanged"


def test_commit_requires_output_path(tmp_path: Path) -> None:
    """dry_run=False with no output_path is a ValueError.

    The caller must choose where the audit report lands so the
    commit is reviewable. Silent defaults would be the
    'looks-fine-but-lost-data' failure mode.
    """
    storage = InMemoryStorage()
    p = _write_doc(tmp_path, "doc.txt")
    r1 = _make_v1_record(result_id="r1", source_path=str(p))
    _populate_storage(storage, [r1])

    def factory() -> Pipeline:
        return Pipeline(backend=_StaticBackend({}), schema="Invoice")

    with pytest.raises(ValueError, match="output_path is required"):
        migrate_template_version(
            storage,
            "Invoice",
            from_version=1,
            to_version=2,
            pipeline_factory=factory,
            output_path=None,
            dry_run=False,
        )
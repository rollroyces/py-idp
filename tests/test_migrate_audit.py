"""Tests for src/idp/migrate_audit.py.

The migration audit walks a SqlStorage DB and flags rows whose
extraction passed v0.1 schema validation but fails v0.2 schema
validation. This is the "do I need to redo any work after upgrading"
question for users coming from v0.1.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from idp.migrate_audit import audit_db
from idp.storage.sql import SqlStorage
from idp.storage.store import StoredResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _populate_db(db_url: str, rows: list[dict]) -> None:
    """Write a list of StoredResult-shaped dicts to a fresh SqlStorage."""
    storage = SqlStorage(db_url)
    for r in rows:
        result = StoredResult(
            id=r.get("id") or f"r-{r['doc_id']}",
            doc_id=r["doc_id"],
            schema_name=r["schema_name"],
            backend_name=r.get("backend_name", "mock"),
            mode=r.get("mode", "ocr_llm"),
            classification=r.get("classification"),
            extraction=r["extraction"],
            confidence=r.get("confidence"),
            validation=r.get("validation", {"passed": True}),
            source_path=r.get("source_path", f"/tmp/{r['doc_id']}.pdf"),
            created_at=r.get("created_at", time.time()),
        )
        storage.put(result)


def _storage_rows(*, doc_id: str, schema_name: str, extraction: dict) -> dict:
    return {
        "doc_id": doc_id,
        "schema_name": schema_name,
        "extraction": extraction,
        "validation": {"passed": True},
        "backend_name": "mock",
    }


# ---------------------------------------------------------------------------
# Empty DB
# ---------------------------------------------------------------------------
def test_audit_db_empty(tmp_path: Path) -> None:
    """An empty DB returns a clean report with 0 rows."""
    db_path = tmp_path / "empty.db"
    db_url = f"sqlite:///{db_path}"
    SqlStorage(db_url)  # creates schema
    report = audit_db(db_url)
    assert report["total_rows"] == 0
    assert report["rows_v01_pass_v02_fail"] == 0
    assert report["affected"] == []


# ---------------------------------------------------------------------------
# Rows that pass both v0.1 and v0.2
# ---------------------------------------------------------------------------
def test_audit_db_compatible_rows_not_flagged(tmp_path: Path) -> None:
    """Rows that pass v0.2 are NOT flagged (they pass both)."""
    db_url = f"sqlite:///{tmp_path / 'ok.db'}"
    # v0.2 Invoice requires invoice_number, vendor_name, total_amount.
    # v0.1 also accepts these (they're non-None Optional fields).
    _populate_db(db_url, [
        _storage_rows(
            doc_id="d1",
            schema_name="Invoice",
            extraction={
                "invoice_number": "INV-001",
                "vendor_name": "Acme",
                "total_amount": 100.0,
            },
        ),
        _storage_rows(
            doc_id="d2",
            schema_name="Invoice",
            extraction={
                "invoice_number": "INV-002",
                "vendor_name": "Globex",
                "total_amount": 200.0,
            },
        ),
    ])
    report = audit_db(db_url)
    assert report["total_rows"] == 2
    assert report["rows_v01_pass_v02_fail"] == 0
    assert report["affected"] == []


# ---------------------------------------------------------------------------
# Rows that pass v0.1 but fail v0.2 (the headline case)
# ---------------------------------------------------------------------------
def test_audit_db_flags_rows_missing_v02_required(tmp_path: Path) -> None:
    """A row missing invoice_number passes v0.1 (all Optional) but fails v0.2."""
    db_url = f"sqlite:///{tmp_path / 'broken.db'}"
    _populate_db(db_url, [
        # v0.1-OK, v0.2-FAILS: missing invoice_number
        _storage_rows(
            doc_id="d1",
            schema_name="Invoice",
            extraction={"vendor_name": "Acme", "total_amount": 100.0},
        ),
    ])
    report = audit_db(db_url)
    assert report["total_rows"] == 1
    assert report["rows_v01_pass_v02_fail"] == 1
    assert len(report["affected"]) == 1
    affected = report["affected"][0]
    assert affected["doc_id"] == "d1"
    assert affected["schema_name"] == "Invoice"
    assert "v02_error" in affected
    # The error mentions the missing field
    assert "invoice_number" in affected["v02_error"].lower() or "Field required" in affected["v02_error"]


def test_audit_db_mixed_compatible_and_incompatible(tmp_path: Path) -> None:
    """Some rows OK, some flagged — only the flagged ones are in affected."""
    db_url = f"sqlite:///{tmp_path / 'mixed.db'}"
    _populate_db(db_url, [
        # OK in both
        _storage_rows(
            doc_id="ok1",
            schema_name="Invoice",
            extraction={"invoice_number": "X", "vendor_name": "V", "total_amount": 1.0},
        ),
        # Fails v0.2
        _storage_rows(
            doc_id="bad1",
            schema_name="Invoice",
            extraction={"vendor_name": "V"},  # missing invoice_number + total_amount
        ),
        # OK in both
        _storage_rows(
            doc_id="ok2",
            schema_name="Contract",
            extraction={"title": "T"},  # all optional
        ),
    ])
    report = audit_db(db_url)
    assert report["total_rows"] == 3
    assert report["rows_v01_pass_v02_fail"] == 1
    assert len(report["affected"]) == 1
    assert report["affected"][0]["doc_id"] == "bad1"


# ---------------------------------------------------------------------------
# Output file writing
# ---------------------------------------------------------------------------
def test_audit_db_writes_output_file(tmp_path: Path) -> None:
    """--output writes a JSON report file."""
    db_url = f"sqlite:///{tmp_path / 'w.db'}"
    _populate_db(db_url, [
        _storage_rows(
            doc_id="d1",
            schema_name="Invoice",
            extraction={"vendor_name": "X"},
        ),
    ])
    out = tmp_path / "report.json"
    report = audit_db(db_url, output_path=str(out))
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["db_url"] == db_url
    assert loaded["total_rows"] == 1
    assert loaded["rows_v01_pass_v02_fail"] == 1
    assert "recommendation" in loaded
    # The in-memory report should be the same as the file
    assert report["total_rows"] == loaded["total_rows"]


# ---------------------------------------------------------------------------
# Schemas not in v01 are skipped
# ---------------------------------------------------------------------------
def test_audit_db_skips_unknown_schemas(tmp_path: Path) -> None:
    """A row with a schema_name not in v01 (no migration story) is skipped."""
    db_url = f"sqlite:///{tmp_path / 'skip.db'}"
    _populate_db(db_url, [
        _storage_rows(
            doc_id="d1",
            schema_name="Receipt",  # not in v01
            extraction={"merchant": "X"},
        ),
    ])
    report = audit_db(db_url)
    assert report["total_rows"] == 1
    # Receipt has no v0.1 counterpart -> not flagged
    assert report["rows_v01_pass_v02_fail"] == 0

# ---------------------------------------------------------------------------
# main() CLI entry point (lines 101-114)
# ---------------------------------------------------------------------------
def test_migrate_audit_main_writes_output(tmp_path):
    """main() parses args, runs audit, prints summary."""
    import sys

    from idp.migrate_audit import main

    # Set up an empty DB with the schema
    db = tmp_path / "idp.db"
    SqlStorage(f"sqlite:///{db}")

    # Invoke main() with sys.argv stub
    out = tmp_path / "report.json"
    saved_argv = sys.argv
    sys.argv = ["migrate_audit", "--db-url", f"sqlite:///{db}", "--output", str(out)]
    try:
        rc = main()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    # Report written
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["total_rows"] == 0
    assert report["rows_v01_pass_v02_fail"] == 0


def test_migrate_audit_main_without_output(tmp_path, capsys):
    """main() prints to stdout if --output not given."""
    import sys

    from idp.migrate_audit import main

    db = tmp_path / "idp.db"
    SqlStorage(f"sqlite:///{db}")

    saved_argv = sys.argv
    sys.argv = ["migrate_audit", "--db-url", f"sqlite:///{db}"]
    try:
        rc = main()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    captured = capsys.readouterr()
    # Without --output, prints JSON summary to stdout
    assert '"total_rows"' in captured.out
    assert '"rows_v01_pass_v02_fail"' in captured.out

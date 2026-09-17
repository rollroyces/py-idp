"""Tests for ``idp.console.pretty_print_result`` (formerly ``idp._util``).

These tests verify the human-readable rendering of a PipelineResult.
Also pins the backward-compat shim ``idp._util`` for the renamed module.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from idp.console import pretty_print_result


def _make_result(
    *,
    source_path: str = "/tmp/inv.pdf",
    schema_name: str = "Invoice",
    backend_name: str = "ollama",
    mode: str = "ocr_llm",
    classification: str = "invoice",
    classification_confidence: float = 0.95,
    validation_passed: bool = True,
    extraction: dict | None = None,
    confidence: dict[str, float] | None = None,
    errors: list[str] | None = None,
    timings: list | None = None,
):
    """Build a mock that pretty_print_result treats as a PipelineResult."""
    result = MagicMock()
    result.document.source_path = source_path
    result.schema_name = schema_name
    result.backend_name = backend_name
    result.mode = mode
    result.document.classification = classification
    result.document.classification_confidence = classification_confidence
    result.validation_passed = validation_passed
    result.document.extraction = extraction or {"vendor_name": "Acme"}
    result.confidence = confidence
    result.document.errors = errors or []
    # Timings: list of objects with .name and .seconds
    if timings is None:
        t1 = MagicMock()
        t1.name = "parse"
        t1.seconds = 0.123
        t2 = MagicMock()
        t2.name = "extract"
        t2.seconds = 0.456
        timings = [t1, t2]
    result.timings = timings
    return result


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_pretty_print_basic(capsys):
    result = _make_result()
    pretty_print_result(result)
    out = capsys.readouterr().out
    # Header
    assert "/tmp/inv.pdf" in out
    assert "schema:    Invoice" in out
    assert "backend:   ollama (ocr_llm)" in out
    assert "validate:  PASS" in out
    # Extraction JSON
    assert '"vendor_name"' in out
    assert '"Acme"' in out
    # Timings
    assert "parse=0.123s" in out
    assert "extract=0.456s" in out


def test_pretty_print_validation_fail(capsys):
    """Failed validation shows 'FAIL'."""
    result = _make_result(validation_passed=False)
    pretty_print_result(result)
    assert "validate:  FAIL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Confidence listing + [REVIEW] marker (line 33)
# ---------------------------------------------------------------------------
def test_pretty_print_confidence_marks_low_values(capsys):
    """Fields with confidence < 0.6 are tagged with [REVIEW]."""
    result = _make_result(confidence={
        "vendor_name": 0.85,    # OK
        "total_amount": 0.55,   # below threshold -> [REVIEW]
        "invoice_number": 0.40, # below threshold -> [REVIEW]
    })
    pretty_print_result(result)
    out = capsys.readouterr().out
    assert "0.85" in out
    assert "0.55 [REVIEW]" in out
    assert "0.40 [REVIEW]" in out
    # Confidence is sorted ascending
    lines = [ln for ln in out.split("\n") if "[REVIEW]" in ln or "0.85" in ln]
    # The first confidence line should be the lowest (0.40)
    assert "0.40" in lines[0]


def test_pretty_print_no_confidence_omits_section(capsys):
    """If result.confidence is None or empty, skip the section."""
    result = _make_result(confidence=None)
    pretty_print_result(result)
    out = capsys.readouterr().out
    assert "confidence (ascending)" not in out


# ---------------------------------------------------------------------------
# Errors section (line 35-37)
# ---------------------------------------------------------------------------
def test_pretty_print_lists_errors(capsys):
    """If document.errors is non-empty, print them."""
    result = _make_result(errors=["backend timeout", "schema mismatch"])
    pretty_print_result(result)
    out = capsys.readouterr().out
    assert "errors (2):" in out
    assert "- backend timeout" in out
    assert "- schema mismatch" in out


def test_pretty_print_no_errors_omits_section(capsys):
    """If document.errors is empty, skip the section."""
    result = _make_result(errors=[])
    pretty_print_result(result)
    assert "errors (" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Extraction with non-serializable values (json.dumps default=str)
# ---------------------------------------------------------------------------
def test_pretty_print_extraction_with_datetime(capsys):
    """Non-JSON-serializable values are coerced via default=str."""
    from datetime import datetime
    dt = datetime(2026, 9, 16, 12, 0, 0)
    result = _make_result(extraction={"date": dt, "vendor": "Acme"})
    pretty_print_result(result)
    out = capsys.readouterr().out
    # datetime serialized via default=str
    assert "2026-09-16" in out


def test_pretty_print_extraction_with_set(capsys):
    """Sets become strings via default=str (sorting/repr aside)."""
    result = _make_result(extraction={"tags": {"a", "b"}})
    pretty_print_result(result)
    out = capsys.readouterr().out
    # set is serializable via default=str
    assert "{'b', 'a'}" in out or "{'a', 'b'}" in out


# ---------------------------------------------------------------------------
# Backward-compat shim (idp._util)
# ---------------------------------------------------------------------------
def test_idp_util_legacy_import_still_works():
    """idp._util.pretty_print_result is a re-export of idp.console."""
    from idp._util import pretty_print_result as legacy
    from idp.console import pretty_print_result as modern

    assert legacy is modern  # same function object
    # Both should work in a from-import scenario
    assert legacy.__module__ == "idp.console"


def test_top_level_pretty_print_result_works():
    """idp.pretty_print_result (top-level) is exposed."""
    # Verify the top-level re-export matches the direct import.
    import idp as _idp
    assert pretty_print_result is _idp.pretty_print_result


# ---------------------------------------------------------------------------
# Realistic end-to-end (using a real Pipeline.run on a doc)
# ---------------------------------------------------------------------------
def test_pretty_print_end_to_end_with_real_pipeline(tmp_path, capsys):
    """pretty_print_result works on a real PipelineResult (not a mock)."""
    from idp import Document, Pipeline
    from idp.core.schemas import Invoice

    f = tmp_path / "inv.txt"
    f.write_text("Invoice INV-001 from Acme. Total: $100.00.")
    res = Pipeline(backend="mock", schema=Invoice).run(Document.from_path(str(f)))
    pretty_print_result(res)
    out = capsys.readouterr().out
    assert "Invoice" in out  # schema name in output
    assert "mock" in out     # backend name in output
    assert "extraction:" in out
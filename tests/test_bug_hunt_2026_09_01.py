"""Regression tests for the bugs found in the 2026-09-01 bug-hunt pass.

B1  validate() passed=True even when Invoice extraction was missing required
    fields, because all fields on Invoice/Contract/BankStatement were Optional.
    Fix: required fields added to the built-in schemas.

B2  Document.from_path(directory) silently returned an empty Document instead
    of raising. Fix: now raises IsADirectoryError.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from idp.core.document import Document
from idp.core.schemas import SCHEMA_REGISTRY, BankStatement, Invoice
from idp.validate.validator import validate


# ---------------------------------------------------------------------------
# B1: built-in schemas have minimal required fields
# ---------------------------------------------------------------------------
def test_invoice_requires_invoice_number():
    """An extraction missing invoice_number is invalid."""
    with pytest.raises(Exception) as exc_info:
        Invoice.model_validate({"vendor_name": "Acme", "total_amount": 100.0})
    assert "invoice_number" in str(exc_info.value)


def test_invoice_requires_vendor_name():
    """An extraction missing vendor_name is invalid."""
    with pytest.raises(Exception) as exc_info:
        Invoice.model_validate({"invoice_number": "INV-1", "total_amount": 100.0})
    assert "vendor_name" in str(exc_info.value)


def test_invoice_requires_total_amount():
    """An extraction missing total_amount is invalid."""
    with pytest.raises(Exception) as exc_info:
        Invoice.model_validate({"invoice_number": "INV-1", "vendor_name": "Acme"})
    assert "total_amount" in str(exc_info.value)


def test_invoice_minimal_valid_dict_passes():
    """Only the three required fields are needed."""
    inv = Invoice.model_validate({
        "invoice_number": "INV-1",
        "vendor_name": "Acme",
        "total_amount": 100.0,
    })
    assert inv.invoice_number == "INV-1"
    assert inv.invoice_date is None
    assert inv.line_items == []


def test_contract_requires_title():
    """A contract without a title is invalid."""
    Contract = SCHEMA_REGISTRY["Contract"]
    with pytest.raises(Exception) as exc_info:
        Contract.model_validate({})
    assert "title" in str(exc_info.value)


def test_bank_statement_requires_account_holder():
    """A statement without an account holder is invalid."""
    with pytest.raises(Exception) as exc_info:
        BankStatement.model_validate({})
    assert "account_holder" in str(exc_info.value)


def test_validate_invoice_missing_required_fails():
    """validate() must report schema_valid=False when required fields are missing.

    This is the bug the original audit caught — before the fix, validate()
    returned passed=True for any extraction because every field was Optional.
    """
    class _Doc:
        extraction_schema = "Invoice"
        extraction = {"total_amount": 100.0}  # missing invoice_number, vendor_name

    d = _Doc()
    validate(d)
    assert d.validation["schema_valid"] is False
    assert d.validation["passed"] is False
    assert len(d.validation["errors"]) > 0


def test_validate_contract_missing_title_fails():
    class _Doc:
        extraction_schema = "Contract"
        extraction = {"effective_date": "2026-01-01"}

    d = _Doc()
    validate(d)
    assert d.validation["schema_valid"] is False
    assert d.validation["passed"] is False


def test_validate_bank_statement_missing_account_holder_fails():
    class _Doc:
        extraction_schema = "BankStatement"
        extraction = {"transactions": []}

    d = _Doc()
    validate(d)
    assert d.validation["schema_valid"] is False
    assert d.validation["passed"] is False


def test_validate_invoice_complete_extraction_passes():
    """Sanity check: a complete extraction still passes after the schema change."""
    class _Doc:
        extraction_schema = "Invoice"
        extraction = {
            "invoice_number": "INV-1",
            "vendor_name": "Acme",
            "total_amount": 540.0,
        }

    d = _Doc()
    validate(d)
    assert d.validation["schema_valid"] is True
    assert d.validation["passed"] is True


# ---------------------------------------------------------------------------
# B2: Document.from_path rejects directories
# ---------------------------------------------------------------------------
def test_from_path_rejects_directory_with_isadirerror():
    """Passing a directory raises IsADirectoryError, not silently returns empty."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(IsADirectoryError) as exc_info:
            Document.from_path(d)
        assert "directory" in str(exc_info.value).lower()


def test_from_path_rejects_symlink_to_directory():
    """A symlink to a directory should also be rejected."""
    with tempfile.TemporaryDirectory() as d:
        real_dir = Path(d) / "real"
        real_dir.mkdir()
        link = Path(d) / "link"
        try:
            link.symlink_to(real_dir)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        with pytest.raises(IsADirectoryError):
            Document.from_path(str(link))


def test_from_path_rejects_missing_file():
    """A nonexistent path still raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Document.from_path("/tmp/this/file/does/not/exist.txt")


def test_from_path_accepts_regular_file():
    """Sanity check: a real file still works after the fix."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("hello world")
        path = f.name
    try:
        doc = Document.from_path(path)
        assert doc.source_path == path
        assert doc.metadata["size"] > 0
    finally:
        Path(path).unlink()


def test_from_path_rejects_fifo():
    """A FIFO / named pipe is not a regular file either."""
    import os
    with tempfile.TemporaryDirectory() as d:
        fifo = Path(d) / "mypipe"
        try:
            os.mkfifo(str(fifo))
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("mkfifo not supported on this platform")
        with pytest.raises(OSError):
            Document.from_path(str(fifo))
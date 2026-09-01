"""Tests for the v0.1 compatibility shim."""
from __future__ import annotations

from idp.compat_v01 import (
    BankStatement_v01,
    Contract_v01,
    Invoice_v01,
    register_compat_schemas,
)
from idp.core.schemas import SCHEMA_REGISTRY, BankStatement, Contract, Invoice
from idp.validate.validator import validate

import pytest


def test_invoice_v01_accepts_empty_dict():
    """The v0.1 shape had all fields optional — empty dict should pass."""
    inv = Invoice_v01.model_validate({})
    assert inv.invoice_number is None
    assert inv.line_items == []


def test_contract_v01_accepts_empty_dict():
    Contract_v01.model_validate({})


def test_bank_statement_v01_accepts_empty_dict():
    BankStatement_v01.model_validate({})


def test_register_compat_schemas_adds_to_registry():
    register_compat_schemas()
    assert "Invoice_v01" in SCHEMA_REGISTRY
    assert "Contract_v01" in SCHEMA_REGISTRY
    assert "BankStatement_v01" in SCHEMA_REGISTRY
    # Original v0.2 schemas still present
    assert "Invoice" in SCHEMA_REGISTRY
    assert "Contract" in SCHEMA_REGISTRY


def test_register_compat_schemas_is_idempotent():
    register_compat_schemas()
    register_compat_schemas()
    # No duplication
    assert SCHEMA_REGISTRY["Invoice_v01"] is Invoice_v01


def test_validate_with_v01_schema_passes_on_empty_dict():
    """The migration use case: validate v0.1 extraction as v0.1 Invoice_v01."""
    class _Doc:
        extraction_schema = "Invoice_v01"
        extraction = {}  # empty — v0.1 was permissive
    register_compat_schemas()
    d = _Doc()
    validate(d)
    assert d.validation["schema_valid"] is True
    assert d.validation["passed"] is True


def test_v02_still_rejects_empty_dict():
    """The breaking change is preserved for the v0.2 schema."""
    class _Doc:
        extraction_schema = "Invoice"
        extraction = {}
    d = _Doc()
    validate(d)
    assert d.validation["schema_valid"] is False


def test_v01_and_v02_diverge_on_partial_dict():
    """A dict with only 'vendor_name' fails v0.2 but passes v0.1."""
    # v0.1: all optional -> passes
    inv_v01 = Invoice_v01.model_validate({"vendor_name": "Acme"})
    assert inv_v01.vendor_name == "Acme"

    # v0.2: missing invoice_number, total_amount -> fails
    with pytest.raises(Exception):
        Invoice.model_validate({"vendor_name": "Acme"})
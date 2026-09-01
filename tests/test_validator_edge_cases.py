"""Tests for validate.validator.numeric_range_rule and edge cases.

These existed in code but were never exercised. Tests here catch the
boundary conditions: what happens with None, what happens at the exact
boundary, what happens with mixed types.
"""
from __future__ import annotations

from idp.validate.validator import (
    numeric_range_rule,
    required_fields_rule,
    validate,
)


# ---------------------------------------------------------------------------
# numeric_range_rule
# ---------------------------------------------------------------------------
def test_numeric_range_field_present_and_in_range():
    r = numeric_range_rule("total_amount", 0.0, 1000.0)
    ok, reason = r({"total_amount": 500.0})
    assert ok is True
    assert reason is None


def test_numeric_range_field_present_above_max():
    r = numeric_range_rule("total_amount", 0.0, 1000.0)
    ok, reason = r({"total_amount": 1500.0})
    assert ok is False
    assert "out of range" in reason
    assert "1500" in reason


def test_numeric_range_field_present_below_min():
    r = numeric_range_rule("total_amount", 10.0, 1000.0)
    ok, reason = r({"total_amount": 5.0})
    assert ok is False
    assert "5" in reason


def test_numeric_range_field_at_exact_boundary_inclusive():
    """At min or max, the field is in range (inclusive boundary)."""
    r = numeric_range_rule("x", 0.0, 100.0)
    assert r({"x": 0.0})[0] is True
    assert r({"x": 100.0})[0] is True
    assert r({"x": 100.01})[0] is False


def test_numeric_range_field_missing_passes_silently():
    """If the field is None/missing, the rule returns (True, None).

    This is by design — pair it with required_fields_rule for required-numeric.
    """
    r = numeric_range_rule("total_amount", 0.0, 1000.0)
    ok, reason = r({})
    assert ok is True
    assert reason is None


def test_numeric_range_field_none_passes():
    r = numeric_range_rule("x", 0.0, 100.0)
    ok, reason = r({"x": None})
    assert ok is True
    assert reason is None


def test_numeric_range_field_non_numeric_rejected():
    """String-where-number-expected -> rule fails (caught at schema validation
    upstream, but defense in depth)."""
    r = numeric_range_rule("total_amount", 0.0, 1000.0)
    ok, reason = r({"total_amount": "not a number"})
    assert ok is False
    assert "not numeric" in reason


def test_numeric_range_int_works():
    """Python int should be accepted as numeric (subclass of float check)."""
    r = numeric_range_rule("qty", 0, 100)
    ok, _ = r({"qty": 50})  # int 50, not float
    assert ok is True


def test_numeric_range_negative_values():
    r = numeric_range_rule("temperature", -50.0, 50.0)
    assert r({"temperature": -25.0})[0] is True
    assert r({"temperature": -100.0})[0] is False


def test_numeric_range_name_attribute_set():
    """The __name__ magic that validate() relies on for diagnostics."""
    r = numeric_range_rule("total_amount", 0.0, 1000.0)
    assert r.__name__ == "range_total_amount"


# ---------------------------------------------------------------------------
# required_fields_rule (also touched up — was missing empty-list case)
# ---------------------------------------------------------------------------
def test_required_fields_with_zero_fields_passes():
    """required_fields_rule() with no fields is a no-op and passes always."""
    r = required_fields_rule()
    assert r({"a": 1, "b": None})[0] is True
    assert r({})[0] is True


def test_required_fields_empty_dict_is_treated_as_missing():
    """Edge case: a nested {} is treated as missing (matches None/[]/'')."""
    r = required_fields_rule("a")
    ok, reason = r({"a": {}})
    # Documenting current behaviour: empty dict is treated as missing.
    # This is by design — pair with schema validation for nested cases.
    assert ok is False
    assert "missing required fields" in reason


# ---------------------------------------------------------------------------
# validate() integration
# ---------------------------------------------------------------------------
def test_validate_with_empty_rule_list_no_schema_returns_passed_false():
    """Documenting the existing behaviour: passed = bool(schema_valid).

    With no schema to check, schema_valid is None and bool(None) is False,
    so validate() reports passed=False even with no rules. This is
    intentional — calling validate() with no schema and no rules is a
    no-op that should not be reported as 'passed'. If you want a 'trivial
    pass', use schema='Invoice' explicitly.
    """
    doc_extraction = {"vendor_name": "Acme", "total_amount": 100.0}
    class _Doc:
        extraction_schema = None
        extraction = doc_extraction
    d = _Doc()
    validate(d, rules=[])
    assert d.validation["schema_valid"] is None
    assert d.validation["rule_results"] == []
    assert d.validation["passed"] is False  # bool(None) is False; intentional


def test_validate_combines_schema_and_rules():
    """Failed rules make validate() pass=False even if schema is valid."""
    class _Doc:
        extraction_schema = "Invoice"
        extraction = {"total_amount": 100.0}

    d = _Doc()
    validate(d, rules=[required_fields_rule("vendor_name")])
    # schema_valid depends on whether Invoice validates; regardless,
    # required_fields_rule("vendor_name") should fail and 'passed' should
    # be False.
    assert d.validation["passed"] is False
    assert any(r["rule"] == "required_vendor_name" for r in d.validation["rule_results"])


# ---------------------------------------------------------------------------
# validate() with raising rules + no-rules default
# ---------------------------------------------------------------------------
def test_validate_rule_raising_is_caught_not_crashed():
    """A rule that raises is caught and reported as a failure, not a crash."""

    def _boom(extraction):
        raise RuntimeError("kaboom")

    class _Doc:
        extraction_schema = "Invoice"
        extraction = {"x": 1}

    d = _Doc()
    validate(d, rules=[_boom])
    assert d.validation["rule_results"][0]["passed"] is False
    assert "rule raised" in d.validation["rule_results"][0]["reason"]
    assert d.validation["passed"] is False


def test_validate_no_rules_arg_uses_none():
    """validate() with no rules argument defaults to an empty rule list."""
    class _Doc:
        extraction_schema = "Invoice"
        # All required fields present so schema_valid is True.
        extraction = {
            "invoice_number": "INV-1",
            "vendor_name": "Acme",
            "total_amount": 100.0,
        }

    d = _Doc()
    validate(d)  # rules defaults to None
    assert d.validation["rule_results"] == []
    assert d.validation["passed"] is True


def test_validate_combined_rules_all_pass():
    """All rules + schema passing yields passed=True."""
    class _Doc:
        extraction_schema = "Invoice"
        extraction = {
            "invoice_number": "INV-1",
            "vendor_name": "Acme",
            "total_amount": 540.0,
        }

    d = _Doc()
    validate(d, rules=[
        required_fields_rule("invoice_number", "vendor_name"),
        numeric_range_rule("total_amount", 0.0, 1000.0),
    ])
    assert d.validation["passed"] is True

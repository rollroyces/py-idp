"""Tests for the eval harness per-field report + regression gate."""
from __future__ import annotations

import pytest

from idp.eval.runner import (
    EvalRegressionError,
    check_regression,
    format_report,
    run_dataset,
    verify_against_thresholds,
)


def _make_results():
    """Build a small fake results dict for unit testing."""
    return {
        "per_strategy": {
            "mock": {
                "schema_valid_rate": 0.6,
                "field_match_rate": 0.5,
                "average_latency_s": 0.1,
                "details": [],
            },
            "ollama": {
                "schema_valid_rate": 1.0,
                "field_match_rate": 0.9,
                "average_latency_s": 1.5,
                "details": [],
            },
        },
        "per_field_accuracy": {
            "invoice_number": {"match": 8, "total": 10, "accuracy": 0.8},
            "total_amount": {"match": 9, "total": 10, "accuracy": 0.9},
            "vendor_name": {"match": 7, "total": 10, "accuracy": 0.7},
        },
    }


def test_check_regression_passes_when_above_threshold():
    results = _make_results()
    failures = check_regression(
        results,
        {
            "per_strategy.mock.schema_valid_rate": 0.5,
            "per_strategy.mock.field_match_rate": 0.4,
            "per_strategy.ollama.schema_valid_rate": 0.95,
            "per_field_accuracy.invoice_number.accuracy": 0.7,
        },
    )
    assert failures == []


def test_check_regression_flags_below_threshold():
    results = _make_results()
    failures = check_regression(
        results,
        {
            "per_strategy.mock.schema_valid_rate": 0.8,  # 0.6 < 0.8 → fail
            "per_strategy.ollama.field_match_rate": 0.95,  # 0.9 < 0.95 → fail
            "per_field_accuracy.vendor_name.accuracy": 0.9,  # 0.7 < 0.9 → fail
        },
    )
    assert len(failures) == 3
    assert any("mock" in f and "schema_valid_rate" in f for f in failures)
    assert any("ollama" in f and "field_match_rate" in f for f in failures)
    assert any("vendor_name" in f for f in failures)


def test_check_regression_ignores_unknown_keys():
    results = _make_results()
    # Unknown metric path → silently skip (don't crash)
    failures = check_regression(
        results,
        {"per_strategy.nonexistent.schema_valid_rate": 0.0},
    )
    assert failures == []


def test_check_regression_strict_equal_satisfies_threshold():
    """A value exactly equal to the threshold is NOT a regression."""
    results = _make_results()
    # mock.schema_valid_rate is 0.6 — threshold 0.6 should pass.
    failures = check_regression(
        results,
        {"per_strategy.mock.schema_valid_rate": 0.6},
    )
    assert failures == []


def test_verify_against_thresholds_raises_on_failure():
    results = _make_results()
    with pytest.raises(EvalRegressionError) as exc_info:
        verify_against_thresholds(
            results,
            {"per_strategy.mock.schema_valid_rate": 0.9},
        )
    # Error message should mention the failing metric
    assert "mock" in str(exc_info.value)


def test_verify_against_thresholds_silent_on_success():
    results = _make_results()
    # All thresholds satisfied — should not raise.
    verify_against_thresholds(
        results,
        {
            "per_strategy.mock.schema_valid_rate": 0.5,
            "per_strategy.ollama.field_match_rate": 0.5,
        },
    )


def test_format_report_runs_without_error():
    results = _make_results()
    out = format_report(results)
    assert "mock" in out
    assert "ollama" in out
    assert "invoice_number" in out
    assert "Per-field accuracy" in out


def test_format_report_includes_percentages():
    results = _make_results()
    out = format_report(results)
    # The accuracy is 0.8 = 80% — should appear
    assert "80" in out or "0.8" in out


def test_format_report_handles_empty_per_field_accuracy():
    """No fields → graceful '(no fields)' line."""
    results = {
        "per_strategy": {
            "mock": {
                "schema_valid_rate": 1.0,
                "field_match_rate": 0.0,
                "average_latency_s": 0.01,
                "details": [],
            },
        },
        "per_field_accuracy": {},
    }
    out = format_report(results)
    assert "Per-field accuracy" in out
    assert "no fields" in out


def test_format_report_lists_all_strategies_and_fields():
    results = _make_results()
    out = format_report(results)
    # All strategy names should appear in the "Strategy 'X':" lines.
    assert "Strategy 'mock':" in out
    assert "Strategy 'ollama':" in out
    # All field names should be in the per-field block.
    assert "invoice_number" in out
    assert "total_amount" in out
    assert "vendor_name" in out


def test_run_dataset_returns_per_field_accuracy(tmp_path):
    """End-to-end smoke: build a tiny dataset, run_dataset should return per_field_accuracy."""
    # Create a 1-case dataset in a temp dir
    ds = tmp_path / "ds"
    ds.mkdir()
    docs = ds / "docs"
    docs.mkdir()
    (docs / "inv.txt").write_text("INVOICE INV-001 ACME $100.00")
    (ds / "cases.jsonl").write_text(
        '{"doc_path": "docs/inv.txt", "schema": "Invoice", "gold": {"invoice_number": "INV-001", "vendor_name": "ACME", "total_amount": 100.0}}\n'
    )
    result = run_dataset(str(ds), strategies=["mock"])
    assert "per_strategy" in result
    assert "per_field_accuracy" in result
    # mock returns empty schema for "extract" prompts, so field accuracy should be 0
    # but the keys should be present
    assert isinstance(result["per_field_accuracy"], dict)
    # Backward-compat keys remain
    assert "rows" in result
    assert "detail" in result
    assert "dataset" in result
    assert "strategies" in result


def test_run_dataset_per_strategy_payload_shape(tmp_path):
    """per_strategy[strat] has the documented fields."""
    ds = tmp_path / "ds"
    ds.mkdir()
    docs = ds / "docs"
    docs.mkdir()
    (docs / "inv.txt").write_text("INVOICE INV-001")
    (ds / "cases.jsonl").write_text(
        '{"doc_path": "docs/inv.txt", "schema": "Invoice", "gold": {"invoice_number": "INV-001"}}\n'
    )
    result = run_dataset(str(ds), strategies=["mock"])
    p = result["per_strategy"]["mock"]
    assert "schema_valid_rate" in p
    assert "field_match_rate" in p
    assert "average_latency_s" in p
    assert "details" in p

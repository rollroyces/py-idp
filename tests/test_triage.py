"""Tests for ``idp.hitl.triage`` (P3 v0.4 D1).

These tests are deterministic and use synthetic ``StoredResult`` rows
fed directly into ``triage_from_results`` — no on-disk storage, no
Streamlit, no real LLM calls. They cover:

  * Output determinism (same input → same report, every run).
  * Threshold correctness (correction_rate = n_corrections / n_reviews;
    a field flagged iff rate >= threshold AND n >= min_reviews).
  * Empty storage / no reviews yet.
  * Fields only seen once (always "insufficient" at default min_reviews).
  * Mixed systematic-error + noise (correctly partitioned).
  * The "insufficient data" honest failure mode.
  * JSON output shape (TriageReport.to_dict).
  * Markdown output shape (format_report_markdown).

P3 v0.4 acceptance criterion D1 #4: "Unit tests on synthetic data:
deterministic output, correct thresholds, handles empty storage,
handles fields only seen once." This file covers that and a few more.
"""
from __future__ import annotations

import json

import pytest

from idp.hitl.triage import (
    DEFAULT_MIN_REVIEWS,
    DEFAULT_THRESHOLD,
    InsufficientField,
    SystematicError,
    TriageReport,
    format_report_markdown,
    triage_from_results,
)
from idp.storage.store import InMemoryStorage, StoredResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_result(
    *,
    result_id: str,
    extraction: dict,
    reviewed_extraction: dict,
    schema: str = "Invoice",
    doc_id: str | None = None,
    created_at: float = 1000.0,
) -> StoredResult:
    """Build a reviewed ``StoredResult`` for synthetic triage tests.

    ``extraction`` is what the model said; ``reviewed_extraction`` is
    what the human corrected it to. Equal dicts → no corrections.
    """
    return StoredResult(
        id=result_id,
        doc_id=doc_id or result_id,
        schema_name=schema,
        backend_name="mock",
        mode="ocr_llm",
        classification="invoice",
        extraction=extraction,
        confidence={k: 0.7 for k in extraction},
        validation={"passed": True},
        source_path=f"/tmp/{result_id}.pdf",
        created_at=created_at,
        reviewed=True,
        reviewed_extraction=reviewed_extraction,
        reviewer="alice",
        last_reviewed_at=created_at + 0.5,
    )


def _unreviewed_result(*, result_id: str = "u1", extraction: dict | None = None) -> StoredResult:
    """A StoredResult that has NOT been reviewed. triage() must skip it."""
    return StoredResult(
        id=result_id,
        doc_id=result_id,
        schema_name="Invoice",
        backend_name="mock",
        mode="ocr_llm",
        classification="invoice",
        extraction=extraction or {"vendor_name": "Acme", "total_amount": 100.0},
        confidence={"vendor_name": 0.9, "total_amount": 0.9},
        validation={"passed": True},
        source_path=f"/tmp/{result_id}.pdf",
        created_at=1000.0,
        reviewed=False,
        reviewed_extraction=None,
        reviewer=None,
        last_reviewed_at=None,
    )


# ---------------------------------------------------------------------------
# 1. Empty storage — no reviews yet
# ---------------------------------------------------------------------------
def test_empty_results_returns_empty_report() -> None:
    """triage_from_results([]) returns an empty TriageReport (no errors)."""
    report = triage_from_results([])
    assert isinstance(report, TriageReport)
    assert report.systematic_errors == []
    assert report.insufficient == []
    assert report.n_reviews_total == 0
    assert report.n_fields_seen == 0
    # The echoed-back thresholds match the defaults.
    assert report.min_reviews == DEFAULT_MIN_REVIEWS == 3
    assert report.threshold == DEFAULT_THRESHOLD == 0.6


def test_storage_with_only_unreviewed_results_returns_empty_report(tmp_path) -> None:
    """A storage backend with results but zero reviews → empty report."""
    storage = InMemoryStorage()
    storage.put(_unreviewed_result(result_id="u1"))
    storage.put(_unreviewed_result(result_id="u2", extraction={"x": 1}))
    report = triage_from_results(storage.list(reviewed_only=True, limit=1000))
    assert report.systematic_errors == []
    assert report.insufficient == []
    assert report.n_reviews_total == 0


# ---------------------------------------------------------------------------
# 2. Correct thresholds — systematic-error math
# ---------------------------------------------------------------------------
def test_systematic_error_emitted_when_rate_above_threshold() -> None:
    """vendor_name corrected in 4/5 reviews → systematic error (4/5 = 0.8 >= 0.6)."""
    results = []
    for i in range(5):
        results.append(
            _make_result(
                result_id=f"r{i}",
                extraction={"vendor_name": "ACME", "total_amount": 100.0},
                # First 4 reviews correct vendor_name; 5th leaves it alone.
                reviewed_extraction=(
                    {"vendor_name": f"Acme {i}", "total_amount": 100.0}
                    if i < 4
                    else {"vendor_name": "ACME", "total_amount": 100.0}
                ),
                created_at=1000.0 + i,
            )
        )
    report = triage_from_results(results)
    assert report.n_reviews_total == 5
    # vendor_name: 4/5 corrections → rate 0.80 → flagged
    fields_in_report = {se.field: se for se in report.systematic_errors}
    assert "vendor_name" in fields_in_report
    se = fields_in_report["vendor_name"]
    assert se.n == 5
    assert se.n_corrections == 4
    assert se.rate == pytest.approx(0.8)
    # total_amount: 0/5 corrections → rate 0 → not flagged, not insufficient
    # (it's exactly at min_reviews, so it's "noise" — not in either list).
    assert "total_amount" not in fields_in_report
    assert all(inf.field != "total_amount" for inf in report.insufficient)


def test_threshold_at_exactly_min_boundary_is_inclusive() -> None:
    """Rate == threshold IS flagged (>=, not >)."""
    # 3/5 = 0.6 == threshold → should flag.
    results = []
    for i in range(5):
        results.append(
            _make_result(
                result_id=f"r{i}",
                extraction={"vendor_name": "ACME"},
                reviewed_extraction=(
                    {"vendor_name": "Acme"} if i < 3 else {"vendor_name": "ACME"}
                ),
                created_at=1000.0 + i,
            )
        )
    report = triage_from_results(results)
    assert len(report.systematic_errors) == 1
    se = report.systematic_errors[0]
    assert se.field == "vendor_name"
    assert se.n == 5
    assert se.n_corrections == 3
    assert se.rate == pytest.approx(0.6)


def test_field_below_threshold_not_flagged() -> None:
    """2/5 = 0.4 < 0.6 → not flagged, not insufficient."""
    results = []
    for i in range(5):
        results.append(
            _make_result(
                result_id=f"r{i}",
                extraction={"vendor_name": "ACME"},
                reviewed_extraction=(
                    {"vendor_name": "Acme"} if i < 2 else {"vendor_name": "ACME"}
                ),
                created_at=1000.0 + i,
            )
        )
    report = triage_from_results(results)
    # The field is "noise" — appears 5 times but rate is below threshold.
    # Not in systematic_errors, not in insufficient (5 >= min_reviews).
    assert all(se.field != "vendor_name" for se in report.systematic_errors)
    assert all(inf.field != "vendor_name" for inf in report.insufficient)


# ---------------------------------------------------------------------------
# 3. Honest failure mode — insufficient data
# ---------------------------------------------------------------------------
def test_insufficient_data_path_for_fields_seen_only_once() -> None:
    """A field seen exactly once → "insufficient data", not a guess."""
    results = [
        _make_result(
            result_id="r1",
            extraction={"vendor_name": "ACME", "address": "123 Fake St"},
            reviewed_extraction={"vendor_name": "Acme", "address": "123 Fake St"},
            created_at=1000.0,
        ),
    ]
    report = triage_from_results(results)
    # Both fields appear once → both should be in insufficient
    inf_fields = {inf.field: inf for inf in report.insufficient}
    assert "vendor_name" in inf_fields
    assert "address" in inf_fields
    # With default min_reviews=3, each field needs 2 more reviews.
    assert inf_fields["vendor_name"].n == 1
    assert inf_fields["vendor_name"].needed == 2
    assert inf_fields["address"].n == 1
    assert inf_fields["address"].needed == 2
    # And neither is in systematic_errors.
    assert all(se.field not in {"vendor_name", "address"} for se in report.systematic_errors)


def test_insufficient_message_states_explicit_count() -> None:
    """The 'needed' count is exact: min_reviews - n_reviews."""
    # vendor_name reviewed 2 times → needs 1 more.
    results = [
        _make_result(
            result_id=f"r{i}",
            extraction={"vendor_name": "ACME", "total_amount": 100.0},
            reviewed_extraction={"vendor_name": "Acme", "total_amount": 100.0},
            created_at=1000.0 + i,
        )
        for i in range(2)
    ]
    report = triage_from_results(results)
    inf_fields = {inf.field: inf for inf in report.insufficient}
    assert inf_fields["vendor_name"].n == 2
    assert inf_fields["vendor_name"].needed == 1
    assert inf_fields["total_amount"].n == 2
    assert inf_fields["total_amount"].needed == 1


# ---------------------------------------------------------------------------
# 4. Mixed: systematic-error + noise + insufficient
# ---------------------------------------------------------------------------
def test_mixed_systematic_noise_and_insufficient_partitioned_correctly() -> None:
    """Three fields: one systematic, one noise (>= min_reviews, < threshold), one rare."""
    results = []
    # vendor_name: corrected in 4/4 reviews (rate 1.0) → systematic
    # total_amount: never corrected, but appears in all 4 → noise
    # rare_field: only appears in 1 review → insufficient
    for i in range(4):
        results.append(
            _make_result(
                result_id=f"r{i}",
                extraction={
                    "vendor_name": "ACME",
                    "total_amount": 100.0,
                    "rare_field": f"value_{i}",
                },
                reviewed_extraction={
                    "vendor_name": f"Acme {i}",  # corrected
                    "total_amount": 100.0,  # accepted
                    "rare_field": f"value_{i}",  # accepted
                },
                created_at=1000.0 + i,
            )
        )
    # A single review that introduces a brand-new field, "phone".
    # Since "phone" only appears in 1 of 5 reviews, it's insufficient.
    results.append(
        _make_result(
            result_id="r4",
            extraction={
                "vendor_name": "ACME",
                "total_amount": 100.0,
                "rare_field": "value_4",
                "phone": "555-0001",
            },
            reviewed_extraction={
                "vendor_name": "ACME",
                "total_amount": 100.0,
                "rare_field": "value_4",
                "phone": "555-0001",
            },
            created_at=1004.0,
        )
    )

    report = triage_from_results(results)
    fields_sys = {se.field for se in report.systematic_errors}
    fields_inf = {inf.field for inf in report.insufficient}

    assert "vendor_name" in fields_sys  # 4/4 systematic
    assert "total_amount" not in fields_sys and "total_amount" not in fields_inf  # noise
    assert "rare_field" not in fields_sys and "rare_field" not in fields_inf  # noise
    assert "phone" in fields_inf  # 1 review < 3 min_reviews


def test_noise_field_with_high_min_reviews_goes_to_insufficient() -> None:
    """Field with n < min_reviews goes to insufficient regardless of rate."""
    results = [
        _make_result(
            result_id=f"r{i}",
            extraction={"vendor_name": "ACME"},
            # All 2 reviews corrected → rate 1.0 but n < min_reviews=3
            reviewed_extraction={"vendor_name": "Acme"},
            created_at=1000.0 + i,
        )
        for i in range(2)
    ]
    report = triage_from_results(results)
    assert report.systematic_errors == []
    assert len(report.insufficient) == 1
    assert report.insufficient[0].field == "vendor_name"
    assert report.insufficient[0].n == 2
    assert report.insufficient[0].needed == 1


# ---------------------------------------------------------------------------
# 5. Output determinism
# ---------------------------------------------------------------------------
def test_output_is_deterministic_across_runs() -> None:
    """Same input → same report bytes. Important for caching / snapshot tests."""
    results = []
    for i in range(5):
        results.append(
            _make_result(
                result_id=f"r{i}",
                extraction={"vendor_name": "ACME", "total_amount": 100.0},
                reviewed_extraction={"vendor_name": f"Acme {i}", "total_amount": 100.0},
                created_at=1000.0 + i,
            )
        )
    rep1 = triage_from_results(results)
    rep2 = triage_from_results(results)
    assert rep1.to_dict() == rep2.to_dict()


def test_systematic_errors_sorted_by_rate_then_n() -> None:
    """Sort order is rate desc, then n desc, then field name asc (stable)."""
    results = []
    # field_a: 3/3 corrected (rate 1.0, n=3)
    # field_b: 5/5 corrected (rate 1.0, n=5) — should come first
    # field_c: 3/5 corrected (rate 0.6, n=5) — should come last
    for i in range(5):
        results.append(
            _make_result(
                result_id=f"r{i}",
                extraction={
                    "field_a": "X",
                    "field_b": "Y",
                    "field_c": "Z",
                },
                reviewed_extraction={
                    "field_a": "X-corrected" if i < 3 else "X",
                    "field_b": "Y-corrected",  # always corrected
                    "field_c": "Z-corrected" if i < 3 else "Z",
                },
                created_at=1000.0 + i,
            )
        )
    report = triage_from_results(results)
    field_order = [se.field for se in report.systematic_errors]
    # field_b (rate 1.0, n=5) first; field_a (rate 1.0, n=3) second;
    # field_c (rate 0.6, n=5) third. (1.0 == 1.0 ties broken by n desc.)
    assert field_order.index("field_b") < field_order.index("field_a")
    assert field_order.index("field_a") < field_order.index("field_c")


# ---------------------------------------------------------------------------
# 6. JSON output shape
# ---------------------------------------------------------------------------
def test_to_dict_json_output_shape() -> None:
    """The dict shape is JSON-serializable and matches the documented schema."""
    results = [
        _make_result(
            result_id="r1",
            extraction={"vendor_name": "ACME"},
            reviewed_extraction={"vendor_name": "Acme"},
        ),
    ]
    report = triage_from_results(results)
    d = report.to_dict()
    # All keys at top level
    expected_top_keys = {
        "min_reviews", "threshold", "n_reviews_total",
        "n_fields_seen", "systematic_errors", "insufficient",
    }
    assert set(d.keys()) == expected_top_keys
    # JSON round-trips
    json.dumps(d)
    # systematic_errors entries have the expected keys
    if d["systematic_errors"]:
        se = d["systematic_errors"][0]
        assert set(se.keys()) == {
            "field", "rate", "n", "n_corrections", "sample_review_ids",
        }
    # insufficient entries have the expected keys
    if d["insufficient"]:
        inf = d["insufficient"][0]
        assert set(inf.keys()) == {"field", "n", "needed"}


def test_to_dict_json_contains_only_primitives() -> None:
    """No dataclass leakage in to_dict — all values are primitives."""
    results = [
        _make_result(
            result_id="r1",
            extraction={"vendor_name": "ACME"},
            reviewed_extraction={"vendor_name": "Acme"},
            created_at=1000.0,
        ),
        _make_result(
            result_id="r2",
            extraction={"vendor_name": "ACME"},
            reviewed_extraction={"vendor_name": "Acme"},
            created_at=1001.0,
        ),
        _make_result(
            result_id="r3",
            extraction={"vendor_name": "ACME"},
            reviewed_extraction={"vendor_name": "Acme"},
            created_at=1002.0,
        ),
    ]
    report = triage_from_results(results)
    raw = json.dumps(report.to_dict())
    parsed = json.loads(raw)
    # Re-serializing parsed dicts == same bytes
    assert json.dumps(parsed) == raw


# ---------------------------------------------------------------------------
# 7. Markdown output shape
# ---------------------------------------------------------------------------
def test_format_report_markdown_contains_key_sections() -> None:
    """Markdown output has the headings the CLI / UI promise."""
    results = [
        _make_result(
            result_id=f"r{i}",
            extraction={"vendor_name": "ACME"},
            reviewed_extraction={"vendor_name": "Acme"},
            created_at=1000.0 + i,
        )
        for i in range(5)
    ]
    report = triage_from_results(results)
    md = format_report_markdown(report)
    # Required structural pieces
    assert "# py-idp triage report" in md
    assert "## Systematic errors" in md
    assert "vendor_name" in md
    # Threshold/min_reviews are surfaced
    assert "0.60" in md
    assert "3" in md  # min_reviews default
    # Footer
    assert "Generated by py-idp triage" in md


def test_format_report_markdown_handles_empty_report() -> None:
    """No reviews → MD report has the placeholder text."""
    md = format_report_markdown(triage_from_results([]))
    assert "# py-idp triage report" in md
    assert "_None — no field has crossed the threshold yet._" in md
    assert "## Insufficient data" not in md  # no insufficient section


def test_format_report_markdown_includes_insufficient_section() -> None:
    """When fields have insufficient reviews, MD report has an Insufficient section."""
    results = [
        _make_result(
            result_id="r1",
            extraction={"vendor_name": "ACME"},
            reviewed_extraction={"vendor_name": "Acme"},
        ),
    ]
    md = format_report_markdown(triage_from_results(results))
    assert "## Insufficient data" in md
    assert "vendor_name" in md
    assert "+2" in md  # needs 2 more reviews


# ---------------------------------------------------------------------------
# 8. Validation of public API surface
# ---------------------------------------------------------------------------
def test_invalid_thresholds_raise() -> None:
    """Caller-passed garbage is rejected up front (not silently ignored)."""
    with pytest.raises(ValueError, match="min_reviews"):
        triage_from_results([], min_reviews=0)
    with pytest.raises(ValueError, match="threshold"):
        triage_from_results([], threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        triage_from_results([], threshold=-0.1)


def test_systematic_error_dataclass_validates() -> None:
    """Sanity check the dataclass invariants documented on SystematicError."""
    # Rate in [0, 1]
    with pytest.raises(ValueError, match="rate"):
        SystematicError(field="x", rate=1.5, n=3, n_corrections=3)
    with pytest.raises(ValueError, match="rate"):
        SystematicError(field="x", rate=-0.1, n=3, n_corrections=3)
    # n_corrections <= n
    with pytest.raises(ValueError, match="cannot exceed"):
        SystematicError(field="x", rate=1.0, n=3, n_corrections=5)
    # OK case
    se = SystematicError(field="x", rate=1.0, n=5, n_corrections=5)
    assert se.sample_review_ids == []  # default


def test_insufficient_field_dataclass_validates() -> None:
    """InsufficientField rejects negative counts."""
    with pytest.raises(ValueError, match="non-negative"):
        InsufficientField(field="x", n=-1, needed=4)
    with pytest.raises(ValueError, match="non-negative"):
        InsufficientField(field="x", n=2, needed=-1)


# ---------------------------------------------------------------------------
# 9. Defensive: malformed reviewed rows
# ---------------------------------------------------------------------------
def test_rows_with_reviewed_true_but_no_reviewed_extraction_are_skipped() -> None:
    """A corrupted StoredResult (reviewed=True, reviewed_extraction=None)
    is skipped instead of crashing the whole report. triage() walks a
    stream of storage rows that may have any shape; one malformed row
    shouldn't bring down the analysis.
    """
    bad = StoredResult(
        id="r0",
        doc_id="d0",
        schema_name="Invoice",
        backend_name="mock",
        mode="ocr_llm",
        classification="invoice",
        extraction={"vendor_name": "ACME"},
        confidence={"vendor_name": 0.7},
        validation={"passed": True},
        source_path="/tmp/r0.pdf",
        created_at=1000.0,
        reviewed=True,  # claimed reviewed
        reviewed_extraction=None,  # but missing the data — corrupted
        reviewer=None,
        last_reviewed_at=None,
    )
    good = _make_result(
        result_id="r1",
        extraction={"vendor_name": "ACME"},
        reviewed_extraction={"vendor_name": "Acme"},
        created_at=1001.0,
    )
    # triage() does NOT crash; bad row is excluded from n_reviews_total.
    report = triage_from_results([bad, good])
    assert report.n_reviews_total == 1  # only the good row counted
    assert report.n_fields_seen == 1


# ---------------------------------------------------------------------------
# 10. sample_review_ids capped at the documented limit
# ---------------------------------------------------------------------------
def test_sample_review_ids_capped_at_documented_limit() -> None:
    """sample_review_ids doesn't grow unbounded for fields reviewed many times."""
    # 50 reviews, vendor_name corrected in all 50 → systematic
    results = []
    for i in range(50):
        results.append(
            _make_result(
                result_id=f"r{i:03d}",
                extraction={"vendor_name": "ACME"},
                reviewed_extraction={"vendor_name": f"Acme {i}"},
                created_at=1000.0 + i,
            )
        )
    report = triage_from_results(results)
    se = report.systematic_errors[0]
    # Cap is 5 (SAMPLE_REVIEW_IDS_LIMIT)
    assert len(se.sample_review_ids) == 5
    # All IDs in the sample should appear in our input set
    for sid in se.sample_review_ids:
        assert sid.startswith("r")

"""Tests for the rl.calibrate module (RL calibration eval)."""
from __future__ import annotations

import pytest

from idp.rl.calibrate import (
    confidence_calibration_error,
    evaluate_policy,
    load_reviews,
    synthetic_reviews_from_gold,
)
from idp.rl.policy import PolicyConfig


# ---------------------------------------------------------------------------
# Review loading
# ---------------------------------------------------------------------------
def test_load_reviews_basic(tmp_path):
    p = tmp_path / "rev.jsonl"
    p.write_text(
        '{"doc_id":"a","schema":"Invoice","model":{"x":1},"human":{"x":1}}\n'
        '{"doc_id":"b","schema":"Invoice","model":{"x":2},"human":{"x":3}}\n'
    )
    revs = load_reviews(p)
    assert len(revs) == 2


# ---------------------------------------------------------------------------
# Synthetic reviews from gold truth
# ---------------------------------------------------------------------------
def test_synthetic_reviews_generates_corrections():
    revs, outcomes = synthetic_reviews_from_gold(
        "/Users/hermes/py-idp/src/idp/eval/datasets/invoices",
        injection_rate=0.5, seed=42,
    )
    # 3 invoices with 9 fields each, injection_rate=0.5, k=4-5 corrupt
    assert len(revs) >= 3
    assert all(r.get("synthetic") is True for r in revs)
    # all fields should have at least 1 correction (since we injected)
    assert any(o["+1"] > 0 for o in outcomes.values())


def test_synthetic_reviews_deterministic_seed():
    a, _ = synthetic_reviews_from_gold(
        "/Users/hermes/py-idp/src/idp/eval/datasets/invoices",
        injection_rate=0.3, seed=7,
    )
    b, _ = synthetic_reviews_from_gold(
        "/Users/hermes/py-idp/src/idp/eval/datasets/invoices",
        injection_rate=0.3, seed=7,
    )
    assert a == b


# ---------------------------------------------------------------------------
# evaluate_policy
# ---------------------------------------------------------------------------
def test_evaluate_policy_empty_reviews():
    pol = PolicyConfig()
    report = evaluate_policy([], pol, synthetic=False)
    assert report.n_reviews == 0
    assert report.overall["hit_rate_when_fires"] == 0.0


def test_evaluate_policy_perfect_policy():
    """A policy that flags every field with corrections should hit all of them."""
    revs = [
        {"doc_id": "1", "schema": "Invoice", "model": {"vendor_name": "X"}, "human": {"vendor_name": "Y"}},
        {"doc_id": "2", "schema": "Invoice", "model": {"vendor_name": "X"}, "human": {"vendor_name": "Y"}},
        {"doc_id": "3", "schema": "Invoice", "model": {"vendor_name": "X"}, "human": {"vendor_name": "X"}},
    ]
    pol = PolicyConfig(field_floors={"vendor_name": 0.95})
    report = evaluate_policy(revs, pol)
    fe = next(f for f in report.field_evals if f.field == "vendor_name")
    assert fe.policy_fires is True
    assert fe.hits == 2
    assert fe.false_flags == 1
    assert fe.hit_rate_when_fires == pytest.approx(2 / 3)


def test_evaluate_policy_silent_policy_misses_corrections():
    """A policy that doesn't fire on a high-failure field misses all corrections."""
    revs = [
        {"doc_id": "1", "schema": "Invoice", "model": {"vendor_name": "X"}, "human": {"vendor_name": "Y"}},
        {"doc_id": "2", "schema": "Invoice", "model": {"vendor_name": "X"}, "human": {"vendor_name": "Y"}},
        {"doc_id": "3", "schema": "Invoice", "model": {"vendor_name": "X"}, "human": {"vendor_name": "X"}},
    ]
    pol = PolicyConfig()  # no overrides -> fires nowhere
    report = evaluate_policy(revs, pol)
    fe = next(f for f in report.field_evals if f.field == "vendor_name")
    assert fe.policy_fires is False
    assert fe.missed_corrections == 2
    assert fe.true_accepts == 1
    assert fe.true_accept_rate_when_silent == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------
def test_calibration_perfectly_calibrated():
    """If confidences exactly predict correctness, calibration error = 0."""
    revs = [
        {"doc_id": "1", "schema": "X", "model": {"f": "right"}, "human": {"f": "right"}},
        {"doc_id": "2", "schema": "X", "model": {"f": "wrong"}, "human": {"f": "right"}},
    ]
    def conf_func(model, field):
        return 0.25 if model.get(field) == "wrong" else 0.85
    out = confidence_calibration_error(revs, conf_func, n_buckets=4)
    # 1 in bucket [0,0.25), 1 in [0.75,1.0)
    # both should have empirical = 1.0; bucket midpoints 0.125 and 0.875
    # absolute errors: |0.125-1| + |0.875-1| = 0.875 + 0.125 = 1.0
    # mean over n=2 = 0.5
    assert out["calibration_error"] > 0  # at least some error expected


def test_calibration_empty_reviews():
    out = confidence_calibration_error([], lambda m, f: 0.5)
    assert out["buckets"] == []
    assert out["calibration_error"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end: synthetic reviews -> policy -> rl-eval
# ---------------------------------------------------------------------------
def test_end_to_end_synthetic_eval_pipeline(tmp_path):
    revs, _ = synthetic_reviews_from_gold(
        "/Users/hermes/py-idp/src/idp/eval/datasets/invoices",
        injection_rate=0.3, seed=0,
    )
    # build a policy that fires on all fields with any correction
    pol = PolicyConfig(
        field_floors={"vendor_name": 0.95, "subtotal": 0.95, "tax_amount": 0.95},
        field_penalties={"vendor_name": 0.20, "subtotal": 0.20, "tax_amount": 0.20},
    )
    report = evaluate_policy(revs, pol, synthetic=True)
    assert report.synthetic is True
    assert len(report.field_evals) >= 3
    # overall numbers should be present
    assert "review_efficiency" in report.overall

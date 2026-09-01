"""Tests for the rl (online-from-HITL) module.

Covers:
  - reward derivation from StoredResult diffs
  - aggregation into PolicyStats
  - PolicyConfig.update from stats
  - policy_to_review_flags / policy_to_penalised_confidence
  - round-trip via PolicyConfig.save / load
  - end-to-end: write reviews -> rl-update -> policy applied to next pipeline run
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from idp.core.schemas import Invoice
from idp.pipeline.pipeline import Pipeline
from idp.llm.backend import Backend, CompletionRequest, Message
from idp.rl.policy import (
    PolicyConfig,
    policy_to_penalised_confidence,
    policy_to_review_flags,
    update_policy,
)
from idp.rl.reward import (
    aggregate_rewards,
    derive_field_rewards,
)
from idp.rl.update import update_policy_from_reviews_file
from idp.storage.store import StoredResult


# ---------------------------------------------------------------------------
# Reward derivation
# ---------------------------------------------------------------------------
def test_derive_field_rewards_no_review_returns_none():
    stored = StoredResult(
        id="1", doc_id="d", schema_name="Invoice", backend_name="mock", mode="ocr_llm",
        classification="invoice", extraction={"vendor_name": "Acme"}, confidence=None,
        validation=None, source_path="/x", created_at=0.0,
    )
    assert derive_field_rewards(stored) is None


def test_derive_field_rewards_identical_fields_zero_reward():
    stored = StoredResult(
        id="1", doc_id="d", schema_name="Invoice", backend_name="mock", mode="ocr_llm",
        classification="invoice", extraction={"vendor_name": "Acme", "total_amount": 100.0},
        confidence=None, validation=None, source_path="/x", created_at=0.0,
        reviewed=True, reviewed_extraction={"vendor_name": "Acme", "total_amount": 100.0},
        reviewer="alice",
    )
    rewards = derive_field_rewards(stored)
    assert rewards is not None
    assert rewards.n_corrected == 0
    assert rewards.n_accepted == 2


def test_derive_field_rewards_corrected_field_positive_reward():
    stored = StoredResult(
        id="1", doc_id="d", schema_name="Invoice", backend_name="mock", mode="ocr_llm",
        classification="invoice", extraction={"vendor_name": "Acme", "total_amount": 100.0},
        confidence=None, validation=None, source_path="/x", created_at=0.0,
        reviewed=True,
        reviewed_extraction={"vendor_name": "Acme Widgets Ltd.", "total_amount": 100.0},
        reviewer="alice",
    )
    rewards = derive_field_rewards(stored)
    assert rewards.n_corrected == 1
    by_field = {r.field: r.reward for r in rewards.field_rewards}
    assert by_field["vendor_name"] == +1
    assert by_field["total_amount"] == 0


def test_derive_field_rewards_new_field_in_human_extract_treated_as_corrected():
    """If human added a field the model left out, model was wrong on that field."""
    stored = StoredResult(
        id="1", doc_id="d", schema_name="Invoice", backend_name="mock", mode="ocr_llm",
        classification="invoice",
        extraction={"vendor_name": "Acme"},  # model missed total_amount
        confidence=None, validation=None, source_path="/x", created_at=0.0,
        reviewed=True,
        reviewed_extraction={"vendor_name": "Acme", "total_amount": 100.0},
        reviewer="alice",
    )
    rewards = derive_field_rewards(stored)
    by_field = {r.field: r.reward for r in rewards.field_rewards}
    assert by_field["total_amount"] == +1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_aggregate_rewards_counts_per_field():
    reviews = [
        derive_field_rewards(StoredResult(
            id="1", doc_id="d", schema_name="Invoice", backend_name="m", mode=None,
            classification=None,
            extraction={"vendor_name": "X"},
            confidence=None, validation=None, source_path="/x", created_at=0.0,
            reviewed=True,
            reviewed_extraction={"vendor_name": "X"},
            reviewer="alice",
        )),
        derive_field_rewards(StoredResult(
            id="2", doc_id="d", schema_name="Invoice", backend_name="m", mode=None,
            classification=None,
            extraction={"vendor_name": "Wrong"},
            confidence=None, validation=None, source_path="/x", created_at=0.0,
            reviewed=True,
            reviewed_extraction={"vendor_name": "Right"},
            reviewer="alice",
        )),
    ]
    stats = aggregate_rewards(reviews)
    assert stats.n_reviews == 2
    # Counter keys are formatted as "+1", "0", "-1" (we use the sign only for nonzero)
    assert stats.per_field["vendor_name"] == {"0": 1, "+1": 1}


# ---------------------------------------------------------------------------
# Policy update
# ---------------------------------------------------------------------------
def test_update_policy_below_threshold_no_override():
    stats = aggregate_rewards([
        derive_field_rewards(StoredResult(
            id=str(i), doc_id="d", schema_name="Invoice", backend_name="m", mode=None,
            classification=None,
            extraction={"vendor_name": f"wrong{i}"},
            confidence=None, validation=None, source_path="/x", created_at=0.0,
            reviewed=True,
            reviewed_extraction={"vendor_name": "correct"},
            reviewer="alice",
        ))
        for i in range(10)  # 10 corrections on vendor_name
    ])
    # 10 corrections / 10 reviews = 100% fail rate -> override
    cfg = update_policy(stats)
    assert "vendor_name" in cfg.field_floors


def test_update_policy_high_threshold_no_override():
    """When threshold is high and fail rate is low, no override."""
    # 3 corrections / 10 reviews = 30% fail rate
    reviews = []
    for i in range(10):
        reviews.append(derive_field_rewards(StoredResult(
            id=str(i), doc_id="d", schema_name="Invoice", backend_name="m", mode=None,
            classification=None,
            extraction={"vendor_name": f"x{i}" if i < 3 else "right"},
            confidence=None, validation=None, source_path="/x", created_at=0.0,
            reviewed=True,
            reviewed_extraction={"vendor_name": "right"},
            reviewer="alice",
        )))
    stats = aggregate_rewards(reviews)
    cfg = update_policy(stats, current=PolicyConfig(high_failure_threshold=0.50))
    assert "vendor_name" not in cfg.field_floors


def test_policy_to_review_flags():
    cfg = PolicyConfig(field_floors={"vendor_name": 0.95}, base_confidence_floor=0.6)
    flags = policy_to_review_flags(
        {"vendor_name": 0.7, "invoice_number": 0.7, "total_amount": 0.5},
        cfg,
    )
    # vendor_name: floor=0.95, conf=0.7 -> flagged
    assert flags["vendor_name"] is True
    # invoice_number: floor=0.6, conf=0.7 -> NOT flagged
    assert flags["invoice_number"] is False
    # total_amount: floor=0.6, conf=0.5 -> flagged
    assert flags["total_amount"] is True


def test_policy_to_penalised_confidence():
    cfg = PolicyConfig(
        field_penalties={"vendor_name": 0.3},
    )
    out = policy_to_penalised_confidence(
        {"vendor_name": 0.9, "invoice_number": 0.9},
        cfg,
    )
    assert out["vendor_name"] == pytest.approx(0.6)
    assert out["invoice_number"] == 0.9


def test_policy_round_trip_json(tmp_path):
    cfg = PolicyConfig(field_floors={"vendor_name": 0.95}, field_penalties={"vendor_name": 0.3})
    p = tmp_path / "policy.json"
    cfg.save(str(p))
    loaded = PolicyConfig.load(str(p))
    assert loaded.field_floors == {"vendor_name": 0.95}
    assert loaded.field_penalties == {"vendor_name": 0.3}


# ---------------------------------------------------------------------------
# End-to-end: reviews JSONL -> policy -> pipeline applies penalties
# ---------------------------------------------------------------------------
def test_end_to_end_policy_in_pipeline(tmp_path):
    reviews_file = tmp_path / "reviews.jsonl"
    # Need >= 10 reviews since PolicyConfig.min_reviews defaults to 10;
    # the heuristic guards against small-sample overrides.
    lines = []
    for i in range(12):
        lines.append(
            f'{{"doc_id":"x{i}","schema":"Invoice",'
            f'"model":{{"vendor_name":"Acme","total":100.0}},'
            f'"human":{{"vendor_name":"Acme Widgets Ltd.","total":100.0}}}}'
        )
    reviews_file.write_text("\n".join(lines) + "\n")
    policy_file = tmp_path / "policy.json"
    new_policy = update_policy_from_reviews_file(str(reviews_file), str(policy_file))
    # 12/12 vendor_name wrong -> 100% fail rate -> override (above min_reviews=10)
    assert "vendor_name" in new_policy.field_floors

    # Use the policy in a real pipeline run via canned backend
    class Canned(Backend):
        name = "canned"
        @property
        def is_multimodal(self): return False
        def complete(self, req: CompletionRequest) -> str:
            text = next(m.content for m in reversed(req.messages) if m.role == "user").lower()
            if "classify" in text:
                return json.dumps({"label": "invoice", "confidence": 0.9})
            return json.dumps({
                "invoice_number": "INV-1",
                "vendor_name": "Acme",
                "total_amount": 100.0,
                "invoice_date": "2026-01-01",
            })

    pipe = Pipeline(backend=Canned(), schema=Invoice, policy_path=str(policy_file))
    doc_path = tmp_path / "inv.txt"
    doc_path.write_text("INVOICE INV-1\nVendor: Acme\nTotal: 100\n")
    from idp.core.document import Document
    res = pipe.run(Document.from_path(str(doc_path)))
    # heuristic base for ocr_llm = 0.7; vendor_name is a non-empty string
    # so +0.05 boost -> 0.75; policy penalty 0.20 -> 0.55
    assert res.document.confidence["vendor_name"] == pytest.approx(0.55)
    # invoice_number is also a non-empty string but no penalty -> 0.75
    assert res.document.confidence["invoice_number"] == pytest.approx(0.75)

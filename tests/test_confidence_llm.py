"""Tests for the LLM self-assess path in assess_confidence.

The heuristic path (use_llm=False) is exercised by other tests; this
file pins the ``use_llm=True`` branch which combines heuristic and
LLM-self-assessed confidence (lines 71-100, 117-128 of
src/idp/assess/confidence.py).
"""
from __future__ import annotations

import json

import pytest

from idp.assess import assess_confidence
from idp.core.document import Document
from idp.llm.backend import Backend, CompletionRequest


class _FakeBackend(Backend):
    """Backend that returns a pre-canned JSON response and records calls."""

    def __init__(self, response_json: dict[str, float] | str):
        self.response_json = response_json
        self.calls: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return "fake-self-assess"

    @property
    def is_multimodal(self) -> bool:
        return False

    def complete(self, req: CompletionRequest) -> str:
        self.calls.append(req)
        if isinstance(self.response_json, dict):
            return json.dumps(self.response_json)
        return self.response_json  # raw string (may be invalid JSON)


@pytest.fixture
def doc(tmp_path) -> Document:
    f = tmp_path / "inv.txt"
    f.write_text("Invoice INV-001 from Acme. Total: $100.00")
    d = Document.from_path(str(f))
    d.extraction = {
        "invoice_number": "INV-001",
        "vendor_name": "Acme",
        "total_amount": 100.0,
    }
    return d


# ---------------------------------------------------------------------------
# use_llm=True happy path (lines 117-128)
# ---------------------------------------------------------------------------
def test_assess_with_llm_blends_heuristic_and_self_assess(doc):
    """use_llm=True returns blended confidence (0.7 * heur + 0.3 * llm)."""
    backend = _FakeBackend({
        "invoice_number": 1.0,
        "vendor_name": 0.8,
        "total_amount": 0.5,
    })
    out = assess_confidence(doc, backend=backend, use_llm=True)
    # Heuristic baseline (mode != 'multimodal' -> base 0.7):
    #   invoice_number (str):   0.7 + 0.05 = 0.75
    #   vendor_name (str):      0.7 + 0.05 = 0.75
    #   total_amount (float):   0.7
    # Then blend: 0.7 * heur + 0.3 * llm
    assert out.confidence["invoice_number"] == pytest.approx(0.7 * 0.75 + 0.3 * 1.0)
    assert out.confidence["vendor_name"] == pytest.approx(0.7 * 0.75 + 0.3 * 0.8)
    assert out.confidence["total_amount"] == pytest.approx(0.7 * 0.70 + 0.3 * 0.5)
    # Backend was called exactly once
    assert len(backend.calls) == 1


def test_assess_with_llm_handles_missing_llm_fields(doc):
    """If LLM self-assess is missing a field, use heuristic value."""
    backend = _FakeBackend({"invoice_number": 1.0})  # vendor_name + total_amount missing
    out = assess_confidence(doc, backend=backend, use_llm=True)
    # For missing LLM fields: 0.7 * heur + 0.3 * 0.5 (default)
    assert out.confidence["vendor_name"] == pytest.approx(0.7 * 0.75 + 0.3 * 0.5)
    assert out.confidence["total_amount"] == pytest.approx(0.7 * 0.70 + 0.3 * 0.5)
    # invoice_number present in both -> exact blend
    assert out.confidence["invoice_number"] == pytest.approx(0.7 * 0.75 + 0.3 * 1.0)


# ---------------------------------------------------------------------------
# Backend that returns invalid JSON -> fallback to {} (lines 98-100)
# ---------------------------------------------------------------------------
def test_assess_with_llm_returns_heuristic_on_bad_json(doc):
    """If backend returns unparseable text, self_assess returns {} (heuristic only)."""
    backend = _FakeBackend("this is not json at all {{{")
    out = assess_confidence(doc, backend=backend, use_llm=True)
    # llm_conf is empty -> blend uses 0.5 default for every field
    heur = {"invoice_number": 0.75, "vendor_name": 0.75, "total_amount": 0.70}
    for k, h in heur.items():
        assert out.confidence[k] == pytest.approx(0.7 * h + 0.3 * 0.5)


def test_assess_with_llm_handles_non_float_values(doc):
    """If backend returns a field with non-numeric value, skip that field.

    The backend returned {'invoice_number': 'high'}; the dict comprehension
    in _llm_self_assess raises ValueError -> except -> return {} ->
    blend uses 0.5 defaults everywhere.
    """
    backend = _FakeBackend({"invoice_number": "high"})
    out = assess_confidence(doc, backend=backend, use_llm=True)
    heur = {"invoice_number": 0.75, "vendor_name": 0.75, "total_amount": 0.70}
    for k, h in heur.items():
        assert out.confidence[k] == pytest.approx(0.7 * h + 0.3 * 0.5)


# ---------------------------------------------------------------------------
# use_llm=True requires a backend (lines 116-117)
# ---------------------------------------------------------------------------
def test_assess_with_llm_but_no_backend_skips_llm_path(doc):
    """If use_llm=True but backend=None, silently skip the LLM branch."""
    out = assess_confidence(doc, backend=None, use_llm=True)
    # Same as heuristic-only path
    assert out.confidence["invoice_number"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Snippet truncation (line 73): raw_text longer than 4000 chars is truncated
# ---------------------------------------------------------------------------
def test_self_assess_truncates_raw_text_to_4000_chars(tmp_path):
    """Backend only sees first 4000 chars of raw_text."""
    f = tmp_path / "long.txt"
    f.write_text("x" * 10000)
    d = Document.from_path(str(f))
    # from_path doesn't auto-load raw_text; pipeline stages set it later.
    # The self_assess helper reads doc.raw_text directly.
    d.raw_text = "x" * 10000
    d.extraction = {"field": "value"}
    backend = _FakeBackend({"field": 1.0})
    assess_confidence(d, backend=backend, use_llm=True)
    user_msg = backend.calls[0].messages[1].content
    # Should contain x * 4000 but NOT x * 4001
    assert "x" * 4000 in user_msg
    assert "x" * 4001 not in user_msg


# ---------------------------------------------------------------------------
# Heuristic: list/dict fields (lines 64-67)
# ---------------------------------------------------------------------------
def test_heuristic_downweights_list_and_dict_fields(tmp_path):
    """Heuristic applies -0.05 penalty for list and dict field values."""
    f = tmp_path / "x.txt"
    f.write_text("x")
    doc2 = Document.from_path(str(f))
    doc2.extraction = {
        "tags": ["a", "b"],          # list -> base - 0.05
        "address": {"city": "NYC"},  # dict -> base - 0.05
        "note": "hello",             # string -> base + 0.05
        "count": 42,                 # number -> base
    }
    out = assess_confidence(doc2, backend=None, use_llm=False)
    # base 0.7 (no multimodal)
    assert out.confidence["tags"] == pytest.approx(0.65)
    assert out.confidence["address"] == pytest.approx(0.65)
    assert out.confidence["note"] == pytest.approx(0.75)
    assert out.confidence["count"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Policy penalty branch (lines 124-128)
# ---------------------------------------------------------------------------
def test_assess_with_policy_applies_penalties(doc):
    """If a PolicyConfig is passed, per-field penalties are applied."""
    from idp.rl.policy import PolicyConfig
    policy = PolicyConfig(
        field_penalties={"vendor_name": 0.20},
        # invoice_number not in policy -> no penalty applied
    )
    out = assess_confidence(doc, backend=None, use_llm=False, policy=policy)
    # vendor_name heuristic was 0.75; with -0.20 penalty -> 0.55
    assert out.confidence["vendor_name"] == pytest.approx(0.55)
    # invoice_number unchanged (not in policy)
    assert out.confidence["invoice_number"] == pytest.approx(0.75)


def test_assess_with_policy_records_error_on_failure(doc):
    """If policy application raises, doc.errors records the failure."""
    from unittest.mock import patch

    # Force policy_to_penalised_confidence to raise
    with patch(
        "idp.rl.policy.policy_to_penalised_confidence",
        side_effect=RuntimeError("policy boom"),
    ):
        from idp.rl.policy import PolicyConfig
        policy = PolicyConfig()
        out = assess_confidence(doc, backend=None, use_llm=False, policy=policy)
    assert any("policy_apply_failed" in e for e in out.errors)
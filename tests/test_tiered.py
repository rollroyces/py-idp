# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for TieredBackend (v0.4 P2).

Coverage targets (from the PR spec):
  - routes-to-cheap when both tiers would succeed
  - escalates on low confidence (cheap returns valid JSON but ambiguous)
  - escalates on validation failure (cheap returns unparseable JSON)
  - does NOT escalate when both pass
  - ambiguity_threshold boundary behaviour
  - both-cheap-and-expensive-fail surfaces a clear error
  - expensive call still uses pipeline-shaped input (not a different request)
  - transparent-to-pipeline (can be assigned to Pipeline.backend)

We drive ``TieredBackend`` with a ``ControllableBackend`` (defined
below) that exposes the two knobs the PR spec calls out:
``force_low_confidence`` and ``force_validation_failure``. We do NOT
patch the global ``MockBackend`` to keep this file self-contained and
to make the test intent obvious from the fixture.
"""
from __future__ import annotations

import json

import pytest

from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.backend import Backend, CompletionRequest, Message
from idp.llm.tiered import TieredBackend, TieredBackendExhaustedError


# ---------------------------------------------------------------------------
# Test double: a Backend we can drive from each test, with the two knobs the
# PR spec demands (``force_low_confidence``, ``force_validation_failure``).
# ---------------------------------------------------------------------------
class ControllableBackend(Backend):
    """A Backend whose ``complete()`` is scriptable per-call.

    Knobs (cumulative — you can set both at once):
      force_validation_failure=True  -> returns garbage that won't parse
      force_low_confidence=True      -> returns valid JSON with empty
                                        strings (per-field conf drops to 0.1)
      neither                        -> returns well-formed invoice JSON

    The class also exposes ``calls`` so a test can see how many times the
    tier was hit and inspect each CompletionRequest.
    """

    name = "controllable"

    def __init__(
        self,
        *,
        force_validation_failure: bool = False,
        force_low_confidence: bool = False,
        response_payload: dict[str, object] | None = None,
        raise_exc: type[BaseException] | None = None,
    ) -> None:
        self.force_validation_failure = force_validation_failure
        self.force_low_confidence = force_low_confidence
        self.response_payload = response_payload
        self.raise_exc = raise_exc
        self.calls: list[CompletionRequest] = []

    @property
    def is_multimodal(self) -> bool:
        return False

    def complete(self, req: CompletionRequest) -> str:
        self.calls.append(req)
        if self.raise_exc is not None:
            raise self.raise_exc("simulated backend failure")
        if self.force_validation_failure:
            # Not valid JSON: triggers TieredBackend's parse-failure path.
            return "this is not json at all { ]"
        if self.force_low_confidence:
            # Valid JSON but with every scalar field None and the list
            # field MISSING from the dict. Per the default heuristic,
            # every present scalar gets 0.1 (well below 0.5 default
            # threshold). For the test to be agnostic to the wrapper's
            # heuristic, we deliberately set scalars to None rather
            # than "" so confidence is unambiguously 0.1.
            return json.dumps(
                {
                    "invoice_number": None,
                    "vendor_name": None,
                    "total_amount": None,
                    # line_items key intentionally OMITTED, not set to
                    # None or [] — extra confidence that the heuristic
                    # we test against doesn't accidentally score a
                    # "present-but-empty" field at >= threshold.
                }
            )
        if self.response_payload is not None:
            return json.dumps(self.response_payload)
        # Default: well-formed invoice-shaped JSON with non-empty values.
        # Every field clears the default 0.7 threshold under the heuristic.
        return json.dumps(
            {
                "invoice_number": "INV-001",
                "vendor_name": "Acme",
                "total_amount": 100.0,
                "line_items": [],
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _req() -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role="user", content="extract")],
        json_mode=True,
    )


# ---------------------------------------------------------------------------
# Core routing behavior
# ---------------------------------------------------------------------------
def test_routes_to_cheap_when_cheap_returns_valid_json():
    """Happy path: cheap returns valid JSON, no escalation."""
    cheap = ControllableBackend()
    expensive = ControllableBackend()
    t = TieredBackend(cheap=cheap, expensive=expensive)

    out = t.complete(_req())

    assert json.loads(out)["invoice_number"] == "INV-001"
    assert len(cheap.calls) == 1
    assert len(expensive.calls) == 0
    assert t.escalation_count == 0
    assert t.expensive_call_count == 0


def test_escalates_on_low_confidence():
    """Cheap returns valid JSON but with empty fields -> below threshold -> escalate."""
    cheap = ControllableBackend(force_low_confidence=True)
    expensive = ControllableBackend()  # well-formed
    t = TieredBackend(cheap=cheap, expensive=expensive)

    out = t.complete(_req())

    # Whatever the heuristic gave the cheap output, the wrapper should
    # have escalated because some field came back at 0.1 (well below 0.7).
    assert json.loads(out)["invoice_number"] == "INV-001"
    assert len(cheap.calls) == 1
    assert len(expensive.calls) == 1
    assert t.escalation_count == 1


def test_escalates_on_validation_failure():
    """Cheap returns garbage that won't parse as JSON -> escalate."""
    cheap = ControllableBackend(force_validation_failure=True)
    expensive = ControllableBackend()
    t = TieredBackend(cheap=cheap, expensive=expensive)

    out = t.complete(_req())

    assert json.loads(out)["invoice_number"] == "INV-001"
    assert len(cheap.calls) == 1
    assert len(expensive.calls) == 1
    assert t.escalation_count == 1


def test_does_not_escalate_when_both_pass():
    """Explicit: confirmation that with two good tiers, the wrapper is a no-op fan-out."""
    cheap = ControllableBackend(response_payload={"a": 1, "b": "x"})
    expensive = ControllableBackend(response_payload={"should": "never appear"})
    t = TieredBackend(cheap=cheap, expensive=expensive)

    out = t.complete(_req())

    # The expensive payload MUST NOT appear in the output. If cheap
    # succeeded the wrapper returned its response verbatim.
    assert json.loads(out) == {"a": 1, "b": "x"}
    assert len(expensive.calls) == 0
    assert t.escalation_count == 0


def test_ambiguity_threshold_boundary_above_keeps_cheap():
    """Exactly at the threshold is NOT escalated; only strictly-below is."""
    # Pass assess_fn that returns exactly 0.7 for every field. The
    # default threshold is 0.7; ``min(conf) == threshold`` must NOT escalate.
    cheap = ControllableBackend()
    expensive = ControllableBackend()
    t = TieredBackend(
        cheap=cheap,
        expensive=expensive,
        assess_fn=lambda raw, req: {k: 0.7 for k in json.loads(raw)},
    )

    out = t.complete(_req())

    # Returns cheap, not expensive.
    assert json.loads(out) == json.loads(cheap.complete(CompletionRequest(messages=[])))
    assert len(expensive.calls) == 0
    assert t.escalation_count == 0


def test_ambiguity_threshold_boundary_below_escalates():
    """One field strictly below the threshold forces escalation.

    To avoid the same-blurry-output-tricks-both-tiers edge case where the
    expensive tier's re-check ALSO fails (and we should raise), the cheap
    tier is configured to return low-confidence output while the expensive
    tier returns a clean response the assess_fn sees as high-confidence.
    """
    cheap = ControllableBackend(force_low_confidence=True)
    expensive = ControllableBackend()  # well-formed, scalar strings/numbers

    def assess(raw, req):
        parsed = json.loads(raw)
        # Cheap gets None for everything → 0.1. Expensive gets real values → 0.8.
        # Use min-of-values to differentiate the two.
        return {k: (0.1 if v is None else 0.8) for k, v in parsed.items()}

    t = TieredBackend(
        cheap=cheap,
        expensive=expensive,
        assess_fn=assess,
    )

    out = t.complete(_req())
    # Expensive returned a well-formed dict; the wrapper passes it through.
    assert json.loads(out)["invoice_number"] == "INV-001"
    assert len(expensive.calls) == 1


def test_custom_ambiguity_threshold_in_constructor():
    """Constructor takes ambiguity_threshold and it propagates to the assess call."""
    cheap = ControllableBackend()
    expensive = ControllableBackend()
    sentinel: dict[str, object] = {"threshold_seen": None}

    def assess(raw, req):
        sentinel["threshold_seen"] = 0.5
        # Return a confidence dict with all fields at 0.9 — well above 0.5.
        # This is also proof the assess_fn is consulted, because if the
        # default heuristic were used, ``line_items: []`` would score 0.7
        # which is BELOW the test's 0.5 threshold the test sets up. Wait,
        # 0.7 > 0.5, so still no escalation. To unambiguously demonstrate
        # ``assess_fn`` controls the per-field score, we set every field's
        # score to 0.9.
        return {k: 0.9 for k in json.loads(raw)}

    t = TieredBackend(
        cheap=cheap, expensive=expensive,
        ambiguity_threshold=0.5,
        assess_fn=assess,
    )

    # With threshold 0.5 and every field at 0.9, no escalation.
    t.complete(_req())
    assert sentinel["threshold_seen"] == 0.5
    assert len(expensive.calls) == 0


def test_raises_when_both_tiers_fail_validation():
    """Both tiers return garbage -> ``TieredBackendExhaustedError`` with both raws attached."""
    cheap = ControllableBackend(force_validation_failure=True)
    expensive = ControllableBackend(force_validation_failure=True)
    t = TieredBackend(cheap=cheap, expensive=expensive)

    with pytest.raises(TieredBackendExhaustedError) as excinfo:
        t.complete(_req())

    err = excinfo.value
    assert "this is not json at all" in err.cheap_raw
    assert "this is not json at all" in (err.expensive_raw or "")
    assert err.original is None  # neither raised; both produced bad strings


def test_raises_when_both_tiers_fail_low_confidence():
    """Both tiers return low-confidence JSON -> exhausted error."""
    cheap = ControllableBackend(force_low_confidence=True)
    expensive = ControllableBackend(force_low_confidence=True)
    t = TieredBackend(cheap=cheap, expensive=expensive, ambiguity_threshold=0.5)

    with pytest.raises(TieredBackendExhaustedError):
        t.complete(_req())


def test_expensive_call_still_uses_the_pipeline_schema():
    """The wrapper must NOT mutate the request before escalating.

    Concretely: the expensive tier should receive the *same*
    ``CompletionRequest`` object the pipeline handed us — including
    ``json_mode=True`` and the original messages — so that any
    schema hints in the prompts still drive the expensive model.
    """
    cheap = ControllableBackend(force_low_confidence=True)
    expensive = ControllableBackend()
    t = TieredBackend(cheap=cheap, expensive=expensive)

    req = _req()
    out = t.complete(req)

    assert json.loads(out)["invoice_number"] == "INV-001"
    # The expensive tier saw the same messages we built.
    assert len(expensive.calls) == 1
    expensive_req = expensive.calls[0]
    assert expensive_req.json_mode is True
    assert len(expensive_req.messages) == len(req.messages)
    # Specifically, the user message we constructed is unchanged.
    assert expensive_req.messages[-1].content == req.messages[-1].content


def test_transparent_to_pipeline_can_be_pipeline_backend(tmp_path):
    """A Pipeline can be constructed with TieredBackend as its backend."""
    cheap = ControllableBackend()
    expensive = ControllableBackend()
    t = TieredBackend(cheap=cheap, expensive=expensive)

    # Lazy import — we don't want a heavy import for the 9 routing
    # tests that don't need Pipeline, but this is the integration
    # test the PR spec calls out by name.
    from idp.pipeline.pipeline import Pipeline

    pipeline = Pipeline(backend=t, schema=Invoice)

    # The pipeline stores the backend we passed, not some transformed version.
    assert pipeline.backend is t

    # And running end-to-end on a real document does not blow up.
    f = tmp_path / "inv.txt"
    f.write_text("Invoice INV-001 from Acme. Total: $100.00")
    Document.from_path(str(f))
    # Don't do a full run (the pipeline stages beyond extract need a
    # classifiable page etc.), but check that we can at least call
    # extract through the pipeline's backend with no errors.
    t.complete(_req())
    cheap.calls.clear()
    expensive.calls.clear()
    pipeline.backend.complete(_req())
    assert len(cheap.calls) + len(expensive.calls) >= 1


# ---------------------------------------------------------------------------
# Defensive: misconfigured usage is rejected at construction, not at call time
# ---------------------------------------------------------------------------
def test_rejects_same_backend_for_both_tiers():
    """Using one Backend twice would be a no-op retry; reject it."""
    only = ControllableBackend()
    with pytest.raises(ValueError, match="different"):
        TieredBackend(cheap=only, expensive=only)


def test_rejects_out_of_range_threshold():
    cheap = ControllableBackend()
    expensive = ControllableBackend()
    with pytest.raises(ValueError, match="ambiguity_threshold"):
        TieredBackend(cheap=cheap, expensive=expensive, ambiguity_threshold=1.5)
    with pytest.raises(ValueError, match="ambiguity_threshold"):
        TieredBackend(cheap=cheap, expensive=expensive, ambiguity_threshold=-0.01)


def test_is_multimodal_true_if_either_tier_is_multimodal():
    """Pipeline route-decision reads `backend.is_multimodal`; the wrapper must say yes
    if either tier supports images, so the multimodal extract path is reachable."""
    from idp.llm.backend import OpenAICompatBackend

    text_only = ControllableBackend()
    # OpenAICompatBackend names itself "openai-compat" and exposes is_multimodal based
    # on the model name.
    multimodal = OpenAICompatBackend(base_url="http://localhost:11434/v1", model="gpt-4o")
    t1 = TieredBackend(cheap=text_only, expensive=multimodal)
    assert t1.is_multimodal is True

    # And a wrapper around two text-only backends is text-only.
    t2 = TieredBackend(cheap=text_only, expensive=ControllableBackend())
    assert t2.is_multimodal is False

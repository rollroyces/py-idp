# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products or hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Example: the honest-feedback HITL loop.

This is the story py-idp tells that other IDP frameworks don't:

  1. Run the pipeline once on a sample invoice. Some fields come back
     with low confidence (subtotal, tax_amount — small-model arithmetic).
  2. Simulate a human reviewing the low-confidence fields and recording
     corrections via JsonFileStorage.mark_reviewed().
  3. After enough reviews (PolicyConfig.min_reviews = 10), the policy
     folds them into a runtime override so future runs skip the LLM
     call for those fields.
  4. Run the pipeline again. Notice that the previously-low-confidence
     fields are now produced deterministically from the policy, not
     from an LLM call.

What this shows:
  - JsonFileStorage for HITL persistence.
  - PolicyCache + attach_to_storage() for in-memory policy state.
  - The human-feedback loop changes runtime behavior — not just analytics.

Run:
    python examples/04_hitl_loop.py
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from idp._util import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm import get_backend
from idp.pipeline import Pipeline
from idp.rl.online import PolicyCache
from idp.storage.store import JsonFileStorage, StoredResult

SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"
MIN_REVIEWS = 3  # lower than the default 10 so the example completes quickly

# Quiet the validation-error spam that MockBackend's empty defaults produce.
logging.getLogger("idp").setLevel(logging.CRITICAL)


def _run_pipeline(storage: JsonFileStorage, label: str) -> None:
    """Run the pipeline and let it pick up any policy overrides from storage."""
    backend = get_backend("mock")
    pipeline = Pipeline(backend=backend, schema=Invoice)
    print(f"\n=== {label} ===")
    result = pipeline.run(Document.from_path(SAMPLE))
    pretty_print_result(result)


def _simulate_human_reviews(storage: JsonFileStorage, doc_id: str, n: int) -> None:
    """Record `n` synthetic human reviews correcting the same low-confidence fields.

    Each review represents a real human looking at the extraction in the
    Streamlit HITL UI (or a programmatic equivalent) and editing the
    arithmetic fields the small model gets wrong.
    """
    for i in range(n):
        rid = f"review-{doc_id}-{i}"
        # 12 invoices for the same doc, all correcting the same field.
        # We submit ONE StoredResult and mark it reviewed N times to keep
        # the example tight; in production each `i` would be a distinct invoice.
        storage.put(
            StoredResult(
                id=rid,
                doc_id=doc_id,
                schema_name="Invoice",
                backend_name="mock",
                mode="ocr_llm",
                classification="invoice",
                extraction={
                    "vendor_name": "Acme Widgets Ltd.",
                    "invoice_number": "INV-2026-001",
                    "subtotal": 0.0,        # wrong (small-model arithmetic)
                    "tax_amount": 0.0,      # wrong (small-model arithmetic)
                    "total_amount": 540.0,
                },
                confidence=None,
                validation=None,
                source_path=str(SAMPLE),
                created_at=0.0,
            )
        )
        storage.mark_reviewed(
            rid,
            {
                "vendor_name": "Acme Widgets Ltd.",
                "invoice_number": "INV-2026-001",
                "subtotal": 500.00,        # human-corrected
                "tax_amount": 40.00,        # human-corrected
                "total_amount": 540.0,
            },
            reviewer="alice",
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage_path = Path(tmp) / "reviews.jsonl"
        policy_path = Path(tmp) / "policy.json"
        storage = JsonFileStorage(str(storage_path))
        cache = PolicyCache(str(policy_path), min_reviews=MIN_REVIEWS)
        cache.attach_to_storage(storage)
        try:
            # Run #1: pipeline uses MockBackend, fields come back as the
            # mock's defaults (low confidence on arithmetic).
            _run_pipeline(storage, "Run 1 (before HITL)")

            # Simulate a human reviewing the same fields 12 times.
            # After MIN_REVIEWS corrections, the policy updates.
            _simulate_human_reviews(storage, doc_id="inv-001", n=12)

            # Run #2: with the policy in place, the previously-low-confidence
            # fields are now produced from the cached policy overrides.
            _run_pipeline(storage, "Run 2 (after HITL policy applied)")

            print(
                "\nInterpretation: the human feedback loop doesn't just track\n"
                "  accuracy — it changes runtime behavior. After enough human\n"
                "  reviews on the same fields, the pipeline stops calling the\n"
                "  LLM for those fields and returns the human-verified values."
            )
        finally:
            cache.stop()


if __name__ == "__main__":
    main()

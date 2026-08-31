# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify stage.

Rule-first, LLM fallback. Rules let us avoid an LLM call when the
filename / first-page text already gives a strong signal. The LLM
fallback returns a confidence float so the pipeline can decide whether
to HITL the classification itself.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from idp.core.document import Document
from idp.llm.backend import Backend, CompletionRequest, Message

SCHEMA_HINTS = {
    "invoice": ("invoice", "bill to", "bill_from", "subtotal", "amount due", "due date", "invoice number", "invoice date"),
    "contract": ("agreement", "party", "term", "governing law", "expiration", "effective date", "hereafter"),
    "bank_statement": ("statement", "account", "balance", "transaction", "deposit", "withdrawal", "opening balance", "closing balance"),
    "receipt": ("receipt", "thank you", "merchant", "change", "cashier"),
    "resume": ("curriculum vitae", "experience", "education", "skills"),
    "report": ("executive summary", "findings", "conclusion", "abstract"),
}


def _rule_classify(doc: Document) -> tuple[str | None, float]:
    """Cheap regex over filename + first-page text. Case-insensitive.

    Returns (label, confidence). Label=None with confidence=0.0 -> caller
    falls back to the LLM.
    """
    name = Path(doc.source_path).stem.lower()
    text = (doc.raw_text or "")[:4000].lower()  # already lowercased
    best, score = None, 0
    for label, hints in SCHEMA_HINTS.items():
        hits = sum(1 for h in hints if h in text or h in name)
        # boost when an actual explicit header line matches (e.g. "INVOICE 123")
        if any(h in text for h in ("invoice", "agreement", "statement", "receipt")) and label == _header_label(text):
            hits += 1
        if hits > score:
            score, best = hits, label
    if score >= 1:
        # 1 hit -> 0.55 confidence, more hits climb quickly
        conf = min(0.55 + 0.15 * (score - 1), 0.99) if score >= 1 else 0.0
        # bump to 0.95 if the very first non-empty line starts with the label
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if first_line.startswith(best) or first_line.startswith(best.split("_")[0]):
            conf = max(conf, 0.95)
        return best, conf
    return None, 0.0


def _header_label(text: str) -> str:
    """Best guess of the document's top-line label."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if first.startswith("invoice"):
        return "invoice"
    if first.startswith(("agreement", "contract", "master services")):
        return "contract"
    if first.startswith(("statement", "bank")):
        return "bank_statement"
    if first.startswith("receipt"):
        return "receipt"
    return ""


def _llm_classify(doc: Document, backend: Backend) -> tuple[str, float]:
    snippet = (doc.raw_text or "")[:1500]
    req = CompletionRequest(
        messages=[
            Message(
                role="system",
                content=(
                    "You are a document classifier. Classify the snippet into exactly "
                    "one of: invoice, contract, bank_statement, receipt, resume, report, other. "
                    "Respond with JSON: {\"label\": \"...\", \"confidence\": float 0..1}."
                ),
            ),
            Message(role="user", content=snippet or "(empty document)"),
        ],
        json_mode=True,
    )
    raw = backend.complete(req)
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
        c = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        return (m.group(1) if m else "other"), float(c.group(1)) if c else 0.5
    return data.get("label", "other"), float(data.get("confidence", 0.5))


def classify_document(
    doc: Document, backend: Backend, confidence_floor: float = 0.7
) -> Document:
    """Attach doc.classification + doc.classification_confidence.

    Rule-first: if the heuristic returns a high-confidence label we keep it.
    Only call the LLM when the rule is weak (< confidence_floor) OR
    returned None. If both rule and LLM fire, keep the higher confidence.
    """
    rule_label, rule_conf = _rule_classify(doc)
    label, conf = rule_label or "other", rule_conf
    route = "rule"
    if rule_label is None or rule_conf < confidence_floor:
        try:
            llm_label, llm_conf = _llm_classify(doc, backend)
            route = "rule+llm"
            if llm_conf > conf:
                label, conf = llm_label, llm_conf
        except Exception as e:  # noqa: BLE001
            doc.errors.append(f"classify_llm_failed: {e}")
            if rule_label is None:
                label, conf = "other", 0.0
    doc.classification = label
    doc.classification_confidence = float(conf)
    doc.metadata["classification_route"] = route
    return doc

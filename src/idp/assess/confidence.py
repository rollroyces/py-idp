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

"""Confidence assessment.

Two strategies:
  - heuristic: per-field conf derived from extraction-mode + type
  - llm_self_assess: ask the model to rate each field as 0..1

LLM self-assessment is calibrated poorly in practice (papers repeatedly
show LLM confidence is overconfident). Heuristic is the default; LLM
self-assess is opt-in via config.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from idp.core.document import Document
from idp.llm.backend import Backend, CompletionRequest, Message

log = logging.getLogger(__name__)


def _heuristic_confidence(doc: Document) -> dict[str, float]:
    """Per-field confidence heuristic. Crude but reproducible.

    Heuristics:
      - multimodal mode + clean digital -> 0.9 base
      - OCR_LLM mode -> 0.7 base
      - per-field: missing/None -> 0.1
      - per-field: present value (string non-empty, number, list, dict) -> base

    NOTE: a numeric `0.0` or `0` is treated the same as any other present
    numeric value (conf = base). If you need to penalize zero/extracted-as-zero
    for compliance reasons, use `assess_confidence(use_llm=True)` or your own
    rule that down-weights fields known to be "should never be zero".
    """
    base = 0.9 if doc.mode == "multimodal" else 0.7
    out: dict[str, float] = {}
    extraction = doc.extraction or {}
    for k, v in extraction.items():
        if v is None or v == "" or v == [] or v == {}:
            out[k] = 0.1
        elif isinstance(v, (int, float)):
            out[k] = max(0.0, min(1.0, base))
        elif isinstance(v, str):
            out[k] = max(0.0, min(1.0, base + 0.05))
        elif isinstance(v, list):
            out[k] = max(0.0, min(1.0, base - 0.05))  # harder
        elif isinstance(v, dict):
            out[k] = max(0.0, min(1.0, base - 0.05))
        else:
            out[k] = base
    return out


def _llm_self_assess(doc: Document, backend: Backend) -> dict[str, float]:
    snippet = json.dumps(doc.extraction or {}, indent=2)
    sample_text = (doc.raw_text or "")[:4000]
    req = CompletionRequest(
        messages=[
            Message(
                role="system",
                content=(
                    "You rate extraction confidence per field. "
                    "Return JSON: {\"field_name\": 0.0-1.0, ...} "
                    "where 1.0 = definitely correct, 0.0 = definitely wrong or absent."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"SOURCE DOCUMENT (truncated):\n{sample_text}\n\n"
                    f"EXTRACTION TO RATE:\n{snippet}"
                ),
            ),
        ],
        json_mode=True,
        temperature=0.0,
    )
    raw = backend.complete(req)
    try:
        return {k: float(v) for k, v in json.loads(raw).items()}
    except Exception as e:  # noqa: BLE001
        log.debug("llm self-assess parse failed: %s", e)
        return {}


def assess_confidence(
    doc: Document,
    backend: Backend | None = None,
    use_llm: bool = False,
    policy: "PolicyConfig | None" = None,
) -> Document:
    """Attach doc.confidence (per-field dict of floats in 0..1).

    If a `policy` is provided (an `idp.rl.PolicyConfig`), apply per-field
    penalties so that fields humans have corrected frequently get
    lower confidence scores and surface to HITL review.
    """
    conf = _heuristic_confidence(doc)
    if use_llm and backend is not None:
        try:
            llm_conf = _llm_self_assess(doc, backend)
            # blend: 70% heuristic (more honest), 30% self-assess
            conf = {k: 0.7 * conf.get(k, 0.5) + 0.3 * llm_conf.get(k, 0.5) for k in conf}
        except Exception as e:  # noqa: BLE001
            doc.errors.append(f"assess_failed: {e}")
    if policy is not None:
        try:
            from idp.rl.policy import policy_to_penalised_confidence
            conf = policy_to_penalised_confidence(conf, policy)
        except Exception as e:  # noqa: BLE001
            doc.errors.append(f"policy_apply_failed: {e}")
    doc.confidence = conf
    return doc

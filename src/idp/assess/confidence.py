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
      - per-field: missing/None -> 0.1, non-empty string -> +0.05, numeric -> +0.0
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
) -> Document:
    """Attach doc.confidence (per-field dict of floats in 0..1)."""
    conf = _heuristic_confidence(doc)
    if use_llm and backend is not None:
        try:
            llm_conf = _llm_self_assess(doc, backend)
            # blend: 70% heuristic (more honest), 30% self-assess
            conf = {k: 0.7 * conf.get(k, 0.5) + 0.3 * llm_conf.get(k, 0.5) for k in conf}
        except Exception as e:  # noqa: BLE001
            doc.errors.append(f"assess_failed: {e}")
    doc.confidence = conf
    return doc

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

"""Mode router: pick multimodal vs OCR+LLM based on doc features.

Empirical finding (arXiv 2509.04469 + 2510.15727, 2025):
  - multimodal VLMs win on clean digital PDFs of <= ~25 pages
  - OCR+LLM wins on noisy scans, complex tables, alphanumeric patterns (IBAN)
"""
from __future__ import annotations

import re

from idp.core.document import Document
from idp.core.types import ExtractionMode

# Tokens that strongly suggest OCR+LLM is safer
_NOISY_TOKENS = (
    "iban",
    "swift",
    "bic",
    "tax id",
    "vat",
    "hs code",
    "passport",
    "vin:",
)
_COMPLEX_TABLE_TOKENS = ("line item", "sku", "qty", "quantity", "unit price", "amount", "subtotal")


def _has_complex_tables(text: str) -> bool:
    """Heuristic: count row-like markdown patterns in the first 4KB."""
    sample = text[:4000].lower()
    hits = sum(sample.count(t) for t in _COMPLEX_TABLE_TOKENS)
    return hits >= 3


def _looks_noisy(text: str) -> bool:
    sample = text[:4000].lower()
    return any(t in sample for t in _NOISY_TOKENS)


def _is_clean_digital(text: str) -> bool:
    """If the parser extracted clean unicode text with reasonable density,
    assume it's a digital PDF (not a scan).
    """
    if not text:
        return False
    printable = sum(c.isprintable() for c in text)
    return (printable / max(len(text), 1)) > 0.95


def choose_mode(doc: Document, backend_is_multimodal: bool = True) -> ExtractionMode:
    """Pick the extraction strategy for this document.

    Returns OCR_LLM if any of:
      - page_count > 25
      - document looks noisy (IBAN/VIN/etc.)
      - tables look complex (line items + prices)
      - backend is text-only
    Otherwise MULTIMODAL.
    """
    text = doc.raw_text or ""
    if not backend_is_multimodal:
        return ExtractionMode.OCR_LLM
    if doc.page_count > 25:
        return ExtractionMode.OCR_LLM
    if _looks_noisy(text):
        return ExtractionMode.OCR_LLM
    if _has_complex_tables(text):
        return ExtractionMode.OCR_LLM
    if not _is_clean_digital(text):
        return ExtractionMode.OCR_LLM
    return ExtractionMode.MULTIMODAL

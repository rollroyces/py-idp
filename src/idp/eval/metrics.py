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

"""Metrics for the eval harness.

Implements:
  - schema_valid_rate:           did Pydantic accept the extraction
  - field_recall / field_precision / field_f1 (micro)
  - exact-match per field
"""
from __future__ import annotations

from typing import Any, Iterable


def _norm(v: Any) -> Any:
    """Normalize values for equality check (lowercase strings, 2dp floats)."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def field_match(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, bool]:
    """Per-field exact-match (normalized)."""
    return {k: _norm(predicted.get(k)) == _norm(gold.get(k)) for k in gold}


def field_scores(per_doc: list[dict[str, bool]]) -> dict[str, float]:
    """Micro-averaged precision/recall/F1 across documents.

    `per_doc` is a list of per-document `field_match()` outputs, one
    per doc, each mapping gold_field_name -> bool (exact match).

    Edge cases:
      - empty input returns all zeros with n=0
      - boolean `False` is counted as a False Negative (the field was
        present in gold and the model didn't match it).
    """
    if not per_doc:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    # union() over possibly-empty iterators is well-defined but an empty
    # per_doc would crash here — guarded above.
    keys: set[str] = set().union(*(d.keys() for d in per_doc))
    tp = fp = fn = 0
    for d in per_doc:
        for k in keys:
            # gold == present in keys; we treat absence as a separate FN below
            if k in d:
                if d[k]:
                    tp += 1
                else:
                    fn += 1
            else:
                # model returned no key for a gold field -> FN
                fn += 1
    # FP only counts extra keys the model emitted beyond the gold set
    for d in per_doc:
        for k in d.keys() - keys:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(per_doc)}


def schema_valid(extractions: Iterable[Any], schema_validator) -> float:
    """% of extractions the schema validates (compares via the caller's validator)."""
    exts = list(extractions)
    if not exts:
        return 0.0
    n = sum(1 for e in exts if schema_validator(e))
    return n / len(exts)

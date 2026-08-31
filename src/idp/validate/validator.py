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

"""Validate stage.

Two passes:
  1. Pydantic schema check (already attempted at extract; redo here as canonical)
  2. User business rules — predicate functions attached via `add_rule`
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from idp.core.document import Document


BusinessRule = Callable[[dict[str, Any]], tuple[bool, str | None]]
"""
A BusinessRule takes the extracted dict and returns (passed, reason_if_failed).
It must NOT raise. Wrap your own exceptions internally.
"""


def _validate_pydantic(doc: Document) -> dict[str, Any]:
    """Re-run Pydantic validation if a schema is registered by name."""
    if not doc.extraction_schema:
        return {"schema_valid": None, "errors": []}
    try:
        from idp.core.schemas import get_schema

        schema = get_schema(doc.extraction_schema)
        schema.model_validate(doc.extraction)
        return {"schema_valid": True, "errors": []}
    except Exception as e:  # noqa: BLE001
        return {"schema_valid": False, "errors": [str(e)]}


def validate(doc: Document, rules: list[BusinessRule] | None = None) -> Document:
    """Attach doc.validation = {schema_valid, errors, rule_results}."""
    result: dict[str, Any] = _validate_pydantic(doc)
    rule_results: list[dict[str, Any]] = []
    if rules:
        for i, rule in enumerate(rules):
            try:
                ok, reason = rule(doc.extraction or {})
                rule_results.append({"rule": rule.__name__, "passed": ok, "reason": reason})
            except Exception as e:  # noqa: BLE001
                rule_results.append(
                    {"rule": rule.__name__, "passed": False, "reason": f"rule raised: {e}"}
                )
    result["rule_results"] = rule_results
    result["passed"] = bool(result.get("schema_valid")) and all(r["passed"] for r in rule_results)
    doc.validation = result
    return doc


# ---------------------------------------------------------------------------
# Built-in rule helpers
# ---------------------------------------------------------------------------
def required_fields_rule(*fields: str) -> BusinessRule:
    """Fail if any of the named fields is None or empty in the extraction."""

    def _rule(extraction: dict[str, Any]) -> tuple[bool, str | None]:
        missing = [f for f in fields if extraction.get(f) in (None, "", [], {})]
        return (not missing), f"missing required fields: {missing}" if missing else None

    _rule.__name__ = f"required_{'_'.join(fields)}"
    return _rule


def numeric_range_rule(field: str, min_v: float, max_v: float) -> BusinessRule:
    """Fail if the named numeric field is outside [min_v, max_v]."""

    def _rule(extraction: dict[str, Any]) -> tuple[bool, str | None]:
        v = extraction.get(field)
        if v is None:
            return True, None
        if not isinstance(v, (int, float)):
            return False, f"{field} is not numeric ({v!r})"
        if v < min_v or v > max_v:
            return False, f"{field}={v} out of range [{min_v}, {max_v}]"
        return True, None

    _rule.__name__ = f"range_{field}"
    return _rule

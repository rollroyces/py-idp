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

"""Run the eval harness against a labeled dataset.

Dataset layout:
  src/idp/eval/datasets/<name>/
      cases.jsonl         # one JSON per line: {doc_path, schema, gold}
      docs/               # supporting files referenced by doc_path

Each `cases.jsonl` line:
  {"doc_path": "docs/inv-001.txt",
   "schema":  "Invoice",
   "gold":    {"invoice_number": "INV-001", "total_amount": 100.00, ...}}

Usage:
  from idp.eval.runner import run_dataset
  res = run_dataset("src/idp/eval/datasets/invoices",
                    strategies=["mock", "ollama"])
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from idp.core.document import Document
from idp.core.schemas import SCHEMA_REGISTRY
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline

log = logging.getLogger(__name__)


def _load_cases(dataset_dir: Path) -> list[dict[str, Any]]:
    cases_file = dataset_dir / "cases.jsonl"
    if not cases_file.exists():
        raise FileNotFoundError(f"no cases.jsonl in {dataset_dir}")
    out = []
    for line in cases_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _schema_validator(schema_name: str):
    schema = SCHEMA_REGISTRY[schema_name]

    def _ok(extraction):
        try:
            schema.model_validate(extraction)
            return True
        except Exception:
            return False

    return _ok


def run_dataset(dataset_dir, strategies: list[str]) -> dict[str, Any]:
    """Run each strategy against the dataset. Returns a metrics dict."""
    from idp.eval.metrics import field_match, field_scores

    dataset_dir = Path(dataset_dir)
    cases = _load_cases(dataset_dir)
    rows: list[dict[str, Any]] = []
    detail: dict[str, list[dict[str, Any]]] = {}

    for strat in strategies:
        backend = get_backend(strat)
        per_doc_matches: list[dict[str, bool]] = []
        per_doc_valid: list[bool] = []
        total_seconds = 0.0
        per_doc_rows: list[dict[str, Any]] = []

        for case in cases:
            schema_name = case["schema"]
            gold = case.get("gold") or {}
            doc_path = dataset_dir / case["doc_path"]
            if not doc_path.exists():
                log.warning("missing: %s", doc_path)
                continue
            doc = Document.from_path(str(doc_path))
            pipe = Pipeline(backend=backend, schema=schema_name)
            t = time.perf_counter()
            try:
                result = pipe.run(doc)
            except Exception as e:  # noqa: BLE001
                log.warning("pipeline failed on %s: %s", doc_path, e)
                continue
            total_seconds += time.perf_counter() - t
            matches = field_match(result.document.extraction or {}, gold)
            per_doc_matches.append(matches)
            per_doc_valid.append(_schema_validator(schema_name)(result.document.extraction or {}))
            per_doc_rows.append(
                {
                    "doc": case["doc_path"],
                    "extraction": result.document.extraction,
                    "gold": gold,
                    "field_match": matches,
                    "validation": result.document.validation,
                    "errors": result.document.errors,
                }
            )

        scores = field_scores(per_doc_matches)
        schema_valid_rate = (
            sum(per_doc_valid) / len(per_doc_valid) if per_doc_valid else 0.0
        )
        rows.append(
            {
                "strategy": strat,
                "schema_valid_rate": schema_valid_rate,
                "field_f1": scores["f1"],
                "field_precision": scores["precision"],
                "field_recall": scores["recall"],
                "n_docs": len(per_doc_rows),
                "avg_sec_per_doc": total_seconds / max(len(per_doc_rows), 1),
            }
        )
        detail[strat] = per_doc_rows

    return {"rows": rows, "detail": detail, "dataset": str(dataset_dir), "strategies": strategies}

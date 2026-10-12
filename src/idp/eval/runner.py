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


__all__ = [
    "EvalRegressionError",
    "check_regression",
    "format_report",
    "run_dataset",
    "verify_against_thresholds",
]


class EvalRegressionError(Exception):
    """Raised when eval results violate configured regression thresholds.

    Use this in CI: catch the exception to print a failure summary, or let
    it propagate to abort the build. The message lists every failing
    threshold so logs are self-explanatory.
    """


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
    """Run each strategy against the dataset. Returns a metrics dict.

    Returns
    -------
    dict
        Backward-compatible keys: ``rows`` (list, per-strategy), ``detail``
        (per-strategy details), ``dataset`` (path), ``strategies`` (list).

        New keys:

        - ``per_strategy``: ``{strategy_name: {"schema_valid_rate": float,
          "field_match_rate": float, "average_latency_s": float,
          "details": [...]}}``
        - ``per_field_accuracy``: ``{field_name: {"match": int, "total": int,
          "accuracy": float}}`` aggregated across all cases AND all strategies
          (so the report is per-field overall, not per-strategy).
    """
    from idp.eval.metrics import field_match, field_scores

    dataset_dir = Path(dataset_dir)
    cases = _load_cases(dataset_dir)
    rows: list[dict[str, Any]] = []
    detail: dict[str, list[dict[str, Any]]] = {}
    per_strategy: dict[str, dict[str, Any]] = {}
    field_match_count: dict[str, int] = {}
    field_total_count: dict[str, int] = {}

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
            per_doc_valid.append(
                _schema_validator(schema_name)(result.document.extraction or {})
            )
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
            # Accumulate per-field accuracy across (doc, strategy) pairs.
            for field_name, is_match in matches.items():
                field_total_count[field_name] = (
                    field_total_count.get(field_name, 0) + 1
                )
                if is_match:
                    field_match_count[field_name] = (
                        field_match_count.get(field_name, 0) + 1
                    )

        scores = field_scores(per_doc_matches)
        schema_valid_rate = (
            sum(per_doc_valid) / len(per_doc_valid) if per_doc_valid else 0.0
        )
        row = {
            "strategy": strat,
            "schema_valid_rate": schema_valid_rate,
            "field_f1": scores["f1"],
            "field_precision": scores["precision"],
            "field_recall": scores["recall"],
            "n_docs": len(per_doc_rows),
            "avg_sec_per_doc": total_seconds / max(len(per_doc_rows), 1),
        }
        rows.append(row)
        detail[strat] = per_doc_rows
        per_strategy[strat] = {
            "schema_valid_rate": schema_valid_rate,
            "field_match_rate": scores["f1"],
            "average_latency_s": row["avg_sec_per_doc"],
            "details": per_doc_rows,
        }

    per_field_accuracy: dict[str, dict[str, Any]] = {}
    for field_name, total in field_total_count.items():
        match_n = field_match_count.get(field_name, 0)
        per_field_accuracy[field_name] = {
            "match": match_n,
            "total": total,
            "accuracy": (match_n / total) if total > 0 else 0.0,
        }

    return {
        "rows": rows,
        "detail": detail,
        "dataset": str(dataset_dir),
        "strategies": strategies,
        "per_strategy": per_strategy,
        "per_field_accuracy": per_field_accuracy,
    }


def _lookup_path(results: dict[str, Any], parts: list[str]) -> Any:
    """Walk `results` by `parts`. Returns sentinel `...` if any step fails."""
    value: Any = results
    for p in parts:
        if isinstance(value, dict) and p in value:
            value = value[p]
        else:
            return _MISSING
    return value


# Sentinel used by check_regression to avoid clashing with valid data.
_MISSING = object()


def check_regression(
    results: dict[str, Any],
    thresholds: dict[str, float],
) -> list[str]:
    """Return a list of regression error messages.

    `thresholds` keys can be:
        'per_strategy.<name>.schema_valid_rate': minimum acceptable rate
        'per_strategy.<name>.field_match_rate':  minimum acceptable rate
        'per_field_accuracy.<field>.accuracy':   minimum acceptable per-field rate

    Returns a list of strings describing each failed threshold. Empty
    list = all good. Use this in CI: non-empty result fails the build.

    Unknown metric paths are silently skipped — they don't crash the
    gate, they're just ignored. This lets the CI config reference
    optional strategies/fields without breaking when they're absent.
    """
    failures: list[str] = []
    for key, threshold in thresholds.items():
        parts = key.split(".")
        value = _lookup_path(results, parts)
        if value is _MISSING:
            # Unknown metric path — silently skip (don't crash)
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            # Non-numeric value (e.g. dict, list) — skip
            continue
        if value < threshold:
            # Render the path's leaf name plus the failing metric for readable logs.
            leaf = parts[-1] if parts else key
            owner = ".".join(parts[:-1])
            failures.append(
                f"{owner}: {leaf}={value} < threshold {threshold}"
            )
    return failures


def verify_against_thresholds(
    results: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    """Raise :class:`EvalRegressionError` if any threshold is violated.

    Convenience wrapper around :func:`check_regression` for use in CI
    scripts that prefer exception flow over return-value handling.
    """
    failures = check_regression(results, thresholds)
    if failures:
        joined = "\n  - ".join(failures)
        raise EvalRegressionError(
            f"Regression thresholds violated:\n  - {joined}"
        )


def format_report(results: dict[str, Any]) -> str:
    """Pretty-print a results dict as a multi-line string.

    Suitable for printing to a console or attaching to CI logs.
    """
    per_strategy = results.get("per_strategy", {}) or {}
    per_field_accuracy = results.get("per_field_accuracy", {}) or {}

    n_strategies = len(per_strategy)
    n_cases = 0
    if per_strategy:
        first = next(iter(per_strategy.values()))
        n_cases = len(first.get("details", []) or [])

    lines = [
        f"Eval Report ({n_cases} cases, {n_strategies} strategies)",
        "====================================",
    ]
    for strat_name, metrics in per_strategy.items():
        svr = float(metrics.get("schema_valid_rate", 0.0) or 0.0)
        fmr = float(metrics.get("field_match_rate", 0.0) or 0.0)
        lat = float(metrics.get("average_latency_s", 0.0) or 0.0)
        lines.append(f"Strategy '{strat_name}':")
        lines.append(f"  schema_valid_rate: {svr:.2f}")
        lines.append(f"  field_match_rate:  {fmr:.2f}")
        lines.append(f"  average_latency_s: {lat:.2f}")

    lines.append("")
    lines.append("Per-field accuracy:")
    if not per_field_accuracy:
        lines.append("  (no fields)")
    else:
        max_name = max(len(name) for name in per_field_accuracy)
        for field_name, info in per_field_accuracy.items():
            match_n = int(info.get("match", 0))
            total = int(info.get("total", 0))
            acc = float(info.get("accuracy", 0.0) or 0.0)
            lines.append(
                f"  {field_name:<{max_name}}: {match_n}/{total} = {acc:.2f}"
            )

    return "\n".join(lines)

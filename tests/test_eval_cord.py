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

"""Tests for the CORD eval (P4 / C2 of v0.4).

These tests cover:

* the manifest loads cleanly and has the documented shape
* the adapter turns the manifest into the runner's ``cases.jsonl``
  shape without losing fields
* the eval runner produces a metrics row on the staged dataset
* the MockBackend baseline is published (i.e. near zero — that is
  the *expected* behaviour for the offline baseline, by design)
* missing files warn-and-skip, they do not crash
* an empty extraction from a mock results in a zerostate (no
  division-by-zero, valid numbers) — guarded by
  ``field_scores({})`` returning 0.0s
* every gold entry in the manifest validates against the
  ``Receipt`` Pydantic schema

The real-backend slow test (``tests/eval_cord/test_eval_real_dataset.py``)
is intentionally NOT in this file. It lives in a separate module
so ``pytest tests/`` (default CI) does not pick it up — see that
file for the ``@pytest.mark.slow`` registration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from idp.core.schemas import Receipt
from idp.eval.metrics import field_match, field_scores
from idp.eval.runner import run_dataset
from tests.eval_cord import (
    cord_dataset_dir,
    load_manifest,
    to_runner_cases,
    write_cases_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_manifest()


@pytest.fixture(scope="module")
def staged_dataset(manifest) -> Path:
    """Materialise the manifest into a runner-shaped temp dataset."""
    return write_cases_jsonl(manifest, target_dir=None)


# ---------------------------------------------------------------------------
# 1. Dataset loads from the manifest
# ---------------------------------------------------------------------------
def test_dataset_loads_from_manifest(manifest) -> None:
    """The manifest is parseable JSON and lists at least 10 docs."""
    assert manifest["dataset"] == "cord"
    assert manifest["schema"] == "Receipt"
    n = manifest["n_docs"]
    assert n == len(manifest["docs"])
    assert n >= 10, f"CORD eval should ship ≥10 docs (got {n})"
    # every entry has the runner-shape keys
    for d in manifest["docs"]:
        assert "doc_path" in d
        assert "schema" in d
        assert d["schema"] == "Receipt"
        assert "gold" in d
        # every doc resolves on disk (relative to the dataset dir)
        path = cord_dataset_dir() / d["doc_path"]
        assert path.exists(), f"manifest doc missing on disk: {path}"


# ---------------------------------------------------------------------------
# 2. Eval runner produces metrics on the CORD set
# ---------------------------------------------------------------------------
def test_eval_runner_produces_metrics(staged_dataset) -> None:
    """``run_dataset`` returns the documented shape on CORD."""
    res = run_dataset(staged_dataset, strategies=["mock"])
    assert "rows" in res
    assert "detail" in res
    assert res["dataset"].endswith("cord") or "cord" in res["dataset"]
    assert res["strategies"] == ["mock"]
    rows = res["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["strategy"] == "mock"
    # the documented metric keys
    for key in (
        "schema_valid_rate",
        "field_f1",
        "field_precision",
        "field_recall",
        "n_docs",
        "avg_sec_per_doc",
    ):
        assert key in row, f"missing metric key: {key}"
    assert row["n_docs"] >= 10, "should have run on ≥10 docs"
    # schema-valid-rate and F1 are 0..1 floats
    for k in ("schema_valid_rate", "field_f1", "field_precision", "field_recall"):
        v = row[k]
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0, f"{k} out of [0, 1]: {v}"
    # detail contains one entry per strategy, with one row per doc
    assert len(res["detail"]["mock"]) == row["n_docs"]


# ---------------------------------------------------------------------------
# 3. MockBackend baseline is published (near-zero, by design)
# ---------------------------------------------------------------------------
def test_mock_backend_baseline_published(staged_dataset) -> None:
    """The MockBackend returns empty Pydantic-style JSON, so field
    F1 is near zero. This is the expected offline baseline — the
    real-backend slow test is what users run to see non-zero numbers.

    We assert the *shape* (numbers in [0, 1], F1 == 0 because no
    fields matched) rather than the precise value, so a future
    MockBackend tweak that still yields zero F1 does not break the
    test.
    """
    res = run_dataset(staged_dataset, strategies=["mock"])
    row = res["rows"][0]
    # The MockBackend's "ideal" mode returns `{}` for extraction;
    # field_match returns False for every gold key, so F1 is 0.
    assert row["field_f1"] == pytest.approx(0.0, abs=0.001)
    assert row["field_recall"] == pytest.approx(0.0, abs=0.001)
    # Precision is also 0: no field the model emitted matched a gold one
    assert row["field_precision"] == pytest.approx(0.0, abs=0.001)
    # The mock produces a syntactically-valid but empty Receipt, so
    # Pydantic validation rejects it (merchant_name is required) and
    # schema_valid_rate should be 0 as well.
    assert row["schema_valid_rate"] == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# 4. Missing files warn-and-skip (no crash)
# ---------------------------------------------------------------------------
def test_missing_files_warn_not_crash(staged_dataset, caplog) -> None:
    """A manifest entry pointing at a non-existent doc must warn and skip,
    not raise.

    We synthesise a temp dataset with one extra case whose doc_path
    does not resolve, run the eval, and assert: no exception, the
    missing case is absent from the detail rows, and a warning was
    logged at WARNING level by the runner.
    """
    extra_dir = staged_dataset  # reuse the staged dir
    cases_file = extra_dir / "cases.jsonl"
    cases = [json.loads(line) for line in cases_file.read_text().splitlines() if line.strip()]
    # add a ghost case
    cases.append(
        {"doc_path": "docs/receipt-does-not-exist.txt", "schema": "Receipt", "gold": {}}
    )
    cases_file.write_text("\n".join(json.dumps(c) for c in cases) + "\n")

    with caplog.at_level(logging.WARNING, logger="idp.eval.runner"):
        res = run_dataset(extra_dir, strategies=["mock"])
    # Did not raise — successful path.
    assert res["rows"], "expected at least one row"
    # The ghost case is NOT in the detail list (it was skipped).
    doc_paths = {d["doc"] for d in res["detail"]["mock"]}
    assert "docs/receipt-does-not-exist.txt" not in doc_paths
    # A WARNING was emitted by the runner with the missing path
    warnings = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING and "missing" in rec.getMessage().lower()
    ]
    assert warnings, "expected at least one 'missing' warning"


# ---------------------------------------------------------------------------
# 5. Empty extraction produces a zerostate (no division-by-zero)
# ---------------------------------------------------------------------------
def test_empty_extraction_skips_with_zerostate() -> None:
    """``field_scores`` on an empty per-doc list returns all zeros.

    The MockBackend emits ``{}`` for every prompt, so every doc's
    extraction is empty. The eval runner must not divide by zero
    when computing precision / recall / F1 over those rows.
    """
    # empty per-doc list -> 0/0 -> 0.0 across the board
    s = field_scores([])
    assert s == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}

    # per-field match with empty gold -> nothing to match
    matches = field_match({}, {"merchant_name": "Acme"})
    assert matches == {"merchant_name": False}

    # per-field match with empty extraction against non-empty gold -> all False
    matches = field_match({"merchant_name": None}, {"merchant_name": "Acme"})
    assert matches == {"merchant_name": False}


# ---------------------------------------------------------------------------
# 6. Ground-truth shape validates against the Receipt schema
# ---------------------------------------------------------------------------
def test_ground_truth_shape_validates_against_receipt_schema(manifest) -> None:
    """Every gold dict in the manifest is a valid ``Receipt``.

    This is the line of defence against typo'd gold values: a
    schema-violating entry would silently be a 0 from the runner
    (and look like a model miss), so we catch it here.
    """
    bad: list[tuple[str, str]] = []
    for entry in manifest["docs"]:
        gold = entry["gold"]
        try:
            Receipt(**gold)
        except Exception as e:  # noqa: BLE001
            bad.append((entry["doc_path"], str(e)))
    assert not bad, f"manifest gold entries failed Receipt validation: {bad}"


# ---------------------------------------------------------------------------
# Bonus coverage — the adapter turns manifest docs into runner cases
# ---------------------------------------------------------------------------
def test_manifest_adapter_roundtrip(manifest) -> None:
    """``to_runner_cases`` preserves ``doc_path``, ``schema``, ``gold``."""
    cases = to_runner_cases(manifest)
    assert len(cases) == manifest["n_docs"]
    for entry, case in zip(manifest["docs"], cases, strict=True):
        assert case["doc_path"] == entry["doc_path"]
        assert case["schema"] == entry["schema"]
        assert case["gold"] == entry["gold"]


# ---------------------------------------------------------------------------
# Bonus coverage — at least 10 receipts on disk (the spec's hard floor)
# ---------------------------------------------------------------------------
def test_at_least_ten_receipts_on_disk() -> None:
    docs = sorted((cord_dataset_dir() / "docs").glob("receipt-*.txt"))
    assert len(docs) >= 10, f"need ≥10 receipts, found {len(docs)}"


# ---------------------------------------------------------------------------
# Bonus coverage — README, LICENSE, ATTRIBUTION present (license audit)
# ---------------------------------------------------------------------------
def test_license_attribution_present() -> None:
    ds = cord_dataset_dir()
    for name in ("README.md", "LICENSE.txt", "ATTRIBUTION.md", "manifest.json"):
        assert (ds / name).exists(), f"{name} missing from CORD dataset dir"
    # ATTRIBUTION.md contains the BibTeX key
    assert "@inproceedings{cord2020" in (ds / "ATTRIBUTION.md").read_text()
    # LICENSE.txt contains the CC-BY-4.0 marker
    assert "Attribution 4.0 International" in (ds / "LICENSE.txt").read_text()
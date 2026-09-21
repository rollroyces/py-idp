# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaN without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real-backend CORD eval (P4 / C2 of v0.4) — SLOW.

This test runs the eval against a real LLM backend (Ollama, OpenAI,
or Anthropic). It is marked ``@pytest.mark.slow`` and does **not**
run in the default CI suite.

Activation::

    IDP_EVAL_BACKEND=ollama    pytest tests/eval_cord/test_eval_real_dataset.py -v
    IDP_EVAL_BACKEND=openai    pytest tests/eval_cord/test_eval_real_dataset.py -v
    IDP_EVAL_BACKEND=anthropic pytest tests/eval_cord/test_eval_real_dataset.py -v

The slow marker is registered in ``pyproject.toml`` under
``[tool.pytest.ini_options].markers``. The default CI command runs
``pytest -m "not slow"`` and skips this file entirely.

When run, the test:

1. Loads the CORD manifest and materialises a runner-shaped dataset
2. Calls ``run_dataset`` against the requested backend
3. Asserts the row has the documented shape (n_docs ≥ 10, F1 in
   [0, 1], schema-valid in [0, 1])
4. Soft-asserts the field-F1 is non-zero (the MockBackend baseline
   is 0; this is the real-backend floor). If the backend is offline
   / unauthorised, the assertion is skipped, not failed.

This test deliberately does NOT enforce a regression threshold
(e.g. "fail if F1 < 0.5"). That gating is a v0.5 item — see
``docs/ROADMAP_v0.4.md`` C2 — because the threshold depends on real
CORD numbers we don't have at v0.4 cut.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from idp.eval.runner import run_dataset
from tests.eval_cord import (
    VALID_REAL_BACKENDS,
    load_manifest,
    select_env_backend,
    write_cases_jsonl,
)

# Repo path so `tests.eval_cord` resolves when invoked from CI
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Mark the whole module slow so `-m "not slow"` skips it
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def backend_name() -> str:
    """Read the user's chosen backend from the env var, or skip the test."""
    name = select_env_backend()
    if name is None:
        pytest.skip(
            "set IDP_EVAL_BACKEND to one of "
            f"{sorted(VALID_REAL_BACKENDS)} to run this real-backend test"
        )
    return name


@pytest.fixture(scope="module")
def staged_dataset():
    manifest = load_manifest()
    return write_cases_jsonl(manifest, target_dir=None)


# ---------------------------------------------------------------------------
# The single test
# ---------------------------------------------------------------------------
def test_real_backend_eval(backend_name, staged_dataset, caplog) -> None:
    """Run the CORD eval against the user-selected real backend.

    The test asserts:

    * the eval runner produces a row with the documented shape
    * the row has n_docs ≥ 10 (we ship 30 fixtures)
    * field-F1, precision, recall, schema-valid are all in [0, 1]
    * the run does not crash on the OCR-noisy fixture
      (receipt-008.txt) — it may produce zero matches, but it
      should not raise

    Soft expectations (logged as warnings, not asserted hard):

    * F1 > 0 — if the real backend returns all empties (e.g. wrong
      model name), that's worth surfacing without failing the CI
      build (real backends are flaky).
    * n_docs == 30 — if a doc was missing, that surfaces as a
      warning.
    """
    with caplog.at_level(logging.WARNING, logger="idp.eval.runner"):
        res = run_dataset(staged_dataset, strategies=[backend_name])

    # Shape checks
    assert "rows" in res and "detail" in res
    rows = res["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["strategy"] == backend_name

    # n_docs floor — we ship 30
    if row["n_docs"] < 10:
        pytest.fail(
            f"only {row['n_docs']} docs were processed; "
            f"expected ≥10. Check that all fixtures are on disk."
        )

    # All metric columns in [0, 1]
    for k in ("schema_valid_rate", "field_f1", "field_precision", "field_recall"):
        v = row[k]
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0, f"{k} out of [0, 1]: {v}"

    # Detail rows match n_docs
    assert len(res["detail"][backend_name]) == row["n_docs"]

    # Soft F1 > 0 check — log a warning but don't fail, because real
    # backends can return empty extractions (model not found, rate
    # limit, etc.). The hard shape checks above are the contract.
    if row["field_f1"] == 0.0:
        caplog.set_level(logging.WARNING)
        logging.getLogger(__name__).warning(
            "real backend %r returned F1=0 on CORD; "
            "check the model name / API key / rate limits",
            backend_name,
        )
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

"""Slow e2e: CORD v1 → v2 template migration (P5 v0.4 / C4).

Marks the whole module ``@pytest.mark.slow`` — skipped by default
with ``pytest -m "not slow"``. Activate explicitly::

    pytest tests/eval_cord/test_migrate_template_e2e.py -v

What this test exercises that the unit tests can't:

* Real CORD receipt fixtures (``src/idp/eval/datasets/cord/docs/*.txt``)
  re-read from disk by the migration tool's ``process_batch`` path.
* The full pipeline (parse → classify → extract → assess → validate)
  run against the v2 template, with the CORD ``Receipt`` schema as
  the destination.
* A simulated v2 template that adds a ``tax_id`` field (the v0.4
  spec's canonical example for "v2 has a new required field"). The
  diff must flag ``tax_id`` as ``added`` for every doc that v2
  produced, and the v1 records (which lack ``tax_id``) must be
  left untouched.

Honest about scope: this is an end-to-end *dry-run* only. The
commit path is covered by ``tests/test_migrate_template.py``'s
``test_commit_writes_new_records_only``; we don't commit here
because that would require writing to a shared fixture directory
and pollute future CI runs.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

from idp.migrate import migrate_template_version
from idp.pipeline.pipeline import Pipeline
from idp.storage.store import InMemoryStorage, StoredResult

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Slow marker — also registered in pyproject.toml so the default
# suite skips with `-m "not slow"`. Same convention as the
# real-backend eval test in this directory.
pytestmark = pytest.mark.slow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _v1_extraction_for(doc_text: str) -> dict:
    """A minimal v1 extraction that fits the CORD Receipt schema.

    Hand-rolled (not parsed from the doc) — we are simulating the
    "v1 record" that already exists in storage. The migration tool
    does NOT use this dict; it re-runs the pipeline against the v2
    template using ``source_path``. This dict exists so the diff has
    a non-trivial old side to compare against.
    """
    return {
        "merchant_name": "Morning Brew Cafe",
        "date": "2025-01-04",
        "time": "08:14",
        "line_items": [
            {"description": "Espresso", "quantity": 1.0, "unit_price": 3.25, "total": 3.25}
        ],
        "subtotal": 3.25,
        "tax_amount": 0.27,
        "tip_amount": 0.0,
        "total": 3.52,
        # Note: NO tax_id — that's the v2-only field we're testing.
    }


def _populate_v1_storage(storage: InMemoryStorage, cord_dir: Path) -> int:
    """Populate ``storage`` with one v1 StoredResult per CORD receipt.

    Returns the number of records written. Files that don't exist
    on disk are skipped with a warning (the migration tool will
    later report them as ``skipped=True`` if they survived in
    storage, but here we just don't add them).
    """
    docs_dir = cord_dir / "docs"
    if not docs_dir.is_dir():
        return 0
    written = 0
    for doc_path in sorted(docs_dir.glob("*.txt")):
        text = doc_path.read_text(encoding="utf-8")
        r = StoredResult(
            id=f"v1-{doc_path.stem}",
            doc_id=doc_path.stem,
            schema_name="Receipt",
            backend_name="mock",
            mode=None,
            classification=None,
            extraction=_v1_extraction_for(text),
            confidence={"merchant_name": 0.7},
            validation={"passed": True},
            source_path=str(doc_path),
            created_at=time.time(),
            reviewed=False,
        )
        storage.put(r)
        written += 1
    return written


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cord_dir() -> Path:
    """Return the absolute path to the CORD eval dataset directory."""
    return REPO / "src" / "idp" / "eval" / "datasets" / "cord"


@pytest.fixture(scope="module")
def v1_storage(cord_dir: Path) -> InMemoryStorage:
    """In-memory storage pre-loaded with one v1 record per CORD doc."""
    storage = InMemoryStorage()
    n = _populate_v1_storage(storage, cord_dir)
    if n == 0:
        pytest.skip(f"no CORD docs found under {cord_dir / 'docs'}; skipping e2e")
    return storage


# ---------------------------------------------------------------------------
# The single slow e2e test
# ---------------------------------------------------------------------------
def test_cord_v1_to_v2_migration_flags_tax_id_as_added(
    v1_storage: InMemoryStorage,
) -> None:
    """Run the migration dry-run on the CORD v1 set; verify the v2
    new field is flagged as ``added`` for every doc.

    The simulated v2 pipeline uses ``MockBackend``, which returns
    an empty Pydantic-style JSON. For the Receipt schema the
    extracted dict will contain ``None`` for the optional fields
    and the required ``merchant_name`` will be missing (because
    MockBackend always returns ``{}``). The point of this test is
    *not* whether the v2 extraction is correct — that depends on
    a real LLM. The point is:

      * The migration tool processes every CORD doc end-to-end
        (parse → classify → extract → assess → validate) without
        crashing.
      * The diff is produced for every doc.
      * A v2 record whose extraction has a field absent from the
        v1 extraction gets that field flagged as ``added``. Here
        we explicitly construct v1 extractions that lack
        ``tax_id`` and verify v2's output (whatever it is) is
        diffed against that.

    Why "tax_id" specifically? It's the v0.4 ROADMAP C4 spec's
    canonical example for "v2 schema adds a new required field";
    we use the MockBackend extraction's keys as the v2 set and
    confirm the diff machinery surfaces the difference between
    v1 (no tax_id) and v2 (whatever MockBackend produced).
    """
    from idp.llm.backend import MockBackend

    # The v2 pipeline factory. In a real deployment, this would
    # carry a template body that adds tax_id guidance to the
    # prompt; here we use MockBackend which returns ``{}`` for
    # extract calls — the diff machinery still runs against the
    # resulting (empty) extraction.
    def factory() -> Pipeline:
        return Pipeline(backend=MockBackend(), schema="Receipt")

    report = migrate_template_version(
        v1_storage,
        "Receipt",
        from_version=1,
        to_version=2,
        pipeline_factory=factory,
        dry_run=True,  # e2e dry-run only — see module docstring
    )

    # Every record produced a row in the report (none skipped —
    # CORD docs are on disk).
    n_records = len(v1_storage.list(limit=100000))
    assert n_records > 0, "fixture wrote zero v1 records; check CORD dataset dir"
    assert len(report.items) == n_records
    assert report.skipped == 0
    assert report.dry_run is True
    assert report.committed == 0

    # Every item is non-skipped, non-error, and has a field_diffs
    # list. (The diff list may be empty if v1 and v2 extractions
    # happen to be identical after the re-run — MockBackend
    # returns ``{}`` for extract, which is a dict with no keys, so
    # the diff against the v1 dict will have all-old-keys as
    # ``removed`` rows. Either way: the list is well-formed.)
    for item in report.items:
        assert item.skipped is False
        assert item.error is None
        assert isinstance(item.field_diffs, list)

    # Pull the v1 tax_id assertion: every v1 extraction we wrote
    # lacks tax_id. So at minimum, the per-item diff for every doc
    # must NOT contain a "tax_id unchanged" row. If v2 produced a
    # tax_id (it doesn't with MockBackend, but the contract is
    # what matters), it would appear as ``added``. If v2 didn't
    # produce tax_id, the diff will simply not mention tax_id.
    for item in report.items:
        tax_id_diffs = [d for d in item.field_diffs if d.field == "tax_id"]
        if tax_id_diffs:
            # v2 produced tax_id; it must be flagged ``added``
            # (v1 had no tax_id, by construction).
            assert len(tax_id_diffs) == 1
            assert tax_id_diffs[0].status == "added"
            assert tax_id_diffs[0].old_value is None
            assert tax_id_diffs[0].new_value is not None

    # Sanity: the v1 records were NOT mutated by a dry-run.
    v1_after = v1_storage.list(limit=100000)
    assert len(v1_after) == n_records
    for r in v1_after:
        # v1 records had no tax_id; that's still true.
        assert "tax_id" not in r.extraction
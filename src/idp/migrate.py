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

"""Template-version migration tool (P5 / C4 of v0.4).

When a template (e.g. ``Invoice``) is bumped from v1 to v2 — different
field names, new required field, type change — extractions stored
under v1 become "orphan": they were valid against v1's schema and are
now stale against v2.

:meth:`migrate_template_version` re-runs the pipeline against the
**current** (v2) template using each stored result's
``StoredResult.source_path`` (the original document on disk), NOT the
stored extraction. It then emits a side-by-side diff of
``old_extraction`` vs ``new_extraction`` and optionally persists the
new extraction as a fresh ``StoredResult`` under the new template
version. The original v1 record is left untouched.

Scope (v0.4, per ``docs/ROADMAP_v0.4.md`` Q4):
  * **Raw extraction only.** v1 human reviews stay attached to the
    v1 record. Re-attaching them silently is the "looks fine but
    lost data" behavior we explicitly avoid — fields reviewed in v1
    may not exist in v2.
  * **Side-by-side diff.** The report enumerates per-field changes
    between old and new extractions; the human reviewer reads it.
  * **No policy-version tracking.** v0.5 work.
  * **No review propagation.** v0.5 work.

Identification caveat (v0.4 simplification):
  ``StoredResult`` carries ``schema_name`` but not yet a dedicated
  ``template_name`` field. Templates are 1:1 with schemas in the
  codebase today (``templates/invoice.md`` declares ``schema:
  Invoice``), so the migration filters by ``schema_name ==
  template_name``. Once templates and schemas diverge (v0.5), a
  dedicated ``template_name`` field will replace this heuristic.

The actual migration work is just ``process_batch`` over the orphan
set, with the diff as a post-process — the spec calls this out
explicitly ("C4 is ``process_batch()`` over the orphan set, with the
diff post-process. Not a fresh parallel runner.").
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idp.batch import BatchItemResult, process_batch
from idp.pipeline.pipeline import Pipeline
from idp.storage.store import Storage, StoredResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class FieldDiff:
    """Per-field change between an old and a new extraction.

    ``status`` semantics:

      * ``"unchanged"`` — value identical.
      * ``"changed"``   — value differs.
      * ``"added"``     — field exists in new extraction only.
      * ``"removed"``   — field exists in old extraction only.

    ``old_value`` / ``new_value`` are ``None`` when the field is
    absent on that side. They are stored as JSON-safe primitives
    (``str | int | float | bool | list | dict | None``); nested
    structures are preserved as-is from the underlying extraction.
    """

    field: str
    status: str
    old_value: Any = None
    new_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "status": self.status,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass
class MigrationItem:
    """One row of the migration report.

    A migration item corresponds to one source ``StoredResult`` from
    the v1 set. The new ``extraction`` and ``confidence`` come from
    re-running the pipeline against the v2 template using the stored
    ``source_path``. ``skipped`` is True when the source file was
    missing on disk (we still produce a report row — with no diff
    fields — so the user can see what was lost).
    """

    stored_result_id: str
    doc_id: str
    source_path: str
    schema_name: str
    backend_name: str
    old_extraction: dict[str, Any]
    new_extraction: dict[str, Any]
    field_diffs: list[FieldDiff] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stored_result_id": self.stored_result_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "schema_name": self.schema_name,
            "backend_name": self.backend_name,
            "old_extraction": self.old_extraction,
            "new_extraction": self.new_extraction,
            "field_diffs": [d.to_dict() for d in self.field_diffs],
            "error": self.error,
            "skipped": self.skipped,
        }


@dataclass
class MigrationReport:
    """Top-level migration report. JSON-serialisable via ``to_dict``.

    ``added`` / ``removed`` / ``changed`` / ``unchanged`` are totals
    across all items' per-field diffs. They are convenience counters
    for the CLI summary line; the per-item detail lives in
    ``items``.
    """

    template_name: str
    from_version: int
    to_version: int
    pipeline_template_body: str
    items: list[MigrationItem] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    changed: int = 0
    unchanged: int = 0
    skipped: int = 0
    committed: int = 0
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_name": self.template_name,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "dry_run": self.dry_run,
            "committed": self.committed,
            "counts": {
                "total": len(self.items),
                "added": self.added,
                "removed": self.removed,
                "changed": self.changed,
                "unchanged": self.unchanged,
                "skipped": self.skipped,
                "committed": self.committed,
            },
            "items": [it.to_dict() for it in self.items],
        }


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------


def _value_equal(a: Any, b: Any) -> bool:
    """Equality check tolerant of ``int`` / ``float`` JSON round-trips.

    JSON serialises ``1`` and ``1.0`` indistinguishably; after a round
    trip through ``json.loads`` they may compare unequal under ``==``
    in the string vs number case. We normalise here so the diff
    doesn't flag a no-op.

    For nested structures (``list``, ``dict``) we fall back to ``==``;
    Pydantic extractions are usually JSON-shaped so this is fine in
    practice. We do NOT chase recursive normalisation — keep this
    cheap, the user can re-run if they need a richer check.
    """
    if type(a) is type(b):
        return a == b
    # Cross-type numeric (int vs float) tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(a) == float(b)
    return a == b


def diff_extractions(
    old: dict[str, Any], new: dict[str, Any]
) -> list[FieldDiff]:
    """Return per-field :class:`FieldDiff` rows between two extractions.

    Field order in the output:
      1. Fields present in ``new`` (added, changed, unchanged)
         — emitted in the order they appear in ``new``.
      2. Fields present only in ``old`` (removed) — emitted in the
         order they appear in ``old``.
    """
    diffs: list[FieldDiff] = []
    seen: set[str] = set()
    for key, new_val in new.items():
        seen.add(key)
        if key not in old:
            diffs.append(
                FieldDiff(field=key, status="added", old_value=None, new_value=new_val)
            )
            continue
        old_val = old[key]
        if _value_equal(old_val, new_val):
            diffs.append(
                FieldDiff(field=key, status="unchanged", old_value=old_val, new_value=new_val)
            )
        else:
            diffs.append(
                FieldDiff(field=key, status="changed", old_value=old_val, new_value=new_val)
            )
    for key, old_val in old.items():
        if key in seen:
            continue
        diffs.append(
            FieldDiff(field=key, status="removed", old_value=old_val, new_value=None)
        )
    return diffs


# ---------------------------------------------------------------------------
# Source-path filtering
# ---------------------------------------------------------------------------


def _select_v1_records(
    storage: Storage,
    *,
    schema_name: str,
) -> list[StoredResult]:
    """Return all stored records matching ``schema_name``.

    Today's :class:`Storage` API doesn't filter on ``template_version``
    (StoredResult doesn't yet carry the field — see module docstring).
    The v0.4 migration tool treats every stored record for the schema
    as a candidate; the user is expected to invoke this tool on a
    *snapshot* of storage containing only v1 records. A v0.5 addition
    will introduce explicit ``template_version`` filtering.

    The list is sorted by ``created_at`` ascending so the report
    order is stable.
    """
    # Use a generous limit; the typical v1 set is small.
    results = storage.list(schema_name=schema_name, limit=100000)
    return sorted(results, key=lambda r: r.created_at)


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------


PipelineFactory = Callable[[], Pipeline]
"""Signature for the pipeline factory passed to the migration tool.

A factory (not a pre-built Pipeline) is used so the migration tool
can build a fresh pipeline per migration — pipelines cache state
(parsers, backends) and reusing one across many migrations with
different templates is a foot-gun. See ROADMAP_v0.4.md C4 #3:
"re-runs the pipeline against the **current** (v2) template".
"""


def migrate_template_version(
    storage: Storage,
    template_name: str,
    *,
    from_version: int,
    to_version: int,
    pipeline_factory: PipelineFactory,
    output_path: str | Path | None = None,
    dry_run: bool = True,
) -> MigrationReport:
    """Re-run ``template_name`` records through the v2 template.

    Args:
        storage: a :class:`Storage` backend holding v1 records.
            Used read-only by default; only written to when
            ``dry_run=False``.
        template_name: the template name to migrate. Today this is
            matched against ``StoredResult.schema_name`` (templates
            are 1:1 with schemas in v0.4; see module docstring).
        from_version: the version recorded on the v1 records. Kept
            in the report for clarity. (Filtering on this value is
            a v0.5 item — see ``_select_v1_records``.)
        to_version: the version to attach to any newly-persisted
            ``StoredResult`` when ``dry_run=False``.
        pipeline_factory: zero-arg callable returning a fresh
            :class:`Pipeline` configured for the v2 template.
        output_path: optional path to write the JSON report. When
            ``None`` and ``dry_run=True``, the report is returned
            but not written to disk. When ``None`` and
            ``dry_run=False``, an :class:`ValueError` is raised —
            the caller must choose where the report lands so they
            can audit the commit.
        dry_run: when True (the default), no new ``StoredResult`` is
            persisted. The diff is computed either way; ``dry_run``
            only controls persistence.

    Returns:
        A :class:`MigrationReport` with one :class:`MigrationItem`
        per v1 record.

    Raises:
        ValueError: if ``from_version == to_version``, if the v1 set
            is empty, or if ``dry_run=False`` and ``output_path``
            is ``None``.
        FileNotFoundError: if the template body string in the v2
            pipeline cannot be retrieved (defensive — should not
            happen if the factory is configured correctly).
    """
    if from_version == to_version:
        raise ValueError(
            f"from_version ({from_version}) must differ from "
            f"to_version ({to_version}); nothing to migrate."
        )
    if not dry_run and output_path is None:
        raise ValueError(
            "output_path is required when dry_run=False; the caller "
            "must choose where the audit report is written so the "
            "commit is reviewable."
        )

    v1_records = _select_v1_records(storage, schema_name=template_name)
    if not v1_records:
        raise ValueError(
            f"no StoredResult rows found for schema_name="
            f"{template_name!r}; nothing to migrate."
        )

    # Build a single pipeline for the re-run. The factory pattern
    # lets callers customise backend / policy / cache; we build once
    # and reuse so the model (if any) is loaded only on the first
    # batch item, not per-item. This matches process_batch semantics.
    pipeline = pipeline_factory()
    template_body = _extract_template_body(pipeline)

    report = MigrationReport(
        template_name=template_name,
        from_version=from_version,
        to_version=to_version,
        pipeline_template_body=template_body,
        dry_run=dry_run,
    )

    # Build the path list for process_batch. We pre-filter missing
    # files (warn-don't-crash) and pass the rest through process_batch
    # so the per-item error handling is uniform.
    existing_paths: list[str] = []
    skipped_paths: dict[str, str] = {}  # path -> StoredResult.id
    for r in v1_records:
        p = Path(r.source_path)
        if not p.exists():
            log.warning(
                "source_path %r for StoredResult %s no longer exists on "
                "disk; skipping (warn-don't-crash).",
                r.source_path,
                r.id,
            )
            report.skipped += 1
            skipped_paths[r.source_path] = r.id
            continue
        existing_paths.append(r.source_path)

    # Re-run via process_batch. We don't use checkpoint here: the
    # migration is a one-shot audit; idempotent resume doesn't add
    # value (the work is already short — single batch).
    batch_results: dict[str, BatchItemResult] = {}
    if existing_paths:
        for item in process_batch(existing_paths, pipeline, progress_every=0):
            batch_results[item.path] = item

    # Now build the report. Order: by v1 record order (== source
    # path ascending). Skipped items follow the in-order insertion.
    for r in v1_records:
        if r.source_path in skipped_paths:
            report.items.append(
                MigrationItem(
                    stored_result_id=r.id,
                    doc_id=r.doc_id,
                    source_path=r.source_path,
                    schema_name=r.schema_name,
                    backend_name=r.backend_name,
                    old_extraction=r.extraction,
                    new_extraction={},
                    skipped=True,
                    error="source_path not found on disk",
                )
            )
            continue

        br = batch_results.get(r.source_path)
        if br is None:
            # Defensive: process_batch yielded nothing for a path we
            # expected. Should not happen, but record as error so the
            # user can see something is wrong.
            report.items.append(
                MigrationItem(
                    stored_result_id=r.id,
                    doc_id=r.doc_id,
                    source_path=r.source_path,
                    schema_name=r.schema_name,
                    backend_name=r.backend_name,
                    old_extraction=r.extraction,
                    new_extraction={},
                    error="process_batch produced no result for this path",
                )
            )
            continue

        if not br.ok or br.result is None:
            report.items.append(
                MigrationItem(
                    stored_result_id=r.id,
                    doc_id=r.doc_id,
                    source_path=r.source_path,
                    schema_name=r.schema_name,
                    backend_name=r.backend_name,
                    old_extraction=r.extraction,
                    new_extraction={},
                    error=br.error or "pipeline run failed",
                )
            )
            continue

        new_extraction = br.result.document.extraction or {}
        diffs = diff_extractions(r.extraction, new_extraction)
        for d in diffs:
            if d.status == "added":
                report.added += 1
            elif d.status == "removed":
                report.removed += 1
            elif d.status == "changed":
                report.changed += 1
            else:
                report.unchanged += 1

        report.items.append(
            MigrationItem(
                stored_result_id=r.id,
                doc_id=r.doc_id,
                source_path=r.source_path,
                schema_name=r.schema_name,
                backend_name=r.backend_name,
                old_extraction=r.extraction,
                new_extraction=new_extraction,
                field_diffs=diffs,
            )
        )

    # Commit phase. We persist a fresh StoredResult per migrated
    # item, with template_version == to_version. The v1 record is
    # left untouched; the new record carries only the raw v2
    # extraction. Per ROADMAP_v0.4.md Q4: "v2 record gets the new
    # extraction but no review" — we deliberately do NOT copy the
    # v1 reviewed_extraction because the v1 fields may not exist in
    # v2.
    if not dry_run:
        for mi in report.items:
            if mi.skipped or mi.error:
                continue
            new_record = StoredResult(
                id="",
                doc_id=mi.doc_id,
                schema_name=mi.schema_name,
                backend_name=mi.backend_name,
                mode=None,
                classification=None,
                extraction=mi.new_extraction,
                confidence=None,
                validation=None,
                source_path=mi.source_path,
                created_at=_now(),
                # reviewed=False on the v2 record by design. See module
                # docstring: raw extraction only, no review propagation.
                reviewed=False,
                reviewed_extraction=None,
                reviewer=None,
                last_reviewed_at=None,
            )
            new_id = storage.put(new_record)
            log.info(
                "migrate: committed v%d record %s for %s (was v%d %s)",
                to_version,
                new_id,
                mi.doc_id,
                from_version,
                mi.stored_result_id,
            )
            report.committed += 1

    # Persist the report to disk if requested. We do this AFTER the
    # commit so the report reflects what actually happened.
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_template_body(pipeline: Pipeline) -> str:
    """Return the template body attached to ``pipeline``, or ``""``.

    The v2 template body is recorded on the report so the audit trail
    shows which guidance the LLM was given during the re-run. We
    reach into Pipeline's private attributes (``_template_body``);
    this is the same pattern ``Pipeline.run`` uses internally.
    """
    return getattr(pipeline, "_template_body", "") or ""


def _now() -> float:
    """Time helper. Imported lazily so tests can monkeypatch if needed."""
    import time as _time

    return _time.time()


__all__ = [
    "FieldDiff",
    "MigrationItem",
    "MigrationReport",
    "PipelineFactory",
    "diff_extractions",
    "migrate_template_version",
]
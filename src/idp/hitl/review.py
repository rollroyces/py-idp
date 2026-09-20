"""Pure data-handling logic for the HITL review UI.

This module is intentionally free of any Streamlit import. It exposes
the data shape and operations that ``idp.hitl.app`` (the Streamlit
app) needs. Keeping it pure means:

  - The HITL data logic is unit-testable without spinning up
    Streamlit + the 20 MB transitive dependencies.
  - The data correctness can be reasoned about without UI context.
  - A future replacement UI (e.g. a FastAPI page) can reuse the
    same logic without duplicating it.

A3 in docs/ROADMAP.md: extract data from idp.hitl.app into a pure
module so it can be tested in isolation.
"""
from __future__ import annotations

import json
from typing import Any

from idp.storage.store import Storage, StoredResult


# ---------------------------------------------------------------------------
# Storage dispatch: SqlStorage.submit_review vs JsonFileStorage.mark_reviewed
# ---------------------------------------------------------------------------
def save_review(
    storage: Storage,
    *,
    result_id: str,
    edited: dict[str, Any],
    reviewer: str,
) -> None:
    """Persist a review, dispatching to the right Storage method.

    SqlStorage exposes ``submit_review`` (atomic, with reviewer table
    and review_edits); legacy storage backends (JsonFileStorage,
    InMemoryStorage) only have ``mark_reviewed``. We prefer
    ``submit_review`` when available because it captures the rich
    audit trail the RL feature depends on.

    Args:
        storage: any Storage instance.
        result_id: the id of the StoredResult being reviewed.
        edited:    the human-edited extraction. The diff between this
                   and the original extraction is what feeds RL.
        reviewer:  free-form string (no auth in this version).

    Returns:
        None. Side effect: storage's review state is updated.
    """
    if hasattr(storage, "submit_review"):
        # SqlStorage exposes submit_review (richer audit trail); legacy
        # Storage backends only have mark_reviewed. We use getattr +
        # setattr to keep mypy happy: the Protocol doesn't promise
        # submit_review, but at runtime we just checked it's there.
        submit = storage.submit_review
        submit(
            result_id=result_id,
            edited=edited,
            reviewer=reviewer,
        )
    else:
        storage.mark_reviewed(result_id, edited, reviewer)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def format_result_label(result: StoredResult) -> str:
    """The label shown in the Streamlit selectbox for picking a result.

    Format: "<id>  —  <schema>  —  <source_path>". Chosen to be readable
    in the dropdown and to be parseable back to a result id (the
    selectbox returns a string, and we recover the StoredResult by
    matching the id prefix).
    """
    return f"{result.id}  —  {result.schema_name}  —  {result.source_path}"


def sort_pending_newest_first(results: list[StoredResult]) -> list[StoredResult]:
    """Sort by created_at descending. Streamlit selectbox is happy with
    any order, but most-recent-first is what a human reviewer wants."""
    return sorted(results, key=lambda r: -r.created_at)


def count_corrections(result: StoredResult) -> int:
    """How many fields did the human change?

    Compares ``extraction`` (model output) to ``reviewed_extraction``
    (human edit) and counts the differing fields. Used by the
    history page to show "fields corrected" per result.

    Returns 0 when the result hasn't been reviewed yet. (Defensive:
    a result with ``reviewed_extraction=None`` but a non-None
    ``extraction`` would otherwise report "all fields corrected",
    which is nonsense.)
    """
    if result.reviewed_extraction is None:
        return 0
    model_ext = result.extraction or {}
    reviewed_ext = result.reviewed_extraction
    return sum(
        1
        for k, v in model_ext.items()
        if reviewed_ext.get(k) != v
    )


def corrected_fields(result: StoredResult) -> set[str]:
    """The set of field names the human actually changed.

    Companion to ``count_corrections`` that returns WHICH fields
    differed rather than just how many. Used by ``idp.hitl.triage``
    to attribute per-field correction rates.

    Behavior:
      * Unreviewed (``reviewed_extraction is None``) → empty set.
      * ``extraction`` is ``None`` → empty set (defensive).
      * Keys present only in ``reviewed_extraction`` are NOT counted
        as corrections; we only know what the model *did* claim a
        value for, not fields the human added. Mirrors
        ``count_corrections``.
      * Fields present in ``extraction`` but missing from
        ``reviewed_extraction`` ARE counted as corrections
        (deletion == change).

    Returns:
        A new ``set[str]`` of field names. Safe to mutate by the caller.
    """
    if result.reviewed_extraction is None:
        return set()
    model_ext = result.extraction or {}
    reviewed_ext = result.reviewed_extraction
    return {k for k, v in model_ext.items() if reviewed_ext.get(k) != v}


# ---------------------------------------------------------------------------
# Field editing: parse / stringify complex values
# ---------------------------------------------------------------------------
def coerce_field_value(raw: str, original: Any) -> tuple[Any, str | None]:
    """Parse a text input back to a structured value, with a JSON fallback.

    The Streamlit UI exposes every field as a ``text_input`` /
    ``text_area``. For dict/list values, the human can edit the JSON
    representation; if it parses, we use the parsed value. If the
    JSON is invalid, we fall back to keeping the raw string and
    returning a warning to surface in the UI.

    Args:
        raw:      the string the human typed into the text widget.
        original: the original model value (used to decide whether
                  JSON parsing makes sense: if the original was a
                  dict/list, we try JSON; otherwise we use raw).

    Returns:
        (parsed_value, warning_or_None). The parsed_value is what
        gets stored in the review; the warning is shown to the user
        in the Streamlit UI ("invalid JSON; saved as raw string").
    """
    if isinstance(original, (dict, list)):
        try:
            return json.loads(raw), None
        except (ValueError, TypeError):
            return raw, f"invalid JSON for {type(original).__name__}; saved as raw string"
    # For scalar values, no parsing needed.
    return raw, None

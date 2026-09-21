"""HITL triage: detect fields the model gets systematically wrong.

P3 v0.4 in docs/ROADMAP_v0.4.md (item D1). The HITL reviewer sees one
document at a time, but the *aggregate* pattern across many reviewed
documents is the signal we actually want:

  "vendor_name was corrected 4/5 times → the model has a systematic
   bug on this field, push back on the prompt/schema instead of
   reviewing it again."

This module walks any ``Storage`` backend, computes per-field correction
rates from reviewed results, and returns a structured report:

  * ``SystematicError`` — field was corrected in >= threshold of
    >= min_reviews reviews.
  * ``InsufficientField`` — field has < min_reviews reviews; the report
    says so honestly rather than guessing.

The module is intentionally Streamlit-free so the same logic powers
the CLI (``idp triage --storage <back>``) and the Streamlit page in
``idp.hitl.app``. The CLI/UI layer only renders the dataclass.

What this module does NOT do (deferred to v0.5):

  * Per-template triage report (today: one report across all schemas).
  * LLM-based confidence overrides (triage uses stored ``confidence``
    as-is; no inference).
  * Automatic field-correction (no ``save_review`` writes happen here).
  * Statistical significance testing (threshold-based, not p-value).

The 0.6 default is documented in
``docs/ROADMAP_v0.4.md`` ("Q5 (D1 systematic-error threshold) —
decided: 0.6"). Override at call time with ``threshold=``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any

from idp.storage.store import Storage, StoredResult

# Default thresholds; see ROADMAP_v0.4.md decisions.
DEFAULT_MIN_REVIEWS = 3
DEFAULT_THRESHOLD = 0.6


# Cap on sample_review_ids we attach to a SystematicError. Keeps the
# report readable when a field is reviewed 200 times; we link to the
# first 5 (deterministic) and call it out.
SAMPLE_REVIEW_IDS_LIMIT = 5


@dataclass
class SystematicError:
    """A field whose correction rate passes the threshold.

    Attributes:
        field:             the field name in the extraction.
        rate:              n_corrections / n_reviews, in [0, 1].
        n:                 total number of times this field appeared
                           in a reviewed extraction.
        n_corrections:     how many of those ``n`` reviews corrected
                           the field.
        sample_review_ids: up to ``SAMPLE_REVIEW_IDS_LIMIT`` review
                           ids where the field was corrected, for
                           the reviewer to drill into.
    """

    field: str
    rate: float
    n: int
    n_corrections: int
    sample_review_ids: list[str] = _dc_field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate!r}")
        if self.n < 0 or self.n_corrections < 0:
            raise ValueError("n and n_corrections must be non-negative")
        if self.n_corrections > self.n:
            raise ValueError(
                f"n_corrections ({self.n_corrections}) cannot exceed n ({self.n})"
            )


@dataclass
class InsufficientField:
    """A field that doesn't yet have enough reviews to triage.

    Honest failure mode for the report: instead of guessing, we surface
    *exactly how many more reviews are needed* before we can say
    anything.
    """

    field: str
    n: int
    needed: int

    def __post_init__(self) -> None:
        if self.n < 0 or self.needed < 0:
            raise ValueError("n and needed must be non-negative")


@dataclass
class FieldStats:
    """Per-field accumulator used while walking the storage.

    Not exposed in the public report; internal to ``triage()``.
    """

    field: str
    n_reviews: int = 0
    n_corrections: int = 0
    sample_corrected_ids: list[str] = _dc_field(default_factory=list)


@dataclass
class TriageReport:
    """Aggregate triage signal across a ``Storage`` backend.

    Attributes:
        systematic_errors: list of ``SystematicError`` sorted by
                           rate desc, then n desc.
        insufficient:      list of ``InsufficientField`` sorted by
                           field name (stable order for testing).
        n_reviews_total:   total reviewed ``StoredResult`` count
                           that fed the report (one per reviewed
                           result, NOT one per field).
        n_fields_seen:     number of distinct field names that
                           appeared in any reviewed extraction.
        min_reviews:       threshold used; echoed for callers.
        threshold:         rate threshold used; echoed for callers.
    """

    systematic_errors: list[SystematicError] = _dc_field(default_factory=list)
    insufficient: list[InsufficientField] = _dc_field(default_factory=list)
    n_reviews_total: int = 0
    n_fields_seen: int = 0
    min_reviews: int = DEFAULT_MIN_REVIEWS
    threshold: float = DEFAULT_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict.

        Used by the ``idp triage --output json`` CLI and by the
        Streamlit page. Dataclass → plain dict so ``json.dumps``
        doesn't choke on the dataclass types.
        """
        return {
            "min_reviews": self.min_reviews,
            "threshold": self.threshold,
            "n_reviews_total": self.n_reviews_total,
            "n_fields_seen": self.n_fields_seen,
            "systematic_errors": [
                {
                    "field": se.field,
                    "rate": se.rate,
                    "n": se.n,
                    "n_corrections": se.n_corrections,
                    "sample_review_ids": list(se.sample_review_ids),
                }
                for se in self.systematic_errors
            ],
            "insufficient": [
                {
                    "field": inf.field,
                    "n": inf.n,
                    "needed": inf.needed,
                }
                for inf in self.insufficient
            ],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def triage(
    storage: Storage,
    *,
    min_reviews: int = DEFAULT_MIN_REVIEWS,
    threshold: float = DEFAULT_THRESHOLD,
) -> TriageReport:
    """Compute per-field correction rates from a ``Storage`` backend.

    Walks every reviewed ``StoredResult`` in ``storage``, computes the
    correction rate per field, and partitions fields into:

      * ``systematic_errors``: rate >= threshold AND n >= min_reviews.
      * ``insufficient``:      n < min_reviews (honest failure mode).

    Fields that are reviewed >= min_reviews but below threshold are
    NOT included in either list — they are "noise" by the report's
    definition, and surfacing them would dilute the high-value signal.
    Callers can re-derive them from the raw counts if needed (out of
    scope for v0.4).

    Args:
        storage:     any ``Storage`` instance. ``storage.list(reviewed_only=True)``
                     is used; we never read unreviewed results.
        min_reviews: minimum reviews for a field to be triaged.
                     Fields below this go to ``insufficient``.
        threshold:   correction-rate cutoff. Fields at or above go
                     to ``systematic_errors``.

    Returns:
        A ``TriageReport`` populated from the storage. Empty lists
        when there are no reviews yet.
    """
    if min_reviews < 1:
        raise ValueError(f"min_reviews must be >= 1, got {min_reviews}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")

    reviewed = storage.list(reviewed_only=True, limit=100000)
    return triage_from_results(
        reviewed,
        min_reviews=min_reviews,
        threshold=threshold,
    )


def triage_from_results(
    results: Iterable[StoredResult],
    *,
    min_reviews: int = DEFAULT_MIN_REVIEWS,
    threshold: float = DEFAULT_THRESHOLD,
) -> TriageReport:
    """Same algorithm as ``triage`` but takes an iterable of results.

    Useful for tests (feed synthetic ``StoredResult`` instances
    without touching a real storage backend) and for callers that
    already have a filtered list.
    """
    if min_reviews < 1:
        raise ValueError(f"min_reviews must be >= 1, got {min_reviews}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")

    stats: dict[str, FieldStats] = {}
    n_reviews_total = 0

    for r in results:
        # Defensive: a row with reviewed=True but reviewed_extraction=None
        # is malformed (shouldn't happen, but a corrupted JSONL could
        # produce it). Skip rather than crash.
        if not r.reviewed or r.reviewed_extraction is None:
            continue
        n_reviews_total += 1
        # All fields in the model's extraction count as "seen" by the
        # reviewer. We compare against the human's reviewed_extraction
        # to count corrections. Mirrors the contract of
        # ``count_corrections``: keys present only in
        # ``reviewed_extraction`` are NOT counted as corrections (we
        # don't know what the model "would have" claimed for them).
        model_ext = r.extraction or {}
        reviewed_ext = r.reviewed_extraction
        corrected = {k for k, v in model_ext.items() if reviewed_ext.get(k) != v}
        for fname in model_ext:
            st = stats.get(fname)
            if st is None:
                st = FieldStats(field=fname)
                stats[fname] = st
            st.n_reviews += 1
            if fname in corrected:
                st.n_corrections += 1
                if len(st.sample_corrected_ids) < SAMPLE_REVIEW_IDS_LIMIT:
                    st.sample_corrected_ids.append(r.id)

    systematic: list[SystematicError] = []
    insufficient: list[InsufficientField] = []

    for st in stats.values():
        if st.n_reviews < min_reviews:
            insufficient.append(
                InsufficientField(
                    field=st.field,
                    n=st.n_reviews,
                    needed=min_reviews - st.n_reviews,
                )
            )
            continue
        rate = st.n_corrections / st.n_reviews if st.n_reviews else 0.0
        if rate >= threshold:
            systematic.append(
                SystematicError(
                    field=st.field,
                    rate=rate,
                    n=st.n_reviews,
                    n_corrections=st.n_corrections,
                    sample_review_ids=list(st.sample_corrected_ids),
                )
            )

    # Stable, deterministic ordering for tests + UI.
    systematic.sort(key=lambda se: (-se.rate, -se.n, se.field))
    insufficient.sort(key=lambda inf: inf.field)

    return TriageReport(
        systematic_errors=systematic,
        insufficient=insufficient,
        n_reviews_total=n_reviews_total,
        n_fields_seen=len(stats),
        min_reviews=min_reviews,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Report formatters (for `idp triage --output md`)
# ---------------------------------------------------------------------------
def format_report_markdown(report: TriageReport) -> str:
    """Render a TriageReport as a human-readable Markdown table.

    Used by ``idp triage --output md`` and the Streamlit Triage page
    (the Streamlit page also calls this when the user wants a copyable
    summary).

    Sections:
      1. Top: thresholds + sample sizes.
      2. Systematic errors (table) — only present if there are any.
      3. Insufficient data (table) — only present if there are any.
      4. Footer with total reviewed count.
    """
    lines: list[str] = []
    lines.append("# py-idp triage report")
    lines.append("")
    lines.append(
        f"- threshold (correction rate cutoff): **{report.threshold:.2f}**"
    )
    lines.append(f"- min_reviews (per field): **{report.min_reviews}**")
    lines.append(f"- reviewed results scanned: **{report.n_reviews_total}**")
    lines.append(f"- distinct fields seen: **{report.n_fields_seen}**")
    lines.append("")

    if report.systematic_errors:
        lines.append("## Systematic errors")
        lines.append("")
        lines.append("These fields were corrected at or above the threshold. "
                     "Treat them as real model bugs and push back on the "
                     "prompt/schema rather than re-correcting.")
        lines.append("")
        lines.append("| field | rate | n | n_corrected | sample review ids |")
        lines.append("|---|---|---|---|---|")
        for se in report.systematic_errors:
            ids = ", ".join(se.sample_review_ids) if se.sample_review_ids else "—"
            lines.append(
                f"| `{se.field}` | {se.rate:.2f} | {se.n} | "
                f"{se.n_corrections} | {ids} |"
            )
        lines.append("")
    else:
        lines.append("## Systematic errors")
        lines.append("")
        lines.append("_None — no field has crossed the threshold yet._")
        lines.append("")

    if report.insufficient:
        lines.append("## Insufficient data")
        lines.append("")
        lines.append(
            "These fields haven't been reviewed enough times to triage. "
            "We refuse to guess — keep reviewing and re-run `idp triage`."
        )
        lines.append("")
        lines.append("| field | n_reviews | more_needed |")
        lines.append("|---|---|---|")
        for inf in report.insufficient:
            lines.append(f"| `{inf.field}` | {inf.n} | +{inf.needed} |")
        lines.append("")

    lines.append("---")
    lines.append(
        f"_Generated by py-idp triage. Reviewed {report.n_reviews_total} "
        f"result(s); {report.n_fields_seen} distinct field(s) seen._"
    )
    return "\n".join(lines) + "\n"

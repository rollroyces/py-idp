"""Auto-template discovery from sample documents (weekend-hack version).

Given 3-5 sample PDFs/text files, propose a single reusable :class:`Template`
that captures the common fields and writes it as a ``.md`` file with
valid frontmatter (compatible with :func:`idp.templates.load_template`).

This is the multi-doc analogue of :func:`idp.discover.discover_schema` —
schema discovery answers "what should I extract from THIS PDF?", template
discovery answers "what should my reusable extraction template look like
across MANY PDFs of the same type?".

Scope (explicit non-goals)
--------------------------
This is a weekend-hack version. It is deliberately NOT production-grade:

* The "fields that appear in N-1 of N" heuristic only looks at the field
  *names* present in each sample's extraction dict. It does NOT check
  whether the LLM actually populated them with real values. With a real
  LLM backend, empty/null extractions get filtered out; with
  ``backend="mock"`` every schema field is "present" by construction,
  so the heuristic passes everything.
* The generated Markdown body is a flat bullet list of field names —
  no field descriptions, no regex hints, no worked examples, no
  "common mistakes" section. Hand-curate those before relying on the
  template.
* Schema name comes from the user-provided ``schema_name_hint`` or a
  majority vote on the per-sample classification. The mock backend's
  classifier always returns ``"other"``, so without a hint the schema
  name falls back to ``"InferredTemplate"``.
* The CLI command has no progress reporting, no resumability, and no
  parallelism. For real workloads use the underlying
  :class:`idp.pipeline.Pipeline` directly.

If you need any of the above, hand-write the template using the
existing :class:`idp.templates.TemplateRegistry` instead.

Usage
-----

    from pathlib import Path
    from idp.template_discovery import discover_template

    template = discover_template(
        samples=["inv-001.pdf", "inv-002.pdf", "inv-003.pdf"],
        output_dir=Path("./templates"),
        schema_name_hint="Invoice",
    )
    # Wrote ./templates/invoice.md

CLI::

    idp discover-template samples/*.pdf --output templates/
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from idp.pipeline.pipeline import Pipeline
from idp.templates import Template

log = logging.getLogger(__name__)


# Lower bound on sample count (matches ROADMAP.md B1: "3-5 sample PDFs").
MIN_SAMPLES = 3
# Upper bound. The heuristic becomes noisier past 5 — extra samples that
# all use a slightly different variant start "averaging out" real fields.
MAX_SAMPLES = 5


@dataclass
class _SampleProfile:
    """What we learned from running the pipeline against one sample."""

    source_path: Path
    fields: list[str]              # top-level extraction keys, original case
    field_set_lower: set[str]      # lowercased, for case-insensitive clustering
    classification: str | None     # pipeline.classification (may be None on failure)
    extraction: dict | None        # full extraction dict (for debugging)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def discover_template(
    samples: list[str],
    output_dir: Path,
    *,
    schema_name_hint: str | None = None,
    backend: str = "mock",
    schema: str = "Invoice",
) -> Template:
    """Cluster fields across ``samples`` and write one template to ``output_dir``.

    Args:
        samples: 3-5 paths to PDF / image / text files of the same document
            type. Less than 3 raises :class:`ValueError`. More than 5
            raises :class:`ValueError` (the heuristic gets noisy beyond 5).
        output_dir: Directory to write the generated ``<name>.md`` into.
            Created if missing.
        schema_name_hint: If provided, used as the template's ``name``
            AND its ``schema`` field. If None, the schema name is the
            majority classification across samples (mock returns
            ``"other"``, so the fallback is ``"InferredTemplate"``).
        backend: Pipeline backend name. Default ``"mock"`` keeps tests
            offline. Real users will want ``"openai"``, ``"ollama"``,
            ``"nanonets"``, etc.
        schema: Pydantic schema name to use as the starting point for
            extraction. Defaults to ``"Invoice"`` because it's the most
            complete built-in schema; override when your samples are
            something else (``"Contract"``, ``"Receipt"``, ...).

    Returns:
        The :class:`Template` that was written. ``template.source_path``
        points at the generated ``.md`` file.

    Raises:
        ValueError: If ``samples`` is empty, has fewer than 3 entries,
            more than 5 entries, or any path does not exist.
        OSError: If the template file can't be written.
    """
    sample_paths = _validate_samples(samples)

    # 1. Run the pipeline against each sample.
    profiles = [_profile_sample(p, backend=backend, schema=schema) for p in sample_paths]

    # 2. Cluster fields with the N-1/N heuristic.
    common_fields = _cluster_common_fields(profiles)
    log.info(
        "discover_template: %d samples -> %d common fields: %s",
        len(profiles), len(common_fields), common_fields,
    )

    # 3. Resolve the schema name (hint wins; else majority classification).
    schema_name = _resolve_schema_name(profiles, hint=schema_name_hint)
    template_name = _slugify(schema_name)

    # 4. Build field_overrides (kept empty for the weekend-hack version —
    # real regex hints / examples should be hand-written).
    field_overrides: dict[str, dict] = {}

    # 5. Render the body.
    body = _render_body(
        schema_name=schema_name,
        common_fields=common_fields,
        sample_paths=sample_paths,
        profiles=profiles,
    )

    # 6. Render the frontmatter.
    frontmatter = {
        "name": template_name,
        "schema": schema_name,
        "version": 1,
        "mime_types": ["application/pdf", "image/jpeg", "image/png"],
        "filename_patterns": [f"*{template_name}*"],
        "classification_hints": _classification_hints(schema_name, profiles),
        "field_overrides": field_overrides,
    }

    # 7. Write the .md file.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{template_name}.md"
    md_text = _render_markdown(frontmatter, body)
    out_path.write_text(md_text, encoding="utf-8")
    log.info("discover_template: wrote %s", out_path)

    # 8. Construct the Template object. Round-trip through the registry
    #    so we get the same validation Template.load() applies — if we
    #    generated invalid frontmatter, this is where it blows up.
    from idp.templates import load_template

    return load_template(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_samples(samples: list[str]) -> list[Path]:
    if not samples:
        raise ValueError(
            "discover_template needs at least one sample; got an empty list"
        )
    if len(samples) < MIN_SAMPLES:
        raise ValueError(
            f"discover_template needs at least {MIN_SAMPLES} samples for the "
            f"N-1 of N heuristic to be meaningful; got {len(samples)}"
        )
    if len(samples) > MAX_SAMPLES:
        raise ValueError(
            f"discover_template accepts at most {MAX_SAMPLES} samples (the "
            f"heuristic gets noisy past 5); got {len(samples)}"
        )
    paths = [Path(s) for s in samples]
    for p in paths:
        if not p.exists():
            raise ValueError(f"sample path does not exist: {p}")
    return paths


# ---------------------------------------------------------------------------
# Per-sample pipeline run
# ---------------------------------------------------------------------------
def _profile_sample(
    path: Path, *, backend: str, schema: str
) -> _SampleProfile:
    """Run the pipeline against ``path`` and capture the field profile."""
    from idp.core.document import Document

    pipeline = Pipeline(backend=backend, schema=schema)
    doc = Document.from_path(path)
    try:
        result = pipeline.run(doc)
    except Exception as e:  # noqa: BLE001
        # One bad sample shouldn't kill the whole discovery. Log and
        # return an empty profile so the clustering excludes it.
        log.warning("discover_template: pipeline failed for %s: %s", path, e)
        return _SampleProfile(
            source_path=path,
            fields=[],
            field_set_lower=set(),
            classification=None,
            extraction=None,
        )

    ext = result.document.extraction or {}
    # Top-level keys only. Mock backend returns nested objects for some
    # fields (e.g. invoice_date: {}); we keep the field name but skip
    # the nested structure.
    fields = list(ext.keys())
    return _SampleProfile(
        source_path=path,
        fields=fields,
        field_set_lower={_normalize_field_name(f) for f in fields},
        classification=result.classification,
        extraction=ext,
    )


# ---------------------------------------------------------------------------
# Field clustering
# ---------------------------------------------------------------------------
def _cluster_common_fields(
    profiles: list[_SampleProfile],
) -> list[str]:
    """Return fields that appear in at least N-1 of N samples.

    Normalization rules:
      - Case-insensitive (``VendorName`` == ``vendorname``).
      - Underscores / spaces / hyphens treated as equivalent
        (``InvoiceNumber`` == ``Invoice_Number`` == ``Invoice Number``).
        Anything non-alphanumeric is collapsed before comparison.

    The output uses the most-common original casing for each surviving
    field. Ties -> first occurrence.
    """
    n = len(profiles)
    if n == 0:
        return []
    threshold = n - 1  # "appears in N-1 of N"
    # If N is small (3-5), N-1/N still means "in all but one" — this is
    # the documented heuristic. With N=3, threshold=2, so a field must
    # appear in >= 2 samples.

    # Count normalized occurrences + track casing variants.
    counts: Counter[str] = Counter()
    casing: dict[str, Counter[str]] = {}
    for prof in profiles:
        for field in prof.fields:
            key = _normalize_field_name(field)
            counts[key] += 1
            casing.setdefault(key, Counter())[field] += 1

    common: list[str] = []
    for key, count in counts.items():
        if count >= threshold:
            # Pick the most common original casing.
            most_common = casing[key].most_common(1)[0][0]
            common.append(most_common)

    # Stable ordering: alphabetical (so the output .md is deterministic).
    return sorted(common)


# Normalize a field name for case/separator-insensitive comparison.
# "Invoice_Number" -> "invoicenumber", "Vendor Name" -> "vendorname",
# "tax-amount" -> "taxamount", "Total Amount!" -> "totalamount".
_FIELD_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_field_name(name: str) -> str:
    """Lowercase + strip all non-alphanumerics for fuzzy comparison.

    This is intentionally aggressive: the goal is to merge what is
    obviously the same field name across casing + separator variants.
    Fields that are still distinct after normalization (e.g.
    ``email`` vs ``emails``) will be kept separate.
    """
    return _FIELD_NAME_NORMALIZE_RE.sub("", name.lower())


# ---------------------------------------------------------------------------
# Schema name resolution
# ---------------------------------------------------------------------------
def _resolve_schema_name(
    profiles: list[_SampleProfile], *, hint: str | None
) -> str:
    """Pick the canonical schema name.

    Priority:
      1. ``hint`` if provided (user override wins).
      2. Majority vote on per-sample classification.
      3. ``"InferredTemplate"`` fallback.
    """
    if hint:
        return hint.strip()

    classifications = [p.classification for p in profiles if p.classification]
    if classifications:
        counter = Counter(classifications)
        top, count = counter.most_common(1)[0]
        if count >= len(profiles) / 2:
            return top.title().replace("_", "").replace(" ", "")

    return "InferredTemplate"


# ---------------------------------------------------------------------------
# Body rendering
# ---------------------------------------------------------------------------
def _render_body(
    *,
    schema_name: str,
    common_fields: list[str],
    sample_paths: list[Path],
    profiles: list[_SampleProfile],
) -> str:
    """Build the Markdown body (no frontmatter)."""
    lines: list[str] = []
    lines.append(f"# {schema_name}")
    lines.append("")
    lines.append(
        f"_Auto-generated from {len(sample_paths)} sample document(s). "
        "Hand-edit before relying on this template._"
    )
    lines.append("")

    # Fields section
    lines.append("## Fields")
    lines.append("")
    if not common_fields:
        lines.append(
            "_No common fields detected across samples. The N-1 of N "
            "heuristic returned an empty set — possibly because all "
            "samples failed extraction or used mutually exclusive schemas._"
        )
    else:
        for f in common_fields:
            # Count how many samples had this field (normalized).
            key = _normalize_field_name(f)
            occurrences = sum(
                1 for p in profiles if key in p.field_set_lower
            )
            lines.append(
                f"* **{f}** — appears in {occurrences} of "
                f"{len(profiles)} samples."
            )
    lines.append("")

    # Fields summary section (mentioned in spec: "body-section-includes-fields-summary")
    lines.append("## Fields summary")
    lines.append("")
    lines.append(f"- Total fields detected: {len(common_fields)}")
    lines.append(f"- Samples analyzed: {len(profiles)}")
    if common_fields:
        lines.append(f"- Field names: {', '.join(common_fields)}")
    lines.append("")

    # Common mistakes placeholder — explicit, so users don't forget
    # to fill it in.
    lines.append("## Common mistakes")
    lines.append("")
    lines.append(
        "_Hand-curate this section. The weekend-hack discovery does not "
        "infer common-mistake patterns from sample data._"
    )
    lines.append("")

    # Sources
    lines.append("## Source samples")
    lines.append("")
    for p in sample_paths:
        lines.append(f"* `{p}`")
    lines.append("")

    return "\n".join(lines)


def _classification_hints(
    schema_name: str, profiles: list[_SampleProfile]
) -> list[str]:
    """Generate a few short classification hints from what we observed."""
    hints: list[str] = []
    if profiles:
        classifications = {
            p.classification for p in profiles if p.classification
        }
        if classifications:
            for c in sorted(classifications):
                if c and c != "other":
                    hints.append(f"Classified as {c!r} in samples")
    if not hints:
        hints.append(f"Documents of type {schema_name}")
        hints.append("Auto-discovered from samples; verify before use")
    return hints


# ---------------------------------------------------------------------------
# Markdown serialization
# ---------------------------------------------------------------------------
def _render_markdown(frontmatter: dict, body: str) -> str:
    """Serialize frontmatter + body into the .md format Template.load reads.

    Format::

        ---
        key: value
        ---
        body
    """
    fm_text = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        default_flow_style=None,
        allow_unicode=True,
        width=100,
    )
    return f"---\n{fm_text}---\n\n{body}"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _slugify(name: str) -> str:
    """Turn ``'Inferred Template'`` into ``'inferred-template'``."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "template"


def _summarize_fields_for_debug(
    profiles: Iterable[_SampleProfile],
) -> dict[str, list[str]]:
    """Map field-name -> [paths that had it]. For diagnostic logging only."""
    out: dict[str, list[str]] = {}
    for prof in profiles:
        for f in prof.fields:
            out.setdefault(f, []).append(str(prof.source_path))
    return out


__all__ = [
    "discover_template",
    "MIN_SAMPLES",
    "MAX_SAMPLES",
]
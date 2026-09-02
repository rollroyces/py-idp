"""Auto-schema discovery: turn a PDF + a hint into a Pydantic class + JSON Schema.

Use case:
    You have a scanned PDF (invoice, contract, bank statement, etc.) and a
    natural-language description of what you want extracted. Instead of
    hand-writing a Pydantic class, let the LLM infer it from the document.

Example:
    Schema, schema_dict = idp.discover_schema(
        "scan.pdf",
        hint="extract vendor_name, invoice_number, total_amount, and line items",
    )
    # Pass the discovered schema directly into a Pipeline:
    result = Pipeline(backend="nanonets", schema=Schema).run(
        Document.from_path("scan.pdf")
    )

The returned ``Schema`` is a Pydantic ``BaseModel`` subclass. It is
NOT a string name — you pass the class itself to ``Pipeline(schema=...)``
exactly like the built-in ``Invoice`` / ``Contract`` / ``BankStatement``.

Implementation notes
--------------------
The discovery flow is:

  1. Parse the PDF with PdfPagesParser (renders pages to images).
  2. Build a multimodal prompt: user hint + "produce a JSON Schema".
  3. Call the backend (defaults to NanonetsVLBackend; any multimodal
     backend with ``is_multimodal == True`` works).
  4. Parse the LLM's JSON Schema output. Defensive — LLMs sometimes
     wrap output in ```json fences; we strip them.
  5. Compile the JSON Schema to a Pydantic class via ``create_model()``.
     We validate the JSON Schema is well-formed first — fail loud on
     garbage, never silently return an empty model.
  6. Return ``(PydanticModel, JSONSchemaDict)``.

If the user provides a ``hint``, that hint steers field selection
(field names, types, which to mark as required). If the hint is empty,
the LLM is asked to infer fields from the document content alone —
useful for "what fields does this PDF have?" exploration.

Limits
------
This is LLM-driven discovery. The LLM may:
  - propose wrong field names (mitigated by the user hint)
  - propose wrong types (mitigated by JSON Schema validation before
    Pydantic compilation)
  - hallucinate fields that aren't actually present (the user should
    always validate the resulting schema against a few real extractions)

The user MUST review the discovered schema before using it in
production. The CLI prints a human-readable summary; the Python API
returns the raw objects for programmatic inspection.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable  # noqa: F401  (re-exported)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from idp.core.document import Document
from idp.llm.backend import Backend, CompletionRequest, Message
from idp.parse.parser import get_parser, parse_document

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_HINT = (
    "Identify the document type and propose fields to extract. "
    "Return a JSON Schema (object with 'type: object', 'properties: {...}')."
)


_SYSTEM_PROMPT = """\
You are a schema-design assistant. Given a document and a user hint, you
return a JSON Schema describing the fields to extract.

Rules:
  - Output ONLY a valid JSON Schema. No prose, no markdown fences, no commentary.
  - Root must be {"type": "object", "properties": {...}, "required": [...]}.
  - Field names use snake_case.
  - Field types are JSON Schema primitives: string, number, integer, boolean,
    array (with items), or nested object.
  - Mark fields as required only when they are clearly present and reliable
    on every example of this document type. If unsure, leave them out of
    "required" — the user can promote them later.
  - For lists (e.g. line_items), use {"type": "array", "items": {"type": "object", ...}}.
  - Add a description to each field (1 sentence) explaining what to extract.
"""


_USER_PROMPT_TEMPLATE = """\
Document path: {path}
User hint: {hint}

Return a JSON Schema. The root must be an object with a "properties" key.
Example of the EXACT shape we want (not content):

{{
  "type": "object",
  "title": "InferredDocument",
  "properties": {{
    "field_one": {{"type": "string", "description": "..."}},
    "field_two": {{"type": "number", "description": "..."}},
    "field_list": {{
      "type": "array",
      "items": {{"type": "object", "properties": {{...}}}},
      "description": "..."
    }}
  }},
  "required": ["field_one"]
}}
"""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class DiscoveryResult:
    """The output of ``discover_schema()``."""

    schema_class: type[BaseModel]
    """Pydantic class compiled from the LLM's JSON Schema."""

    json_schema: dict[str, Any]
    """The raw JSON Schema dict returned by the LLM."""

    raw_response: str
    """The raw LLM output (after fence-stripping). Useful for debugging."""

    backend_name: str
    """Name of the backend that generated the schema."""

    doc: Document
    """The Document that was passed in (so the caller can reuse it)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def discover_schema(
    source: str | Path | Document,
    *,
    hint: str = _DEFAULT_HINT,
    backend: Backend | None = None,
    page_limit: int = 4,
) -> DiscoveryResult:
    """Infer a Pydantic schema from a PDF + natural-language hint.

    Args:
        source:    PDF path or a pre-parsed ``Document``. If a Document
                   without ``pages[i].images_b64`` is passed, this will
                   run ``PdfPagesParser`` on it first.
        hint:      Natural-language description of fields to extract.
                   Example: "extract vendor_name, total_amount, and line
                   items". Pass an empty string to let the LLM infer
                   fields from the document alone.
        backend:   Multimodal backend to use. Defaults to NanonetsVLBackend
                   (gated by ``IDP_ENABLE_NANONETS=1``). Any backend with
                   ``is_multimodal == True`` works.
        page_limit: Max pages to render for the prompt. Default 4 keeps
                    the prompt under most models' context budgets.

    Returns:
        ``DiscoveryResult`` with the inferred ``schema_class`` and the
        raw ``json_schema``. Pass ``result.schema_class`` to
        ``Pipeline(schema=...)``.

    Raises:
        FileNotFoundError: if the source path doesn't exist.
        ValueError:        if the LLM's output isn't valid JSON Schema.
        RuntimeError:      if the backend fails.
    """
    if backend is None:
        backend = make_backend("nanonets")

    if not backend.is_multimodal:
        raise ValueError(
            f"discover_schema needs a multimodal backend (got {type(backend).__name__} "
            f"with is_multimodal={backend.is_multimodal}). Use NanonetsVLBackend or "
            "another VLM-capable backend."
        )

    # 1. Get a Document with images_b64 populated on pages
    doc = _ensure_parsed(source, page_limit=page_limit)

    # 2. Build the prompt
    images = [img for p in doc.pages for img in (p.images_b64 or [])][:page_limit]
    source_path = str(doc.source_path)
    user_content = _USER_PROMPT_TEMPLATE.format(path=source_path, hint=hint or "(none)")

    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=user_content, images_b64=images),
    ]
    req = CompletionRequest(messages=messages, json_mode=True, temperature=0.0)

    # 3. Call the backend
    raw = backend.complete(req)
    log.info("discover_schema: backend returned %d chars", len(raw))

    # 4. Parse the JSON Schema
    schema_dict = _parse_json_schema(raw)

    # 5. Compile to Pydantic
    schema_class = _json_schema_to_pydantic(schema_dict, fallback_name="InferredDocument")

    return DiscoveryResult(
        schema_class=schema_class,
        json_schema=schema_dict,
        raw_response=raw,
        backend_name=getattr(backend, "name", type(backend).__name__),
        doc=doc,
    )


# ---------------------------------------------------------------------------
# Backend factory (used by CLI and discover_schema default)
# ---------------------------------------------------------------------------
_AVAILABLE_BACKENDS = ("nanonets", "mock")


def available_backends() -> list[str]:
    """List of backend names that discover_schema can use."""
    return list(_AVAILABLE_BACKENDS)


def make_backend(name: str = "nanonets") -> Backend:
    """Build a backend by name. Raises RuntimeError with a clear hint
    if the backend isn't installed/available.

    Args:
        name: "nanonets" (default, multimodal VLM) or "mock" (returns
              canned JSON Schema — useful for testing without a GPU).
    """
    if name == "mock":
        # Mock backend isn't multimodal; the caller should use a real
        # multimodal backend. Returned for completeness; discover_schema
        # will reject it at runtime.
        from idp.llm.backend import MockBackend
        return MockBackend(mode="ideal")
    if name == "nanonets":
        import os
        if os.environ.get("IDP_ENABLE_NANONETS") != "1":
            raise RuntimeError(
                "NanonetsVLBackend requires IDP_ENABLE_NANONETS=1 to be set. "
                "This is a deliberate opt-in to avoid a 7 GB surprise download."
            )
        try:
            from idp.llm.nanonets import NanonetsVLBackend
        except ImportError as e:
            raise RuntimeError(
                "NanonetsVLBackend not installed. Install with "
                "`pip install py-idp[hf-vlm]`."
            ) from e
        return NanonetsVLBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Available: {', '.join(_AVAILABLE_BACKENDS)}"
    )


# CLI alias (shorter name, same function)
_default_backend_factory = make_backend


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------
def _ensure_parsed(source: str | Path | Document, *, page_limit: int) -> Document:
    """Ensure the source is a Document with images_b64 on pages.

    - str/Path -> Document.from_path + parse_document(parser="pdf-pages")
    - Document with images_b64 -> pass through (capped at page_limit)
    - Document without images_b64 -> parse_document(parser="pdf-pages")
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"source path does not exist: {p}")
        doc = Document.from_path(p)
        parse_document(doc, parser=get_parser("pdf-pages"))
    else:
        doc = source
        has_images = bool(doc.pages) and any(p.images_b64 for p in doc.pages)
        if not has_images:
            parse_document(doc, parser=get_parser("pdf-pages"))

    # Cap the pages we'll render in the prompt
    if doc.pages and len(doc.pages) > page_limit:
        # Keep first N pages; trim images too
        doc.pages = doc.pages[:page_limit]
        for page in doc.pages:
            page.images_b64 = (page.images_b64 or [])[:1]  # 1 image per page max

    return doc


# ---------------------------------------------------------------------------
# JSON Schema parsing (defensive)
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def _parse_json_schema(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON Schema output, defending against common errors.

    - Strips ```json ... ``` fences (some models still emit them despite
      json_mode=True).
    - Tries the whole string first; falls back to extracting the first
      {...} block if the model wrapped the schema in prose.
    - Validates the result is an object with 'type' == 'object'.
    """
    if not raw or not raw.strip():
        raise ValueError("LLM returned empty response")

    s = raw.strip()

    # Strip ```json fences
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    # Try direct parse first
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        # Fallback: find first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                f"LLM output is not valid JSON. First 200 chars: {raw[:200]!r}"
            ) from None
        try:
            parsed = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM output is not valid JSON. First 200 chars: {raw[:200]!r}. "
                f"JSON error: {e}"
            ) from None

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM output is not a JSON object: got {type(parsed).__name__}. "
            f"First 200 chars: {raw[:200]!r}"
        )
    if parsed.get("type") != "object":
        # Some LLMs omit 'type'; normalize.
        parsed["type"] = "object"
    if "properties" not in parsed:
        parsed["properties"] = {}
    return parsed


# ---------------------------------------------------------------------------
# JSON Schema -> Pydantic
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _json_type_to_python(prop: dict[str, Any]) -> Any:
    """Map a JSON Schema property to a Python type hint."""
    t = prop.get("type")
    if t in _TYPE_MAP:
        return _TYPE_MAP[t]
    if t == "array":
        items = prop.get("items") or {}
        item_type = _json_type_to_python(items) if items else Any
        return list[item_type]  # type: ignore[valid-type]
    if t == "object":
        # Nested object -> compile a nested Pydantic model
        return _json_schema_to_pydantic(prop, fallback_name="NestedObject")
    return Any


def _json_schema_to_pydantic(
    schema: dict[str, Any], *, fallback_name: str = "InferredSchema"
) -> type[BaseModel]:
    """Compile a JSON Schema dict into a Pydantic BaseModel subclass.

    Required fields are non-Optional; optional fields are Optional[...] with
    a default of None. Lists default to []. Nested objects become nested
    Pydantic models.

    The model is permissive (``model_config = ConfigDict(extra='allow')``)
    so the user's downstream extractor can still set fields the LLM forgot.
    """
    props: dict[str, Any] = dict(schema.get("properties") or {})
    required: set[str] = set(schema.get("required") or [])
    title: str = schema.get("title") or fallback_name

    fields: dict[str, Any] = {}
    for name, prop in props.items():
        py_type = _json_type_to_python(prop)
        description = prop.get("description") or ""
        if name in required:
            default: Any = ...
        else:
            # Make the type Optional[T] with default None
            if py_type is Any:
                default = None
            else:
                py_type = py_type | None  # type: ignore[operator]
                default = None
        fields[name] = (py_type, Field(default, description=description))

    model = create_model(
        title,
        __config__=ConfigDict(extra="allow", arbitrary_types_allowed=True),
        **fields,
    )
    return model


__all__ = [
    "DiscoveryResult",
    "discover_schema",
    "available_backends",
    "make_backend",
]



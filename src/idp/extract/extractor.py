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

"""Extract stage.

This is the core of the framework. Two call paths:
  - multimodal: pass page images to a VLM (Qwen2.5-VL, GPT-4o, Claude Vision)
  - ocr_llm:    pass parsed text to an LLM

Both paths produce output that should validate against the user's
Pydantic schema. Validation is handled in the next stage; here we
just collect the JSON dict.

Prompt design follows the AWS GenAI IDP Accelerator pattern:
  system: role + behavior
  user:   instructions + JSON Schema + output rules + content
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from idp.core.document import Document
from idp.core.types import ExtractionMode
from idp.llm.backend import Backend, CompletionRequest, Message

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise document-extraction assistant. Your job is to read \
the provided document content and return a single JSON object that exactly matches the \
requested schema.

Rules:
1. Set a field to `null` if it is not present in the document. Do NOT fabricate.
2. Preserve original spelling, formatting, and currency symbols from the source.
3. If a number is ambiguous, set it to `null` rather than guess.
4. Output ONLY valid JSON — no commentary, no markdown fences, no apologies.
5. If multiple values exist for a single field (e.g. multiple dates), pick the \
first explicit occurrence in document order.
"""


def _build_messages(
    schema: type[BaseModel],
    text: str,
    images_b64: list[str],
    extra_instructions: str = "",
) -> list[Message]:
    """Build the chat messages for an extraction call."""
    schema_name = schema.__name__
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    user_content = f"""Extract data from the document below into the schema `{schema_name}`.

Output JSON Schema:
{schema_json}

Output rules:
- Return a single JSON object — no prose, no markdown fences.
- Numbers stay numeric. Strings stay strings. Dates stay in their original format.
- Missing fields -> `null`. Do NOT invent.
- For arrays (e.g. line items), include every row you can identify in document order.

{extra_instructions}

Document content:
\"\"\"
{text[:8000]}
\"\"\""""
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user_content, images_b64=images_b64),
    ]


def _safe_load(raw: str) -> dict[str, Any]:
    """Parse a model's JSON response into a dict.

    Differences from `_safe_json` (the llm helper):
      - only used by the extract stage, where `dict[str, Any]` is the
        schema target, and we WANT to fail loudly when the model emits
        non-JSON or text before/after JSON.
      - Returns `{"_error": ..., "_raw": ...}` on any parse failure so
        the caller's `schema.model_validate` raises — which then surfaces
        in `doc.errors` instead of being silently treated as a valid
        empty extraction.
    """
    s = (raw or "").strip()
    if not s:
        return {"_error": "empty model output", "_raw": raw}
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"_value": v}
    except Exception:  # noqa: BLE001
        # Only try the brace-fallback when the model wrapped the JSON in
        # markdown. Otherwise we'd silently eat garbage and Pydantic would
        # see `{"_raw": ...}` as a valid extraction.
        m = re.search(r"^```(?:json)?\s*(\{.*\})\s*```", raw or "", re.S)
        if m:
            try:
                v = json.loads(m.group(1))
                return v if isinstance(v, dict) else {"_value": v}
            except Exception:  # noqa: BLE001
                pass
        return {"_error": "could not parse JSON", "_raw": raw}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def render_first_n_pages_to_images(doc: Document, n: int = 3) -> list[str]:
    """Best-effort render of the first n pages to base64 PNGs.

    Returns an empty list when:
      - the doc is not a PDF
      - pdf2image / poppler isn't available
      - rendering fails for any reason

    Failures log a warning (visible at INFO+); they are NOT silently
    swallowed because the user might be expecting a multimodal call and
    need to know the multimodal path degraded to text-only.
    """
    if doc.extension != "pdf":
        return []
    if n <= 0:
        return []
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError as e:
        log.warning(
            "could not render PDF pages to images (install pdf2image + poppler "
            "to use multimodal extraction): %s",
            e,
        )
        return []
    out: list[str] = []
    try:
        pages = convert_from_path(doc.source_path, dpi=150, first_page=1, last_page=n)
        from io import BytesIO
        for img in pages:
            buf = BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
    except Exception as e:  # noqa: BLE001
        log.warning("pdf rendering failed for %s: %s", doc.source_path, e)
    return out


# ---------------------------------------------------------------------------
# Extract entrypoint
# ---------------------------------------------------------------------------
def extract(
    doc: Document,
    schema: Type[BaseModel],
    backend: Backend,
    mode: ExtractionMode | str | None = None,
    extra_instructions: str = "",
) -> Document:
    """Run the extraction stage. Sets doc.extraction and doc.extraction_schema."""
    if mode is None:
        mode = doc.mode or ExtractionMode.OCR_LLM
    if isinstance(mode, str):
        mode = ExtractionMode(mode)

    images_b64: list[str] = []
    text = doc.raw_text or ""
    if mode == ExtractionMode.MULTIMODAL and backend.is_multimodal:
        images_b64 = render_first_n_pages_to_images(doc, n=3)

    # Short-circuit on empty inputs: don't burn an LLM call when the
    # parser found nothing. The caller still gets a schema-shaped stub
    # back so downstream stages behave consistently, and the short-circuit
    # is logged via doc.errors so it's visible in the HITL review.
    if not text and not images_b64:
        doc.errors.append(
            "extract_skipped: empty document (no text from parser, no images); "
            "returned schema-shaped nulls without calling LLM"
        )
        validated: dict[str, Any] = {}
        try:
            validated = schema.model_validate({}).model_dump(mode="json")
        except Exception:  # noqa: BLE001
            # Schema requires non-Optional fields; return raw nulls.
            validated = {"_raw": "<empty input>"}
        doc.extraction = validated
        doc.extraction_schema = schema.__name__
        doc.mode = mode.value if isinstance(mode, ExtractionMode) else str(mode)
        return doc

    messages = _build_messages(schema, text, images_b64, extra_instructions=extra_instructions)
    req = CompletionRequest(messages=messages, json_mode=True, temperature=0.0)
    raw = ""
    try:
        raw = backend.complete(req)
    except Exception as e:  # noqa: BLE001
        doc.errors.append(f"extract_failed: {e}")
        raw = ""

    # Coerce through the user's schema; keep raw as a fallback. If the
    # backend returned garbage instead of JSON, surface that — DON'T let
    # `_safe_load`'s `{"_error": ..., "_raw": ...}` shape be silently
    # coerced into a "valid" empty extraction by the user's `Optional`
    # fields. That was a data-loss bug in v0.1.
    extracted = _safe_load(raw)
    raw_had_error = "_error" in extracted

    validated: dict[str, Any] = {}
    try:
        validated_obj = schema.model_validate(extracted)
        validated = validated_obj.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        log.debug("schema validation at extract time failed: %s", e)
        validated = extracted if isinstance(extracted, dict) else {"_raw": str(extracted)}
        doc.errors.append(f"extract_schema_unvalidated: {e}")

    # Loud failure surfacing: if the backend produced non-JSON AND Pydantic
    # didn't raise, the result might still be a dict-shaped "looks valid"
    # extraction. Mark it explicitly so the caller can short-circuit to
    # HITL or retry.
    if raw_had_error:
        doc.errors.append(
            f"extract_json_parse_failed: {extracted.get('_error')!r}; "
            "extraction is fall-back nulls, do not trust without review"
        )
        # collapse the extraction to an explicit marker so HITL sees red
        if isinstance(validated, dict) and validated and all(v is None for v in validated.values()):
            validated = {"_parse_failed": True, "_raw": extracted.get("_raw")}

    doc.extraction = validated
    doc.extraction_schema = schema.__name__
    doc.mode = mode.value if isinstance(mode, ExtractionMode) else str(mode)
    return doc

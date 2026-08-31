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
    """Parse JSON robustly, tolerating ```json fences and trailing prose."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # try direct
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        # grab the first {...} block
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
        return {"_raw": raw, "_error": "could not parse JSON"}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def render_first_n_pages_to_images(doc: Document, n: int = 3) -> list[str]:
    """Best-effort render of the first n pages to base64 PNGs.

    Falls back to empty list if no renderer is available (PDFs without
    poppler/pdf2image installed). Multimodal calls then degrade gracefully
    to text-only via the LLM backend.
    """
    out: list[str] = []
    try:
        import pdf2image  # type: ignore
        from pdf2image import convert_from_path  # type: ignore

        if doc.extension != "pdf":
            return []
        pages = convert_from_path(doc.source_path, dpi=150, first_page=1, last_page=n)
        for img in pages:
            from io import BytesIO

            buf = BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
    except Exception as e:  # noqa: BLE001
        log.debug("could not render pages to images: %s", e)
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

    messages = _build_messages(schema, text, images_b64, extra_instructions=extra_instructions)
    req = CompletionRequest(messages=messages, json_mode=True, temperature=0.0)
    try:
        raw = backend.complete(req)
        extracted = _safe_load(raw)
    except Exception as e:  # noqa: BLE001
        doc.errors.append(f"extract_failed: {e}")
        extracted = {"_error": str(e)}

    # coerce through the user's schema; keep raw as a fallback
    validated: dict[str, Any] = {}
    try:
        validated = schema.model_validate(extracted).model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        log.debug("schema validation at extract time failed: %s", e)
        validated = extracted if isinstance(extracted, dict) else {"_raw": str(extracted)}
        doc.errors.append(f"extract_schema_unvalidated: {e}")

    doc.extraction = validated
    doc.extraction_schema = schema.__name__
    doc.mode = mode.value if isinstance(mode, ExtractionMode) else str(mode)
    return doc

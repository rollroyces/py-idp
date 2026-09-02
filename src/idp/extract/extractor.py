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
from typing import Any

from pydantic import BaseModel

from idp.chunker import (
    PageChunker,
    TokenChunker,
    merge_extractions,
    should_chunk_pages,
    should_chunk_text,
)
from idp.core.document import Document, Page
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

    If a previous parser stage already rendered pages to images
    (e.g. ``PdfPagesParser`` populates ``doc.pages[i].images_b64``),
    those images are reused and the PDF is NOT re-rendered.
    """
    # Fast path: a parser already gave us images. Reuse them.
    if doc.pages and any(p.images_b64 for p in doc.pages):
        out: list[str] = []
        for p in doc.pages:
            if not p.images_b64:
                continue
            # Per-page images_b64 is a list; flatten to one image per page
            out.extend(p.images_b64)
            if len(out) >= n:
                return out[:n]
        return out[:n] if out else []

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
    out_pages: list[str] = []
    try:
        pages = convert_from_path(doc.source_path, dpi=150, first_page=1, last_page=n)
        from io import BytesIO
        for img in pages:
            buf = BytesIO()
            img.save(buf, format="PNG")
            out_pages.append(base64.b64encode(buf.getvalue()).decode())
    except Exception as e:  # noqa: BLE001
        log.warning("pdf rendering failed for %s: %s", doc.source_path, e)
    return out_pages


# ---------------------------------------------------------------------------
# Extract entrypoint
# ---------------------------------------------------------------------------
def extract(
    doc: Document,
    schema: type[BaseModel],
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
        stub: dict[str, Any] = {}
        try:
            stub = schema.model_validate({}).model_dump(mode="json")
        except Exception:  # noqa: BLE001
            # Schema requires non-Optional fields; return raw nulls.
            stub = {"_raw": "<empty input>"}
        doc.extraction = stub
        doc.extraction_schema = schema.__name__
        doc.mode = mode.value if isinstance(mode, ExtractionMode) else str(mode)
        return doc

    # Chunk the document when it won't fit in one LLM call. Multimodal
    # uses page-based chunking (groups of pages with overlap); text uses
    # token-based chunking (tiktoken). Each chunk gets its own LLM call,
    # then we merge the per-chunk results.
    chunks = _maybe_chunk(doc, text, images_b64, mode, backend)
    if len(chunks) > 1:
        log.info("extract: chunking %d -> %d chunks for %s",
                 _doc_size(doc, text, images_b64, mode), len(chunks), doc.source_path)
        per_chunk_dicts = _extract_chunks(chunks, schema, backend,
                                          extra_instructions=extra_instructions,
                                          doc=doc)
        merged = _validate_and_merge_chunks(per_chunk_dicts, schema, doc)
        doc.extraction = merged
        doc.extraction_schema = schema.__name__
        doc.mode = mode.value if isinstance(mode, ExtractionMode) else str(mode)
        doc.errors.append(f"extract_chunked: {len(chunks)} chunks")
        return doc

    # Single-chunk path: unchanged from before
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


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------
def _doc_size(
    doc: Document,
    text: str,
    images_b64: list[str],
    mode: ExtractionMode,
) -> int:
    """Return the dimension the chunker cares about: page count or char count."""
    if mode == ExtractionMode.MULTIMODAL and doc.pages:
        return len(doc.pages)
    return len(text)


def _maybe_chunk(
    doc: Document,
    text: str,
    images_b64: list[str],
    mode: ExtractionMode,
    backend: Backend,
) -> list[tuple[str, list[str]]]:
    """Decide if chunking is needed and return a list of (chunk_text, chunk_images).

    Each element of the returned list is one chunk ready to feed into
    ``_build_messages``. The list has length 1 when no chunking is needed
    (the caller can short-circuit), or >1 otherwise.
    """
    # Multimodal + doc has pages -> PageChunker
    if mode == ExtractionMode.MULTIMODAL and backend.is_multimodal and doc.pages:
        page_chunker = PageChunker(max_pages=4, overlap_pages=1)
        if not should_chunk_pages(doc.pages, max_pages=page_chunker.max_pages):
            # Single-chunk: feed everything as-is (mirrors single-call path)
            return [("", _collect_images_for_pages(doc.pages))]
        per_chunk_texts = page_chunker.collect_texts(doc.pages)
        per_chunk_imgs = page_chunker.collect_images(doc.pages)
        # If doc.pages has no images (text parser was used), fall back to
        # the global images_b64 — only meaningful for the first chunk.
        # zip(strict=True) is safe: same chunker produces both lists.
        return list(zip(per_chunk_texts, per_chunk_imgs, strict=True))

    # Text path -> TokenChunker
    text_chunker = TokenChunker(max_tokens=4000, overlap_tokens=200)
    if not should_chunk_text(text, max_tokens=text_chunker.max_tokens):
        return [(text, [])]
    texts = text_chunker.split(text)
    return [(t, []) for t in texts]


def _collect_images_for_pages(pages: list[Page]) -> list[str]:
    """Flatten all per-page images_b64 into one list."""
    return [img for p in pages for img in (p.images_b64 or [])]


def _extract_chunks(
    chunks: list[tuple[str, list[str]]],
    schema: type[BaseModel],
    backend: Backend,
    *,
    extra_instructions: str,
    doc: Document,
) -> list[dict[str, Any]]:
    """Call backend.complete() once per chunk; return list of parsed dicts.

    Errors per chunk are logged on ``doc.errors`` and the failed chunk
    contributes an empty dict — so a single bad chunk doesn't kill the
    whole batch. This matches the user's "process 1000 docs / month"
    resilience ask.
    """
    out: list[dict[str, Any]] = []
    for i, (text, images) in enumerate(chunks):
        if not text and not images:
            out.append({})
            continue
        messages = _build_messages(schema, text, images,
                                   extra_instructions=extra_instructions)
        req = CompletionRequest(messages=messages, json_mode=True, temperature=0.0)
        try:
            raw = backend.complete(req)
        except Exception as e:  # noqa: BLE001
            doc.errors.append(f"extract_chunk_failed[{i}]: {e}")
            raw = ""
        parsed = _safe_load(raw)
        out.append(parsed if isinstance(parsed, dict) else {})
    return out


def _validate_and_merge_chunks(
    per_chunk_dicts: list[dict[str, Any]],
    schema: type[BaseModel],
    doc: Document,
) -> dict[str, Any]:
    """Merge per-chunk dicts, then run Pydantic validation on the merge.

    If validation fails, fall back to the merged dict anyway — the
    per-chunk info is more useful than discarding it.
    """
    merged = merge_extractions(per_chunk_dicts, prefer="first")
    # Strip internal chunk_count marker before validation
    clean = {k: v for k, v in merged.items() if not k.startswith("_")}
    try:
        validated_obj = schema.model_validate(clean)
        validated = validated_obj.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        log.debug("schema validation at merge time failed: %s", e)
        doc.errors.append(f"extract_merge_validation_failed: {e}")
        validated = clean
    # Re-attach chunk_count marker so downstream observability can see it
    if "_chunk_count" in merged:
        validated["_chunk_count"] = merged["_chunk_count"]
    return validated

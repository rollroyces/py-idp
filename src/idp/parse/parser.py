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

"""Parse stage.

Three parsers ship out of the box, all conforming to the same interface:
    parse(path: str | Path) -> ParsedDocument

ParsedDocument = dict with keys:
    text:    str  (full document text)
    pages:   list[dict]   (one entry per page, with text + image_path)
    tables:  list[dict]   (extracted tables in markdown + rows)
    metadata: dict

Parsers:
  - DoclingParser  (best PDF tables; requires 'docling' extra)
  - PdfPlumberParser (lightweight fallback, no extra deps)
  - PlainTextParser (txt/md/eml)

Strategy:
  Use Docling when the doc is a PDF and the extra is installed; fall back
  to pdfplumber otherwise. TXT/HTML/email go through dedicated handlers.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Protocol

from idp.core.document import Document, Page

log = logging.getLogger(__name__)


class Parser(Protocol):
    def parse(self, path: str | Path) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Plain text / email / markdown
# ---------------------------------------------------------------------------
class PlainTextParser:
    """Handles .txt, .md, .markdown, .eml by splitting on blank lines for pages."""

    name = "plain"

    def parse(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Synthetic "pages" = chunks of ~3000 chars (LLM token windowing heuristic)
        chunk = 3000
        pages: list[dict[str, Any]] = []
        for i in range(0, len(text), chunk):
            seg = text[i : i + chunk]
            pages.append({"page": len(pages) + 1, "text": seg, "image_path": None})
        return {
            "text": text,
            "pages": pages,
            "tables": [],
            "metadata": {"parser": "plain", "size": p.stat().st_size},
        }


# ---------------------------------------------------------------------------
# pdfplumber fallback (lightweight, no docling dep)
# ---------------------------------------------------------------------------
class PdfPlumberParser:
    """Lightweight PDF parser using pdfplumber. Good enough for text-heavy PDFs."""

    name = "pdfplumber"

    def parse(self, path: str | Path) -> dict[str, Any]:
        import pdfplumber

        p = Path(path)
        pages: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        full_text_parts: list[str] = []
        with pdfplumber.open(str(p)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                full_text_parts.append(txt)
                # naive table extraction on each page
                page_tables = []
                try:
                    for t in page.extract_tables() or []:
                        page_tables.append(t)
                        tables.append({"page": i, "rows": t})
                except Exception as e:  # noqa: BLE001
                    log.debug("table extraction failed on page %d: %s", i, e)
                pages.append({"page": i, "text": txt, "image_path": None})
        return {
            "text": "\n\n".join(full_text_parts),
            "pages": pages,
            "tables": tables,
            "metadata": {"parser": "pdfplumber", "size": p.stat().st_size},
        }


# ---------------------------------------------------------------------------
# Docling (default for PDFs when available)
# ---------------------------------------------------------------------------
class DoclingParser:
    """IBM Docling wrapper. Best PDF table + reading-order extraction.

    Heavy install (~500MB). Falls back to pdfplumber if not present.
    """

    name = "docling"

    def __init__(self):
        try:
            from docling.document_converter import (
                DocumentConverter,  # type: ignore[import-not-found]
            )
        except ImportError as e:
            raise ImportError(
                "Install Docling: pip install py-idp[docling]"
            ) from e
        self._converter = DocumentConverter()

    def parse(self, path: str | Path) -> dict[str, Any]:
        from docling_core.types.doc.base import (
            ImageRefMode,  # type: ignore[import-not-found]  # noqa: F401
        )

        result = self._converter.convert(str(path))
        doc = result.document
        text = doc.export_to_markdown()
        # Docling produces a single concatenated text for v1; split by form-feed
        # or by '# Page X' markers if present.
        pages: list[dict[str, Any]] = []
        for i, page in enumerate(doc.pages, start=1):
            page_text_parts: list[str] = []
            for item in page.main_text or []:
                if hasattr(item, "text"):
                    page_text_parts.append(item.text)
            pages.append(
                {
                    "page": i,
                    "text": "\n".join(page_text_parts),
                    "image_path": None,
                }
            )
        # Docling returns tables at the top level
        tables = []
        for i, t in enumerate(getattr(doc, "tables", []) or []):
            tables.append(
                {
                    "page": i,
                    "rows": getattr(t, "data", []),
                    "markdown": t.export_to_markdown() if hasattr(t, "export_to_markdown") else "",
                }
            )
        return {
            "text": text,
            "pages": pages if pages else [{"page": 1, "text": text, "image_path": None}],
            "tables": tables,
            "metadata": {"parser": "docling", "size": Path(path).stat().st_size},
        }


# ---------------------------------------------------------------------------
# Parser router
# ---------------------------------------------------------------------------
def get_parser(name: str = "auto") -> Parser:
    """Resolve a parser. 'auto' picks: docling > pdfplumber > plain."""
    if name == "auto":
        try:
            import docling  # type: ignore[import-not-found]  # noqa: F401

            return DoclingParser()
        except ImportError:
            pass
        try:
            import pdfplumber  # noqa: F401

            return PdfPlumberParser()
        except ImportError:
            pass
        return PlainTextParser()
    if name == "docling":
        return DoclingParser()
    if name == "pdfplumber":
        return PdfPlumberParser()
    if name == "plain":
        return PlainTextParser()
    raise ValueError(f"Unknown parser: {name}")


def parse_document(doc: Document, parser: Parser | None = None) -> Document:
    """Run a parser and attach the result onto the Document.

    If `parser` is None, auto-resolves by file extension (NOT just
    'try docling first'): plain text files go to PlainTextParser,
    pdfs to docling > pdfplumber.
    """
    if parser is None:
        parser = _auto_pick(doc)
    try:
        result = parser.parse(doc.source_path)
    except Exception as e:
        doc.errors.append(f"parse_failed: {e}")
        result = {"text": "", "pages": [], "tables": [], "metadata": {"error": str(e)}}
    doc.parsed_pages = result["pages"]
    doc.raw_text = result["text"]
    doc.metadata["parser"] = result["metadata"].get(
        "parser", parser.name if hasattr(parser, "name") else "?"
    )
    # populate doc.pages for downstream stages
    doc.pages = []
    for p in result["pages"]:
        doc.pages.append(
            Page(
                page_number=p["page"],
                text=p["text"],
                image_path=p.get("image_path"),
            )
        )
    return doc


def _auto_pick(doc: Document) -> Parser:
    """Choose parser by file extension. PDF -> docling > pdfplumber; else plain.

    PlainTextParser handles: txt, md, html, eml, csv, log, and any
    extension we don't otherwise recognize.
    """
    ext = (doc.extension or "").lower()
    if ext == "pdf":
        try:
            return DoclingParser()
        except ImportError:
            return PdfPlumberParser()
    if ext in ("png", "jpg", "jpeg", "tiff", "tif", "bmp"):
        # Raster images: no PDF parser will help — use plain (OCR is optional).
        # OCR path would go here; for now route to plain stub.
        with contextlib.suppress(ImportError):
            import pytesseract  # type: ignore[import-not-found]  # noqa: F401
    # Everything else (txt, md, html, eml, csv, no-ext, docx, xlsx, ...) -> plain
    return PlainTextParser()

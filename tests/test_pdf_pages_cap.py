"""Regression tests for IDP_MAX_PDF_PAGES enforcement in parse_document (P1 audit fix).

The setting exists in Settings (default 100, env IDP_MAX_PDF_PAGES) but
was previously NEVER enforced — a 10,000-page PDF would happily trigger
Docling for 30 minutes, exhausting memory. The fix:

  * New helper ``_count_pdf_pages(path) -> int | None`` uses pdfplumber
    (cheap page-tree read, no text extraction) to get the count.
  * ``parse_document`` calls it before invoking the parser and raises
    ``DocumentParseError`` when the cap is exceeded.
  * If pdfplumber isn't installed, the check is skipped silently
    (returns None) — best-effort, not a hard security boundary.

These tests:
  * Build real PDF files with a known page count.
  * Confirm the cap fires with a small IDP_MAX_PDF_PAGES (e.g. 2) and
    that a within-cap PDF still parses.
  * Confirm the cap is skipped when pdfplumber is unavailable.
"""
from __future__ import annotations

from typing import Any

import pytest


def _make_pdf(path, num_pages: int) -> str:
    """Create a minimal PDF with ``num_pages`` blank pages.

    Uses the simplest possible structure that pdfplumber can parse.
    The minimal PDF has 5 required trailer objects; for blank pages we
    just duplicate the page reference N times in /Pages /Kids.
    """
    # PDF 1.4 spec minimal structure. We embed num_pages blank /Page
    # entries. Content is intentionally trivial (just a single stream
    # per page that's empty). This produces a real PDF that pdfplumber
    # can open and count.
    lines = [
        "%PDF-1.4",
        "1 0 obj",
        "<< /Type /Catalog /Pages 2 0 R >>",
        "endobj",
        "2 0 obj",
    ]
    # Build a /Pages tree referencing pages 3..(2+num_pages)
    page_refs = " ".join(f"{i} 0 R" for i in range(3, 3 + num_pages))
    lines.append(f"<< /Type /Pages /Kids [{page_refs}] /Count {num_pages} >>")
    lines.append("endobj")

    # Each page: /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
    body = []
    xref_offsets: list[int] = []
    pos = 0
    out = "\n".join(lines) + "\n"
    pos = len(out.encode("latin-1"))

    for i in range(3, 3 + num_pages):
        xref_offsets.append(pos)
        page_obj = (
            f"{i} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
            "endobj\n"
        )
        out += page_obj
        pos += len(page_obj.encode("latin-1"))

    xref_pos = pos
    out += f"xref\n0 {3 + num_pages}\n"
    out += "0000000000 65535 f \n"
    for off in xref_offsets:
        out += f"{off:010d} 00000 n \n"
    out += (
        "trailer\n"
        f"<< /Size {3 + num_pages} /Root 1 0 R >>\n"
        "startxref\n"
        f"{xref_pos}\n"
        "%%EOF\n"
    )
    path.write_bytes(out.encode("latin-1"))
    return str(path)


# ---------------------------------------------------------------------------
# _count_pdf_pages
# ---------------------------------------------------------------------------
def test_count_pdf_pages_returns_correct_count(tmp_path):
    from idp.parse.parser import _count_pdf_pages

    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, 3)
    n = _count_pdf_pages(pdf)
    assert n == 3, f"expected 3 pages, got {n}"


def test_count_pdf_pages_returns_none_when_pdfplumber_missing(monkeypatch, tmp_path):
    """If pdfplumber isn't installed, return None (skip the check)."""
    from idp.parse import parser as parser_module

    # Simulate ImportError by patching the import.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pdfplumber" or name.startswith("pdfplumber."):
            raise ImportError("simulated missing pdfplumber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, 3)
    assert parser_module._count_pdf_pages(pdf) is None


def test_count_pdf_pages_returns_none_for_corrupt_file(tmp_path):
    from idp.parse.parser import _count_pdf_pages

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a real pdf")
    # pdfplumber raises on corrupt files; we surface None (best-effort).
    assert _count_pdf_pages(bad) is None


# ---------------------------------------------------------------------------
# parse_document enforces the cap
# ---------------------------------------------------------------------------
def test_parse_document_rejects_oversized_pdf(tmp_path, monkeypatch):
    """A 5-page PDF must fail parse when IDP_MAX_PDF_PAGES=2."""
    monkeypatch.setenv("IDP_MAX_PDF_PAGES", "2")
    from idp.config import Settings

    Settings.load()  # ensure validator sees the env (no-op for our check)
    from idp.core.document import Document
    from idp.errors import DocumentParseError
    from idp.parse.parser import PlainTextParser, parse_document

    pdf = tmp_path / "big.pdf"
    _make_pdf(pdf, 5)
    doc = Document.from_path(str(pdf))
    # Use plain parser so we don't need docling; the cap fires before
    # the parser is invoked regardless.
    with pytest.raises(DocumentParseError) as excinfo:
        parse_document(doc, parser=PlainTextParser())
    assert "5" in str(excinfo.value)
    assert "IDP_MAX_PDF_PAGES=2" in str(excinfo.value)


def test_parse_document_allows_under_cap_pdf(tmp_path, monkeypatch):
    """A 3-page PDF passes when IDP_MAX_PDF_PAGES=10."""
    monkeypatch.setenv("IDP_MAX_PDF_PAGES", "10")
    from idp.core.document import Document
    from idp.parse.parser import PlainTextParser, parse_document

    pdf = tmp_path / "small.pdf"
    _make_pdf(pdf, 3)
    doc = Document.from_path(str(pdf))
    # Should not raise. Plain parser on a 3-page blank PDF returns "" text
    # but the parse call itself succeeds.
    out = parse_document(doc, parser=PlainTextParser())
    assert out is not None


def test_parse_document_no_cap_means_no_check(tmp_path, monkeypatch):
    """If the env var isn't set, the default cap (100) is used."""
    monkeypatch.delenv("IDP_MAX_PDF_PAGES", raising=False)
    from idp.core.document import Document
    from idp.parse.parser import PlainTextParser, parse_document

    pdf = tmp_path / "ok.pdf"
    _make_pdf(pdf, 3)
    doc = Document.from_path(str(pdf))
    # Should not raise (3 < 100 default).
    out = parse_document(doc, parser=PlainTextParser())
    assert out is not None


def test_parse_document_skips_cap_when_pdfplumber_missing(tmp_path, monkeypatch):
    """Best-effort: if pdfplumber isn't installed, the check is skipped."""
    monkeypatch.setenv("IDP_MAX_PDF_PAGES", "2")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pdfplumber" or name.startswith("pdfplumber."):
            raise ImportError("simulated missing pdfplumber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from idp.core.document import Document
    from idp.parse.parser import PlainTextParser, parse_document

    pdf = tmp_path / "big.pdf"
    _make_pdf(pdf, 5)
    doc = Document.from_path(str(pdf))
    # Should NOT raise even though the PDF has 5 pages (cap=2). The
    # check is skipped silently when pdfplumber is unavailable.
    out = parse_document(doc, parser=PlainTextParser())
    assert out is not None


def test_parse_document_non_pdf_not_subject_to_cap(tmp_path, monkeypatch):
    """Only PDF files are gated by the page cap; txt/md/etc. are not."""
    monkeypatch.setenv("IDP_MAX_PDF_PAGES", "2")
    from idp.core.document import Document
    from idp.parse.parser import PlainTextParser, parse_document

    txt = tmp_path / "doc.txt"
    txt.write_text("hello world " * 10000, encoding="utf-8")
    doc = Document.from_path(str(txt))
    # 10000 lines is a lot, but it's a .txt — cap doesn't apply.
    out = parse_document(doc, parser=PlainTextParser())
    assert out is not None


def test_parse_document_at_exact_cap_is_allowed(tmp_path, monkeypatch):
    """Cap is strict greater-than: pages == cap should pass, pages == cap+1 fails."""
    monkeypatch.setenv("IDP_MAX_PDF_PAGES", "3")
    from idp.core.document import Document
    from idp.parse.parser import PlainTextParser, parse_document

    pdf = tmp_path / "exact.pdf"
    _make_pdf(pdf, 3)
    doc = Document.from_path(str(pdf))
    # 3 == 3 -> allowed
    out = parse_document(doc, parser=PlainTextParser())
    assert out is not None

    # 4 > 3 -> rejected
    pdf4 = tmp_path / "over.pdf"
    _make_pdf(pdf4, 4)
    doc4 = Document.from_path(str(pdf4))
    from idp.errors import DocumentParseError

    with pytest.raises(DocumentParseError):
        parse_document(doc4, parser=PlainTextParser())
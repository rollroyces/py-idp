"""E3: parser coverage — PdfPlumber + Docling (skip-if-missing).

PdfPlumber is installed in the test env; Docling is optional.
Tests use a real generated PDF fixture so the PdfPlumber path actually runs.
"""
from __future__ import annotations

import pytest

from idp.core.document import Document
from idp.parse.parser import (
    DoclingParser,
    PdfPlumberParser,
    PlainTextParser,
    get_parser,
    parse_document,
)


def _make_pdf(tmp_path, lines: list[str]) -> str:
    """Generate a minimal one-page PDF with pdfplumber is NOT enough — we
    need a real PDF file. Use pdfplumber? No — use a tiny hand-written PDF.

    We build a minimal valid PDF by hand. This is a text-only PDF with one
    page and no fonts, which pdfplumber can read the text of via its
    low-level parser only if there's an actual content stream.

    Simpler: use pdfplumber to GENERATE? pdfplumber can't write.

    Fallback: use `reportlab` if available, else a fixture shipped in-tree.
    """
    # Use reportlab if available
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(path), pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        for line in lines:
            c.drawString(72, y, line)
            y -= 20
        c.save()
        return str(path)
    except ImportError:
        pytest.skip("reportlab not installed; cannot generate PDF fixture")


@pytest.fixture
def sample_pdf(tmp_path):
    return _make_pdf(tmp_path, [
        "INVOICE INV-2026-001",
        "Vendor: Acme Widgets Ltd.",
        "Total: $540.00",
        "Currency: USD",
    ])


# ---------------------------------------------------------------------------
# PdfPlumberParser
# ---------------------------------------------------------------------------
def test_pdfplumber_parses_real_pdf(sample_pdf):
    p = PdfPlumberParser()
    result = p.parse(sample_pdf)
    assert result["text"], "pdfplumber should extract text from a real PDF"
    assert "Acme" in result["text"] or "INVOICE" in result["text"]
    assert len(result["pages"]) >= 1
    assert result["metadata"]["parser"] == "pdfplumber"


def test_pdfplumber_returns_empty_on_non_pdf(tmp_path):
    txt = tmp_path / "x.txt"
    txt.write_text("hello")
    p = PdfPlumberParser()
    # pdfplumber raises on non-PDF; parse_document should catch and record error
    doc = Document.from_path(str(txt))
    parse_document(doc, parser=p)
    assert "parse_failed" in doc.errors[0]


# ---------------------------------------------------------------------------
# PlainTextParser
# ---------------------------------------------------------------------------
def test_plain_text_parser_pages_split(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("word " * 10000)  # ~50KB
    p = PlainTextParser()
    result = p.parse(str(big))
    assert len(result["pages"]) > 1
    assert result["metadata"]["parser"] == "plain"


def test_plain_text_parser_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    p = PlainTextParser()
    result = p.parse(str(empty))
    assert result["text"] == ""
    assert len(result["pages"]) == 0


# ---------------------------------------------------------------------------
# get_parser auto-resolve
# ---------------------------------------------------------------------------
def test_get_parser_auto_resolves_to_pdfplumber():
    """pdfplumber is installed -> auto should pick it for PDFs (docling absent)."""
    try:
        import docling  # noqa: F401
        has_docling = True
    except ImportError:
        has_docling = False
    p = get_parser("auto")
    if has_docling:
        assert isinstance(p, DoclingParser)
    else:
        assert isinstance(p, PdfPlumberParser)


def test_get_parser_explicit_plain():
    assert isinstance(get_parser("plain"), PlainTextParser)


def test_get_parser_unknown_raises():
    with pytest.raises(ValueError):
        get_parser("nonexistent-parser")


# ---------------------------------------------------------------------------
# DoclingParser (skip if not installed)
# ---------------------------------------------------------------------------
def test_docling_parser_import_guard():
    """DoclingParser.__init__ raises ImportError with a helpful message when
    docling isn't installed — the parser should never silently no-op."""
    try:
        import docling  # noqa: F401
        has_docling = True
    except ImportError:
        has_docling = False
    if has_docling:
        pytest.skip("docling installed; cannot test the missing-dependency path")
    with pytest.raises(ImportError, match="docling"):
        DoclingParser()


# ---------------------------------------------------------------------------
# parse_document error handling on a nonexistent path
# ---------------------------------------------------------------------------
def test_parse_document_nonexistent_file(tmp_path):
    missing = tmp_path / "nope.pdf"
    doc = Document.from_path(str(missing)) if missing.exists() else _doc_with_fake_path(str(missing))
    parse_document(doc, parser=PlainTextParser())
    assert "parse_failed" in doc.errors[0]


def _doc_with_fake_path(path: str) -> Document:
    # Document.from_path raises if the file doesn't exist; construct directly
    return Document(source_path=path, doc_id="fake")


# ---------------------------------------------------------------------------
# DoclingParser.parse() — mocked (lines 133-165)
# ---------------------------------------------------------------------------
def test_docling_parser_parse_with_mocked_converter(tmp_path):
    """DoclingParser.parse() handles a mock conversion result correctly."""
    from unittest.mock import MagicMock

    from idp.parse.parser import DoclingParser

    parser = DoclingParser.__new__(DoclingParser)  # bypass __init__ (no docling)
    # Build a mock that mimics docling's ConversionResult.document
    mock_page = MagicMock()
    mock_page.main_text = [MagicMock(text="Page one line 1"), MagicMock(text="Page one line 2")]
    mock_table = MagicMock()
    mock_table.data = [["a", "b"], ["1", "2"]]
    mock_table.export_to_markdown.return_value = "| a | b |\n| 1 | 2 |"
    mock_doc = MagicMock()
    mock_doc.pages = [mock_page]
    mock_doc.tables = [mock_table]
    mock_doc.export_to_markdown.return_value = "# full markdown\npage 1 text"
    mock_result = MagicMock()
    mock_result.document = mock_doc
    parser._converter = MagicMock()
    parser._converter.convert.return_value = mock_result

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = parser.parse(str(pdf))
    assert out["text"] == "# full markdown\npage 1 text"
    assert len(out["pages"]) == 1
    assert out["pages"][0]["page"] == 1
    assert "Page one line 1" in out["pages"][0]["text"]
    assert out["pages"][0]["image_path"] is None
    assert len(out["tables"]) == 1
    assert out["tables"][0]["rows"] == [["a", "b"], ["1", "2"]]
    assert out["metadata"]["parser"] == "docling"
    assert out["metadata"]["size"] > 0


def test_docling_parser_parse_handles_table_without_export_method(tmp_path):
    """Tables lacking export_to_markdown get an empty markdown field."""
    from unittest.mock import MagicMock

    from idp.parse.parser import DoclingParser

    parser = DoclingParser.__new__(DoclingParser)
    mock_doc = MagicMock()
    mock_doc.pages = []
    # A "table" object without export_to_markdown method
    mock_table = MagicMock(spec=["data"])  # only has .data
    mock_table.data = [["x"]]
    mock_doc.tables = [mock_table]
    mock_doc.export_to_markdown.return_value = "full text"
    mock_result = MagicMock()
    mock_result.document = mock_doc
    parser._converter = MagicMock()
    parser._converter.convert.return_value = mock_result

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = parser.parse(str(pdf))
    assert out["tables"][0]["markdown"] == ""
    # pages is empty -> fall back to a single full-text page
    assert len(out["pages"]) == 1
    assert out["pages"][0]["text"] == "full text"


def test_docling_parser_parse_handles_page_items_without_text_attr(tmp_path):
    """Page items without .text attribute are skipped gracefully."""
    from unittest.mock import MagicMock

    from idp.parse.parser import DoclingParser

    parser = DoclingParser.__new__(DoclingParser)
    # page item has no .text attr -> the hasattr() check filters it
    mock_page_item_no_text = MagicMock(spec=[])  # no .text attribute
    mock_page_item_with_text = MagicMock(text="present")
    mock_page = MagicMock()
    mock_page.main_text = [mock_page_item_no_text, mock_page_item_with_text]
    mock_doc = MagicMock()
    mock_doc.pages = [mock_page]
    mock_doc.tables = []
    mock_doc.export_to_markdown.return_value = "all"
    mock_result = MagicMock()
    mock_result.document = mock_doc
    parser._converter = MagicMock()
    parser._converter.convert.return_value = mock_result

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = parser.parse(str(pdf))
    assert out["pages"][0]["text"] == "present"  # only the one with .text


def test_docling_parser_parse_handles_no_tables(tmp_path):
    """When doc.tables is None or empty, no tables section returned."""
    from unittest.mock import MagicMock

    from idp.parse.parser import DoclingParser

    parser = DoclingParser.__new__(DoclingParser)
    mock_doc = MagicMock()
    mock_doc.pages = []
    mock_doc.tables = None  # explicitly None
    mock_doc.export_to_markdown.return_value = "x"
    mock_result = MagicMock()
    mock_result.document = mock_doc
    parser._converter = MagicMock()
    parser._converter.convert.return_value = mock_result

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = parser.parse(str(pdf))
    assert out["tables"] == []

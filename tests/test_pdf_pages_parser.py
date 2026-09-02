"""Tests for PdfPagesParser — all with mocked pdf2image since poppler
is a system dep that may not be available in CI.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from idp.parse.pdf_pages import PdfPagesParser, _pil_to_data_uri


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_construction_defaults():
    p = PdfPagesParser()
    assert p.dpi == 200
    assert p.max_side == 448
    assert p.fmt == "PNG"
    assert p.name == "pdf-pages"


def test_construction_custom_params():
    p = PdfPagesParser(dpi=300, max_side=1024, fmt="JPEG")
    assert p.dpi == 300
    assert p.max_side == 1024
    assert p.fmt == "JPEG"


# ---------------------------------------------------------------------------
# Image-encoding helper (no pdf2image needed)
# ---------------------------------------------------------------------------
def test_pil_to_data_uri_png():
    """Encode a PIL image as a data:image/png;base64 URI."""
    fake_img = MagicMock()
    fake_img.size = (100, 200)
    # Image.save(buf, format="PNG", optimize=True) is called; return
    # a deterministic byte string by stubbing the save method.
    fake_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"FAKE_PNG"))
    import PIL.Image
    with patch.object(PIL.Image, "open"):
        uri = _pil_to_data_uri(fake_img, max_side=448)
    assert uri.startswith("data:image/png;base64,")
    payload = uri.split(",", 1)[1]
    assert base64.b64decode(payload) == b"FAKE_PNG"
    # 200 < 448, so no resize should happen
    fake_img.resize.assert_not_called()


def test_pil_to_data_uri_resizes_when_too_large():
    """Images with longest side > max_side get resized before encoding."""
    fake_img = MagicMock()
    fake_img.size = (1000, 500)  # longest = 1000, scale to 448
    fake_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import PIL.Image
    with patch.object(PIL.Image, "open"):
        _pil_to_data_uri(fake_img, max_side=448)
    fake_img.resize.assert_called_once()
    new_size = fake_img.resize.call_args[0][0]
    assert new_size[0] == 448
    assert new_size[1] == 224  # 500 * 448/1000


def test_pil_to_data_uri_jpeg_strips_alpha():
    """JPEG format converts RGBA -> RGB to avoid PIL error."""
    fake_img = MagicMock()
    fake_img.size = (100, 100)
    fake_img.mode = "RGBA"
    fake_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import PIL.Image
    with patch.object(PIL.Image, "open"):
        uri = _pil_to_data_uri(fake_img, max_side=448, fmt="JPEG")
    assert "image/jpeg" in uri
    # The RGBA image should have been converted to RGB
    fake_img.convert.assert_called_once_with("RGB")


# ---------------------------------------------------------------------------
# parse() with mocked pdf2image
# ---------------------------------------------------------------------------
def _make_fake_pdf_page(width: int = 100, height: int = 200) -> MagicMock:
    """A MagicMock that quacks like a PIL Image."""
    img = MagicMock()
    img.size = (width, height)
    img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"FAKE"))
    return img


def test_parse_returns_dict_with_one_image_per_page(tmp_path):
    """A 2-page PDF produces 2 pages, each with 1 image."""
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake")  # content doesn't matter; mocked
    fake_pages = [_make_fake_pdf_page(), _make_fake_pdf_page()]

    # Pre-import the modules so the lazy imports inside parse() can find them.
    # Then patch them at their source location.
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=fake_pages), \
         patch.object(PIL.Image, "open"):
        parser = PdfPagesParser(dpi=200, max_side=448)
        result = parser.parse(pdf_path)

    assert result["text"] == ""
    assert result["metadata"]["parser"] == "pdf-pages"
    assert result["metadata"]["dpi"] == 200
    assert result["metadata"]["max_side"] == 448
    assert len(result["pages"]) == 2
    for p in result["pages"]:
        assert p["text"] == ""
        assert len(p["images_b64"]) == 1
        assert p["images_b64"][0].startswith("data:image/png;base64,")


def test_parse_resizes_each_page_individually(tmp_path):
    """A page larger than max_side gets resized; small ones don't."""
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")  # content doesn't matter; we mock the render
    big_page = _make_fake_pdf_page(width=1000, height=500)
    small_page = _make_fake_pdf_page(width=100, height=200)

    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[big_page, small_page]), \
         patch.object(PIL.Image, "open"):
        parser = PdfPagesParser(max_side=448)
        parser.parse(pdf_path)

    # big page resized
    big_page.resize.assert_called_once()
    # small page not resized
    small_page.resize.assert_not_called()


def test_parse_propagates_to_document_pages(tmp_path):
    """parse_document wires parser output -> Document.pages[i].images_b64."""
    from idp.core.document import Document
    from idp.parse.parser import get_parser, parse_document

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_pages = [_make_fake_pdf_page(), _make_fake_pdf_page()]

    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=fake_pages), \
         patch.object(PIL.Image, "open"):
        # Get a real PdfPagesParser instance, then pass it to parse_document
        doc = Document.from_path(pdf_path)
        result_doc = parse_document(doc, parser=get_parser("pdf-pages"))

    assert len(result_doc.pages) == 2
    for p in result_doc.pages:
        assert len(p.images_b64) == 1
        assert p.text == ""  # pdf-pages produces no text


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_parse_missing_pdf2image_raises_clear_error(tmp_path):
    """If pdf2image is not installed, the error mentions pip install + poppler."""
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    with patch.dict("sys.modules", {"pdf2image": None}):
        parser = PdfPagesParser()
        with pytest.raises(ImportError) as exc_info:
            parser.parse(pdf_path)
    msg = str(exc_info.value)
    assert "pdf2image" in msg
    assert "poppler" in msg.lower()
    # install hints
    assert "pip install" in msg


def test_parse_non_pdf_extension_raises_value_error(tmp_path):
    """Only .pdf files are accepted (not .txt with PDF-looking content)."""
    txt = tmp_path / "looks-like-pdf.txt"
    txt.write_text("not a pdf")
    parser = PdfPagesParser()
    with pytest.raises(ValueError) as match:
        parser.parse(txt)
    assert "pdf" in str(match.value).lower()


def test_parse_missing_file_raises_filenotfound(tmp_path):
    parser = PdfPagesParser()
    with patch.dict("sys.modules", {"pdf2image": MagicMock()}), \
         pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "does-not-exist.pdf")


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------
def test_get_parser_resolves_pdf_pages_string():
    """The router knows 'pdf-pages' as a valid parser name."""
    from idp.parse.parser import get_parser
    with patch("idp.parse.pdf_pages.PdfPagesParser") as MockCls:
        MockCls.return_value = "fake-parser"
        p = get_parser("pdf-pages")
    assert p == "fake-parser"
    MockCls.assert_called_once()
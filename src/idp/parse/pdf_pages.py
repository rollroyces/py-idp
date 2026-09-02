"""PdfPagesParser: render each PDF page to a base64-encoded PNG.

Unlike ``PlainTextParser`` and ``PdfPlumberParser`` (which extract text),
this parser emits **page images** that multimodal backends (notably
``NanonetsVLBackend``) can consume directly.

The output is the standard parser dict shape:
    {
        "text": "<empty string — images only>",
        "pages": [
            {"page": 1, "text": "", "image_path": None, "images_b64": ["data:image/png;base64,..."]},
            ...
        ],
        "tables": [],
        "metadata": {"parser": "pdf-pages", "size": <bytes>},
    }

The ``Document`` (after ``parse_document(doc, parser="pdf-pages")``) has
``doc.pages[i].images_b64`` populated. Multimodal backends then read
those images without any extra glue.

Requires:
  - pdf2image (``pip install pdf2image``)
  - poppler (system dep)
      macOS:  brew install poppler
      Debian: apt install poppler-utils

Heavy imports are lazy: this module only imports pdf2image on
``.parse()`` call, so the rest of py-idp imports stay fast and
people who don't need PDF-to-image don't pay the import cost.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


DEFAULT_DPI = 200
DEFAULT_MAX_SIDE = 448  # match NanonetsVLBackend's default


class PdfPagesParser:
    """Render each PDF page to a base64 PNG data URI.

    The output is what ``NanonetsVLBackend`` (and any future multimodal
    backend) consumes. text is empty because the point of this parser
    is the image path — use ``PlainTextParser`` or ``PdfPlumberParser``
    if you need text.
    """

    name = "pdf-pages"

    def __init__(
        self,
        dpi: int = DEFAULT_DPI,
        max_side: int = DEFAULT_MAX_SIDE,
        fmt: str = "PNG",
    ) -> None:
        """
        Args:
            dpi:     rendering resolution. 200 is a good default for
                     invoices (sharp text at modest file size). Increase
                     for dense forms with small print.
            max_side: longest side, in pixels, to resize each page to.
                      448 is the NanonetsVLBackend default and a good
                      M4 default. Larger = more memory + slower.
            fmt:     image format for the data URI (PNG recommended;
                     JPEG is smaller but loses text edges).
        """
        self.dpi = dpi
        self.max_side = max_side
        self.fmt = fmt

    def parse(self, path: str | Path) -> dict[str, Any]:
        """Render each page of the PDF at ``path`` to a base64 PNG data URI.

        Raises:
            ImportError: if pdf2image is not installed.
            FileNotFoundError: if the PDF doesn't exist or isn't a PDF.
            Exception: any pdf2image error (corrupt PDF, missing poppler, etc.)
                        is propagated — caller's process_batch should catch.
        """
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(
                "PdfPagesParser requires pdf2image. "
                "Install with:  pip install pdf2image  (and install poppler "
                "system-wide: brew install poppler on macOS, apt install "
                "poppler-utils on Debian/Ubuntu)"
            ) from e

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(
                f"PdfPagesParser only handles .pdf files, got {p.suffix!r}"
            )

        # pdf2image is the bottleneck. Heavy on memory for big PDFs.
        # first_page / last_page can be used to chunk very large docs;
        # we render all pages by default.
        pil_pages = convert_from_path(str(p), dpi=self.dpi)
        log.info("PdfPagesParser: rendered %d pages from %s", len(pil_pages), p)

        pages: list[dict[str, Any]] = []
        for i, img in enumerate(pil_pages, start=1):
            b64 = _pil_to_data_uri(img, max_side=self.max_side, fmt=self.fmt)
            # We don't fill width/height from PIL — those are stored on
            # Page, not the parser dict. The router / Document code
            # can populate them if needed.
            pages.append({
                "page": i,
                "text": "",
                "image_path": None,
                "images_b64": [b64],
            })

        return {
            "text": "",
            "pages": pages,
            "tables": [],
            "metadata": {
                "parser": "pdf-pages",
                "size": p.stat().st_size,
                "dpi": self.dpi,
                "max_side": self.max_side,
            },
        }


def _pil_to_data_uri(img: Any, max_side: int, fmt: str = "PNG") -> str:
    """Resize a PIL Image to max_side and encode as a base64 data URI.

    Imports PIL lazily — non-multimodal code paths don't pay the
    import cost.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "PdfPagesParser requires Pillow for image encoding. "
            "Install with:  pip install Pillow  (included in [hf-vlm])"
        ) from e

    # Resize if needed
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (max(1, int(img.size[0] * scale)),
                    max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)  # type: ignore[attr-defined]

    buf = BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    elif fmt.upper() in ("JPEG", "JPG"):
        # JPEG needs RGB (no alpha)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        mime = "image/jpeg"
    else:
        img.save(buf, format=fmt)
        mime = f"image/{fmt.lower()}"

    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


__all__ = ["DEFAULT_DPI", "DEFAULT_MAX_SIDE", "PdfPagesParser"]
"""Tests for auto-schema discovery (idp.discover)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from idp.discover import (
    DiscoveryResult,
    _json_schema_to_pydantic,
    _parse_json_schema,
    available_backends,
    discover_schema,
    make_backend,
)


# ---------------------------------------------------------------------------
# _parse_json_schema — defensive parsing
# ---------------------------------------------------------------------------
def test_parse_json_schema_clean_object():
    """A clean JSON Schema parses directly."""
    raw = json.dumps({"type": "object", "properties": {"x": {"type": "string"}}})
    parsed = _parse_json_schema(raw)
    assert parsed["type"] == "object"
    assert "properties" in parsed
    assert "x" in parsed["properties"]


def test_parse_json_schema_strips_fences():
    """```json ... ``` fences are stripped before parsing."""
    raw = '```json\n{"type": "object", "properties": {"x": {"type": "string"}}}\n```'
    parsed = _parse_json_schema(raw)
    assert parsed["type"] == "object"


def test_parse_json_schema_strips_fences_case_insensitive():
    """```JSON ... ``` (uppercase) is also handled."""
    raw = '```JSON\n{"type": "object"}\n```'
    parsed = _parse_json_schema(raw)
    assert parsed["type"] == "object"


def test_parse_json_schema_extracts_block_from_prose():
    """If the LLM wraps the schema in prose, we extract the first {...} block."""
    raw = 'Here is the schema:\n{"type": "object", "properties": {"x": {"type": "string"}}}\nHope this helps!'
    parsed = _parse_json_schema(raw)
    assert parsed["type"] == "object"
    assert "x" in parsed["properties"]


def test_parse_json_schema_normalizes_missing_type():
    """If 'type' is missing, we add type=object (most LLMs omit it)."""
    raw = json.dumps({"properties": {"x": {"type": "string"}}})
    parsed = _parse_json_schema(raw)
    assert parsed["type"] == "object"


def test_parse_json_schema_normalizes_missing_properties():
    """If 'properties' is missing, we add an empty dict."""
    raw = json.dumps({"type": "object"})
    parsed = _parse_json_schema(raw)
    assert parsed["properties"] == {}


def test_parse_json_schema_empty_raises():
    """Empty response raises ValueError with a clear message."""
    with pytest.raises(ValueError, match="empty"):
        _parse_json_schema("")


def test_parse_json_schema_garbage_raises():
    """Non-JSON garbage raises ValueError with the first 200 chars for debugging."""
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_json_schema("hello there friend")


def test_parse_json_schema_non_object_raises():
    """A JSON array (not object) raises — discover_schema needs an object schema."""
    with pytest.raises(ValueError, match="not a JSON object"):
        _parse_json_schema("[1, 2, 3]")


# ---------------------------------------------------------------------------
# _json_schema_to_pydantic — compilation
# ---------------------------------------------------------------------------
def test_compile_basic_string_field():
    """A simple string property compiles to an Optional[str] field."""
    schema = {
        "type": "object",
        "properties": {"vendor_name": {"type": "string", "description": "who"}},
    }
    M = _json_schema_to_pydantic(schema)
    assert issubclass(M, BaseModel)
    assert "vendor_name" in M.model_fields
    # Default is None (not required)
    inst = M()
    assert inst.vendor_name is None


def test_compile_required_field_has_no_default():
    """A field in 'required' gets default=... (raises ValidationError when missing)."""
    from pydantic import ValidationError
    schema = {
        "type": "object",
        "properties": {"invoice_number": {"type": "string"}},
        "required": ["invoice_number"],
    }
    M = _json_schema_to_pydantic(schema)
    with pytest.raises(ValidationError):
        M()
    inst = M(invoice_number="INV-1")
    assert inst.invoice_number == "INV-1"


def test_compile_numeric_types():
    """number -> float, integer -> int."""
    schema = {
        "type": "object",
        "properties": {
            "total": {"type": "number"},
            "count": {"type": "integer"},
            "ratio": {"type": "number", "description": "0.0-1.0"},
        },
    }
    M = _json_schema_to_pydantic(schema)
    inst = M(total=100.0, count=5, ratio=0.5)
    assert inst.total == 100.0
    assert inst.count == 5


def test_compile_array_field_with_object_items():
    """An array of objects compiles to list[NestedModel]."""
    schema = {
        "type": "object",
        "properties": {
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["amount"],
                },
            }
        },
    }
    M = _json_schema_to_pydantic(schema)
    inst = M(line_items=[{"description": "Widget", "amount": 10.0}])
    # Items become NestedModel Pydantic instances, not raw dicts
    assert inst.line_items[0].amount == 10.0
    assert inst.line_items[0].description == "Widget"
    # Round-trip via model_dump -> dict
    dumped = inst.model_dump()
    assert dumped["line_items"][0]["amount"] == 10.0


def test_compile_uses_title():
    """The schema 'title' becomes the class name."""
    schema = {
        "type": "object",
        "title": "CustomSchemaName",
        "properties": {"x": {"type": "string"}},
    }
    M = _json_schema_to_pydantic(schema)
    assert M.__name__ == "CustomSchemaName"


def test_compile_uses_fallback_title():
    """When no title, use the fallback name."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    M = _json_schema_to_pydantic(schema)
    assert M.__name__ == "InferredSchema"


def test_compile_descriptions_preserved():
    """JSON Schema 'description' becomes Pydantic Field description."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string", "description": "important field"}},
    }
    M = _json_schema_to_pydantic(schema)
    assert M.model_fields["x"].description == "important field"


# ---------------------------------------------------------------------------
# make_backend / available_backends
# ---------------------------------------------------------------------------
def test_available_backends_lists():
    assert "nanonets" in available_backends()
    assert "mock" in available_backends()


def test_make_backend_nanonets_requires_env_var(monkeypatch):
    """make_backend('nanonets') without IDP_ENABLE_NANONETS raises."""
    monkeypatch.delenv("IDP_ENABLE_NANONETS", raising=False)
    with pytest.raises(RuntimeError, match="IDP_ENABLE_NANONETS"):
        make_backend("nanonets")


def test_make_backend_nanonets_with_env_var(monkeypatch):
    """make_backend('nanonets') with env var returns NanonetsVLBackend."""
    monkeypatch.setenv("IDP_ENABLE_NANONETS", "1")
    b = make_backend("nanonets")
    # We don't actually load the model, just construct
    assert type(b).__name__ == "NanonetsVLBackend"


def test_make_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        make_backend("not-a-real-backend")


# ---------------------------------------------------------------------------
# discover_schema — end-to-end with mocked backend
# ---------------------------------------------------------------------------
def _mock_backend(returning_json: str) -> MagicMock:
    """A mock multimodal backend that returns the given JSON string."""
    b = MagicMock()
    b.is_multimodal = True
    b.complete = MagicMock(return_value=returning_json)
    b.name = "MockBackend"
    return b


def test_discover_schema_with_mock_backend_returns_pydantic(tmp_path):
    """End-to-end with a mocked Nanonets: PDF in, Pydantic class out."""
    # Create a dummy PDF (content doesn't matter; pdf2image is mocked)
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")

    schema_response = json.dumps({
        "type": "object",
        "title": "MockedInvoice",
        "properties": {
            "vendor_name": {"type": "string", "description": "vendor"},
            "total_amount": {"type": "number", "description": "total"},
            "line_items": {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        },
        "required": ["vendor_name", "total_amount"],
    })

    backend = _mock_backend(schema_response)

    # Mock pdf2image.convert_from_path so we don't need poppler
    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    fake_page.convert = MagicMock(return_value=fake_page)

    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page, fake_page]), \
         patch.object(PIL.Image, "open"):
        result = discover_schema(str(pdf), hint="extract vendor and total", backend=backend)

    assert isinstance(result, DiscoveryResult)
    assert issubclass(result.schema_class, BaseModel)
    assert result.schema_class.__name__ == "MockedInvoice"
    assert "vendor_name" in result.schema_class.model_fields
    assert "total_amount" in result.schema_class.model_fields
    assert "line_items" in result.schema_class.model_fields
    assert result.backend_name == "MockBackend"

    # The class is usable: required field enforced, optional field is None
    inst = result.schema_class(vendor_name="Acme", total_amount=100.0)
    assert inst.vendor_name == "Acme"
    assert inst.line_items is None  # optional, default None
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        result.schema_class()  # missing required fields


def test_discover_schema_handles_fenced_response(tmp_path):
    """LLM returns ```json ... ``` fences — we strip them."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    schema_obj = {"type": "object", "properties": {"y": {"type": "string"}}}
    fenced = f"```json\n{json.dumps(schema_obj)}\n```"
    backend = _mock_backend(fenced)
    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page]), \
         patch.object(PIL.Image, "open"):
        result = discover_schema(str(pdf), backend=backend)
    assert "y" in result.schema_class.model_fields


def test_discover_schema_invalid_response_raises(tmp_path):
    """LLM returns garbage — discover_schema raises ValueError with context."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    backend = _mock_backend("this is not JSON at all")
    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page]), \
         patch.object(PIL.Image, "open"), pytest.raises(ValueError, match="not valid JSON"):
        discover_schema(str(pdf), backend=backend)


def test_discover_schema_empty_response_raises(tmp_path):
    """Empty LLM response raises ValueError."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    backend = _mock_backend("")
    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page]), \
         patch.object(PIL.Image, "open"), pytest.raises(ValueError, match="empty"):
        discover_schema(str(pdf), backend=backend)


def test_discover_schema_rejects_non_multimodal_backend(tmp_path):
    """Passing a text-only backend raises ValueError with a clear message."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    b = MagicMock()
    b.is_multimodal = False
    b.complete = MagicMock(return_value="{}")
    with pytest.raises(ValueError, match="multimodal"):
        discover_schema(str(pdf), backend=b)


def test_discover_schema_uses_default_backend(monkeypatch, tmp_path):
    """When backend=None, make_backend('nanonets') is called."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    schema_obj = {"type": "object", "properties": {"z": {"type": "string"}}}
    backend = _mock_backend(json.dumps(schema_obj))

    monkeypatch.setattr("idp.discover.make_backend", lambda name="nanonets": backend)

    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page]), \
         patch.object(PIL.Image, "open"):
        result = discover_schema(str(pdf))  # backend=None -> default
    assert "z" in result.schema_class.model_fields
    backend.complete.assert_called_once()


def test_discover_schema_passes_hint_to_backend(tmp_path):
    """The user hint is included in the prompt sent to the backend."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    backend = _mock_backend(json.dumps({"type": "object", "properties": {}}))
    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page]), \
         patch.object(PIL.Image, "open"):
        discover_schema(str(pdf), hint="extract only the date field", backend=backend)

    # Inspect the request the backend received
    req = backend.complete.call_args[0][0]
    user_msg = next(m for m in req.messages if m.role == "user")
    assert "extract only the date field" in user_msg.content


def test_discover_schema_accepts_pre_parsed_document(tmp_path):
    """A Document with images_b64 already populated is accepted as input."""
    from idp.core.document import Document, Page
    doc = Document(
        source_path=str(tmp_path / "x.pdf"),
        doc_id="pre",
        pages=[Page(page_number=1, images_b64=["data:image/png;base64,AAA"])],
    )
    backend = _mock_backend(json.dumps({
        "type": "object",
        "properties": {"k": {"type": "string"}},
    }))
    result = discover_schema(doc, backend=backend)
    assert "k" in result.schema_class.model_fields
    # Document was reused
    assert result.doc is doc


def test_discover_schema_missing_file_raises(tmp_path):
    """A non-existent path raises FileNotFoundError before any LLM call."""
    backend = _mock_backend("{}")
    with pytest.raises(FileNotFoundError):
        discover_schema(str(tmp_path / "does-not-exist.pdf"), backend=backend)
    backend.complete.assert_not_called()


def test_discover_schema_caps_pages_to_page_limit(tmp_path):
    """A 20-page PDF has page_limit applied (default 4 pages)."""
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(b"%PDF")
    backend = _mock_backend(json.dumps({
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }))

    # Mock pdf2image to return 20 pages
    pages = []
    for _ in range(20):
        p = MagicMock()
        p.size = (50, 50)
        p.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
        pages.append(p)

    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=pages), \
         patch.object(PIL.Image, "open"):
        discover_schema(str(pdf), page_limit=3, backend=backend)

    # Only 3 pages should be in the prompt's images
    req = backend.complete.call_args[0][0]
    user_msg = next(m for m in req.messages if m.role == "user")
    assert len(user_msg.images_b64 or []) == 3


def test_discover_schema_can_pass_to_pipeline(tmp_path):
    """The discovered schema_class is a real Pydantic model usable by Pipeline."""
    from idp.pipeline import Pipeline
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    backend = _mock_backend(json.dumps({
        "type": "object",
        "title": "TestInvoice",
        "properties": {"vendor_name": {"type": "string"}},
        "required": ["vendor_name"],
    }))
    fake_page = MagicMock()
    fake_page.size = (100, 100)
    fake_page.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"X"))
    import pdf2image
    import PIL.Image
    with patch.object(pdf2image, "convert_from_path", return_value=[fake_page]), \
         patch.object(PIL.Image, "open"):
        result = discover_schema(str(pdf), backend=backend)

    # Construct a Pipeline with the discovered class as schema
    pipe = Pipeline(backend="mock", schema=result.schema_class)
    assert pipe.schema is result.schema_class
    assert issubclass(pipe.schema, BaseModel)
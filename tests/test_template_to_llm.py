"""Tests for template body actually reaching the LLM in extract() and Pipeline.run().

The earlier templates test (test_templates_and_errors.py) covered the
registry, API endpoints, and routing. This file covers the
**end-to-end behavior**: did the template body actually appear in the
prompt the LLM saw?
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from idp.core.document import Document
from idp.extract import extract
from idp.extract.extractor import _build_messages
from idp.llm.backend import Backend, CompletionRequest
from idp.pipeline.pipeline import Pipeline
from idp.templates import Template, TemplateRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _CaptureBackend(Backend):
    """Backend that records the messages it was called with, then returns a stub.

    This is the *only* reliable way to assert "the template body reached
    the LLM" — anything else is checking the wrong layer. A mock that
    just returns a JSON dict would pass tests but not actually verify
    the wiring.
    """

    def __init__(self, response: str = "{}") -> None:
        self.requests: list[CompletionRequest] = []
        self._response = response

    @property
    def is_multimodal(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "capture-mock"

    def complete(self, req: CompletionRequest) -> str:
        self.requests.append(req)
        return self._response

    async def acomplete(self, req: CompletionRequest) -> str:
        return self.complete(req)


class _Schema(BaseModel):
    invoice_number: str | None = None
    total: float | None = None


def _make_doc(tmp_path: Path, text: str = "Invoice #INV-001, total: $100") -> Document:
    """Write a real file and return a Document pointing at it.

    The pipeline auto-picks the parser by file extension. We use .txt
    so PlainTextParser reads the file back. If the file doesn't exist
    the parser swallows the error and the document's raw_text is
    cleared — the LLM gets called with "(empty document)" and the
    test fails for the wrong reason.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "test.txt"
    p.write_text(text, encoding="utf-8")
    return Document(source_path=str(p), doc_id="t1", raw_text=text)


# ---------------------------------------------------------------------------
# Unit tests: _build_messages with template_body
# ---------------------------------------------------------------------------
def test_build_messages_no_template() -> None:
    """Empty template_body: no 'Template-specific guidance' block appears."""
    msgs = _build_messages(_Schema, "text", [], template_body="")
    user_msg = msgs[1]
    assert "Template-specific guidance" not in user_msg.content


def test_build_messages_with_template_body() -> None:
    """Non-empty template_body: appears in the user message, before the schema."""
    body = "Look for vendor_name near the top."
    msgs = _build_messages(_Schema, "text", [], template_body=body)
    user_msg = msgs[1]
    assert "Template-specific guidance" in user_msg.content
    assert body in user_msg.content
    # Template must come BEFORE the JSON Schema in the prompt
    template_pos = user_msg.content.find("Template-specific guidance")
    schema_pos = user_msg.content.find("Output JSON Schema:")
    assert template_pos < schema_pos, (
        f"template at {template_pos} should precede schema at {schema_pos}"
    )


def test_build_messages_template_capped_at_2000_tokens() -> None:
    """A 50k-char template body must be truncated to ~2000 tokens."""
    huge = "x" * 50_000
    msgs = _build_messages(_Schema, "text", [], template_body=huge)
    user_msg = msgs[1]
    # The full 50k of "x" should NOT all be in the content
    # (we cap at ~2000 tokens ≈ 8000 chars)
    assert user_msg.content.count("x") < 50_000
    # And the cap should leave at least *some* content
    assert "x" in user_msg.content


# ---------------------------------------------------------------------------
# Integration: extract() with template_body actually sends it
# ---------------------------------------------------------------------------
def test_extract_sends_template_body_to_backend(tmp_path: Path) -> None:
    """The litmus test: with a template body set, the LLM sees it."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001", "total": 100.0}')
    doc = _make_doc(tmp_path)
    template = Template(
        name="invoice",
        schema="Invoice",
        body="Always look for invoice_number near the top-right corner.",
        source_path=Path("/fake/invoice.md"),
        version=2,
    )
    extract(doc, _Schema, backend, template_body=template.body,
            template_name=template.name, template_version=template.version)
    assert len(backend.requests) == 1
    content = backend.requests[0].messages[1].content
    assert "near the top-right corner" in content


def test_extract_without_template_does_not_send_one(tmp_path: Path) -> None:
    """Without a template, no 'Template-specific guidance' block."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001"}')
    doc = _make_doc(tmp_path)
    extract(doc, _Schema, backend)
    content = backend.requests[0].messages[1].content
    assert "Template-specific guidance" not in content


def test_extract_records_template_provenance_on_doc(tmp_path: Path) -> None:
    """template_name and template_version end up on doc for audit."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001"}')
    doc = _make_doc(tmp_path)
    extract(doc, _Schema, backend, template_body="b",
            template_name="invoice", template_version=3)
    assert doc.template_name == "invoice"
    assert doc.template_version == 3


# ---------------------------------------------------------------------------
# Integration: Pipeline(template=...) wires through to extract
# ---------------------------------------------------------------------------
def test_pipeline_with_template_object(tmp_path: Path) -> None:
    """Pipeline(template=Template(...)) -> template body reaches LLM."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001", "total": 100.0}')
    template = Template(
        name="invoice",
        schema="Invoice",
        body="This template has UNIQUE marker text: zebra-finch-9876.",
        source_path=Path("/fake/i.md"),
    )
    pipe = Pipeline(backend=backend, schema=_Schema, template=template)
    doc = _make_doc(tmp_path)
    result = pipe.run(doc)
    # Backend was called and saw the template body
    assert len(backend.requests) == 1
    assert "zebra-finch-9876" in backend.requests[0].messages[1].content
    # Result records the template name + version
    assert result.template_name == "invoice"
    assert result.template_version == template.version
    # to_dict includes them
    d = result.to_dict()
    assert d["template_name"] == "invoice"
    assert d["template_version"] == template.version


def test_pipeline_with_template_name_via_registry(tmp_path: Path) -> None:
    """Pipeline(template='invoice') + set_template_registry() resolves at run() time."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001", "total": 100.0}')
    d = tmp_path / "templates"
    d.mkdir()
    (d / "invoice.md").write_text(
        "---\n"
        "name: invoice\n"
        "schema: Invoice\n"
        "version: 5\n"
        "---\n"
        "REGISTRY-BODY-MARKER-canary-1234.\n",
        encoding="utf-8",
    )
    registry = TemplateRegistry.load(d)
    pipe = Pipeline(backend=backend, schema=_Schema, template="invoice")
    pipe.set_template_registry(registry)
    doc = _make_doc(tmp_path)
    result = pipe.run(doc)
    # Registry body was used (not empty default)
    assert "REGISTRY-BODY-MARKER-canary-1234" in backend.requests[0].messages[1].content
    assert result.template_name == "invoice"
    assert result.template_version == 5


def test_pipeline_template_name_missing_runs_without_template(tmp_path: Path) -> None:
    """If the name isn't in the registry, run() proceeds without template."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001"}')
    pipe = Pipeline(backend=backend, schema=_Schema, template="nonexistent")
    pipe.set_template_registry(TemplateRegistry({}))  # empty
    doc = _make_doc(tmp_path)
    result = pipe.run(doc)
    # No template body sent
    assert "Template-specific guidance" not in backend.requests[0].messages[1].content
    # Result has no template provenance
    assert result.template_name is None


def test_pipeline_without_template_unchanged(tmp_path: Path) -> None:
    """Pipeline(template=None) is byte-identical to old behavior."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001"}')
    pipe = Pipeline(backend=backend, schema=_Schema)
    doc = _make_doc(tmp_path)
    pipe.run(doc)
    content = backend.requests[0].messages[1].content
    assert "Template-specific guidance" not in content


# ---------------------------------------------------------------------------
# Hot reload: changes to the .md file between run() calls are picked up
# ---------------------------------------------------------------------------
def test_template_hot_reload_between_runs(tmp_path: Path) -> None:
    """Modify the .md file between two run() calls; second run sees new body."""
    backend = _CaptureBackend(response='{"invoice_number": "INV-001"}')
    d = tmp_path / "templates"
    d.mkdir()
    template_path = d / "invoice.md"
    template_path.write_text(
        "---\nname: invoice\nschema: Invoice\n---\nVERSION-1-MARKER\n",
        encoding="utf-8",
    )
    registry = TemplateRegistry.load(d, watch=True)
    pipe = Pipeline(backend=backend, schema=_Schema, template="invoice")
    pipe.set_template_registry(registry)
    doc1 = _make_doc(tmp_path / "d1", "doc 1")
    pipe.run(doc1)
    # Pipeline calls backend twice: once for classify, once for extract.
    # The template body only goes into the extract call.
    extract_call_1 = backend.requests[1]  # 0=classify, 1=extract
    assert "VERSION-1-MARKER" in extract_call_1.messages[1].content

    # Edit the file and bump mtime
    template_path.write_text(
        "---\nname: invoice\nschema: Invoice\n---\nVERSION-2-MARKER\n",
        encoding="utf-8",
    )
    new_mtime = template_path.stat().st_mtime + 1
    os.utime(template_path, (new_mtime, new_mtime))

    doc2 = _make_doc(tmp_path / "d2", "doc 2")
    pipe.run(doc2)
    # Same indexing: requests[2] is the 2nd run's classify, [3] is extract
    extract_call_2 = backend.requests[3]
    assert "VERSION-2-MARKER" in extract_call_2.messages[1].content


# ---------------------------------------------------------------------------
# Chunked path: template body included in EVERY chunk
# ---------------------------------------------------------------------------
def test_template_body_in_all_chunks(tmp_path: Path) -> None:
    """A multi-chunk extraction must include the template body in every chunk's call.

    Without this, chunks 2..N would lose the field descriptions and
    common-mistakes notes, and accuracy on those pages would tank.
    """
    backend = _CaptureBackend(
        response=json.dumps({"invoice_number": "INV-001", "total": 100.0, "line_items": []})
    )
    long_text = "word " * 5000  # ~5000 tokens, will force multiple chunks
    doc = _make_doc(tmp_path, text=long_text)
    template_body = "TEMPLATE-MARKER-finch-7777. Always extract line items."
    extract(doc, _Schema, backend, template_body=template_body)
    # Backend was called multiple times (one per chunk)
    assert len(backend.requests) > 1, "expected chunked extraction"
    # Every chunk's call had the template body
    for i, req in enumerate(backend.requests):
        assert "TEMPLATE-MARKER-finch-7777" in req.messages[1].content, (
            f"chunk {i} missing template body"
        )

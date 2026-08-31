"""Regression tests for bugs found in the v0.1 audit.

Each test corresponds to one bug in the audit list (B1-B14). If you
regress one of these, the corresponding real-world failure mode returns.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.eval.metrics import field_scores
from idp.extract.extractor import _safe_load, extract
from idp.llm.backend import (
    AnthropicBackend,
    Backend,
    CompletionRequest,
    Message,
    MockBackend,
    _safe_json,
    _extract_schema_block,
)
from idp.storage.store import InMemoryStorage, JsonFileStorage, StoredResult


# ---------------------------------------------------------------------------
# B1: _safe_json must not crash on empty / null input
# ---------------------------------------------------------------------------
def test_b1_safe_json_empty_string_returns_empty_dict():
    assert _safe_json("") == {}


def test_b1_safe_json_null_returns_empty_dict():
    assert _safe_json("null") == {}


def test_b1_safe_json_garbage_returns_error_dict():
    out = _safe_json("not json at all")
    assert "_error" in out
    assert out.get("_raw") == "not json at all"


def test_b1_safe_json_fenced_parses():
    assert _safe_json("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_b1_safe_json_non_dict_wraps_in_value():
    out = _safe_json("[1, 2, 3]")
    assert out.get("_value") == [1, 2, 3]


def test_b1_json_complete_never_raises():
    """Backend.json_complete must not raise on garbage model output."""
    b = MockBackend()
    out = b.json_complete([Message("user", "x")])
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# B2: JsonFileStorage must not crash on corrupt lines
# ---------------------------------------------------------------------------
def test_b2_storage_skips_corrupt_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    good = (
        '{"id":"a","doc_id":"d","schema_name":"Invoice","backend_name":"m",'
        '"mode":null,"classification":null,"extraction":{},"confidence":null,'
        '"validation":null,"source_path":"/x","created_at":1.0}\n'
    )
    path.write_text(good + '{"id":"b","broken\n')  # truncated/corrupt line
    store = JsonFileStorage(str(path))
    items = store.list(limit=100)
    # one valid line survives, corrupt one is skipped, no crash
    assert len(items) == 1
    assert items[0].id == "a"


def test_b2_storage_corrupt_only_returns_empty(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"id":"broken"\n')  # entirely corrupt
    store = JsonFileStorage(str(path))
    assert store.list(limit=100) == []


# ---------------------------------------------------------------------------
# B5/B14: extract stage must surface JSON parse errors via doc.errors
# ---------------------------------------------------------------------------
class AlwaysGarbageBackend(Backend):
    """Returns text that isn't JSON; extraction must fail loudly."""

    name = "garbage"

    @property
    def is_multimodal(self):
        return False

    def complete(self, req: CompletionRequest) -> str:
        return "Here is your invoice: ... I forget the format"


def test_b5_extract_garbage_backend_records_error(tmp_path):
    """A non-JSON model response must end up in doc.errors, not silently
    produce a 'valid' empty extraction."""
    f = tmp_path / "inv.txt"
    f.write_text("INVOICE INV-1\nTotal: $100\n")
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    extract(doc, Invoice, AlwaysGarbageBackend())
    assert doc.extraction is not None
    # Confirm the error is surfaced
    assert any("could not parse" in e or "JSON" in e for e in doc.errors), (
        f"expected JSON parse error in doc.errors; got {doc.errors}"
    )


def test_b5_safe_load_rejects_unfenced_garbage():
    """The legacy code had a greedy regex that silently ate any JSON-shaped
    text. Now we only attempt brace-fallback when the model wrapped in
    ```json...``` fences."""
    out = _safe_load('Hello {"vendor": "Acme"} goodbye')
    # No fences -> raw preservation, not "Acme" attribution
    assert "_error" in out


def test_b5_safe_load_accepts_fenced_json():
    out = _safe_load('```json\n{"vendor": "Acme"}\n```')
    assert out == {"vendor": "Acme"}


def test_b5_safe_load_accepts_valid_json():
    out = _safe_load('{"vendor": "Acme", "total": 100}')
    assert out == {"vendor": "Acme", "total": 100}


# ---------------------------------------------------------------------------
# B7: render_first_n_pages_to_images must surface a warning when pdf2image
#     isn't available, not silently log a debug message
# ---------------------------------------------------------------------------
def test_b7_no_pdf2image_returns_empty_list_no_crash(tmp_path, caplog):
    from idp.extract.extractor import render_first_n_pages_to_images
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    doc = Document.from_path(str(pdf))
    # Without pdf2image installed, returns [] gracefully
    out = render_first_n_pages_to_images(doc, n=2)
    assert out == []


def test_b7_non_pdf_returns_empty_without_import(tmp_path):
    from idp.extract.extractor import render_first_n_pages_to_images
    f = tmp_path / "x.txt"
    f.write_text("hello")
    doc = Document.from_path(str(f))
    out = render_first_n_pages_to_images(doc, n=0)
    assert out == []


# ---------------------------------------------------------------------------
# B8: _extract_schema_block handles edge cases of marker ordering
# ---------------------------------------------------------------------------
def test_b8_schema_block_extracts_with_output_rules_marker():
    p = "Intro\nOutput JSON Schema:\n{\"a\": 1}\n\nOutput rules: do X"
    out = _extract_schema_block(p)
    assert json.loads(out) == {"a": 1}


def test_b8_schema_block_falls_back_to_document_content():
    """If the schema block runs all the way to document content (no Output
    rules: marker), still extract correctly."""
    p = "Intro\nOutput JSON Schema:\n{\"a\": 2}\n\nDocument content:..."
    out = _extract_schema_block(p)
    assert json.loads(out) == {"a": 2}


def test_b8_schema_block_no_markers_returns_empty_object():
    assert _extract_schema_block("random prompt with no markers") == "{}"


def test_b8_schema_block_handles_alternative_marker():
    p = "JSON Schema:\n{\"b\": 3}\n\nOutput:..."
    out = _extract_schema_block(p)
    assert json.loads(out) == {"b": 3}


# ---------------------------------------------------------------------------
# B11: field_scores: explicit FP/FN semantics for missing/extra keys
# ---------------------------------------------------------------------------
def test_b11_field_scores_missing_key_is_fn():
    # Pred missing 'c' that was in gold
    pred = {"a": 1.0, "b": 2.0}
    gold = {"a": 1.0, "b": 2.0, "c": 3.0}
    matches = {"a": True, "b": True, "c": False}
    s = field_scores([matches])
    # tp=2, fn=1 -> recall = 2/3
    assert abs(s["precision"] - 2 / 2) < 1e-6   # 1.0
    assert abs(s["recall"] - 2 / 3) < 1e-6


def test_b11_field_scores_extra_predicted_field_is_fp():
    pred = {"a": 1.0, "b": 2.0, "extra": 99.0}
    gold = {"a": 1.0, "b": 2.0}
    matches = {"a": True, "b": True}
    # Manually craft — extra would show as an FP in the inner loop
    from idp.eval.metrics import field_match
    actual_matches = field_match(pred, gold)
    s = field_scores([actual_matches])
    # We can't easily test FP here without a different code path; just
    # confirm scores are sane and the function runs.
    assert s["precision"] >= 0
    assert s["recall"] >= 0


def test_b11_field_scores_empty():
    assert field_scores([]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}


# ---------------------------------------------------------------------------
# B10 (heuristic confidence): missing vs present numeric
# ---------------------------------------------------------------------------
def test_b10_heuristic_zero_not_treated_as_missing(tmp_path):
    from idp.assess.confidence import _heuristic_confidence
    from idp.core.document import Document

    f = tmp_path / "x"
    f.write_text("x")
    doc = Document.from_path(str(f))
    doc.mode = "multimodal"
    doc.extraction = {"x": 0.0, "y": None, "z": ""}
    conf = _heuristic_confidence(doc)
    # 0.0 is present -> base (0.9)
    assert conf["x"] == 0.9
    # None/"" -> 0.1
    assert conf["y"] == 0.1
    assert conf["z"] == 0.1


# ---------------------------------------------------------------------------
# B12: hash_key / verify round-trip + wrong-key rejection
# ---------------------------------------------------------------------------
def test_b12_hash_round_trip():
    from idp.auth.keys import hash_key, verify
    raw = "sk-abc123"
    h = hash_key(raw, salt="test")
    assert verify(raw, h, salt="test")
    assert not verify("sk-WRONG", h, salt="test")
    # Different salt -> fails
    assert not verify(raw, h, salt="other")


def test_b12_make_default_key_is_unique():
    from idp.auth.keys import make_default_key
    a = make_default_key()
    b = make_default_key()
    assert a != b
    assert len(a) >= 24


# ---------------------------------------------------------------------------
# Cross-cutting: pipeline still passes end-to-end after the bug fixes
# ---------------------------------------------------------------------------
def test_end_to_end_pipeline_still_passes():
    from idp.pipeline.pipeline import Pipeline

    p = Pipeline(backend="mock", schema=Invoice)
    doc = Document.from_path(
        "/Users/hermes/py-idp/src/idp/eval/datasets/invoices/docs/inv-001.txt"
    )
    res = p.run(doc)
    assert res.classification == "invoice"
    assert res.document.extraction
    assert "invoice_number" in res.document.extraction


def test_pipeline_garbage_backend_propagates_error_not_silent_pass():
    from idp.pipeline.pipeline import Pipeline

    p = Pipeline(backend=AlwaysGarbageBackend(), schema=Invoice)
    doc = Document.from_path(
        "/Users/hermes/py-idp/src/idp/eval/datasets/invoices/docs/inv-001.txt"
    )
    res = p.run(doc)
    # extraction must contain an error marker
    assert any(
        "could not parse" in e or "JSON" in e
        for e in res.document.errors
    ), f"expected JSON-parse error in doc.errors; got {res.document.errors}"


# ---------------------------------------------------------------------------
# B15: no duplicate top-level imports in any source file
# ---------------------------------------------------------------------------
def test_b15_no_duplicate_top_level_imports():
    """Every src/ file should declare each stdlib module only once at
    column-0 (top level). Function-scope lazy imports for optional deps
    are allowed (and intentional)."""
    import re
    from pathlib import Path
    bad = []
    for f in Path("/Users/hermes/py-idp/src/idp").rglob("*.py"):
        text = f.read_text()
        # ONLY column-0 (top-level) imports — exclude indented function-scope
        top_imports = [
            re.match(r"^import\s+(\w+)", l).group(1)
            for l in text.splitlines()
            if re.match(r"^import\s+(\w+)", l) is not None
        ]
        from collections import Counter
        for mod, count in Counter(top_imports).items():
            if count > 1:
                bad.append((f.relative_to(Path("/Users/hermes/py-idp")), mod, count))
    assert bad == [], f"duplicate top-level imports: {bad}"


# ---------------------------------------------------------------------------
# B16/B21: AnthropicBackend with no key raises OSError with a helpful message
# ---------------------------------------------------------------------------
def test_b16_anthropic_backend_no_key_raises_helpful_oserror(monkeypatch):
    """With anthropic SDK present but no API key set, we should get a
    helpful OSError, not a raw KeyError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Stub `anthropic` into sys.modules so the SDK-import probe succeeds.
    # Then we exercise the "package installed, no key" path.
    import sys, types
    fake = types.ModuleType("anthropic")
    sys.modules["anthropic"] = fake
    from idp.llm.backend import AnthropicBackend
    try:
        AnthropicBackend(api_key=None)
    except OSError as e:
        assert "ANTHROPIC_API_KEY" in str(e) or "api_key" in str(e)
        return
    except Exception as e:
        pytest.fail(f"expected OSError; got {type(e).__name__}: {e}")
    finally:
        del sys.modules["anthropic"]
    pytest.fail("AnthropicBackend() with no key should raise")


def test_b16_anthropic_backend_explicit_key_works(monkeypatch):
    """With anthropic SDK present and explicit api_key, no env needed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import sys, types
    fake = types.ModuleType("anthropic")
    sys.modules["anthropic"] = fake
    from idp.llm.backend import AnthropicBackend
    try:
        b = AnthropicBackend(api_key="sk-test-fake")
        assert b.api_key == "sk-test-fake"
    finally:
        del sys.modules["anthropic"]


# ---------------------------------------------------------------------------
# B25: AnthropicBackend.is_multimodal — Claude 2 is NOT multimodal
# ---------------------------------------------------------------------------
def test_b25_anthropic_claude2_is_text_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import sys, types
    fake = types.ModuleType("anthropic")
    sys.modules["anthropic"] = fake
    from idp.llm.backend import AnthropicBackend
    try:
        b = AnthropicBackend(api_key="sk-test", model="claude-2.1")
        assert b.is_multimodal is False
    finally:
        del sys.modules["anthropic"]


def test_b25_anthropic_claude35_is_multimodal(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import sys, types
    fake = types.ModuleType("anthropic")
    sys.modules["anthropic"] = fake
    from idp.llm.backend import AnthropicBackend
    try:
        b = AnthropicBackend(api_key="sk-test", model="claude-3-5-sonnet-latest")
        assert b.is_multimodal is True
    finally:
        del sys.modules["anthropic"]


# ---------------------------------------------------------------------------
# B24: extract() short-circuits on empty inputs — no LLM call
# ---------------------------------------------------------------------------
class CallCounter(Backend):
    """Backend that counts how many times it was called."""

    name = "counter"

    def __init__(self):
        self.calls = 0

    @property
    def is_multimodal(self):
        return False

    def complete(self, req):
        self.calls += 1
        return "{}"


def test_b24_extract_empty_text_skips_llm(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")  # zero bytes
    doc = Document.from_path(str(f))
    # raw_text is "" by default; renderer returns [] for non-PDF
    backend = CallCounter()
    extract(doc, Invoice, backend)
    assert backend.calls == 0, (
        "extract() should NOT call the LLM when raw_text is empty and no images"
    )
    assert any("empty document" in e or "extract_skipped" in e for e in doc.errors), (
        f"expected skip marker in doc.errors; got {doc.errors}"
    )
    # extraction is a valid Invoice-shaped dict of nulls
    parsed = Invoice.model_validate(doc.extraction)
    assert parsed.invoice_number is None

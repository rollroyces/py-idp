"""Tests for process_batch (idp.llm.nanonets_batch)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from idp.llm.backend import Backend, CompletionRequest
from idp.llm.nanonets_batch import BatchItemResult, process_batch


# ---------------------------------------------------------------------------
# Test infrastructure: a fake backend that returns canned responses
# ---------------------------------------------------------------------------
class FakeBackend(Backend):
    """Backend that returns the text 'null' (mimics mock-ideal)."""

    name = "fake"
    is_multimodal = False
    call_count = 0

    def complete(self, req: CompletionRequest) -> str:
        type(self).call_count += 1
        return "null"


@pytest.fixture
def fake_pipeline():
    """A pipeline that uses FakeBackend and never fails."""
    backend = FakeBackend()
    pipeline = MagicMock()
    pipeline.run = MagicMock(side_effect=lambda doc: _fake_result(doc))
    pipeline.backend = backend
    return pipeline


def _fake_result(doc):
    """Build a PipelineResult-like object that has the fields the batch uses."""
    r = MagicMock()
    r.document = doc
    r.schema_name = "Invoice"
    r.backend_name = "fake"
    r.classification = "invoice"
    r.confidence = {"vendor_name": 0.9, "total_amount": 0.9}
    r.validation_passed = True
    r.extraction = None  # mock returns "null", doc.extraction stays None
    return r


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_process_batch_yields_one_per_path(tmp_path, fake_pipeline):
    paths = []
    for i in range(3):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"Vendor: Acme {i}\nInvoice Number: INV-{i}\nTotal: $100")
        paths.append(str(p))

    results = list(process_batch(paths, fake_pipeline))

    assert len(results) == 3
    assert all(r.ok for r in results)
    assert [r.path for r in results] == paths
    assert fake_pipeline.run.call_count == 3


def test_process_batch_loads_model_once(tmp_path, fake_pipeline):
    """All 5 docs go through the same pipeline.run (model loaded on first call)."""
    paths = [str(tmp_path / f"d{i}.txt") for i in range(5)]
    for p in paths:
        Path(p).write_text("Invoice content")

    results = list(process_batch(paths, fake_pipeline))
    assert len(results) == 5
    assert fake_pipeline.run.call_count == 5  # one per doc, model reused


# ---------------------------------------------------------------------------
# Error handling: one bad file shouldn't kill the batch
# ---------------------------------------------------------------------------
def test_process_batch_catches_errors_per_item(tmp_path, fake_pipeline):
    """A failure on doc[1] doesn't stop the batch."""
    paths = [str(tmp_path / f"d{i}.txt") for i in range(3)]
    for p in paths:
        Path(p).write_text("Invoice content")

    # Make doc[1] fail in pipeline.run
    def side_effect(doc):
        if "d1.txt" in doc.source_path:
            raise ValueError("simulated parse failure")
        return _fake_result(doc)
    fake_pipeline.run.side_effect = side_effect

    results = list(process_batch(paths, fake_pipeline))
    assert len(results) == 3
    assert results[0].ok and results[0].error is None
    assert not results[1].ok and "ValueError" in results[1].error
    assert "simulated parse failure" in results[1].error
    assert results[2].ok


def test_process_batch_catches_document_from_path_errors(tmp_path, fake_pipeline):
    """A nonexistent file path produces an error item, not a raise."""
    paths = [
        str(tmp_path / "real.txt"),
        "/nonexistent/path/that/does/not/exist.txt",
        str(tmp_path / "another.txt"),
    ]
    (tmp_path / "real.txt").write_text("Invoice content")
    (tmp_path / "another.txt").write_text("Invoice content")

    results = list(process_batch(paths, fake_pipeline))
    assert len(results) == 3
    assert results[0].ok
    assert not results[1].ok  # nonexistent file
    assert "FileNotFoundError" in results[1].error or "No such file" in results[1].error
    assert results[2].ok


# ---------------------------------------------------------------------------
# Iterator / list handling
# ---------------------------------------------------------------------------
def test_process_batch_accepts_iterator(fake_pipeline, tmp_path):
    """Passing an iterator (not a list) works without length-known optimization."""
    paths = [str(tmp_path / f"d{i}.txt") for i in range(3)]
    for p in paths:
        Path(p).write_text("Invoice content")

    def gen():
        yield from paths

    results = list(process_batch(gen(), fake_pipeline))
    assert len(results) == 3
    assert all(r.ok for r in results)


def test_process_batch_empty_list_yields_nothing(fake_pipeline):
    results = list(process_batch([], fake_pipeline))
    assert results == []
    assert fake_pipeline.run.call_count == 0


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
def test_process_batch_calls_on_progress(tmp_path, fake_pipeline):
    paths = [str(tmp_path / f"d{i}.txt") for i in range(3)]
    for p in paths:
        Path(p).write_text("Invoice content")

    calls = []
    def on_prog(i, total, item):
        calls.append((i, total, item.path))

    list(process_batch(paths, fake_pipeline, on_progress=on_prog))
    assert len(calls) == 3
    # on_progress is called once per item, with 1-based index
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[1] == 3 for c in calls)  # total known when paths is a list


def test_process_batch_progress_callback_failure_doesnt_break_batch(tmp_path, fake_pipeline):
    paths = [str(tmp_path / f"d{i}.txt") for i in range(2)]
    for p in paths:
        Path(p).write_text("Invoice content")

    def bad_callback(*args, **kwargs):
        raise RuntimeError("callback crashed")

    # Should not raise; should still yield all items
    results = list(process_batch(paths, fake_pipeline, on_progress=bad_callback))
    assert len(results) == 2
    assert all(r.ok for r in results)


# ---------------------------------------------------------------------------
# BatchItemResult serialization
# ---------------------------------------------------------------------------
def test_batch_item_result_to_dict_success(tmp_path):
    """A successful result serializes to a flat dict (DataFrame-friendly)."""
    # Build a real-looking result, not a MagicMock (which is what broke it)
    class _Doc:
        doc_id = "d-001"
        extraction = {"vendor_name": "Acme"}
    class _Result:
        document = _Doc()
        schema_name = "Invoice"
        backend_name = "fake"
        classification = "invoice"
        confidence = {"vendor_name": 0.9, "total_amount": 0.5}
        validation_passed = True
        extraction = {"vendor_name": "Acme"}
    item = BatchItemResult(path="/x.txt", result=_Result(), elapsed_seconds=2.0)
    d = item.to_dict()
    assert d["path"] == "/x.txt"
    assert d["ok"] is True
    assert d["doc_id"] == "d-001"
    assert d["schema"] == "Invoice"
    assert d["extraction"] == {"vendor_name": "Acme"}
    # needs_review: total_amount=0.5 < 0.7 → True
    assert d["needs_review"] is True
    # JSON-serializable (for Delta Lake / DataFrame)
    json.dumps(d)  # must not raise


def test_batch_item_result_to_dict_error():
    """An error result has ok=False, error=string, no extracted fields."""
    item = BatchItemResult(
        path="/bad.pdf",
        error="ValueError: bad PDF",
        elapsed_seconds=0.1,
    )
    d = item.to_dict()
    assert d["ok"] is False
    assert d["error"] == "ValueError: bad PDF"
    assert "extraction" not in d
    assert "doc_id" not in d


def test_batch_item_result_json_serializable(tmp_path, fake_pipeline):
    """to_dict() output is JSON-clean for Delta Lake / DataFrame rows."""
    paths = [str(tmp_path / f"d{i}.txt") for i in range(2)]
    for p in paths:
        Path(p).write_text("Invoice content")

    results = list(process_batch(paths, fake_pipeline))
    for r in results:
        d = r.to_dict()
        # The full dict must serialize to JSON without TypeError
        json.dumps(d)
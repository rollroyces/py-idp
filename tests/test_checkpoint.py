"""Tests for idp.checkpoint and process_batch() checkpoint integration."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from idp.checkpoint import CheckpointEntry, CheckpointStore
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.nanonets_batch import process_batch
from idp.pipeline import Pipeline


# ---------------------------------------------------------------------------
# CheckpointEntry — serialization round-trip
# ---------------------------------------------------------------------------
def test_entry_round_trip_with_all_fields():
    e = CheckpointEntry(
        path="/tmp/x.pdf",
        ok=True,
        error=None,
        result={"vendor_name": "Acme", "amount": 100.0},
        elapsed_seconds=1.5,
        timestamp=1234567890.0,
    )
    s = e.to_json()
    parsed = CheckpointEntry.from_json(s)
    assert parsed.path == e.path
    assert parsed.ok == e.ok
    assert parsed.error is None
    assert parsed.result == e.result
    assert parsed.elapsed_seconds == e.elapsed_seconds
    assert parsed.timestamp == e.timestamp


def test_entry_round_trip_with_error():
    e = CheckpointEntry(
        path="/tmp/x.pdf",
        ok=False,
        error="RuntimeError: connection timeout",
        result=None,
    )
    s = e.to_json()
    parsed = CheckpointEntry.from_json(s)
    assert parsed.ok is False
    assert parsed.error == "RuntimeError: connection timeout"
    assert parsed.result is None


def test_entry_default_timestamp_set_on_to_json():
    """When timestamp=0, record() sets it to time.time() automatically."""
    e = CheckpointEntry(path="/tmp/x.pdf", ok=True)
    assert e.timestamp == 0.0
    e.timestamp = 9999999999.0
    # Non-zero timestamp is preserved
    e2 = CheckpointEntry.from_json(e.to_json())
    assert e2.timestamp == 9999999999.0


# ---------------------------------------------------------------------------
# CheckpointStore — basic CRUD
# ---------------------------------------------------------------------------
def test_store_empty_file_is_empty(tmp_path):
    s = CheckpointStore(tmp_path / "cp.jsonl")
    assert len(s) == 0
    assert "/tmp/x" not in s
    s.close()


def test_store_record_and_contains(tmp_path):
    s = CheckpointStore(tmp_path / "cp.jsonl")
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    s.record(CheckpointEntry(path="/tmp/b.pdf", ok=False, error="timeout"))
    assert "/tmp/a.pdf" in s
    assert "/tmp/b.pdf" in s
    assert "/tmp/c.pdf" not in s
    assert len(s) == 2
    s.close()


def test_store_seen_paths_returns_set(tmp_path):
    s = CheckpointStore(tmp_path / "cp.jsonl")
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    s.record(CheckpointEntry(path="/tmp/b.pdf", ok=True))
    paths = s.seen_paths()
    assert isinstance(paths, set)
    assert paths == {"/tmp/a.pdf", "/tmp/b.pdf"}
    # Mutating the returned set doesn't affect the store
    paths.add("/tmp/c.pdf")
    assert "/tmp/c.pdf" not in s
    s.close()


def test_store_skips_malformed_lines(tmp_path):
    path = tmp_path / "cp.jsonl"
    path.write_text(
        '{"path": "/tmp/a.pdf", "ok": true}\n'
        'this is garbage\n'
        '{"path": "/tmp/b.pdf", "ok": true}\n'
        '{"malformed": no closing brace\n'
    )
    s = CheckpointStore(path)
    assert len(s) == 2
    assert "/tmp/a.pdf" in s
    assert "/tmp/b.pdf" in s
    s.close()


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "cp.jsonl"
    s1 = CheckpointStore(path)
    s1.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    s1.close()

    s2 = CheckpointStore(path)
    assert "/tmp/a.pdf" in s2
    s2.close()


def test_store_overwrites_same_path(tmp_path):
    """Re-recording a path keeps both lines; __len__ reports the deduped count."""
    s = CheckpointStore(tmp_path / "cp.jsonl")
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=False, error="timeout"))
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True, result={"x": 1}))
    # File has two entries (append-only); len() dedupes
    assert len(s) == 1
    # seen_paths() returns the latest single entry
    paths = s.seen_paths()
    assert paths == {"/tmp/a.pdf"}
    # But the raw file has 2 lines (debuggability)
    raw = (tmp_path / "cp.jsonl").read_text()
    assert raw.count("\n") == 2
    s.close()


def test_store_clear_truncates(tmp_path):
    s = CheckpointStore(tmp_path / "cp.jsonl")
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    s.clear()
    assert len(s) == 0
    assert "/tmp/a.pdf" not in s
    s.close()


def test_store_stats(tmp_path):
    s = CheckpointStore(tmp_path / "cp.jsonl")
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    s.record(CheckpointEntry(path="/tmp/b.pdf", ok=False, error="timeout"))
    s.record(CheckpointEntry(path="/tmp/c.pdf", ok=True))
    stats = s.stats()
    assert stats["total"] == 3
    assert stats["ok"] == 2
    assert stats["errors"] == 1
    assert stats["first_ts"] is not None
    assert stats["last_ts"] is not None
    s.close()


def test_store_stats_empty(tmp_path):
    s = CheckpointStore(tmp_path / "cp.jsonl")
    stats = s.stats()
    assert stats == {"total": 0, "ok": 0, "errors": 0, "first_ts": None, "last_ts": None}
    s.close()


def test_store_context_manager(tmp_path):
    path = tmp_path / "cp.jsonl"
    with CheckpointStore(path) as s:
        s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    # After exit, the store can be reopened
    s2 = CheckpointStore(path)
    assert "/tmp/a.pdf" in s2
    s2.close()


def test_store_atomic_single_line_write(tmp_path):
    """A record() call writes exactly one JSON line per call (crash-safe)."""
    s = CheckpointStore(tmp_path / "cp.jsonl")
    s.record(CheckpointEntry(path="/tmp/a.pdf", ok=True))
    s.close()
    raw = (tmp_path / "cp.jsonl").read_text()
    # Single trailing newline; one entry
    assert raw.count("\n") == 1
    assert raw.endswith("\n")


# ---------------------------------------------------------------------------
# process_batch() — checkpoint integration
# ---------------------------------------------------------------------------
def _make_pipeline_with_mock(mock_responses: list[str]):
    """Build a Pipeline that returns the given JSON for each call."""
    call_count = [0]

    def complete(req):
        response = mock_responses[call_count[0] % len(mock_responses)]
        call_count[0] += 1
        return response

    inner = MagicMock()
    inner.name = "test-mock"
    inner.is_multimodal = False
    inner.complete = complete

    return Pipeline(backend=inner, schema=Invoice), call_count


def test_process_batch_with_checkpoint_records_each_doc(tmp_path):
    """Every processed doc (success or failure) gets recorded in the checkpoint."""
    cp_path = tmp_path / "cp.jsonl"
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")

    # 3 docs, all succeed
    pipeline, calls = _make_pipeline_with_mock([
        json.dumps({"vendor_name": "Acme", "invoice_number": "INV-1",
                   "total_amount": 100.0, "line_items": []}),
    ] * 5)

    paths = [str(tmp_path / f"doc_{i}.pdf") for i in range(3)]
    for p in paths:
        Path_ = __import__("pathlib").Path
        Path_(p).write_bytes(b"%PDF")

    results = list(process_batch(
        paths, pipeline,
        progress_every=0,
        checkpoint=cp_path,
    ))

    assert len(results) == 3
    # Backend called once per doc
    assert calls[0] == 3

    # Checkpoint has all 3 paths
    cp = CheckpointStore(cp_path)
    for p in paths:
        assert p in cp
    cp.close()


def test_process_batch_records_failures_too(tmp_path):
    """Failed docs also get recorded — so re-runs don't retry them."""
    cp_path = tmp_path / "cp.jsonl"
    paths = [str(tmp_path / f"doc_{i}.pdf") for i in range(3)]
    for p in paths:
        Path_ = __import__("pathlib").Path
        Path_(p).write_bytes(b"%PDF")

    # Mock the pipeline's run() method directly so we control success/failure
    pipeline = MagicMock()
    pipeline.run = MagicMock(side_effect=[
        _make_pipeline_result_success(),
        RuntimeError("LLM down"),
        _make_pipeline_result_success(),
    ])

    results = list(process_batch(paths, pipeline, progress_every=0, checkpoint=cp_path))

    assert len(results) == 3
    assert sum(1 for r in results if r.ok) == 2
    assert sum(1 for r in results if not r.ok) == 1

    # All 3 paths recorded, including the failed one
    cp = CheckpointStore(cp_path)
    for p in paths:
        assert p in cp, f"{p} missing from checkpoint"
    cp.close()


def _make_pipeline_result_success():
    """Build a PipelineResult that looks successful."""
    from idp.pipeline.pipeline import PipelineResult, StageTiming
    doc = Document(source_path="x.pdf", doc_id="x", raw_text="Acme INV-1 $100")
    doc.extraction = {"vendor_name": "Acme", "invoice_number": "INV-1",
                      "total_amount": 100.0, "line_items": []}
    return PipelineResult(
        document=doc,
        schema_name="Invoice",
        timings=[StageTiming(name="parse", seconds=0.01)],
        backend_name="mock",
        mode="ocr_llm",
        classification="invoice",
        confidence={"vendor_name": 0.95},
        validation_passed=True,
    )


def test_process_batch_skips_already_done_paths(tmp_path):
    """Idempotent resume: re-running skips paths in the checkpoint."""
    cp_path = tmp_path / "cp.jsonl"
    paths = [str(tmp_path / f"doc_{i}.pdf") for i in range(5)]
    for p in paths:
        Path_ = __import__("pathlib").Path
        Path_(p).write_bytes(b"%PDF")

    pipeline, calls = _make_pipeline_with_mock([
        json.dumps({"vendor_name": "Acme", "invoice_number": "INV-1",
                   "total_amount": 100.0, "line_items": []}),
    ] * 10)

    # First run: process all 5
    results1 = list(process_batch(paths, pipeline, progress_every=0, checkpoint=cp_path))
    assert len(results1) == 5
    assert calls[0] == 5

    # Second run: all 5 should be skipped (idempotent)
    pipeline2, calls2 = _make_pipeline_with_mock([
        json.dumps({"vendor_name": "Other", "invoice_number": "INV-2",
                   "total_amount": 200.0, "line_items": []}),
    ] * 10)
    results2 = list(process_batch(paths, pipeline2, progress_every=0, checkpoint=cp_path))

    assert len(results2) == 0  # no items yielded (all skipped)
    assert calls2[0] == 0  # backend never called


def test_process_batch_partial_resume(tmp_path):
    """Simulate mid-batch failure: re-run completes the rest, doesn't redo done docs."""
    cp_path = tmp_path / "cp.jsonl"
    paths = [str(tmp_path / f"doc_{i}.pdf") for i in range(10)]
    for p in paths:
        Path_ = __import__("pathlib").Path
        Path_(p).write_bytes(b"%PDF")

    success_result = _make_pipeline_result_success()

    # First run: process 4, then simulate a crash on the 5th
    pipeline = MagicMock()
    call_count = [0]

    def run_with_crash(doc):
        call_count[0] += 1
        if call_count[0] == 5:
            raise RuntimeError("simulated mid-batch crash")
        return success_result

    pipeline.run = run_with_crash

    items = []
    try:
        for item in process_batch(paths, pipeline, progress_every=0, checkpoint=cp_path):
            items.append(item)
            if not item.ok:
                # Simulate "client crashes mid-batch" by raising here
                raise RuntimeError("simulated client crash")
    except RuntimeError as e:
        if "simulated client crash" not in str(e):
            raise

    # 5 items processed (4 ok + 1 fail), then we crashed before the 6th
    assert len(items) == 5
    assert call_count[0] == 5  # 5 backend calls

    # Second run: should skip the 5 already-done, process only the remaining 5
    pipeline2 = MagicMock()
    call_count2 = [0]

    def run_2(doc):
        call_count2[0] += 1
        return success_result

    pipeline2.run = run_2

    items2 = list(process_batch(paths, pipeline2, progress_every=0, checkpoint=cp_path))
    assert len(items2) == 5  # the remaining 5
    assert call_count2[0] == 5  # backend called only for those 5


def test_process_batch_without_checkpoint_is_unchanged(tmp_path):
    """Without checkpoint, all docs are processed every run."""
    paths = [str(tmp_path / f"doc_{i}.pdf") for i in range(3)]
    for p in paths:
        Path_ = __import__("pathlib").Path
        Path_(p).write_bytes(b"%PDF")

    pipeline, calls = _make_pipeline_with_mock([
        json.dumps({"vendor_name": "Acme", "invoice_number": "INV-1",
                   "total_amount": 100.0, "line_items": []}),
    ] * 5)

    # First run: 3 calls
    list(process_batch(paths, pipeline, progress_every=0))
    # Second run without checkpoint: 3 more calls
    list(process_batch(paths, pipeline, progress_every=0))

    assert calls[0] == 6  # no caching


def test_process_batch_accepts_string_checkpoint_path(tmp_path):
    """checkpoint='path/to/file.jsonl' should work as well as a CheckpointStore."""
    cp_path = str(tmp_path / "cp.jsonl")
    paths = [str(tmp_path / f"doc_{i}.pdf") for i in range(2)]
    for p in paths:
        Path_ = __import__("pathlib").Path
        Path_(p).write_bytes(b"%PDF")

    pipeline, calls = _make_pipeline_with_mock([
        json.dumps({"vendor_name": "Acme", "invoice_number": "INV-1",
                   "total_amount": 100.0, "line_items": []}),
    ] * 5)

    list(process_batch(paths, pipeline, progress_every=0, checkpoint=cp_path))

    # Reopen from disk and verify
    cp = CheckpointStore(cp_path)
    assert len(cp) == 2
    cp.close()

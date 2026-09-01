"""E5: stage-timing benchmark (pytest-benchmark).

Guards against perf regressions in the CPU-bound stages (everything
except the LLM call). These are the microsecond-level operations that
should stay microsecond-level; if a future change makes `parse` take
50ms on a .txt file, these tests fail.

LLM calls are NOT benchmarked here (they're external + non-deterministic).
"""
from __future__ import annotations

from pathlib import Path

from idp.assess import assess_confidence
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.backend import MockBackend
from idp.parse.parser import PlainTextParser, parse_document
from idp.validate import validate

SAMPLE = Path("/Users/hermes/py-idp/src/idp/eval/datasets/invoices/docs/inv-001.txt")


def test_bench_plain_text_parse(benchmark):
    p = PlainTextParser()
    result = benchmark(lambda: p.parse(str(SAMPLE)))
    assert result["text"]


def test_bench_parse_document(benchmark):
    doc = Document.from_path(str(SAMPLE))
    result = benchmark(lambda: parse_document(doc, parser=PlainTextParser()))
    assert result.raw_text


def test_bench_assess_confidence(benchmark):
    doc = Document.from_path(str(SAMPLE))
    doc.mode = "ocr_llm"
    # Use a minimal valid Invoice (required fields only)
    doc.extraction = {
        "invoice_number": "INV-1",
        "vendor_name": "Acme",
        "total_amount": 540.0,
    }
    result = benchmark(lambda: assess_confidence(doc))
    assert result.confidence


def test_bench_validate(benchmark):
    doc = Document.from_path(str(SAMPLE))
    doc.extraction = {"invoice_number": "INV-1", "total_amount": 540.0}
    doc.extraction_schema = "Invoice"
    result = benchmark(lambda: validate(doc))
    assert result.validation


def test_bench_mock_extract_pipeline_no_llm(benchmark):
    """Full pipeline with MockBackend (no network) — measures framework overhead."""
    from idp.pipeline.pipeline import Pipeline

    p = Pipeline(backend=MockBackend(), schema=Invoice)
    result = benchmark(lambda: p.run(Document.from_path(str(SAMPLE))))
    assert result.document.extraction is not None

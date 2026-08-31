"""End-to-end pipeline tests with the mock backend."""
from pathlib import Path

from idp.classify.classifier import _rule_classify
from idp.core.document import Document
from idp.core.schemas import Contract, Invoice
from idp.pipeline.pipeline import Pipeline


def _doc_with_text(rel: str) -> Document:
    p = Path("src/idp/eval/datasets") / rel
    d = Document.from_path(str(p))
    d.raw_text = p.read_text()
    return d


def test_rule_classifies_invoice():
    doc = _doc_with_text("invoices/docs/inv-001.txt")
    label, conf = _rule_classify(doc)
    assert label == "invoice"
    assert conf >= 0.7


def test_rule_classifies_contract():
    doc = _doc_with_text("contracts/docs/svc-001.txt")
    label, conf = _rule_classify(doc)
    assert label == "contract"
    assert conf >= 0.7


def test_pipeline_invoice_runs_end_to_end():
    path = Path("src/idp/eval/datasets/invoices/docs/inv-001.txt")
    p = Pipeline(backend="mock", schema=Invoice)
    res = p.run(Document.from_path(path))
    assert res.classification == "invoice", res.classification
    assert res.document.extraction is not None
    assert "invoice_number" in res.document.extraction


def test_pipeline_contract_runs_end_to_end():
    path = Path("src/idp/eval/datasets/contracts/docs/svc-001.txt")
    p = Pipeline(backend="mock", schema=Contract)
    res = p.run(Document.from_path(path))
    assert res.classification == "contract", res.classification
    assert res.document.extraction is not None
    assert "effective_date" in res.document.extraction

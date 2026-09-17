"""End-to-end pipeline tests with the mock backend."""
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# _rule_classify edge cases (lines 76, 78, 101-104, 126-129)
# ---------------------------------------------------------------------------
def test_rule_classify_boosts_confidence_to_95_when_first_line_matches(tmp_path):
    """First line starting with the inferred label → 0.95 confidence."""
    from idp.classify.classifier import _rule_classify

    f = tmp_path / "inv.txt"
    f.write_text("INVOICE INV-001 from Acme\nTotal: $100")
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    label, conf = _rule_classify(doc)
    assert label == "invoice"
    assert conf == pytest.approx(0.95)


def test_rule_classify_first_line_matches_label_with_underscore_split(tmp_path):
    """First line starts with the schema's first underscore-split segment."""
    from idp.classify.classifier import _rule_classify

    f = tmp_path / "stmt.txt"
    f.write_text("Bank Statement for Acme\nAccount: 1234")
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    label, conf = _rule_classify(doc)
    assert label == "bank_statement"
    assert conf == pytest.approx(0.95)


def test_rule_classify_returns_none_when_no_match(tmp_path):
    """If no keyword matches, returns (None, 0.0)."""
    from idp.classify.classifier import _rule_classify

    f = tmp_path / "x.txt"
    f.write_text("Random text with no keywords\nNothing here")
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    label, conf = _rule_classify(doc)
    assert label is None
    assert conf == 0.0


def test_rule_classify_handles_empty_text(tmp_path):
    """Empty document text -> no match, no crash."""
    from idp.classify.classifier import _rule_classify

    f = tmp_path / "x.txt"
    f.write_text("")
    doc = Document.from_path(str(f))
    doc.raw_text = ""
    label, conf = _rule_classify(doc)
    assert label is None
    assert conf == 0.0


def test_header_label_invoice():
    from idp.classify.classifier import _header_label
    # _header_label receives lowercased text (see _rule_classify line 46)
    assert _header_label("invoice inv-001\n...") == "invoice"


def test_header_label_contract():
    from idp.classify.classifier import _header_label
    assert _header_label("agreement between parties\n...") == "contract"
    assert _header_label("contract for services\n...") == "contract"
    assert _header_label("master services agreement\n...") == "contract"


def test_header_label_bank_statement():
    from idp.classify.classifier import _header_label
    assert _header_label("statement for the period\n...") == "bank_statement"
    assert _header_label("bank statement january 2024\n...") == "bank_statement"


def test_header_label_receipt():
    from idp.classify.classifier import _header_label
    assert _header_label("receipt for purchase\n...") == "receipt"


def test_header_label_returns_empty_for_unknown():
    from idp.classify.classifier import _header_label
    assert _header_label("random title\nbody") == ""


def test_header_label_skips_blank_lines():
    """Leading blank lines are skipped to find first non-empty line."""
    from idp.classify.classifier import _header_label
    assert _header_label("\n\n\ninvoice inv-001\n...") == "invoice"


# ---------------------------------------------------------------------------
# classify_document with LLM fallback (lines 126-129)
# ---------------------------------------------------------------------------
def test_classify_document_uses_llm_when_rule_weak(tmp_path):
    """When rule returns low confidence, LLM is consulted."""
    from unittest.mock import MagicMock

    from idp.classify.classifier import classify_document

    f = tmp_path / "doc.txt"
    f.write_text("Random document about nothing in particular.")  # no keywords
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    backend = MagicMock()
    backend.name = "fake"
    backend.complete.return_value = '{"label": "report", "confidence": 0.85}'

    out = classify_document(doc, backend, confidence_floor=0.7)
    # Rule was weak; LLM took over
    assert out.classification == "report"
    assert out.classification_confidence == 0.85
    assert out.metadata["classification_route"] == "rule+llm"


def test_classify_document_records_error_when_llm_fails(tmp_path):
    """If LLM throws and rule is weak, doc.errors records the failure."""
    from unittest.mock import MagicMock

    from idp.classify.classifier import classify_document

    f = tmp_path / "doc.txt"
    f.write_text("Random text without keywords.")
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    backend = MagicMock()
    backend.complete.side_effect = RuntimeError("LLM exploded")

    out = classify_document(doc, backend, confidence_floor=0.7)
    # Rule was None (no keywords), LLM failed → falls back to ("other", 0.0)
    assert out.classification == "other"
    assert out.classification_confidence == 0.0
    assert any("classify_llm_failed" in e for e in out.errors)


def test_classify_document_skips_llm_when_rule_strong(tmp_path):
    """When rule returns high confidence (≥ floor), LLM is NOT called."""
    from unittest.mock import MagicMock

    from idp.classify.classifier import classify_document

    f = tmp_path / "doc.txt"
    f.write_text("Invoice INV-001 from Acme\nTotal: $100")  # clear invoice match
    doc = Document.from_path(str(f))
    doc.raw_text = f.read_text()
    backend = MagicMock()
    backend.complete.side_effect = AssertionError("LLM must not be called")

    out = classify_document(doc, backend, confidence_floor=0.7)
    assert out.classification == "invoice"
    assert out.classification_confidence >= 0.95
    # Route should be "rule" only (no LLM)
    assert out.metadata["classification_route"] == "rule"


# ---------------------------------------------------------------------------
# _llm_classify JSON fallback (lines 101-104)
# ---------------------------------------------------------------------------
def test_llm_classify_falls_back_to_regex_on_invalid_json(tmp_path):
    """If backend returns non-JSON, parse label/confidence via regex fallback."""
    from unittest.mock import MagicMock

    from idp.classify.classifier import _llm_classify

    f = tmp_path / "x.txt"
    f.write_text("placeholder")
    doc = Document.from_path(str(f))
    backend = MagicMock()
    backend.complete.return_value = 'Some preamble "label": "invoice", "confidence": 0.77 trailing text'

    label, conf = _llm_classify(doc, backend)
    assert label == "invoice"
    assert conf == 0.77


def test_llm_classify_falls_back_to_defaults_on_no_match(tmp_path):
    """If backend returns garbage with no label/confidence, return (other, 0.5)."""
    from unittest.mock import MagicMock

    from idp.classify.classifier import _llm_classify

    f = tmp_path / "x.txt"
    f.write_text("placeholder")
    doc = Document.from_path(str(f))
    backend = MagicMock()
    backend.complete.return_value = "totally unstructured text with no JSON"

    label, conf = _llm_classify(doc, backend)
    assert label == "other"
    assert conf == 0.5

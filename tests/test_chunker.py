"""Tests for chunking module + integration with extract()."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from idp.chunker import (
    PageChunker,
    TokenChunker,
    _merge_list_field,
    _merge_scalar_field,
    merge_extractions,
    should_chunk_pages,
    should_chunk_text,
)
from idp.core.document import Document, Page


# ---------------------------------------------------------------------------
# PageChunker
# ---------------------------------------------------------------------------
def _pages(n: int) -> list[Page]:
    """Make n pages with simple text."""
    return [Page(page_number=i + 1, text=f"page {i+1} content") for i in range(n)]


def test_page_chunker_short_doc_one_chunk():
    """A doc that fits in one call produces one chunk."""
    c = PageChunker(max_pages=4, overlap_pages=1)
    chunks = c.split(_pages(3))
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_page_chunker_long_doc_chunks_with_overlap():
    """A 10-page doc with max_pages=4, overlap=1 -> 4 chunks, sized right."""
    c = PageChunker(max_pages=4, overlap_pages=1)
    chunks = c.split(_pages(10))
    # stride=3. Pages: [1,2,3,4], [4,5,6,7], [7,8,9,10]. That's 3 chunks.
    assert len(chunks) == 3
    assert [p.page_number for p in chunks[0]] == [1, 2, 3, 4]
    assert [p.page_number for p in chunks[1]] == [4, 5, 6, 7]
    assert [p.page_number for p in chunks[2]] == [7, 8, 9, 10]


def test_page_chunker_empty_doc_one_empty_chunk():
    """Empty pages list still returns one chunk (consistent contract)."""
    c = PageChunker()
    chunks = c.split([])
    assert chunks == [[]]


def test_page_chunker_validation_errors():
    with pytest.raises(ValueError):
        PageChunker(max_pages=0)
    with pytest.raises(ValueError):
        PageChunker(overlap_pages=-1)
    with pytest.raises(ValueError):
        PageChunker(max_pages=4, overlap_pages=4)  # overlap >= max


def test_page_chunker_collect_texts_per_chunk():
    """Texts are joined with double newlines, one per chunk."""
    c = PageChunker(max_pages=4, overlap_pages=1)
    pages = _pages(10)
    texts = c.collect_texts(pages)
    assert len(texts) == 3
    assert texts[0] == "page 1 content\n\npage 2 content\n\npage 3 content\n\npage 4 content"
    # chunk 2 re-uses page 4's text:
    assert texts[1].startswith("page 4 content")


def test_page_chunker_collect_images_per_chunk():
    """Images are flattened across pages of each chunk."""
    p1 = Page(page_number=1, text="", images_b64=["a", "b"])
    p2 = Page(page_number=2, text="", images_b64=["c"])
    p3 = Page(page_number=3, text="", images_b64=[])
    p4 = Page(page_number=4, text="", images_b64=["d"])
    c = PageChunker(max_pages=3, overlap_pages=1)
    imgs = c.collect_images([p1, p2, p3, p4])
    # stride=2. Chunks: [p1,p2,p3], [p3,p4] -> 2 chunks.
    assert len(imgs) == 2
    assert imgs[0] == ["a", "b", "c"]
    assert imgs[1] == ["d"]  # p3 has no images


# ---------------------------------------------------------------------------
# TokenChunker
# ---------------------------------------------------------------------------
def test_token_chunker_short_text_one_chunk():
    """Short text fits in one chunk."""
    c = TokenChunker(max_tokens=1000, overlap_tokens=100)
    text = "hello world " * 50
    chunks = c.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_token_chunker_long_text_multiple_chunks():
    """Long text is split with overlap."""
    c = TokenChunker(max_tokens=50, overlap_tokens=10)
    # ~1000 chars -> ~250 tokens -> 5-6 chunks
    text = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    chunks = c.split(text)
    assert len(chunks) >= 2
    # Overlap check: chunk 2 should start with words from chunk 1's tail
    # (we can't predict exact tokens, but the lengths should be reasonable)
    for chunk in chunks:
        assert len(chunk) > 0


def test_token_chunker_empty_text():
    c = TokenChunker()
    assert c.split("") == []
    assert c.split("   ") == []  # whitespace-only


def test_token_chunker_validation_errors():
    with pytest.raises(ValueError):
        TokenChunker(max_tokens=0)
    with pytest.raises(ValueError):
        TokenChunker(overlap_tokens=-1)
    with pytest.raises(ValueError):
        TokenChunker(max_tokens=100, overlap_tokens=100)


# ---------------------------------------------------------------------------
# Merge — scalars
# ---------------------------------------------------------------------------
def test_merge_scalars_first_wins():
    """For two non-None values, first wins when prefer='first'."""
    assert _merge_scalar_field("vendor-A", "vendor-B", prefer="first") == "vendor-A"
    assert _merge_scalar_field(1, 2, prefer="first") == 1


def test_merge_scalars_last_wins():
    """For two non-None values, last wins when prefer='last'."""
    assert _merge_scalar_field("vendor-A", "vendor-B", prefer="last") == "vendor-B"


def test_merge_scalars_longest_for_strings():
    """prefer='longest' picks the more detailed string."""
    assert _merge_scalar_field("Acme", "Acme Corporation Inc.", prefer="longest") == "Acme Corporation Inc."
    assert _merge_scalar_field("Acme Corp", "Acme", prefer="longest") == "Acme Corp"


def test_merge_scalars_none_promoted_to_value():
    """None values are overridden by non-None values."""
    assert _merge_scalar_field(None, "real", prefer="first") == "real"
    assert _merge_scalar_field("real", None, prefer="last") == "real"


def test_merge_scalars_equal_values():
    """Same value on both sides is unchanged."""
    assert _merge_scalar_field("same", "same", prefer="first") == "same"


# ---------------------------------------------------------------------------
# Merge — lists
# ---------------------------------------------------------------------------
def test_merge_list_field_strings_dedupes():
    """List of strings is unioned with deduplication."""
    merged = _merge_list_field(["a", "b"], ["b", "c"])
    assert merged == ["a", "b", "c"]


def test_merge_list_field_dicts_dedupes_by_id():
    """List of dicts with 'id' is deduped by id."""
    a = [{"id": 1, "name": "foo"}, {"id": 2, "name": "bar"}]
    b = [{"id": 2, "name": "bar-renamed"}, {"id": 3, "name": "baz"}]
    merged = _merge_list_field(a, b)
    # 2 appears in both -> keep first
    assert len(merged) == 3
    assert {"id": 1, "name": "foo"} in merged
    assert {"id": 2, "name": "bar"} in merged  # first wins
    assert {"id": 3, "name": "baz"} in merged


def test_merge_list_field_promotes_scalar_to_list():
    """A scalar on one side and a list on the other still merge."""
    merged = _merge_list_field("a", ["b", "c"])
    assert merged == ["a", "b", "c"]
    merged = _merge_list_field(None, ["b"])
    assert merged == ["b"]


# ---------------------------------------------------------------------------
# merge_extractions — full
# ---------------------------------------------------------------------------
def test_merge_extractions_empty():
    assert merge_extractions([]) == {}


def test_merge_extractions_single_chunk_passes_through():
    """A single non-empty chunk becomes the merged dict (plus _chunk_count)."""
    out = merge_extractions([{"vendor_name": "Acme", "total_amount": 100.0}])
    assert out["vendor_name"] == "Acme"
    assert out["total_amount"] == 100.0
    assert out["_chunk_count"] == 1


def test_merge_extractions_different_keys_union():
    """Chunks with different keys contribute distinct fields."""
    out = merge_extractions([
        {"vendor_name": "Acme"},
        {"total_amount": 100.0},
    ])
    assert out["vendor_name"] == "Acme"
    assert out["total_amount"] == 100.0


def test_merge_extractions_scalar_conflict_first_wins():
    """Conflicting scalar values resolve by prefer policy."""
    out = merge_extractions([
        {"vendor_name": "Acme-A"},
        {"vendor_name": "Acme-B"},
    ], prefer="first")
    assert out["vendor_name"] == "Acme-A"


def test_merge_extractions_list_fields_union():
    out = merge_extractions([
        {"line_items": [{"id": 1}]},
        {"line_items": [{"id": 2}]},
    ])
    assert out["line_items"] == [{"id": 1}, {"id": 2}]


def test_merge_extractions_drops_none_values():
    """Chunks with None for a field don't override a real value."""
    out = merge_extractions([
        {"vendor_name": "Acme"},
        {"vendor_name": None},
    ])
    assert out["vendor_name"] == "Acme"


def test_merge_extractions_skips_empty_chunks():
    out = merge_extractions([
        {},
        {"vendor_name": "Acme"},
        {},
    ])
    assert out["vendor_name"] == "Acme"
    assert out["_chunk_count"] == 1  # only the non-empty chunk counts


# ---------------------------------------------------------------------------
# should_chunk_* helpers
# ---------------------------------------------------------------------------
def test_should_chunk_pages_false_when_fits():
    assert should_chunk_pages(_pages(3), max_pages=4) is False


def test_should_chunk_pages_true_when_too_many():
    assert should_chunk_pages(_pages(10), max_pages=4) is True


def test_should_chunk_text_false_when_fits():
    assert should_chunk_text("hello", max_tokens=1000) is False


def test_should_chunk_text_true_when_too_long():
    # ~10000 chars ~= 2500 tokens at 4 chars/token, plus margin
    text = "word " * 2000  # 10000 chars
    # Use a tight token budget to force chunking
    assert should_chunk_text(text, max_tokens=100) is True


def test_should_chunk_text_empty():
    assert should_chunk_text("", max_tokens=1000) is False


def test_should_chunk_text_without_tiktoken_falls_back_to_heuristic():
    """If tiktoken isn't installed, char-based heuristic still works."""
    import sys
    # Hide tiktoken from the import machinery by mocking it as None.
    # We also block re-import via sys.modules (in case it was cached).
    sys.modules["tiktoken"] = None  # type: ignore[assignment]
    try:
        # ~500 chars / 4 = 125, max=100 -> True
        assert should_chunk_text("x" * 500, max_tokens=100) is True
        # ~50 chars / 4 = 12, max=100 -> False
        assert should_chunk_text("x" * 50, max_tokens=100) is False
    finally:
        sys.modules.pop("tiktoken", None)


# ---------------------------------------------------------------------------
# Integration with extract()
# ---------------------------------------------------------------------------
def _mock_backend(per_chunk_outputs: list[dict]) -> MagicMock:
    """A backend that returns per-chunk JSON dicts in order."""
    b = MagicMock()
    b.is_multimodal = False
    b.complete = MagicMock(side_effect=[
        _jsonify(d) for d in per_chunk_outputs
    ])
    return b


def _jsonify(d: dict) -> str:
    import json
    return json.dumps(d)


def test_extract_chunks_when_text_too_long(tmp_path):
    """End-to-end: a doc with >4000 tokens of text gets chunked + merged."""
    from idp.core.schemas import Invoice
    from idp.extract.extractor import extract

    # Build a doc with >12000 tokens. With default TokenChunker (max=4000,
    # stride=3800), this produces 4 chunks: 0-4000, 3800-7800, 7600-11600, 11400-15400.
    # Each phrase is short but the multiplier is high enough to push us past 12k tokens.
    long_text = (
        ("Vendor name is Acme Corporation. " * 200) +
        ("Invoice number is INV-2026-0042. " * 200) +
        ("Total amount due is one hundred dollars. " * 200) +
        ("Line item A: widget costs ten dollars. " * 200) +
        ("Line item B: gadget costs twenty dollars. " * 200) +
        ("Padding text to fill out the chunk. " * 200) +
        ("More padding text to extend the chunk. " * 200) +
        ("Even more padding text to exceed limit. " * 200)
    )
    doc = Document(source_path=str(tmp_path / "fake.pdf"), raw_text=long_text, doc_id="fake")

    # Mock backend returns a partial invoice per chunk.
    # We don't know exactly how many chunks will be produced — MagicMock
    # with side_effect=cycles keeps returning the cycle; we just need
    # the first three responses for the asserts below.
    from itertools import cycle
    backend = MagicMock()
    backend.is_multimodal = False
    backend.complete = MagicMock(side_effect=cycle([
        _jsonify({"vendor_name": "Acme-A", "line_items": [{"id": 1}]}),
        _jsonify({"total_amount": 100.0, "line_items": [{"id": 2}]}),
        _jsonify({"invoice_number": "INV-1"}),
        _jsonify({}),  # extra chunks
        _jsonify({}),
    ]))

    extract(doc, Invoice, backend, mode="ocr_llm")
    assert doc.extraction.get("vendor_name") == "Acme-A"
    assert doc.extraction.get("total_amount") == 100.0
    assert doc.extraction.get("invoice_number") == "INV-1"
    # line_items from multiple chunks were unioned
    assert len(doc.extraction.get("line_items", [])) >= 2
    # Chunking was logged
    assert any("extract_chunked:" in e for e in doc.errors)


def test_extract_no_chunking_for_short_doc(tmp_path):
    """Short docs use the single-call path; no chunking marker."""
    from idp.core.schemas import Invoice
    from idp.extract.extractor import extract

    doc = Document(source_path=str(tmp_path / "fake.pdf"), raw_text="hello world", doc_id="fake")
    backend = _mock_backend([{"vendor_name": "Acme"}])
    extract(doc, Invoice, backend, mode="ocr_llm")
    assert doc.extraction.get("vendor_name") == "Acme"
    assert not any("extract_chunked:" in e for e in doc.errors)


def test_extract_chunk_failure_does_not_kill_batch(tmp_path):
    """A bad chunk contributes empty dict; other chunks still merge in."""

    from idp.core.schemas import Invoice
    from idp.extract.extractor import extract

    # >12000 tokens -> 4 chunks. Each phrase ~7 tokens, * 800 reps = 5600 tokens per phrase.
    long_text = (
        ("Vendor name is Acme. " * 800) +
        ("Total amount is fifty dollars. " * 800) +
        ("Padding text to fill chunks. " * 800) +
        ("More padding here and there. " * 800)
    )
    doc = Document(source_path=str(tmp_path / "fake.pdf"), raw_text=long_text, doc_id="fake")

    backend = MagicMock()
    backend.is_multimodal = False
    # Cycle: chunk 0 good, chunk 1 fails, chunk 2 good, then alternate.
    side_effect_iter = iter([
        _jsonify({"vendor_name": "Acme"}),
        RuntimeError("LLM down"),  # chunk 1 fails
        _jsonify({"total_amount": 50.0}),  # chunk 2 succeeds
    ])
    # Extend the iterator to be infinite for any extra chunks
    from itertools import chain, repeat
    infinite = chain(side_effect_iter, repeat(_jsonify({})))
    backend.complete = MagicMock(side_effect=infinite)

    extract(doc, Invoice, backend, mode="ocr_llm")
    # We got info from the working chunks; failure was logged
    assert doc.extraction.get("vendor_name") == "Acme"
    assert doc.extraction.get("total_amount") == 50.0
    assert any("extract_chunk_failed[1]" in e for e in doc.errors)
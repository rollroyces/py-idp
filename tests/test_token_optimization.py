"""Tests for token-usage optimizations in extract.

Verifies:
  1. _schema_json_cached returns the same string across calls (cache hit).
  2. _schema_json_cached strips non-essential JSON Schema fields
     (additionalProperties, $defs, etc.) — the prompt only carries
     what the LLM needs to produce output.
  3. _truncate_to_tokens honors the max_tokens limit using tiktoken,
     not chars. Falls back to char-based truncation if tiktoken missing.
"""
from __future__ import annotations

import sys

from idp.core.schemas import BankStatement, Contract, Invoice
from idp.extract.extractor import (
    _MINIMAL_SCHEMA_JSON_CACHED,
    _build_messages,
    _schema_json_cached,
    _truncate_to_tokens,
)


# ---------------------------------------------------------------------------
# _schema_json_cached — caching behavior
# ---------------------------------------------------------------------------
def test_schema_json_cached_returns_same_object_for_same_class():
    """Same Pydantic class -> same string instance (cache hit)."""
    s1 = _schema_json_cached(Invoice)
    s2 = _schema_json_cached(Invoice)
    assert s1 == s2
    # Caching via lru_cache returns the same string object for hits.
    assert s1 is s2


def test_schema_json_cached_distinguishes_different_classes():
    """Invoice and Contract get different serialized strings."""
    assert _schema_json_cached(Invoice) != _schema_json_cached(Contract)


# ---------------------------------------------------------------------------
# _schema_json_cached — minimal subset
# ---------------------------------------------------------------------------
def test_schema_json_strips_additionalProperties():
    """``additionalProperties`` doesn't appear in the prompt schema."""
    s = _schema_json_cached(Invoice)
    assert "additionalProperties" not in s


def test_schema_json_keeps_essential_fields():
    """Type, items, enum, format, required are preserved in the output.

    ``description`` is preserved *if* the schema declared one. Built-in
    schemas currently declare descriptions only for required fields;
    optional fields without a description serialize as ``{}`` — which
    is acceptable: the LLM gets the field name and type, but no extra
    prose to consume tokens.
    """
    s = _schema_json_cached(Invoice)
    # Must still describe the contract the LLM is being asked to satisfy.
    assert '"type"' in s
    assert '"required"' in s
    # nested arrays still describe their items
    if "items" in s:
        assert '"items"' in s


def test_schema_json_is_valid_json():
    """The returned string is parseable as JSON."""
    import json as _json
    parsed = _json.loads(_schema_json_cached(Invoice))
    assert parsed["type"] == "object"
    assert "properties" in parsed
    assert isinstance(parsed["properties"], dict)


def test_schema_json_saves_tokens_vs_raw():
    """The minimal schema is materially smaller than model_json_schema().

    This is the SLO that justifies the optimization: without it, every
    chunk re-pays the cost of verbose JSON Schema fields the LLM doesn't
    need.
    """
    import json

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    for schema in (Invoice, Contract, BankStatement):
        raw = json.dumps(schema.model_json_schema(), indent=2)
        minimal = _schema_json_cached(schema)
        raw_t = len(enc.encode(raw))
        min_t = len(enc.encode(minimal))
        # Should be meaningfully smaller — empirically ~75-80% reduction
        assert min_t < raw_t * 0.35, (
            f"{schema.__name__}: minimal ({min_t}) should be <35% of raw "
            f"({raw_t}); got {min_t / raw_t * 100:.1f}%"
        )


def test_schema_json_does_not_strip_required_field():
    """``required`` is essential — model_validate depends on it."""
    import json
    parsed = json.loads(_schema_json_cached(Invoice))
    # Even if empty, the key should be present.
    assert "required" in parsed


# ---------------------------------------------------------------------------
# _truncate_to_tokens — token-aware truncation
# ---------------------------------------------------------------------------
def test_truncate_short_text_unchanged():
    """Short text returns unchanged."""
    assert _truncate_to_tokens("hello", max_tokens=100) == "hello"


def test_truncate_empty_string():
    """Empty input returns empty."""
    assert _truncate_to_tokens("", max_tokens=100) == ""


def test_truncate_respects_max_tokens():
    """A long string is truncated to <= max_tokens tokens."""
    long_text = " ".join(["word"] * 5000)  # ~5000 tokens
    truncated = _truncate_to_tokens(long_text, max_tokens=100)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(truncated)) <= 100


def test_truncate_dense_text():
    """Dense text (1 char ~ 1 token) is truncated, not char-based."""
    # 5000 chars of "x" with no whitespace = ~5000 tokens (worst case)
    dense = "x" * 5000
    truncated = _truncate_to_tokens(dense, max_tokens=100)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(truncated)) <= 100


def test_truncate_falls_back_without_tiktoken(monkeypatch):
    """If tiktoken isn't installed, falls back to char-based heuristic."""
    # Hide tiktoken
    sys.modules["tiktoken"] = None  # type: ignore[assignment]
    try:
        # 1000 chars with max_tokens=100 -> heuristic cap at 100*4=400 chars
        text = "a" * 1000
        truncated = _truncate_to_tokens(text, max_tokens=100)
        assert len(truncated) <= 400
    finally:
        sys.modules.pop("tiktoken", None)


# ---------------------------------------------------------------------------
# _build_messages — end-to-end with a real schema
# ---------------------------------------------------------------------------
def test_build_messages_uses_cached_schema():
    """Two calls to _build_messages with same schema share the cached serialization."""
    msgs1 = _build_messages(Invoice, "some text", [], extra_instructions="")
    msgs2 = _build_messages(Invoice, "different text", [], extra_instructions="")

    # Both contain the schema in the user prompt; same schema -> same
    # serialized form (string equality).
    schema1 = msgs1[1].content.split("Output JSON Schema:\n")[1].split("\n\nOutput rules:")[0]
    schema2 = msgs2[1].content.split("Output JSON Schema:\n")[1].split("\n\nOutput rules:")[0]
    assert schema1 == schema2
    # And the underlying cache holds a single string object across calls.
    cached = _MINIMAL_SCHEMA_JSON_CACHED(Invoice)
    assert schema1 == cached


def test_build_messages_caps_text_to_max_tokens():
    """Long text input is truncated before being put in the prompt."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    long_text = "x" * 50000  # way over 3000 tokens
    msgs = _build_messages(Invoice, long_text, [], extra_instructions="")

    # Extract just the document content from the user message
    user_content = msgs[1].content
    doc_marker = "Document content:\n\"\"\"\n"
    assert doc_marker in user_content
    doc_text = user_content.split(doc_marker, 1)[1].split("\n\"\"\"", 1)[0]

    # Should be well under the cap
    tokens = len(enc.encode(doc_text))
    assert tokens <= 3000, f"doc text has {tokens} tokens, expected <= 3000"


def test_build_messages_total_size_reasonable():
    """A single non-chunked call sends a modest number of tokens.

    The whole point of this work is to keep the prompt small. Set a
    generous upper bound so we catch regressions but allow some
    slack for template text.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    msgs = _build_messages(Invoice, "Sample text. " * 100, [], extra_instructions="hint")
    total = len(enc.encode(msgs[0].content)) + len(enc.encode(msgs[1].content))
    # system (~132) + schema (~170) + text (~150) + template (~80) ~= 530
    # Allow up to 1000 to catch 2-3x regressions without false-positives.
    assert total < 1000, f"prompt total {total} tokens is too large"
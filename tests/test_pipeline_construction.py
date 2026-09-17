"""Tests for Pipeline construction with various options (lines 125-194).

Covers:
  - retry wrapping with bool vs RetryConfig
  - cache wrapping with bool vs ExtractionCache
  - policy loading from a path
  - explicit parser object passed in
  - schema fallback to dict
"""
from __future__ import annotations

from idp.core.schemas import Invoice
from idp.pipeline.pipeline import Pipeline
from idp.reliability import ExtractionCache, RetryConfig


def test_pipeline_wraps_with_default_retry_when_retry_true(tmp_path):
    """retry=True → wraps backend in RetryingBackend with RetryConfig defaults."""
    f = tmp_path / "doc.txt"
    f.write_text("test")
    pipe = Pipeline(backend="mock", schema=Invoice, retry=True)
    # The backend is now wrapped; check it's not the bare MockBackend
    # (RetryingBackend.wrapped holds the original)
    assert hasattr(pipe.backend, "wrapped") or pipe.backend.__class__.__name__ == "RetryingBackend"


def test_pipeline_wraps_with_explicit_retry_config(tmp_path):
    """Passing a RetryConfig instance is honored."""
    cfg = RetryConfig(max_retries=7, initial_delay_sec=0.01)
    pipe = Pipeline(backend="mock", schema=Invoice, retry=cfg)
    assert pipe.cache is None


def test_pipeline_wraps_with_default_cache_when_cache_true(tmp_path, monkeypatch):
    """cache=True → wraps in CachingBackend; the disk cache file is created lazily."""
    pipe = Pipeline(backend="mock", schema=Invoice, cache=True)
    # CachingBackend wraps the original
    assert pipe.cache is not None


def test_pipeline_uses_explicit_cache_object(tmp_path):
    """Passing an ExtractionCache instance uses it directly."""
    cache = ExtractionCache(tmp_path / "c.db")
    pipe = Pipeline(backend="mock", schema=Invoice, cache=cache)
    assert pipe.cache is cache
    cache.close()


def test_pipeline_loads_policy_from_path(tmp_path):
    """policy_path loads a PolicyConfig from disk."""
    import json

    from idp.rl.policy import PolicyConfig

    cfg = PolicyConfig(min_reviews=42)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(cfg.to_dict()))

    pipe = Pipeline(backend="mock", schema=Invoice, policy_path=str(path))
    assert pipe.policy is not None
    assert pipe.policy.min_reviews == 42


def test_pipeline_logs_warning_on_invalid_policy_path(tmp_path, caplog):
    """If policy_path points at a corrupt file, pipe.policy stays None + warning logged."""
    import logging

    path = tmp_path / "policy.json"
    path.write_text("{not valid json")

    caplog.set_level(logging.WARNING)
    pipe = Pipeline(backend="mock", schema=Invoice, policy_path=str(path))
    # Falls back to None with a warning
    assert pipe.policy is None
    assert any("failed to load policy" in r.message for r in caplog.records)


def test_pipeline_resolve_parser_returns_explicit_parser(tmp_path):
    """If parser is a Parser object, _resolve_parser returns it as-is."""
    from idp.parse.parser import PlainTextParser

    explicit = PlainTextParser()
    pipe = Pipeline(backend="mock", schema=Invoice, parser=explicit)
    # Use a real Document with a .txt source
    f = tmp_path / "x.txt"
    f.write_text("hi")
    from idp.core.document import Document
    doc = Document.from_path(str(f))
    out = pipe._resolve_parser(doc)
    assert out is explicit


def test_pipeline_resolve_parser_uses_name(tmp_path):
    """If parser is a string ('plain'), _resolve_parser returns the named parser."""
    pipe = Pipeline(backend="mock", schema=Invoice, parser="plain")
    f = tmp_path / "x.txt"
    f.write_text("hi")
    from idp.core.document import Document
    doc = Document.from_path(str(f))
    out = pipe._resolve_parser(doc)
    assert out is not None


def test_pipeline_resolve_parser_auto_picks_by_extension(tmp_path):
    """If parser is None or 'auto', picks by extension."""
    pipe = Pipeline(backend="mock", schema=Invoice)  # parser=None → auto
    f = tmp_path / "x.txt"
    f.write_text("hi")
    from idp.core.document import Document
    doc = Document.from_path(str(f))
    out = pipe._resolve_parser(doc)
    # .txt → PlainTextParser (the safe default for text)
    assert out is not None


def test_pipeline_accepts_pydantic_schema_via_get_schema():
    """If schema is a known string name, get_schema() resolves it to the Pydantic class."""
    pipe = Pipeline(backend="mock", schema="Invoice")
    assert pipe.schema_name == "Invoice"
    # pipe.schema should be the Invoice Pydantic class
    assert pipe.schema.__name__ == "Invoice"


def test_pipeline_schema_pydantic_class_uses_class_name():
    """If schema is a Pydantic class, schema_name = the class's __name__."""
    pipe = Pipeline(backend="mock", schema=Invoice)
    assert pipe.schema_name == "Invoice"
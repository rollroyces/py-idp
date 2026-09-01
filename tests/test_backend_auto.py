"""Tests for get_backend('auto') fallback behaviour.

Covers:
  - No API keys  -> mock (NEW: was ollama, which fails without a server)
  - OPENAI_API_KEY only  -> openai
  - ANTHROPIC_API_KEY only  -> anthropic
  - OLLAMA_HOST  -> ollama
  - IDP_BACKEND explicit  -> wins
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all LLM-related env vars."""
    for var in (
        "IDP_BACKEND", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "OLLAMA_HOST", "IDP_OLLAMA",
    ):
        monkeypatch.delenv(var, raising=False)
    # Re-import the module so module-level os.environ reads happen fresh
    sys.modules.pop("idp.llm.backend", None)
    import idp.llm.backend as b
    importlib.reload(b)
    yield b


def test_auto_with_no_keys_falls_back_to_mock(clean_env):
    backend = clean_env.get_backend("auto")
    # MockBackend subclasses Backend; name attribute is 'mock'
    assert backend.name == "mock"


def test_auto_with_openai_key_picks_openai(monkeypatch, clean_env):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = clean_env.get_backend("auto")
    # The dispatcher resolves "openai" to OpenAICompatBackend (name='openai-compat')
    assert backend.name == "openai-compat"


def test_auto_with_anthropic_key_picks_anthropic(monkeypatch, clean_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # anthropic extra is not installed in dev env; ImportError is the
    # documented behaviour. Verify that's what happens (better than silently
    # picking the wrong backend).
    with pytest.raises(ImportError, match=r"anthropic"):
        clean_env.get_backend("auto")


def test_auto_with_ollama_host_picks_ollama(monkeypatch, clean_env):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    backend = clean_env.get_backend("auto")
    # Ollama also dispatches through OpenAICompatBackend
    assert backend.name == "openai-compat"


def test_explicit_idp_backend_overrides_autodetect(monkeypatch, clean_env):
    monkeypatch.setenv("IDP_BACKEND", "mock-random")
    backend = clean_env.get_backend("auto")
    # MockBackend's `name` attribute is always "mock" — discriminator is the mode
    from idp.llm.backend import MockBackend
    assert isinstance(backend, MockBackend)
    assert backend.mode == "random"


def test_anthropic_wins_over_openai_when_both_set(monkeypatch, clean_env):
    """If both keys are set, anthropic has priority."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with pytest.raises(ImportError, match=r"anthropic"):
        clean_env.get_backend("auto")
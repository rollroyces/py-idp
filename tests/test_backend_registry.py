"""Tests for the register_backend() decorator + registry.

Covers:
  - register_backend adds to _REGISTRY; cls.name is set
  - get_backend("openai") returns OpenAICompatBackend (alias resolves)
  - get_backend("ollama") / "vllm" / "lm-studio" / "compat" -> OpenAICompatBackend
  - get_backend("mock-random") / "mock-ideal" / "mock-omits" -> MockBackend with right mode
  - get_backend("unknown") raises ValueError listing registered names
  - get_backend("auto") with no env vars returns MockBackend
  - get_backend("auto") with ANTHROPIC_API_KEY returns AnthropicBackend (or ImportError)
  - get_backend("china:qwen") returns OpenAICompatBackend pointing at DashScope
  - get_backend("slowmock") without IDP_ENABLE_SLOWMOCK=1 raises ValueError
  - get_backend("nanonets") without IDP_ENABLE_NANONETS=1 raises ValueError
  - list_backends() returns sorted list and is non-empty
  - Registering a new backend via decorator + get_backend("newname") works
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all LLM-related env vars; reload backend module so the
    module-level reads re-execute.

    We DO NOT reload idp.llm.china — its captured OpenAICompatBackend
    reference would diverge from the freshly loaded class. Instead we
    test the alias / mock paths directly and skip china-specific asserts
    (covered by tests/test_china_providers.py).
    """
    for var in (
        "IDP_BACKEND", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "OLLAMA_HOST", "IDP_OLLAMA", "IDP_ENABLE_SLOWMOCK",
        "IDP_ENABLE_NANONETS", "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # Reload only the backend module — china keeps its original
    # OpenAICompatBackend reference, which still resolves correctly
    # because reload() updates the same module object's attributes.
    sys.modules.pop("idp.llm.backend", None)
    import idp.llm.backend as b
    importlib.reload(b)
    yield b


def test_register_backend_stores_in_registry_and_sets_name(clean_env):
    b = clean_env
    initial = len(b._REGISTRY)

    @b.register_backend("__test_dummy__")
    class DummyBackend(b.Backend):
        def complete(self, req):
            return "dummy"

    assert "__test_dummy__" in b._REGISTRY
    assert b._REGISTRY["__test_dummy__"] is DummyBackend
    assert DummyBackend.name == "__test_dummy__"
    # Not removed from registry (other tests may rely on order). Just
    # verify our backend can be looked up.
    inst = b.get_backend("__test_dummy__")
    assert isinstance(inst, DummyBackend)
    assert inst.complete(None) == "dummy"  # type: ignore[arg-type]
    assert len(b._REGISTRY) == initial + 1


def test_alias_openai_returns_openai_compat(clean_env):
    b = clean_env
    backend = b.get_backend("openai")
    assert isinstance(backend, b.OpenAICompatBackend)
    assert backend.name == "openai-compat"


def test_alias_ollama_returns_openai_compat(clean_env):
    b = clean_env
    backend = b.get_backend("ollama")
    assert isinstance(backend, b.OpenAICompatBackend)


def test_alias_vllm_returns_openai_compat(clean_env):
    b = clean_env
    backend = b.get_backend("vllm")
    assert isinstance(backend, b.OpenAICompatBackend)


def test_alias_lm_studio_returns_openai_compat(clean_env):
    b = clean_env
    backend = b.get_backend("lm-studio")
    assert isinstance(backend, b.OpenAICompatBackend)


def test_alias_compat_returns_openai_compat(clean_env):
    b = clean_env
    backend = b.get_backend("compat")
    assert isinstance(backend, b.OpenAICompatBackend)


def test_alias_mock_random_returns_mock_random_mode(clean_env):
    b = clean_env
    backend = b.get_backend("mock-random")
    assert isinstance(backend, b.MockBackend)
    assert backend.mode == "random"


def test_alias_mock_ideal_returns_mock_ideal_mode(clean_env):
    b = clean_env
    backend = b.get_backend("mock-ideal")
    assert isinstance(backend, b.MockBackend)
    assert backend.mode == "ideal"


def test_alias_mock_omits_returns_mock_omits_mode(clean_env):
    b = clean_env
    backend = b.get_backend("mock-omits")
    assert isinstance(backend, b.MockBackend)
    assert backend.mode == "omits"


def test_mock_alias_does_not_override_explicit_mode(clean_env):
    """If the caller passes mode= explicitly, alias shouldn't override."""
    b = clean_env
    backend = b.get_backend("mock-random", mode="ideal")
    assert backend.mode == "ideal"


def test_unknown_backend_raises_with_registered_names(clean_env):
    b = clean_env
    with pytest.raises(ValueError) as exc_info:
        b.get_backend("totally-unknown-backend-xyz")
    msg = str(exc_info.value)
    assert "totally-unknown-backend-xyz" in msg
    # Lists registered names (both canonical and aliases)
    assert "mock" in msg
    assert "openai-compat" in msg
    assert "anthropic" in msg


def test_auto_with_no_keys_returns_mock(clean_env):
    b = clean_env
    backend = b.get_backend("auto")
    assert isinstance(backend, b.MockBackend)
    assert backend.mode == "ideal"


def test_auto_with_anthropic_key_returns_anthropic(clean_env, monkeypatch):
    b = clean_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # anthropic extra may not be installed in dev env; ImportError is OK
    # here (same behaviour as the pre-refactor factory).
    try:
        backend = b.get_backend("auto")
        assert isinstance(backend, b.AnthropicBackend)
    except ImportError:
        pytest.skip("anthropic extra not installed")


def test_china_qwen_returns_openai_compat_with_dashscope(clean_env, monkeypatch):
    """china:qwen should resolve to an OpenAICompatBackend pointing at
    DashScope. We duck-type instead of isinstance() because the china
    module captures the OpenAICompatBackend reference at import time.
    Full coverage of get_china_backend lives in tests/test_china_providers.py.
    """
    b = clean_env
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    backend = b.get_backend("china:qwen")
    # Duck-type: must expose the same attributes as OpenAICompatBackend
    assert hasattr(backend, "base_url")
    assert hasattr(backend, "model")
    assert "dashscope" in backend.base_url.lower()
    # And it must respond to .complete (signature matches Backend ABC)
    assert callable(backend.complete)


def test_slowmock_without_env_var_raises(clean_env):
    b = clean_env
    with pytest.raises(ValueError, match="slowmock"):
        b.get_backend("slowmock")


def test_nanonets_without_env_var_raises(clean_env):
    b = clean_env
    with pytest.raises(ValueError, match="nanonets"):
        b.get_backend("nanonets")


def test_list_backends_returns_sorted_nonempty(clean_env):
    b = clean_env
    result = b.list_backends()
    assert isinstance(result, list)
    assert len(result) >= 2  # at least mock + openai-compat (+anthropic)
    assert result == sorted(result)
    assert "mock" in result
    assert "openai-compat" in result


def test_registry_aliases_map_is_preserved(clean_env):
    """The aliases dict is exposed and contains the legacy entries."""
    b = clean_env
    assert "openai" in b._REGISTRY_ALIASES
    assert b._REGISTRY_ALIASES["openai"] == "openai-compat"
    assert "ollama" in b._REGISTRY_ALIASES
    assert "vllm" in b._REGISTRY_ALIASES
    assert "lm-studio" in b._REGISTRY_ALIASES
    assert "compat" in b._REGISTRY_ALIASES
    assert "mock-ideal" in b._REGISTRY_ALIASES
    assert "mock-random" in b._REGISTRY_ALIASES
    assert "mock-omits" in b._REGISTRY_ALIASES


def test_decorator_does_not_clobber_explicit_class_name(clean_env):
    """If the subclass already has a non-default name, the decorator
    must not overwrite it."""
    b = clean_env

    @b.register_backend("__explicit_name_test__")
    class Custom(b.Backend):
        name = "i-set-this-myself"  # should be preserved

        def complete(self, req):
            return "x"

    # The canonical registered name still works
    inst1 = b.get_backend("__explicit_name_test__")
    # ...but the class's name attribute reflects the developer's choice
    assert Custom.name == "i-set-this-myself"
    assert isinstance(inst1, Custom)

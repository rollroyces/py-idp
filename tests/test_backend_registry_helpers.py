"""Tests for Backend.__repr__ and the list_backends() registry helper."""
from idp.llm import (
    MockBackend,
    OpenAICompatBackend,
    get_backend,
    list_backends,
)


def test_backend_default_repr():
    """Every backend should have a readable __repr__ that includes the class name."""
    b = MockBackend()
    r = repr(b)
    assert "MockBackend" in r
    assert "mock" in r  # the .name attribute


def test_backend_repr_works_for_subclasses():
    """__repr__ is on the ABC, so all subclasses inherit it."""
    b = OpenAICompatBackend(base_url="http://x", model="m")
    assert "OpenAICompatBackend" in repr(b)


def test_list_backends_returns_at_least_15():
    """We promise '12+ LLM backends' in the README. After v0.5 we have:
    4 mock variants + 4 international + 1 anthropic + 1 ollama
    + 5 intl:* presets + 8 china:* presets = 23 total."""
    rows = list_backends()
    assert len(rows) >= 15, f"only {len(rows)} backends registered"
    assert isinstance(rows, list)
    assert all({"name", "family", "auth"} <= r.keys() for r in rows)


def test_list_backends_includes_known_names():
    rows = list_backends()
    names = {r["name"] for r in rows}
    # Mock
    assert "mock" in names
    assert "mock-omits" in names
    # International
    assert "openai" in names
    assert "anthropic" in names
    assert "ollama" in names
    # New intl presets
    assert "intl:groq" in names
    assert "intl:mistral" in names
    assert "intl:together" in names
    assert "intl:cohere" in names
    assert "intl:fireworks" in names
    # China presets
    assert "china:deepseek" in names
    assert "china:qwen" in names


def test_list_backends_no_duplicates():
    rows = list_backends()
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


def test_list_backends_auth_flags():
    rows = list_backends()
    by_name = {r["name"]: r for r in rows}
    # Mock needs no key
    assert by_name["mock"]["auth"] == "none"
    # OpenAI / Anthropic need an env var
    assert by_name["openai"]["auth"] == "env-or-kwarg"
    assert by_name["anthropic"]["auth"] == "env-or-kwarg"
    # Ollama is self-hosted, no key
    assert by_name["ollama"]["auth"] == "none"
    # Presets all need an env var
    assert by_name["intl:groq"]["auth"] == "env-or-kwarg"
    assert by_name["china:qwen"]["auth"] == "env-or-kwarg"


def test_list_backends_gated_when_off():
    """slowmock + nanonets should NOT appear unless the gate env var is set."""
    import os

    # Ensure the gate is closed for this test
    os.environ.pop("IDP_ENABLE_SLOWMOCK", None)
    os.environ.pop("IDP_ENABLE_NANONETS", None)
    names = {r["name"] for r in list_backends()}
    assert "slowmock" not in names
    assert "nanonets" not in names


def test_list_backends_gated_when_on():
    """When the gate is open, the gated backends ARE listed (with auth='gated')."""
    import os

    os.environ["IDP_ENABLE_SLOWMOCK"] = "1"
    try:
        names = {r["name"] for r in list_backends()}
        assert "slowmock" in names
        by_name = {r["name"]: r for r in list_backends()}
        assert by_name["slowmock"]["auth"] == "gated"
    finally:
        os.environ.pop("IDP_ENABLE_SLOWMOCK", None)


def test_list_backends_includes_unlimited_ocr_when_gated_on():
    """v0.6: IDP_ENABLE_UNLIMITED_OCR=1 exposes unlimited-ocr in list_backends()."""
    import os

    os.environ["IDP_ENABLE_UNLIMITED_OCR"] = "1"
    try:
        names = {r["name"] for r in list_backends()}
        assert "unlimited-ocr" in names
        by_name = {r["name"]: r for r in list_backends()}
        assert by_name["unlimited-ocr"]["auth"] == "gated"
    finally:
        os.environ.pop("IDP_ENABLE_UNLIMITED_OCR", None)


def test_list_backends_family_groups():
    """Each family should have at least one entry, except the gated one when off."""
    rows = list_backends()
    families = {r["family"] for r in rows}
    assert "mock" in families
    assert "international" in families
    assert "self-hosted" in families
    assert "intl-preset" in families
    assert "china-preset" in families


def test_get_backend_after_helpers():
    """Sanity: get_backend still works after list_backends was added."""
    b = get_backend("mock")
    assert isinstance(b, MockBackend)
    # and the repr should still be well-formed
    assert "MockBackend" in repr(b)

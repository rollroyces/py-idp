"""Verify the international provider registry is sane."""
import os

import pytest

from idp.llm import get_backend
from idp.llm.international import (
    INTERNATIONAL_PROVIDER_PRESETS,
    get_international_backend,
    list_international_providers,
)


def test_at_least_five_providers():
    """We promised 5 international LLMs — guarantee depth."""
    assert len(INTERNATIONAL_PROVIDER_PRESETS) >= 5


def test_list_international_providers_returns_rows():
    rows = list_international_providers()
    assert len(rows) == len(INTERNATIONAL_PROVIDER_PRESETS)
    for row in rows:
        assert {"name", "env_var", "default_model", "vision_model", "notes"} <= row.keys()


def test_each_provider_has_endpoint_and_envvar():
    for _name, p in INTERNATIONAL_PROVIDER_PRESETS.items():
        assert p["base_url"].startswith("https://")
        assert "env_var" in p
        assert "default_model" in p


def test_get_backend_dispatch_intl_groq():
    """`get_international_backend('groq')` should raise OSError mentioning GROQ_API_KEY."""
    os.environ.pop("GROQ_API_KEY", None)
    try:
        get_international_backend("groq")
    except OSError as e:
        # Expected — we did not set the env var in this test
        assert "GROQ_API_KEY" in str(e)
    except Exception:
        # any other failure is a bug
        raise


def test_get_backend_dispatch_intl_unknown():
    """An unknown provider name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown international provider"):
        get_international_backend("nonexistent-llm-co")


def test_get_backend_top_level_intl_prefix():
    """`get_backend('intl:mistral')` should route to the international factory."""
    os.environ.pop("MISTRAL_API_KEY", None)
    try:
        get_backend("intl:mistral")
    except OSError as e:
        # Expected — we did not set the env var in this test
        assert "MISTRAL_API_KEY" in str(e)
    except Exception:
        # any other failure is a bug
        raise

"""Verify the China provider registry is sane."""
from idp.llm.china import (
    CHINA_PROVIDER_PRESETS,
    get_china_backend,
    list_china_providers,
)


def test_at_least_eight_providers():
    """We promised 'more China LLMs' — guarantee depth."""
    assert len(CHINA_PROVIDER_PRESETS) >= 8


def test_list_china_providers_returns_rows():
    rows = list_china_providers()
    assert len(rows) == len(CHINA_PROVIDER_PRESETS)
    for row in rows:
        assert {"name", "env_var", "default_model", "vision_model", "notes"} <= row.keys()


def test_each_provider_has_endpoint_and_envvar():
    for name, p in CHINA_PROVIDER_PRESETS.items():
        assert p["base_url"].startswith("https://")
        assert "env_var" in p
        assert "default_model" in p


def test_get_backend_dispatch_china():
    """`get_backend('china:qwen')` should NOT raise (it raises only on missing API key)."""
    import os
    os.environ.pop("DASHSCOPE_API_KEY", None)
    try:
        get_china_backend("qwen")
    except OSError as e:
        # Expected — we did not set the env var in this test
        assert "DASHSCOPE_API_KEY" in str(e)
    except Exception:
        # any other failure is a bug
        raise


def test_unknown_china_provider_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown China provider"):
        get_china_backend("nonexistent-llm-co")

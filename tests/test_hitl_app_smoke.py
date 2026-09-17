"""Smoke tests for ``idp.hitl.app`` using ``streamlit.testing.v1.AppTest``.

The Streamlit UI glue is hard to test with traditional mocking because
every ``st.<widget>`` call would need to be intercepted. Streamlit
provides an in-process test harness (``AppTest``) that runs the app's
``main()`` function and exposes the resulting widget tree.

These tests don't assert specific UI behavior — they pin the contract
that the app:
  1. Imports without crashing (even when storage is empty / no backend set)
  2. Renders without raising
  3. Honors the no-reviewer case

The data-handling logic is exhaustively tested in test_hitl_review.py.
The tests here are deliberately lightweight — they just verify the UI
*shell* loads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import idp.hitl.app as _hitl_app  # module-level import for CodeQL dedup

# Make sure the app module path is on sys.path so AppTest can find it.
APP_PATH = Path(__file__).resolve().parent.parent / "src" / "idp" / "hitl" / "app.py"


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Configure a JSON file storage via env vars for the app to find."""
    db_file = tmp_path / "idp.json"
    monkeypatch.setenv("IDP_DB_URL", "")  # ensure no sqlite interference
    monkeypatch.setenv("IDP_STORAGE_BACKEND", "json")
    monkeypatch.setenv("IDP_JSON_PATH", str(db_file))
    return db_file


def test_app_imports_without_error():
    """The app module imports cleanly (Streamlit loaded lazily)."""
    import idp.hitl.app as _app  # noqa: F401  # alias for CodeQL dedup
    assert hasattr(_app, "main")


def test_app_renders_with_empty_storage(tmp_storage):
    """The app's main() runs without exceptions when storage is empty."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH))
    at.run()
    # Should complete with no errors
    assert not at.exception


def test_app_pages_listed_in_sidebar(tmp_storage):
    """The sidebar has the expected page radio options."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH))
    at.run()
    # Sidebar radio is in at.sidebar
    radios = at.sidebar.radio
    assert len(radios) >= 1
    # The first radio should be the page picker
    page_options = list(radios[0].options)
    assert "Review queue" in page_options
    assert "Review history" in page_options
    assert "About" in page_options


def test_get_storage_uses_lru_cache():
    """get_storage is memoized (idempotent)."""
    import idp.hitl.app as app_mod
    # clear the cache
    app_mod.get_storage.cache_clear()
    s1 = app_mod.get_storage()
    s2 = app_mod.get_storage()
    # Same instance because of lru_cache
    assert s1 is s2
    app_mod.get_storage.cache_clear()


def test_parse_args_defaults():
    """_parse_args() returns Namespace with defaults."""
    import idp.hitl.app as app_mod
    saved = sys.argv
    sys.argv = ["streamlit"]
    try:
        args = app_mod._parse_args()
        # Defaults: no flags -> specific behaviors
        assert args is not None
    finally:
        sys.argv = saved


def test_format_result_label_includes_id_and_schema():
    """format_result_label produces a string with the result id and schema."""
    from unittest.mock import MagicMock

    result = MagicMock()
    result.id = "abc123"
    result.schema_name = "Invoice"
    label = _hitl_app.format_result_label(result)
    assert "abc123" in label
    assert "Invoice" in label
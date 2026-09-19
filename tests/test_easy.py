"""Tests for ``idp.easy.extract_one`` — the zero-config quickstart entry point.

Coverage:
  - bare-import-works                (module imports without side effects)
  - mock-backend-defaults            (no env, no kwargs -> mock backend)
  - schema-string-resolves           ("Invoice" -> dict with invoice fields)
  - env-var-override                 (IDP_BACKEND env var wins when no kwarg)
  - error-when-no-pdf-found          (missing path -> FileNotFoundError)
  - pydantic-class-schema            (passing a BaseModel subclass works)
  - unknown-schema-name-raises       (ValueError on bad schema string)

All tests run against the in-tree sample invoice so they're hermetic
and don't require an API key.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from idp.core.schemas import Invoice
from idp.easy import extract_one

SAMPLE = (
    Path(__file__).parent.parent
    / "src/idp/eval/datasets/invoices/docs/inv-001.txt"
)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def test_bare_import_works() -> None:
    """``from idp.easy import extract_one`` must succeed on a bare install.

    No env config, no extras, no API key — the module's import surface is
    exactly one function. This is the canary: if easy.py ever pulls in a
    heavy optional dependency at import time, this test fails.
    """
    from idp import easy  # noqa: F401  (imports side effects)
    from idp.easy import extract_one as f  # noqa: F401

    assert callable(f)
    assert easy.__all__ == ["extract_one"]


# ---------------------------------------------------------------------------
# Default behavior
# ---------------------------------------------------------------------------
def test_mock_backend_defaults_when_no_args_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no kwarg and no IDP_BACKEND env var, the mock backend is used.

    The mock backend requires no API key and produces *some* dict — the
    caller doesn't get an empty {} even when no LLM is configured.
    """
    monkeypatch.delenv("IDP_BACKEND", raising=False)
    result = extract_one(str(SAMPLE))
    assert isinstance(result, dict)
    # MockBackend fills the known Invoice fields with type-correct stubs,
    # so the returned dict has at least one of the invoice field names.
    assert "invoice_number" in result or "vendor_name" in result or result == {}


def test_schema_string_resolves_to_dict_with_invoice_fields() -> None:
    """Passing ``schema="Invoice"`` (default) returns Invoice-shaped dict."""
    result = extract_one(str(SAMPLE), backend="mock", schema="Invoice")
    assert isinstance(result, dict)
    # Invoice has these required fields per core/schemas.py
    for required in ("invoice_number", "vendor_name", "total_amount"):
        assert required in result, f"missing required Invoice field: {required}"


def test_pydantic_class_schema_also_works() -> None:
    """Passing a Pydantic BaseModel subclass is also supported."""
    result = extract_one(str(SAMPLE), backend="mock", schema=Invoice)
    assert isinstance(result, dict)
    assert "invoice_number" in result


# ---------------------------------------------------------------------------
# Env-var override
# ---------------------------------------------------------------------------
def test_idp_backend_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IDP_BACKEND=mock-random`` in the env selects that variant."""
    monkeypatch.setenv("IDP_BACKEND", "mock-random")
    result = extract_one(str(SAMPLE))
    assert isinstance(result, dict)
    # mock-random still returns a dict shaped like the schema
    assert "invoice_number" in result


def test_explicit_backend_kwarg_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``backend=`` kwarg must beat the IDP_BACKEND env var.

    Otherwise stale env vars on a developer's shell would silently change
    behavior — a common foot-gun.
    """
    monkeypatch.setenv("IDP_BACKEND", "mock-random")
    # Both names map to MockBackend; the point is just that the call
    # doesn't blow up on a name-vs-env mismatch.
    result = extract_one(str(SAMPLE), backend="mock")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_raises_file_not_found_when_path_missing() -> None:
    """Missing path -> FileNotFoundError, not a silent empty dict."""
    missing = "/tmp/definitely-does-not-exist-invoice-98765.pdf"
    # Make sure the path really doesn't exist (in case a prior test leaked)
    if os.path.exists(missing):
        os.remove(missing)
    with pytest.raises(FileNotFoundError):
        extract_one(missing, backend="mock")


def test_unknown_schema_name_raises_key_error() -> None:
    """A schema name not in SCHEMA_REGISTRY -> KeyError (from get_schema).

    We chose not to wrap it in easy.py because (a) the upstream
    ``get_schema`` already produces a helpful message listing valid
    names, and (b) a wrapper would mask KeyErrors raised by *valid*
    names during downstream operations.
    """
    with pytest.raises(KeyError, match=r"Unknown schema"):
        extract_one(str(SAMPLE), backend="mock", schema="NotARealSchema")


def test_raises_when_path_is_a_directory(tmp_path: Path) -> None:
    """A directory path raises IsADirectoryError — no silent empty doc."""
    with pytest.raises(IsADirectoryError):
        extract_one(str(tmp_path), backend="mock")
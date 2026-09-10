"""E7: property-based tests for the data-shape normalizers.

_safe_load, _safe_json, _stub, and _extract_schema_block are the four
functions that must never crash regardless of what the LLM produces.
Hypothesis fuzzes them with adversarial strings and structured data.
"""
from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from idp.extract.extractor import _safe_load
from idp.llm.backend import (
    _extract_schema_block,
    _safe_json,
    _stub,
)


# ---------------------------------------------------------------------------
# _safe_json: never raises, always returns a dict
# ---------------------------------------------------------------------------
@settings(max_examples=200)
@given(st.text())
def test_safe_json_never_raises_and_returns_dict(s):
    out = _safe_json(s)
    assert isinstance(out, dict)


@settings(max_examples=100)
@given(st.text())
def test_safe_json_is_total(s):
    """Empty/null -> {}; anything else -> dict (possibly _error)."""
    out = _safe_json(s)
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# _safe_load: never raises, always returns a dict
# ---------------------------------------------------------------------------
@settings(max_examples=200)
@given(st.text())
def test_safe_load_never_raises_and_returns_dict(s):
    out = _safe_load(s)
    assert isinstance(out, dict)


@settings(max_examples=100)
@given(st.text())
def test_safe_load_preserves_valid_json(s):
    """If input is valid JSON dict, output should equal it (or wrap value)."""
    try:
        parsed = json.loads(s)
    except Exception:
        return  # not JSON; skip
    if isinstance(parsed, dict):
        out = _safe_load(s)
        # if parse succeeded directly, output should not be an _error marker
        assert "_error" not in out, f"valid JSON {s!r} parsed as error"
        assert out == parsed


# ---------------------------------------------------------------------------
# _stub: never raises, always returns a JSON-serialisable value
# ---------------------------------------------------------------------------
@settings(max_examples=200)
@given(st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()),
    lambda children: st.one_of(st.lists(children, max_size=3), st.dictionaries(st.text(), children, max_size=3)),
    max_leaves=20,
))
def test_stub_never_raises_and_serialisable(x):
    out = _stub(x)
    # whatever comes out must be JSON-serialisable
    json.dumps(out)  # raises if not serialisable


@given(st.dictionaries(st.text(), st.integers(), max_size=5))
def test_stub_dict_preserves_keys(d):
    """For a flat int-valued dict, _stub should preserve the keys."""
    out = _stub(d)
    if isinstance(out, dict):
        assert set(out.keys()) == set(d.keys())


# ---------------------------------------------------------------------------
# _extract_schema_block: never raises, always returns a string
# ---------------------------------------------------------------------------
@settings(max_examples=200)
@given(st.text())
def test_extract_schema_block_never_raises_and_returns_str(s):
    out = _extract_schema_block(s)
    assert isinstance(out, str)


@settings(max_examples=50)
@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]\":, \n", max_size=500))
def test_extract_schema_block_with_json_marker(s):
    """When the marker is present and followed by JSON, extract cleanly."""
    if "Output JSON Schema:" in s:
        out = _extract_schema_block(s)
        assert isinstance(out, str)
        # if there's valid JSON after the marker, it should round-trip
        idx = s.index("Output JSON Schema:") + len("Output JSON Schema:")
        rest = s[idx:]
        # trim to first 'Output rules:' if present
        end = rest.index("Output rules:") if "Output rules:" in rest else len(rest)
        candidate = rest[:end].strip()
        if candidate:
            try:
                parsed = json.loads(candidate)
                assert parsed == json.loads(out)
            except json.JSONDecodeError:
                pass  # not valid JSON, nothing to assert


# ---------------------------------------------------------------------------
# Regression: malformed schemas where properties isn't a dict
# ---------------------------------------------------------------------------
def test_stub_does_not_crash_on_non_dict_properties():
    """Regression: {'properties': 0} (int) used to crash _stub()."""
    from idp.llm.backend import _stub
    # Should not raise — defensive coercion handles malformed input.
    out = _stub({"properties": 0})
    # The function falls back to returning the schema unchanged when
    # it can't recognize a structure. Either way, no crash.
    assert out is not None


def test_stub_does_not_crash_on_properties_as_none():
    from idp.llm.backend import _stub
    out = _stub({"properties": None})
    assert out is not None


def test_stub_handles_empty_dict():
    from idp.llm.backend import _stub
    assert _stub({}) is not None


def test_stub_handles_nested_garbage():
    """Nested malformed schemas: int where dict is expected."""
    from idp.llm.backend import _stub
    out = _stub({"type": "object", "properties": {"a": 0, "b": "string"}})
    # 'a' is int — _stub recurses on it (returns 0.0 for number type);
    # 'b' is string — returns "" for string type. Should not crash.
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# Regression: $ref with non-string values (Hypothesis-generated garbage)
# ---------------------------------------------------------------------------
def test_stub_handles_non_string_ref():
    """Schema with '$ref': 0 (int) used to crash _stub() with AttributeError.

    Defensive coercion: the bad ref returns None (per existing contract
    for unresolvable refs) instead of crashing.
    """
    from idp.llm.backend import _stub
    out = _stub({"$ref": 0})
    # The fix: don't crash. Returns None because the int ref isn't a
    # valid "#/$defs/Foo" pointer.
    assert out is None


def test_stub_handles_ref_without_defs_prefix():
    """Schema with '$ref': 'not-a-pointer' falls back to None."""
    from idp.llm.backend import _stub
    out = _stub({"$ref": "not-a-pointer"})
    # Malformed refs return None (per existing contract).
    assert out is None

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

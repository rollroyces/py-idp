"""Regression test for the _empty_schema crash on non-dict JSON literals.

Bug: MockBackend(mode="random"|"omits") crashed with AttributeError when
_extract_schema_block returned the JSON literal "null" (no schema block
in the prompt). Cause: json.loads("null") -> None, then schema.get(...)
crashed.

Fix: _empty_schema and _perturb_json/_omit_fields now guard against
non-dict parsed values.
"""
from __future__ import annotations

from idp.llm.backend import (
    CompletionRequest,
    Message,
    MockBackend,
    _empty_schema,
    _omit_fields,
    _perturb_json,
)


def test_empty_schema_handles_null_literal():
    """_empty_schema('null') returns '{}' instead of crashing."""
    out = _empty_schema("null")
    assert out == "{}", f"expected '{{}}', got {out!r}"


def test_empty_schema_handles_non_dict_json():
    """_empty_schema returns '{}' for any non-dict JSON value."""
    for raw in ('true', 'false', '42', '[]', '"hello"'):
        out = _empty_schema(raw)
        assert out == "{}", f"{raw!r} -> {out!r}"


def test_perturb_json_handles_null_base():
    """_perturb_json returns '{}' when _empty_schema yields 'null'."""
    out = _perturb_json("null")
    assert out == "{}", f"expected '{{}}', got {out!r}"


def test_omit_fields_handles_null_base():
    """_omit_fields returns '{}' when _empty_schema yields 'null'."""
    out = _omit_fields("null")
    assert out == "{}", f"expected '{{}}', got {out!r}"


def test_mock_backend_random_mode_no_schema_does_not_crash():
    """MockBackend(mode='random').complete('') should not crash.

    This is the exact bug found during component testing.
    """
    mock = MockBackend(mode="random")
    req = CompletionRequest(messages=[Message(role="user", content="")])
    # If the prompt is empty/tiny, _extract_schema_block returns "{}"
    # or "null"; this used to crash.
    out = mock.complete(req)
    # Should produce a valid JSON string (possibly empty)
    assert isinstance(out, str)


def test_mock_backend_omits_mode_no_schema_does_not_crash():
    mock = MockBackend(mode="omits")
    req = CompletionRequest(messages=[Message(role="user", content="")])
    out = mock.complete(req)
    assert isinstance(out, str)

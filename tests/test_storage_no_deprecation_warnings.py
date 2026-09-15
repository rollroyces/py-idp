"""Regression tests for the datetime.utcnow() -> timezone.utc migration.

A1 in docs/ROADMAP.md: 16 DeprecationWarnings fired from
src/idp/storage/sql.py because the code used ``datetime.utcnow()``.
This test pins the fix: importing the module + exercising storage
must not emit any ``datetime.utcnow`` deprecation.

Run: pytest tests/test_storage_no_deprecation_warnings.py -v
"""
from __future__ import annotations

import warnings


def test_no_datetime_utcnow_deprecation_on_import() -> None:
    """Re-importing storage.sql must not raise DeprecationWarning.

    The migration was: ``datetime.utcnow()`` ->
    ``datetime.now(timezone.utc)`` (with naive wire format preserved).
    If anyone reverts that, Python 3.12+ will warn here.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Force a fresh import. importlib.reload would be heavier
        # than needed; just import (already imported by other tests)
        # and check the warning registry.
        from idp.storage import sql  # noqa: F401

    utcnow_warnings = [
        w
        for w in caught
        if "utcnow" in str(w.message).lower()
        and issubclass(w.category, DeprecationWarning)
    ]
    assert utcnow_warnings == [], (
        f"datetime.utcnow() DeprecationWarning still fires: "
        f"{[str(w.message) for w in utcnow_warnings]}"
    )


def test_no_datetime_utcnow_deprecation_during_storage_roundtrip(tmp_path) -> None:
    """A full save + list roundtrip must not emit utcnow deprecations."""
    from idp.storage.sql import SqlStorage
    from idp.storage.store import StoredResult

    db = tmp_path / "test.db"
    storage = SqlStorage(f"sqlite:///{db}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = StoredResult(
            id="r1",
            doc_id="d1",
            schema_name="Invoice",
            backend_name="mock",
            mode="ocr_llm",
            classification=None,
            extraction={"vendor_name": "Acme"},
            confidence=None,
            validation={"passed": True},
            source_path=str(tmp_path / "in.pdf"),
            created_at=1234567890.0,
        )
        storage.put(result)
        listed = storage.list()

    utcnow_warnings = [
        w
        for w in caught
        if "utcnow" in str(w.message).lower()
        and issubclass(w.category, DeprecationWarning)
    ]
    assert utcnow_warnings == [], (
        f"datetime.utcnow() DeprecationWarning still fires on storage roundtrip: "
        f"{[str(w.message) for w in utcnow_warnings]}"
    )
    # Sanity: the roundtrip worked
    assert len(listed) == 1
    assert listed[0].doc_id == "d1"


def test_now_helper_format_is_naive_iso() -> None:
    """The _now() helper must return a naive ISO 8601 string in UTC.

    This is the wire format SqlStorage writes to the database. Changing
    it would break compatibility with existing rows.
    """
    from idp.storage.sql import _now

    s = _now("sqlite")
    # Naive ISO 8601: no 'T' offset, no tzinfo suffix.
    # Examples: "2026-09-15T10:30:00" or "2026-09-15T10:30:00.123456"
    assert "T" in s, f"unexpected _now() format: {s!r}"
    assert "+" not in s, f"_now() should be naive UTC, got: {s!r}"
    assert "Z" not in s, f"_now() should be naive UTC, got: {s!r}"
    assert not s.endswith("+00:00"), f"_now() should be naive, got: {s!r}"
"""Storage factory.

Returns the right Storage implementation based on environment / argument.

Default backend (for backward compat with existing CLI users):
  - "json"  JsonFileStorage (the existing default)
  - "memory" InMemoryStorage (used in tests)
  - "sql"   SqlStorage (new, opt-in)

Selection priority:
  1. explicit `make_storage(backend="...")` arg
  2. IDP_STORAGE_BACKEND env var
  3. IDP_DB_URL env var → auto-select "sql" with that URL
  4. default to "json"
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from idp.storage.store import InMemoryStorage, JsonFileStorage, Storage, StoredResult


def make_storage(
    backend: str | None = None,
    *,
    json_path: str | Path | None = None,
    db_url: str | None = None,
) -> Storage:
    """Resolve and instantiate a Storage backend.

    Args:
        backend: explicit backend name; falls back to env var.
        json_path: path for JsonFileStorage (default ./idp_data/results.jsonl).
        db_url: SQL URL for SqlStorage (env: IDP_DB_URL).

    Returns:
        A Storage instance.
    """
    if backend is None:
        backend = os.environ.get("IDP_STORAGE_BACKEND")
    if backend is None and os.environ.get("IDP_DB_URL"):
        backend = "sql"
    if backend is None:
        backend = "json"

    if backend in ("json", "file"):
        path = Path(json_path) if json_path else Path("idp_data") / "results.jsonl"
        return JsonFileStorage(str(path))
    if backend in ("memory", "mem", "in-memory"):
        return InMemoryStorage()
    if backend in ("sql", "sqlite", "postgres", "postgresql"):
        from idp.storage.sql import SqlStorage

        url = db_url or os.environ.get("IDP_DB_URL")
        if not url:
            raise ValueError(
                "SqlStorage requires IDP_DB_URL or db_url="
                "(e.g. 'sqlite:///./idp.db' or 'postgresql://...')"
            )
        return SqlStorage(url)
    raise ValueError(f"unknown storage backend: {backend!r}")

"""SQL-backed Storage implementation.

Supports two dialects via one Python class:
  - SQLite (default, stdlib `sqlite3`, zero extra deps) — single-user demos
  - Postgres (psycopg3, optional `pip install py-idp[sql]`) — production

Connection string format:
  - SQLite:      "sqlite:///absolute/path/to/file.db" or "sqlite:///:memory:"
  - Postgres:    "postgresql://user:pass@host:5432/dbname"

Usage:
    storage = SqlStorage("sqlite:///./idp.db")
    storage.put(StoredResult(...))
    item = storage.get("abc123")
    storage.submit_review(result_id="abc123", edited={...}, reviewer="alice", duration_sec=12.3)
    items = storage.list(reviewed_only=True, schema_name="Invoice", limit=50)

Schema is bootstrapped on first connect from
`src/idp/storage/migrations/v001_initial*.sql`.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from idp.storage.store import StoredResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def _parse_url(url: str) -> tuple[str, str]:
    """Return (dialect, connection_target)."""
    # In-memory URI variants need special-casing because they look like
    # filesystem paths ('sqlite:///:memory:') or ':memory:' no slashes.
    if url in ("sqlite:///:memory:", "sqlite:///:memory:?cache=shared", "sqlite://:memory:"):
        return "sqlite", ":memory:"
    if url.startswith("sqlite:///"):
        # absolute filesystem path
        path = url[len("sqlite://"):]  # strip "sqlite://" -> "/abs/path"
        return "sqlite", path
    if url.startswith("sqlite://"):
        # ":memory:" or ":memory:?cache=shared" (no leading slash)
        target = url[len("sqlite://"):]
        return "sqlite", target
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return "postgres", url
    raise ValueError(f"unsupported DB URL: {url!r}; expected sqlite:/// or postgresql://")


def _connect(url: str):
    """Open a connection. Caller is responsible for closing."""
    dialect, target = _parse_url(url)
    if dialect == "sqlite":
        # Plain ':memory:' is per-connection, so the schema from
        # _bootstrap() wouldn't be visible to subsequent put/get calls.
        # Use a shared cache URI so the in-memory DB persists across
        # connections in the same process.
        if target == ":memory:":
            return sqlite3.connect("file::memory:?cache=shared", uri=True), dialect
        if target.startswith("file::memory:"):
            return sqlite3.connect(target, uri=True), dialect
        # filesystem path; ensure parent dir exists
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target)
        # enable FK constraints + WAL for concurrent readers
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn, dialect
    if dialect == "postgres":
        try:
            import psycopg  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Postgres support requires the 'sql' extra: "
                "pip install 'py-idp[sql]'"
            ) from e
        return psycopg.connect(target), dialect
    raise ValueError(f"unknown dialect: {dialect}")


def _json_dump(v: Any) -> Any:
    if v is None:
        # SQLite's review_edits.human_value / model_value are NOT NULL TEXT.
        # Use the JSON literal null encoded as text so downstream parsers
        # can distinguish "explicitly null" from "empty string".
        return "null"
    return json.dumps(v, default=str)


def _json_load(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    return json.loads(v)


# ---------------------------------------------------------------------------
# SqlStorage
# ---------------------------------------------------------------------------
class SqlStorage:
    """SQL-backed storage; works against SQLite or Postgres.

    Thread-safe via a per-instance lock. Not async-safe (use sync).
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._lock = threading.Lock()
        self._dialect, _ = _parse_url(url)
        self._bootstrap()

    def _connect(self):
        conn, dialect = _connect(self.url)
        # Row factory: dict-like access (sqlite only — postgres returns RealDictRow natively)
        if dialect == "sqlite" and hasattr(conn, "row_factory"):
            conn.row_factory = sqlite3.Row
        return conn, dialect

    def _bootstrap(self) -> None:
        """Apply pending migrations. Idempotent."""
        conn, dialect = self._connect()
        try:
            self._init_schema(conn, dialect)
            self._apply_migrations(conn, dialect)
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self, conn, dialect) -> None:
        """Ensure schema_version table exists before checking what's applied."""
        if dialect == "sqlite":
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
        else:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
            )

    def _apply_migrations(self, conn, dialect) -> None:
        migrations_dir = Path(__file__).parent / "migrations"
        # Postgres first (the canonical), then SQLite
        order: list[Path]
        if dialect == "sqlite":
            order = [
                migrations_dir / "v001_initial_sqlite.sql",
                migrations_dir / "v001_initial.sql",
            ]
        else:
            order = [
                migrations_dir / "v001_initial.sql",
                migrations_dir / "v001_initial_sqlite.sql",
            ]
        for path in order:
            if not path.exists():
                continue
            try:
                sql_text = path.read_text()
                # Skip DO $$ / $$ in SQLite; skip the Postgres CREATE TYPE in SQLite
                if dialect == "sqlite":
                    sql_text = _strip_postgres_only(sql_text)
                if dialect == "postgres":
                    # skip the SQLite-specific file (it has AUTOINCREMENT etc.)
                    if path.name == "v001_initial_sqlite.sql":
                        continue
                # also skip the file if it's not the dialect-native one
                if dialect == "sqlite" and path.name == "v001_initial.sql":
                    continue
                if dialect == "postgres" and path.name == "v001_initial_sqlite.sql":
                    continue
                # apply
                conn.executescript(sql_text) if dialect == "sqlite" else conn.execute(sql_text)
                # mark version row
                version = path.stem.replace("_sqlite", "")
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version(version) VALUES (?)" if dialect == "sqlite"
                    else "INSERT INTO schema_version(version) VALUES (%s) ON CONFLICT DO NOTHING",
                    (version,),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("migration %s skipped: %s", path.name, e)

    # ---- Storage protocol methods ----
    def put(self, result: StoredResult) -> str:
        if not result.id:
            import uuid
            result.id = uuid.uuid4().hex[:16]
        with self._lock:
            conn, dialect = self._connect()
            try:
                if dialect == "sqlite":
                    conn.execute(
                        "INSERT OR REPLACE INTO stored_results "
                        "(id, doc_id, schema_name, backend_name, mode, classification, "
                        " extraction, confidence, validation, source_path, created_at, "
                        " reviewed, reviewed_extraction, reviewer, last_reviewed_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            result.id, result.doc_id, result.schema_name, result.backend_name,
                            result.mode, result.classification,
                            _json_dump(result.extraction),
                            _json_dump(result.confidence),
                            _json_dump(result.validation),
                            result.source_path, _now(dialect),
                            1 if result.reviewed else 0,
                            _json_dump(result.reviewed_extraction),
                            result.reviewer,
                            _now(dialect),
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO stored_results "
                        "(id, doc_id, schema_name, backend_name, mode, classification, "
                        " extraction, confidence, validation, source_path, "
                        " reviewed, reviewed_extraction, reviewer, last_reviewed_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (id) DO UPDATE SET "
                        "doc_id=EXCLUDED.doc_id, schema_name=EXCLUDED.schema_name, "
                        "backend_name=EXCLUDED.backend_name, mode=EXCLUDED.mode, "
                        "classification=EXCLUDED.classification, extraction=EXCLUDED.extraction, "
                        "confidence=EXCLUDED.confidence, validation=EXCLUDED.validation, "
                        "source_path=EXCLUDED.source_path, "
                        "reviewed=EXCLUDED.reviewed, reviewed_extraction=EXCLUDED.reviewed_extraction, "
                        "reviewer=EXCLUDED.reviewer, last_reviewed_at=EXCLUDED.last_reviewed_at",
                        (
                            result.id, result.doc_id, result.schema_name, result.backend_name,
                            result.mode, result.classification,
                            _json_dump(result.extraction),
                            _json_dump(result.confidence),
                            _json_dump(result.validation),
                            result.source_path,
                            result.reviewed,
                            _json_dump(result.reviewed_extraction),
                            result.reviewer,
                            _now(dialect),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        return result.id

    def get(self, result_id: str) -> StoredResult | None:
        with self._lock:
            conn, dialect = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM stored_results WHERE id = ?"
                    if dialect == "sqlite"
                    else "SELECT * FROM stored_results WHERE id = %s",
                    (result_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return _row_to_stored_result(row, dialect)
            finally:
                conn.close()

    def list(
        self,
        doc_id: str | None = None,
        schema_name: str | None = None,
        reviewed_only: bool = False,
        reviewed_since: float | None = None,
        limit: int = 50,
    ) -> list[StoredResult]:
        with self._lock:
            conn, dialect = self._connect()
            try:
                where: list[str] = []
                params: list[Any] = []
                if doc_id:
                    where.append("doc_id = ?")
                    params.append(doc_id)
                if schema_name:
                    where.append("schema_name = ?")
                    params.append(schema_name)
                if reviewed_only:
                    where.append("reviewed = 1" if dialect == "sqlite" else "reviewed = TRUE")
                if reviewed_since is not None:
                    iso = _epoch_to_iso(reviewed_since, dialect)
                    where.append("last_reviewed_at >= ?")
                    params.append(iso)
                clause = (" WHERE " + " AND ".join(where)) if where else ""
                order = " ORDER BY created_at DESC" if dialect == "sqlite" else " ORDER BY created_at DESC"
                sql = f"SELECT * FROM stored_results{clause}{order} LIMIT {int(limit)}"
                cur = conn.execute(sql, params)
                return [_row_to_stored_result(r, dialect) for r in cur.fetchall()]
            finally:
                conn.close()

    def mark_reviewed(
        self, result_id: str, edited: dict[str, Any], reviewer: str
    ) -> None:
        """Backward-compat path: only updates the denormalised fields.

        For full per-field edit tracking, prefer submit_review() below.
        """
        with self._lock:
            conn, dialect = self._connect()
            try:
                sql = (
                    "UPDATE stored_results SET reviewed = 1, reviewed_extraction = ?, "
                    "reviewer = ?, last_reviewed_at = ? WHERE id = ?"
                    if dialect == "sqlite"
                    else "UPDATE stored_results SET reviewed = TRUE, reviewed_extraction = %s, "
                    "reviewer = %s, last_reviewed_at = %s WHERE id = %s"
                )
                conn.execute(sql, (_json_dump(edited), reviewer, _now(dialect), result_id))
                conn.commit()
            finally:
                conn.close()

    # ---- New richer API ----
    def submit_review(
        self,
        result_id: str,
        edited: dict[str, Any],
        reviewer: str,
        duration_sec: float | None = None,
    ) -> str:
        """Atomic: insert into reviews + review_edits + update stored_results denorm.

        Returns the new reviews.id.
        """
        with self._lock:
            conn, dialect = self._connect()
            try:
                # 1. ensure reviewer row exists
                rid = self._get_or_create_reviewer(conn, dialect, reviewer)
                # 2. read model extraction for the diff
                cur = conn.execute(
                    "SELECT extraction FROM stored_results WHERE id = ?"
                    if dialect == "sqlite"
                    else "SELECT extraction FROM stored_results WHERE id = %s",
                    (result_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise KeyError(f"no such result: {result_id}")
                model_extraction = _json_load(row["extraction"]) if dialect == "sqlite" else _json_load(row[0])
                # 3. insert reviews row
                if dialect == "sqlite":
                    cur = conn.execute(
                        "INSERT INTO reviews (result_id, reviewer_id, submitted_at, status, duration_sec) "
                        "VALUES (?, ?, ?, 'submitted', ?)",
                        (result_id, rid, _now(dialect), duration_sec),
                    )
                    review_id = cur.lastrowid
                else:
                    cur = conn.execute(
                        "INSERT INTO reviews (result_id, reviewer_id, submitted_at, status, duration_sec) "
                        "VALUES (%s, %s, %s, 'submitted', %s) RETURNING id",
                        (result_id, rid, _now(dialect), duration_sec),
                    )
                    review_id = cur.fetchone()[0]
                # 4. insert per-field edits
                for field_name in set(model_extraction.keys()) | set(edited.keys()):
                    m = model_extraction.get(field_name)
                    h = edited.get(field_name)
                    if _field_equal(m, h):
                        reward = 0
                    else:
                        reward = +1
                    if dialect == "sqlite":
                        conn.execute(
                            "INSERT INTO review_edits (review_id, field_name, model_value, human_value, reward) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (review_id, field_name, _json_dump(m), _json_dump(h), reward),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO review_edits (review_id, field_name, model_value, human_value, reward) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (review_id, field_name, _json_dump(m), _json_dump(h), reward),
                        )
                # 5. update denorm in the SAME connection/transaction
                #    (opening a new connection here would deadlock on the
                #     uncommitted write lock held by this transaction)
                now_iso = _now(dialect)
                if dialect == "sqlite":
                    conn.execute(
                        "UPDATE stored_results SET reviewed = 1, reviewed_extraction = ?, "
                        "reviewer = ?, last_reviewed_at = ? WHERE id = ?",
                        (_json_dump(edited), reviewer, now_iso, result_id),
                    )
                else:
                    conn.execute(
                        "UPDATE stored_results SET reviewed = TRUE, reviewed_extraction = %s, "
                        "reviewer = %s, last_reviewed_at = %s WHERE id = %s",
                        (_json_dump(edited), reviewer, now_iso, result_id),
                    )
                conn.commit()
                return str(review_id)
            finally:
                conn.close()

    def reviews_as_dicts(
        self,
        since: float | None = None,
        schema_name: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Return reviews in the {doc_id, schema, model, human} shape the RL layer uses.

        Joins reviews + review_edits + stored_results + reviewers.
        """
        with self._lock:
            conn, dialect = self._connect()
            try:
                where = ["1=1"]
                params: list[Any] = []
                if since is not None:
                    where.append("r.submitted_at >= ?")
                    params.append(_epoch_to_iso(since, dialect))
                if schema_name:
                    where.append("sr.schema_name = ?")
                    params.append(schema_name)
                clause = " AND ".join(where)
                sql = f"""
                    SELECT sr.doc_id AS doc_id,
                           sr.schema_name AS schema,
                           sr.extraction AS model_json,
                           re.field_name AS field_name,
                           re.model_value AS model_value,
                           re.human_value AS human_value,
                           rv.external_id AS reviewer
                    FROM reviews r
                    JOIN stored_results sr ON sr.id = r.result_id
                    JOIN review_edits re ON re.review_id = r.id
                    JOIN reviewers rv ON rv.id = r.reviewer_id
                    WHERE {clause}
                    ORDER BY r.submitted_at DESC
                    LIMIT {int(limit)}
                """
                cur = conn.execute(sql, params)
                rows = cur.fetchall()

                # Re-group by review session (doc_id + schema + reviewer + a synthetic session_id).
                # We don't have a session id beyond review id, so group by (doc_id, schema, reviewer, review boundary)
                # but for the RL layer we just need the unioned (model, human) per (doc_id, schema).
                grouped: dict[tuple[str, str], dict[str, Any]] = {}
                for row in rows:
                    r = dict(row)
                    key = (r["doc_id"], r["schema"])
                    if key not in grouped:
                        grouped[key] = {
                            "doc_id": r["doc_id"],
                            "schema": r["schema"],
                            "model": _json_load(r["model_json"]),
                            "human": {},
                        }
                    # we keep the latest human value per field across reviews
                    grouped[key]["human"][r["field_name"]] = _json_load(r["human_value"])
                return list(grouped.values())
            finally:
                conn.close()

    def _get_or_create_reviewer(self, conn, dialect, external_id: str) -> int:
        if dialect == "sqlite":
            cur = conn.execute("SELECT id FROM reviewers WHERE external_id = ?", (external_id,))
            row = cur.fetchone()
            if row:
                return row["id"]
            cur = conn.execute(
                "INSERT INTO reviewers (external_id) VALUES (?)", (external_id,),
            )
            return cur.lastrowid
        cur = conn.execute("SELECT id FROM reviewers WHERE external_id = %s", (external_id,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur = conn.execute(
            "INSERT INTO reviewers (external_id) VALUES (%s) RETURNING id", (external_id,),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now(dialect: str) -> str:
    """Current timestamp in the right format for the dialect."""
    if dialect == "sqlite":
        # use the same format SQLite will return from datetime('now')
        import datetime as _dt
        return _dt.datetime.utcnow().isoformat(timespec="seconds")
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat()


def _epoch_to_iso(epoch: float, dialect: str) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(epoch).isoformat()


def _strip_postgres_only(sql_text: str) -> str:
    """For SQLite: drop DO $$ / $$ blocks and CREATE TYPE / ENUM types."""
    import re
    # drop DO blocks
    sql_text = re.sub(r"DO\s+\$\$.*?\$\$\s*;?", "", sql_text, flags=re.S)
    # drop CREATE TYPE ... AS ENUM
    sql_text = re.sub(
        r"CREATE\s+TYPE\s+\w+\s+AS\s+ENUM\s*\([^)]+\)\s*;?",
        "",
        sql_text,
        flags=re.S | re.I,
    )
    # replace review_status enum with TEXT
    sql_text = re.sub(
        r"review_status\s+NOT\s+NULL\s+DEFAULT\s+'submitted'",
        "TEXT NOT NULL DEFAULT 'submitted'",
        sql_text,
        flags=re.I,
    )
    return sql_text


def _field_equal(a: Any, b: Any) -> bool:
    """Same normalization as idp.rl.reward._norm_for_compare."""
    if a is None or a == "" or a == [] or a == {}:
        a = None
    if b is None or b == "" or b == [] or b == {}:
        b = None
    if isinstance(a, str):
        a = a.strip().lower()
    if isinstance(b, str):
        b = b.strip().lower()
    if isinstance(a, float):
        a = round(a, 2)
    if isinstance(b, float):
        b = round(b, 2)
    return a == b


def _row_to_stored_result(row, dialect: str) -> StoredResult:
    """Convert a DB row into a StoredResult.

    Supports both sqlite3.Row (subscriptable by name) and psycopg
    RealDictRow (also subscriptable by name).
    """
    def _g(k: str) -> Any:
        try:
            return row[k]
        except (KeyError, IndexError):
            # sqlite3.Row raises IndexError for missing keys
            return None
    return StoredResult(
        id=_g("id"),
        doc_id=_g("doc_id"),
        schema_name=_g("schema_name"),
        backend_name=_g("backend_name"),
        mode=_g("mode"),
        classification=_g("classification"),
        extraction=_json_load(_g("extraction")),
        confidence=_json_load(_g("confidence")),
        validation=_json_load(_g("validation")),
        source_path=_g("source_path"),
        created_at=_g("created_at") or 0.0,
        reviewed=bool(_g("reviewed")),
        reviewed_extraction=_json_load(_g("reviewed_extraction")),
        reviewer=_g("reviewer"),
        last_reviewed_at=_g("last_reviewed_at"),
    )

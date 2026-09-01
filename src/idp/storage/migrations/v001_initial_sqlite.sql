-- =====================================================================
-- py-idp storage backend — SQLite schema (single-file default)
-- File: src/idp/storage/migrations/v001_initial_sqlite.sql
-- =====================================================================
-- This schema is the canonical SQLite form. Postgres form is in
-- v001_initial.sql and differs in TIMESTAMP / JSON / SERIAL types.

CREATE TABLE IF NOT EXISTS reviewers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id   TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stored_results (
    id                 TEXT PRIMARY KEY,
    doc_id             TEXT NOT NULL,
    schema_name        TEXT NOT NULL,
    backend_name       TEXT NOT NULL,
    mode               TEXT,
    classification     TEXT,
    extraction         TEXT NOT NULL,            -- JSON
    confidence         TEXT,                     -- JSON
    validation         TEXT,                     -- JSON
    source_path        TEXT NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    reviewed           INTEGER NOT NULL DEFAULT 0,
    reviewed_extraction TEXT,                    -- JSON
    reviewer           TEXT,                     -- FK by external_id
    last_reviewed_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_stored_results_doc_id ON stored_results(doc_id);
CREATE INDEX IF NOT EXISTS idx_stored_results_schema ON stored_results(schema_name);
CREATE INDEX IF NOT EXISTS idx_stored_results_reviewed ON stored_results(reviewed);
CREATE INDEX IF NOT EXISTS idx_stored_results_created_at ON stored_results(created_at);

CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id     TEXT NOT NULL REFERENCES stored_results(id) ON DELETE CASCADE,
    reviewer_id   INTEGER NOT NULL REFERENCES reviewers(id),
    started_at    TEXT NOT NULL DEFAULT (datetime('now')),
    submitted_at  TEXT,
    status        TEXT NOT NULL DEFAULT 'submitted',  -- in_progress|submitted|rejected
    duration_sec  REAL
);

CREATE INDEX IF NOT EXISTS idx_reviews_result_id ON reviews(result_id);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id);

CREATE TABLE IF NOT EXISTS review_edits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id     INTEGER NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
    field_name    TEXT NOT NULL,
    model_value   TEXT NOT NULL,
    human_value   TEXT NOT NULL,
    reward        INTEGER NOT NULL             -- -1 | 0 | +1 (currently 0 | +1)
);

CREATE INDEX IF NOT EXISTS idx_review_edits_review ON review_edits(review_id);
CREATE INDEX IF NOT EXISTS idx_review_edits_field ON review_edits(field_name);

CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

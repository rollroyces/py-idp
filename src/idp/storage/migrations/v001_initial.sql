-- =====================================================================
-- py-idp storage backend — Postgres schema (production)
-- File: src/idp/storage/migrations/v001_initial.sql
-- =====================================================================
-- Same logical schema as SQLite, with Postgres-idiomatic types.
-- Use this when IDP_DB_URL starts with postgresql://.

DO $$ BEGIN
    CREATE TYPE review_status AS ENUM ('in_progress', 'submitted', 'rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS reviewers (
    id            BIGSERIAL PRIMARY KEY,
    external_id   TEXT NOT NULL UNIQUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stored_results (
    id                  TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    schema_name         TEXT NOT NULL,
    backend_name        TEXT NOT NULL,
    mode                TEXT,
    classification      TEXT,
    extraction          JSONB NOT NULL,
    confidence          JSONB,
    validation          JSONB,
    source_path         TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed            BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_extraction JSONB,
    reviewer            TEXT,
    last_reviewed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_stored_results_doc_id ON stored_results(doc_id);
CREATE INDEX IF NOT EXISTS idx_stored_results_schema ON stored_results(schema_name);
CREATE INDEX IF NOT EXISTS idx_stored_results_reviewed ON stored_results(reviewed);
CREATE INDEX IF NOT EXISTS idx_stored_results_created_at ON stored_results(created_at);

CREATE TABLE IF NOT EXISTS reviews (
    id            BIGSERIAL PRIMARY KEY,
    result_id     TEXT NOT NULL REFERENCES stored_results(id) ON DELETE CASCADE,
    reviewer_id   BIGINT NOT NULL REFERENCES reviewers(id),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at  TIMESTAMPTZ,
    status        review_status NOT NULL DEFAULT 'submitted',
    duration_sec  DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_reviews_result_id ON reviews(result_id);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id);

CREATE TABLE IF NOT EXISTS review_edits (
    id            BIGSERIAL PRIMARY KEY,
    review_id     BIGINT NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
    field_name    TEXT NOT NULL,
    model_value   TEXT NOT NULL,
    human_value   TEXT NOT NULL,
    reward        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_edits_review ON review_edits(review_id);
CREATE INDEX IF NOT EXISTS idx_review_edits_field ON review_edits(field_name);

CREATE TABLE IF NOT EXISTS schema_version (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

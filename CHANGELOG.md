# CHANGELOG

All notable changes to py-idp are documented here. Versions follow
[Semantic Versioning](https://semver.org/). The first number is bumped
on breaking API changes; the second on backward-compatible features;
the third on bugfixes.

## [0.2.0] — 2026-09-01 — Production hardening

### Added

* **`idp.errors`**: a typed exception hierarchy (`IDPError` base +
  `ConfigurationError`, `RateLimitedError`, `BackendUnavailableError`,
  `StorageError`, `DocumentParseError`, `SchemaValidationError`).
  Catch `IDPError` for framework-level handling without swallowing
  unrelated exceptions.
* **`idp.metrics`**: lightweight in-process metrics with thread-safe
  counters, gauges, histograms, and Prometheus text-format export
  at `GET /metrics`.
* **`idp.config`**: validated `Settings` loaded from env. All
  variables are `IDP_`-prefixed. Bad values raise `ConfigurationError`
  at startup so misconfigured deployments fail fast instead of at the
  first request.
* **`idp.ratelimit`**: per-key + global sliding-window rate limiter
  (1-minute windows). Raises `RateLimitedError` (→ HTTP 429) when
  exceeded.
* **`idp.api`**: production-hardened FastAPI app replacing the
  `examples/api.py` demo:
  - `/healthz` (liveness), `/readyz` (readiness), `/version`,
    `/metrics`, `/extract`
  - Auth via `X-API-Key` header **or** `Authorization: Bearer …`
  - `IDP_MAX_UPLOAD_BYTES` enforced via both `Content-Length` header
  and actual stream read (defends against missing/lying headers)
  - Structured request logging with metrics
  - CORS support via `IDP_CORS_ORIGINS` (comma-separated)
* **`idp._logging`**: `LOG_LEVEL` + `LOG_FORMAT=human|json`
  configuration helpers.

### Changed

* **Invoice** schema: `invoice_number`, `vendor_name`, `total_amount`
  are now **required**. Partial extraction still works for everything
  else.
* **Contract** schema: `title` is now **required**.
* **BankStatement** schema: `account_holder` is now **required**.
* **Document.from_path()** now raises `IsADirectoryError` for
  directories and `OSError` for non-regular files (FIFOs, sockets,
  devices) instead of silently returning an empty Document.
* Bumped to v0.2.0.

### Fixed

* **`_stub` crashed on malformed `enum` values** (dict/set/bool/empty
  list). Caught by Hypothesis property test. Now falls back to `None`
  on any non-list enum.

## [0.1.0] — 2026-08

Initial release. Six-stage pipeline (parse → classify → extract →
assess → validate → HITL), mock + OpenAI-compatible + Anthropic
backends, in-process queue, JSON + SQLite + in-memory storage,
human-in-the-loop Streamlit UI, reinforcement-learning policy from
review feedback.
# CHANGELOG

All notable changes to py-idp are documented here. Versions follow
[Semantic Versioning](https://semver.org/). The first number is bumped
on breaking API changes; the second on backward-compatible features;
the third on bugfixes.

## [0.3.6] — 2026-09-16 — `idp batch` CLI, batch module home

### Added

* **`idp batch` CLI** — new command for running the pipeline over
  many documents. Replaces the previous pattern of looping
  `idp run` or calling `process_batch` directly. Example:
  ```bash
  idp batch /mnt/inbox/ --backend ollama --schema Invoice \
            --output out.jsonl --report report.json --dlq dlq.jsonl
  ```
  Sources can be paths, directories (recursive scan for *.pdf /
  *.png / *.jpg / *.tiff / *.txt), or `@<file>` (path list).
  Produces per-doc JSONL, an aggregate summary (throughput,
  p50/p95 latency, error histogram), and a dead-letter queue for
  failed docs.

* **`idp.process_batch` and `idp.BatchItemResult` exposed at top
  level** — `from idp import process_batch` works directly. The
  helper was already there; now it's discoverable at the top
  of the package.

* **`idp.cli_sources` module** — small path-collection helper
  (`collect_paths`) used by the `idp batch` CLI, exposed for
  scripts that want the same path-discovery behavior without
  going through Typer.

### Changed

* **`idp.llm.nanonets_batch` → `idp.batch`** — the existing
  `process_batch` helper moved to its proper home. The old path
  is kept as a 12-line re-export shim
  (`from idp.llm.nanonets_batch import process_batch` still
  works). The old name was misleading — the helper works with
  any backend, not just Nanonets.

* **CI ruff strictness** — `.github/workflows/tests.yml` and the
  local ruff now agree on F841 (unused local) and I001
  (import-order). The batch CLI's file-handle code was
  restructured to satisfy F841 without a `del` workaround.

### Tests

* 654 → 665 (+11): `tests/test_cli_batch.py` covers
  `collect_paths` (file / dir / @file / dedup / errors) and the
  full CLI surface (`--help`, `--dry-run`, output files,
  missing-source exit codes).

## [0.3.5] — 2026-09-15 — Tier-A roadmap fixes, Mermaid CI guard

### Fixed

* **`datetime.utcnow()` deprecation** — `src/idp/storage/sql.py`
  migrated from `datetime.utcnow()` (deprecated in Python 3.12+)
  to `datetime.now(timezone.utc)`. The naive ISO 8601 wire format
  is preserved. Adds a regression test (`tests/test_storage_no_deprecation_warnings.py`)
  that pins the fix and would fail if anyone reverts it. **0
  `datetime.utcnow` deprecations from our code in CI.**

* **`count_corrections` reported all fields as "corrected" for
  unreviewed results** — `src/idp/hitl/review.py` (extracted from
  `src/idp/hitl/app.py` as part of A3) now returns 0 when
  `reviewed_extraction is None`. The previous behavior was
  `n_corrections == len(extraction)`, which is nonsense for an
  un-reviewed result. Caught by `test_count_corrections_zero_when_no_review`.

* **HITL data-handling logic now unit-testable** — `idp.hitl.app`
  was 0% covered because its logic was intertwined with the
  Streamlit UI. Extracted into `idp.hitl/review.py` (pure, no
  Streamlit import). The pure module is at **100% coverage** with
  18 tests. The Streamlit app now imports the helpers; this is
  the right factoring for a future FastAPI replacement UI.

### Added

* **SQLite + Postgres dialect parity tests** — `tests/test_sql_dialect.py`
  (23 tests) covers `_parse_url` for both schemes, `_connect`
  mocked for both, `_strip_postgres_only` for the syntax it actually
  handles (DO `$` blocks, CREATE TYPE, review_status ENUM), and
  schema bootstrap. Documents the current scope of
  `_strip_postgres_only` (it does NOT strip RETURNING / ON CONFLICT
  — those would require a real Postgres to verify safely).

* **CLI test coverage 83% → 97%** — `tests/test_cli.py` now exercises
  the actual `discover-schema` flow (not just `--help`), the
  `--storage` URL vs path heuristic in `rl-update`, and the
  `--synthetic without --fixtures` error in `rl-eval`. 6 new tests.

* **`@mermaid-js/mermaid-cli` validation in CI** — new workflow
  `.github/workflows/validate-mermaid.yml` runs `mmdc` on every
  Mermaid block in the 3 READMEs and fails the build on parse
  errors. This prevents the exact class of bug we shipped in
  PR #15 (unquoted braces in node labels were mis-parsed as
  diamond-shape openers, breaking the entire diagram). Renders
  are also uploaded as artifacts for visual review.

### Tests

* 654 tests pass (was 613), +41 new
* mypy clean across **61 source files** (was 60)
* ruff clean
* 0 `datetime.utcnow` deprecations from our code (verified with
  `pytest -W error::DeprecationWarning`)

## [0.3.4] — 2026-09-14 — Tier-1 test coverage, branch protection, CI bots

### Added

* **CLI test coverage** — `tests/test_cli.py` (17 tests) covers every
  command in `idp.pipeline.cli`: `run`, `schemas`, `providers`,
  `discover-schema`, `eval`, `rl-update`, `rl-eval`, `serve`. Brings
  the CLI from 0% to 83% line coverage. Tests use Typer's CliRunner
  so we exercise the actual CLI surface without subprocess overhead.
* **`InProcessQueue` test coverage** — `tests/test_queue.py` (12 tests)
  covers job submit/status/list, worker run/failure/recovery, and
  start/stop idempotency. Brings `queue/jobs.py` from 0% to 97%.
* **Eval harness test coverage** — `tests/test_eval_runner.py` (11
  tests) covers dataset loading, schema validation, multiple
  strategies, missing-doc skipping, and end-to-end runs against the
  bundled invoices dataset. Brings `eval/runner.py` from 0% to 96%.
* **Migration audit test coverage** — `tests/test_migrate_audit.py`
  (6 tests) covers empty DB, all-compatible rows, the headline
  case (v0.1-OK but v0.2-fails), mixed DBs, output-file writing, and
  schema-name skipping. Brings `migrate_audit.py` from 0% to 76%.

### Bots and branch protection

* **Branch protection on `main`** — required status checks enforced,
  linear history required, force-pushes disabled, branch deletion
  disabled, conversation resolution required. Enforced for admins too,
  so even direct pushes by the owner need a PR review.
* **CodeQL workflow** (`.github/workflows/codeql.yml`) — weekly
  security scan + per-PR scan using GitHub's default security-and-
  quality query pack. Runs only on Python (no JS/TS to analyze).
* **CI matrix expanded** — `tests.yml` now runs on Python 3.10 / 3.11
  / 3.12 / 3.13 / 3.14 (was 3.10–3.12). Matches `pyproject.toml`'s
  `python = ">=3.10"` declaration and includes the latest stable.
* **CI build-artifact guard** — `tests.yml` runs a shell check at the
  start of every CI job that fails the build if `site/` or other
  build dirs are tracked. Mirrors the local `.githooks/pre-commit`
  hook so PRs from forks (where the local hook isn't installed) get
  caught too.
* **Concurrency** — `.github/workflows/tests.yml` cancels in-progress
  runs for the same branch when a new commit is pushed. Saves CI
  minutes.

### Tests

* 46 new tests added across 4 new files. Coverage of the 4
  user-facing modules that previously had 0% is now between 76% and
  97%.

## [0.3.3] — 2026-09-14 — Templates wire to LLM, structured error envelope, README translations

### Added

* **Template body now reaches the LLM** — `Pipeline(template=...)`
  threads the template's Markdown body into every extraction call's
  user message. Field descriptions, worked examples, and
  common-mistakes notes from your `templates/*.md` files are prepended
  to the prompt before the JSON Schema. The body is capped at ~2000
  tokens to protect the 16k-context Nanonets-OCR2 budget, and is
  included in **every** chunk call (not just the first), so multi-chunk
  documents keep full template context throughout.
* **`Pipeline(template="name")` + `set_template_registry(registry)`**
  — resolve the template by name at run() time. Hot-reload works
  transparently when the registry was loaded with `watch=True`. Missing
  template name → logs a warning and proceeds without one.
* **Template provenance on `Document` and `PipelineResult`** — the
  matched template's name and version are recorded on
  `Document.template_name` / `Document.template_version` and surfaced
  in `PipelineResult` + `to_dict()` for audit. Answers "which template
  version produced this row?" without re-running.
* **Structured error envelope** — every `IDPError` subtype now carries
  a stable machine-readable `code` (`IDP-RATE-001`, `IDP-PARSE-001`,
  `IDP-SCHEMA-001`, `IDP-BACKEND-001`, `IDP-STORE-001`,
  `IDP-CONF-001`, `IDP-TMPL-001`, `IDP-TMPL-404`) and an
  `http_status`. The API's per-subtype exception handlers emit a
  uniform JSON shape: `{"error": {"code", "message", "type",
  "request_id", "details?"}}`. `RateLimitedError` adds a
  `Retry-After: 60` header.
* **`X-Request-ID` middleware** — every request gets a UUID (or
  echoes the caller's header). Surfaces in the response header AND
  every error envelope so client/server logs can be correlated.
* **API endpoints for templates**:
  * `GET /templates` — list summary (name, schema, version, patterns)
  * `GET /templates/{name}` — full template including body
  * `POST /extract` is now template-aware: when `schema_name` is
    omitted, the server picks a template by filename + MIME and uses
    its body as LLM context. The matched template is returned as
    `template_used` in the response.
* **Configuration** — `IDP_TEMPLATE_DIR` (default `./templates`),
  `IDP_TEMPLATE_RELOAD` (default `false`; dev mode only).
* **README translations** — Traditional Chinese (`README.zh-TW.md`)
  and Simplified Chinese (`README.zh-CN.md`) at full feature parity
  with the English version. Language switcher at the top of each
  README.

### Tests

* 12 new tests in `tests/test_template_to_llm.py` cover: template
  body reaching the LLM, template body in every chunk, hot-reload
  between run() calls, provenance on `Document` and `PipelineResult`,
  Pipeline API (`template=Template`, `template="name"` with
  registry), token cap on huge bodies, missing-template fallback.

## [0.3.2] — 2026-09-10 — Reliability, performance, observability, schema discovery

### Added

* **`Pipeline(retry=...)`** — `RetryingBackend` wraps any Backend with
  exponential-backoff retries on transient errors. Defaults: 4 attempts,
  1s → 2s → 4s → 8s with ±20% jitter, capped at 30s. Auth and bad-request
  errors fail fast. See `idp.reliability.RetryConfig`.
* **`Pipeline(cache=...)`** — `ExtractionCache` is a disk-backed SQLite
  cache keyed by sha256 of `(schema_name, backend_name, request)`. Same
  input → no LLM call. Default location `~/.cache/idp/extract.db`.
  `cache.stats()` returns entries, total_hits, per-schema breakdown.
  See `idp.reliability.ExtractionCache`.
* **`Pipeline(cache=True, retry=True)` composes** — cache is applied
  AFTER retry so cached hits skip the retry loop entirely.
* **`process_batch(checkpoint=...)`** — `CheckpointStore` is an
  append-only JSONL ledger of processed paths. Re-running with the same
  checkpoint path skips already-done paths (idempotent). The
  `record()` method holds an `flock` across write+flush+fsync for
  crash-safe single-line appends. `archive_at_start=True` rotates the
  ledger between runs (timestamped backup file). See `idp.checkpoint`.
* **`discover_schema` — hint grounding** — `DiscoveryResult.hint_grounding`
  is a new dict with `hint_tokens`, `schema_fields`, `grounded`
  (list of `(hint_token, schema_field, "exact"|"fuzzy")` triples),
  `ungrounded`, and `grounding_score` (0.0 = none of your hint tokens
  appear, 1.0 = perfect match). When the score is below 0.5, a
  warning is logged listing the ungrounded hint tokens. Doesn't fix
  wrong field names — makes the wrongness observable so you know to
  verify.
* **`@lru_cache`-cached schema serialization** for `_build_messages()`
  in `extract.py`. The JSON Schema dump for a Pydantic class is
  cached per-class; only fields the LLM needs to produce output
  (type/description/items/enum/format/required) are sent, dropping
  `$defs`, `additionalProperties`, and other verbose fields. Measured
  ~75-80% reduction on built-in schemas (Invoice 785 → 169 tokens).
  For a 20-field custom schema, ~21K tokens of schema overhead per
  13-chunk document is paid ONCE instead of N times.
* **Token-aware text truncation** (`_truncate_to_tokens`) replaces
  `text[:N_CHARS]` with `tiktoken`-based exact-token truncation.
  Avoids silent budget overshoot on dense text (numbers, currency,
  base64) where chars ~= tokens.
* **RAM optimizations**: lazy streamlit import in `idp.hitl.app`
  (saves ~20 MB of protobuf etc. when importing the package without
  running the UI), `JsonFileStorage` in-memory cache (saves ~40 MB of
  re-parsing on long batch reads), and intermediate-string cleanup
  in `extract()`.

### Changed

* `MockBackend` reports `is_multimodal=True` so it works in any
  multimodal pipeline without code changes.
* CLI defaults to Nanonets backend when `IDP_ENABLE_NANONETS=1`;
  explicit CLI error if not set.

### Fixed

* `_stub()` in `llm.backend` previously crashed on malformed JSON
  Schema inputs like `{'properties': 0}` (int instead of dict). Now
  defensive: falls back to returning the raw schema when no
  recognizable structure is found. Caught by Hypothesis property
  tests.
* `MockBackend` can now be passed directly to `Pipeline(schema=...)`
  via `Pipeline(backend="mock")` — was returning `None` for `is_multimodal`.

## [0.3.1] — 2026-09-10 — Discoverability & pre-existing-bug pass

### Added

* `CITATION.cff` — GitHub now shows a "Cite this repository" button and
  surfaces py-idp in academic dependency graphs.
* `docs/SECURITY.md` — supported-versions table, private reporting
  channel, scope, and a 72 h / 14 d SLA for critical issues. Surfaces
  the GitHub Security tab and enables private vulnerability reports.
* `.github/ISSUE_TEMPLATE/{bug_report,feature_request,question}.yml`
  — structured intake; first-time contributors get the right fields
  on the first try.

### Changed

* README badge row now links to live targets: CI to the Actions page,
  PyPI/downloads to the package stats, stars to the stargazers page,
  plus a `Cite this repository` badge for `CITATION.cff`.
* New README sections: **Security**, **Citing**, **Contributing** —
  each links to the corresponding repo file.
* pyproject keywords expanded to cover `rag`, `multimodal`,
  `information-extraction`, `intelligent-document-processing` so
  `pip search` and GitHub topic graph pick the project up.

### Fixed

* `tiktoken` is now a real `[project] dependency` (was promised by
  the README but never declared; fresh installs and CI failed with
  `ModuleNotFoundError` on `idp.chunker.TokenChunker`).
* New `[pdf-render]` extra (`pdf2image`, `Pillow`) for the multimodal
  path (`PdfPagesParser`, `NanonetsVLBackend`); declared in
  `pyproject.toml`, installed by CI.
* `.github/workflows/tests.yml` install line now uses
  `.[dev,eval,api,pdf-render]` so `test_security_adversarial.py`
  collects cleanly.
* Six test files no longer hard-code `/Users/hermes/py-idp/...` as
  the repo path; they derive `REPO` from `Path(__file__)`. Tests now
  pass on any machine, not just the author's.
* `PolicyCache.flush_now()` race with the background flusher: a
  caller that flushes synchronously and then reads the policy file
  can no longer observe a stray `.tmp` because the flusher re-checks
  `_dirty.is_set()` inside the lock and skips no-op writes.

## [0.3.0] — 2026-09-02 — Self-hosted VLM backend

### Added

* **`NanonetsVLBackend`** in `idp.llm.nanonets` — self-hosted,
  offline OCR + extraction via `nanonets/Nanonets-OCR2-3B` (Qwen2.5-VL
  fine-tune on Hugging Face). No API key, no cloud egress. ~7 GB
  weights cached at `~/.cache/huggingface/hub/`. Runs on Apple
  Silicon (MPS) or CUDA. Lazy model load on first `.complete()` call.
  Configurable `device`, `dtype`, `max_image_side`, `load_in_4bit`.

* **`[hf-vlm]` extra** in `pyproject.toml`:
  `torch`, `transformers>=4.45`, `accelerate`, `safetensors`,
  `Pillow`, `huggingface-hub`.

* **Gated by `IDP_ENABLE_NANONETS=1`** to avoid surprise ~7 GB
  downloads. `get_backend("nanonets")` raises a clear error if the
  gate is off or `[hf-vlm]` is not installed.

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
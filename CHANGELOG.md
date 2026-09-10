# CHANGELOG

All notable changes to py-idp are documented here. Versions follow
[Semantic Versioning](https://semver.org/). The first number is bumped
on breaking API changes; the second on backward-compatible features;
the third on bugfixes.

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
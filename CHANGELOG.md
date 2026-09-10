# CHANGELOG

All notable changes to py-idp are documented here. Versions follow
[Semantic Versioning](https://semver.org/). The first number is bumped
on breaking API changes; the second on backward-compatible features;
the third on bugfixes.

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
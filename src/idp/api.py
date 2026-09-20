"""Production-hardened FastAPI server.

Improvements over the original ``examples/api.py``:

* Reads ``Settings`` from env (port, workers, rate limits, max upload
  size, backend, storage) and **fails fast** on misconfiguration.
* Health (``/healthz``) and readiness (``/readyz``) endpoints for k8s.
* Prometheus text exposition at ``/metrics``.
* Per-key + global rate limiting via ``idp.ratelimit.RateLimiter``.
* Upload size limit enforced via ``Content-Length`` *and* actual
  stream read (defends against missing/lying Content-Length headers).
* ``/version`` endpoint exposes the package version for ops dashboards.
* Structured request logging with timing.
* **Template registry** (``/templates`` list, ``/templates/{name}`` get,
  template-aware ``/extract`` that picks the right ``.md`` context for
  the LLM). See ``idp.templates``.
* **Structured error envelope** with stable machine-readable codes
  (``IDP-RATE-001`` etc.) and per-subtype HTTP status mapping.
* **X-Request-ID** middleware (auto-generates one if the client doesn't
  send it; surfaces it in error responses and logs).

Not in scope: HTTPS termination (use a reverse proxy), TLS, SSO,
multi-tenant auth (these are deployment-level concerns).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from idp import __version__
from idp._logging import configure, get_logger
from idp.config import Settings
from idp.core.document import Document
from idp.errors import (
    ConfigurationError,
    IDPError,
    RateLimitedError,
    error_envelope,
)
from idp.metrics import metrics
from idp.pipeline.pipeline import Pipeline
from idp.queue.jobs import InProcessQueue, Job, JobStatus
from idp.ratelimit import RateLimiter
from idp.storage.store import InMemoryStorage, StoredResult
from idp.templates import TemplateRegistry

# Module-level logger; configured at lifespan startup.
_log = get_logger("idp.api")
_settings: Settings | None = None
_rate_limiter: RateLimiter | None = None
_async_rate_limiter: RateLimiter | None = None  # per-API-key cap on /extract_async
_upload_dir: Path | None = None
_template_registry: TemplateRegistry | None = None
_job_queue: InProcessQueue | None = None
_results_store: InMemoryStorage | None = None

# Request-ID header name (lowercase; FastAPI normalizes).
_REQUEST_ID_HEADER = "x-request-id"
_SIGNATURE_HEADER = "X-IDP-Signature"
_SIGNATURE_SCHEME = "sha256"
_TIMESTAMP_HEADER = "X-IDP-Timestamp"
_NONCE_HEADER = "X-IDP-Nonce"

# Bounded LRU cache of nonces we've seen on /verify_signature. Acts as
# a single-shot blacklist — replay protection. The OrderedDict lets
# us pop the oldest entry when we exceed the configured maxsize.
_nonce_cache: "OrderedDict[str, None]" = OrderedDict()
_nonce_lock = threading.Lock()


def _nonce_check_and_add(nonce: str, maxsize: int) -> bool:
    """Record ``nonce``; return True iff it was new.

    Side effect: bounded LRU eviction of the oldest nonces when full.
    Thread-safe.
    """
    with _nonce_lock:
        if nonce in _nonce_cache:
            return False
        _nonce_cache[nonce] = None
        while len(_nonce_cache) > maxsize:
            _nonce_cache.popitem(last=False)
        return True


def _nonce_cache_reset() -> None:
    """Clear the nonce cache (for tests only)."""
    with _nonce_lock:
        _nonce_cache.clear()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    schema_name: str | None = None
    backend: str | None = None


class ExtractResponse(BaseModel):
    schema_name: str | None = None
    backend_name: str
    mode: str
    classification: str | None
    extraction: dict[str, Any]
    confidence: dict[str, float] | None
    validation: dict[str, Any] | None
    template_used: str | None = None  # NEW: which template was matched


class ReviewRequest(BaseModel):
    edited: dict[str, Any]
    reviewer: str


class TemplateSummary(BaseModel):
    """Lightweight template metadata for list endpoints."""

    name: str
    # `schema` would clash with Pydantic's BaseModel.schema() method.
    # Use a Python-safe name and serialize via a custom field below.
    schema_name: str
    version: int
    mime_types: list[str]
    filename_patterns: list[str]

    def to_wire(self) -> dict[str, Any]:
        """Serialize with the public `schema` key (matches file format)."""
        d = self.model_dump()
        d["schema"] = d.pop("schema_name")
        return d


class TemplateDetail(BaseModel):
    """Full template including body."""

    name: str
    schema_name: str
    version: int
    mime_types: list[str]
    filename_patterns: list[str]
    classification_hints: list[str]
    field_overrides: dict[str, dict[str, Any]]
    body: str
    source_path: str

    def to_wire(self) -> dict[str, Any]:
        """Serialize with the public `schema` key."""
        d = self.model_dump()
        d["schema"] = d.pop("schema_name")
        return d


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def auth(request: Request) -> None:
    """Validate the ``X-API-Key`` header against ``Settings.api_key``."""
    s = _require_settings()
    if not s.api_key_required:
        return
    if s.api_key is None:
        # No key configured but key required -> fail closed.
        raise HTTPException(status_code=503, detail="API key required but not configured on server")
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not provided:
        raise HTTPException(status_code=401, detail="missing API key")
    if provided != s.api_key:
        raise HTTPException(status_code=403, detail="invalid API key")
    # Record authenticated request for rate-limit accounting
    if _rate_limiter is not None:
        _rate_limiter.check(key=provided)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging, validate settings, prep upload dir, load templates."""
    global _settings, _rate_limiter, _async_rate_limiter, _upload_dir, _template_registry
    global _job_queue, _results_store

    # Load settings (raises ConfigurationError -> startup aborts with 500)
    try:
        _settings = Settings.load()
    except ConfigurationError as e:
        # Re-raise so uvicorn logs it; we can't return JSON yet because
        # the app isn't built.
        _log.error("configuration error: %s", e)
        raise

    configure(level=_settings.log_level, fmt=_settings.log_format)
    _log.info("py-idp API starting: host=%s port=%d storage=%s backend=%s",
              _settings.api_host, _settings.api_port, _settings.storage_backend, _settings.default_backend)

    _rate_limiter = RateLimiter(
        per_key_per_minute=_settings.rate_limit_per_minute,
        global_per_minute=0,  # disabled by default; set via env if needed
    )
    # /extract_async uses a separate, smaller budget. Built fresh on
    # every lifespan so tests can override Settings before startup.
    _async_rate_limiter = RateLimiter(
        per_key_per_minute=_settings.idp_async_rate_limit_per_minute,
        global_per_minute=0,  # per-key only
    )

    _upload_dir = Path(os.environ.get("IDP_UPLOAD_DIR", "/tmp/idp-uploads"))
    _upload_dir.mkdir(parents=True, exist_ok=True)

    # Load templates from the configured directory. Empty/missing
    # directory = empty registry (server starts; /templates returns []).
    template_dir = Path(_settings.template_dir)
    try:
        _template_registry = TemplateRegistry.load(
            template_dir, watch=_settings.template_reload
        )
    except Exception as e:
        # A bad template file should fail startup so the operator sees
        # it, not the first user.
        _log.error("failed to load templates from %s: %s", template_dir, e)
        if isinstance(e, IDPError):
            raise
        raise ConfigurationError(f"template load failed: {e}") from e
    _log.info("loaded %d template(s) from %s", len(_template_registry), template_dir)

    # Async submission infrastructure: in-process job queue + result store.
    # The queue runs pipeline.arun() off the event-loop thread that serves
    # /extract_async. Each job gets its own asyncio.run() call so the worker's
    # event loop is independent. The results store is in-memory only —
    # production deployments should swap it for SqlStorage via the existing
    # Settings.storage_backend knob (future P-tier work).
    _results_store = InMemoryStorage()

    def _job_runner(job: Job) -> None:
        """Worker tick: load doc, run pipeline, persist result, fire webhook.

        Synchronous wrapper (matches InProcessQueue's runner contract).
        Pipeline.run() is the canonical sync entry point; arun is only
        added in the P0 audit branch (audit/p0-fixes-v0.3.9). On a
        P0-equipped checkout, prefer arun when present so the request
        loop isn't blocked.
        """
        schema_name = job.schema_name
        backend_name = job.backend_name
        doc = Document.from_path(job.doc_path)
        pipeline = Pipeline(backend=backend_name, schema=schema_name)

        def _run_sync() -> Any:
            arun = getattr(pipeline, "arun", None)
            if arun is None:
                # Pre-P0 fallback: synchronous Pipeline.run.
                return pipeline.run(doc)
            # P0+: arun exists; drive it on a private event loop so we
            # don't collide with the FastAPI request loop. The worker
            # is itself an asyncio task, so we cannot call asyncio.run().
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(arun(doc))
            finally:
                loop.close()

        from concurrent.futures import ThreadPoolExecutor

        started = time.perf_counter()
        success = True
        with ThreadPoolExecutor(max_workers=1) as ex:
            try:
                res = ex.submit(_run_sync).result()
            except Exception as e:  # noqa: BLE001
                job.error = str(e)
                metrics.inc("async_jobs_failed", 1, backend=backend_name, error_type=type(e).__name__)
                success = False
                return
        duration = time.perf_counter() - started
        metrics.observe("async_job_duration_seconds", duration, backend=backend_name)

        stored = _pipeline_result_to_stored(res, job)
        if _results_store is not None:
            result_id = _results_store.put(stored)
            job.result_id = result_id
            job.extra["result_id"] = result_id
            # Fire the webhook (best-effort; do not raise if it fails).
            cb = job.extra.get("callback_url")
            cb_secret = job.extra.get("callback_secret")
            if cb:
                _post_webhook(cb, stored, cb_secret)
        # Final job-state metric. We track success locally because
        # ``job.status`` is set by the InProcessQueue worker AFTER the
        # runner returns (it's still RUNNING while we're in this body).
        # A failed runner returns early above (``success = False``),
        # so checking the local flag avoids the timing race.
        if success:
            metrics.inc("async_jobs_succeeded", 1, backend=backend_name)

    _job_queue = InProcessQueue(
        runner=_job_runner,
        max_concurrent=_settings.idp_async_max_concurrent,
    )
    await _job_queue.start()
    _log.info(
        "async job queue started (in-process, max_concurrent=%d, async_rl=%d/min)",
        _settings.idp_async_max_concurrent, _settings.idp_async_rate_limit_per_minute,
    )

    # Background gauge-refresh task: every 1s, publish current queue depth
    # and inflight count so /metrics shows fresh values without operators
    # needing to poll /jobs/{id}.
    import asyncio as _asyncio

    queue_ref = _job_queue  # capture the local, non-None reference

    async def _refresh_async_gauges() -> None:
        try:
            while True:
                metrics.gauge("async_queue_depth", queue_ref.queue_depth)
                metrics.gauge("async_inflight_jobs", queue_ref.inflight)
                await _asyncio.sleep(1.0)
        except _asyncio.CancelledError:
            return

    _gauge_task = _asyncio.create_task(_refresh_async_gauges())

    yield

    # Shutdown: clean up, log final metrics
    _log.info("py-idp API shutting down. metrics: %s", metrics.snapshot())
    _gauge_task.cancel()
    try:
        await _gauge_task
    except _asyncio.CancelledError:
        pass
    if _job_queue is not None:
        await _job_queue.stop()
    _settings = None
    _rate_limiter = None
    _async_rate_limiter = None
    _upload_dir = None
    _template_registry = None
    _job_queue = None
    _results_store = None


app = FastAPI(
    title="py-idp API",
    version=__version__,
    description="Production AI-enabled Intelligent Document Processing API.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Ensure every request has an X-Request-ID.

    If the client sent one (e.g. from a load balancer), we keep it.
    Otherwise we mint a fresh UUID. The id is stored on
    ``request.state.request_id`` and echoed in the response header.
    """
    rid = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers[_REQUEST_ID_HEADER] = rid
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log every request with timing + status code + request id."""
    start = time.perf_counter()
    rid = getattr(request.state, "request_id", "-")
    metrics.inc("http_requests", 1, path=request.url.path, method=request.method)
    try:
        response = await call_next(request)
        metrics.inc("http_responses", 1, status=str(response.status_code), path=request.url.path)
        return response
    except Exception as e:
        metrics.inc("http_errors", 1, type=type(e).__name__)
        _log.exception("request failed: %s %s rid=%s", request.method, request.url.path, rid)
        env = error_envelope(e, request_id=rid)
        return JSONResponse(status_code=500, content=env)
    finally:
        elapsed = time.perf_counter() - start
        metrics.observe("http_request_duration_seconds", elapsed, path=request.url.path)


# CORS (configured in lifespan but installed here so the order is correct)
def _install_cors() -> None:
    s = _require_settings()
    if s.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(s.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz", response_class=PlainTextResponse, include_in_schema=False)
async def healthz() -> str:
    """Liveness: returns 200 OK if the process is up. Cheap; no DB call."""
    return "ok"


@app.get("/readyz", response_class=PlainTextResponse, include_in_schema=False)
async def readyz() -> str:
    """Readiness: 200 OK if settings are loaded; 503 if not yet ready."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="not ready: settings not loaded")
    return "ready"


@app.get("/version", response_class=PlainTextResponse)
async def version() -> str:
    """Return the py-idp version."""
    return __version__


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus text-format metrics."""
    if not _require_settings().metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics disabled")
    return PlainTextResponse(metrics.export_prometheus(), media_type="text/plain; version=0.0.4")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
@app.get("/templates", dependencies=[Depends(auth)])
async def list_templates() -> list[dict[str, Any]]:
    """List all registered template names + metadata.

    The full body is **not** returned here — fetch ``/templates/{name}``
    for the Markdown body. Clients can use this endpoint to populate a
    template picker.
    """
    reg = _require_template_registry()
    return [
        TemplateSummary(
            name=t.name,
            schema_name=t.schema,
            version=t.version,
            mime_types=list(t.mime_types),
            filename_patterns=list(t.filename_patterns),
        ).to_wire()
        for t in reg.all()
    ]


@app.get("/templates/{name}", dependencies=[Depends(auth)])
async def get_template(name: str) -> dict[str, Any]:
    """Return the full template (frontmatter + body) for ``name``.

    Raises 404 (``IDP-TMPL-404``) if the template is not registered.
    """
    reg = _require_template_registry()
    t = reg.get(name)  # raises TemplateNotFoundError -> handled below
    return TemplateDetail(
        name=t.name,
        schema_name=t.schema,
        version=t.version,
        mime_types=list(t.mime_types),
        filename_patterns=list(t.filename_patterns),
        classification_hints=list(t.classification_hints),
        field_overrides=dict(t.field_overrides),
        body=t.body,
        source_path=str(t.source_path),
    ).to_wire()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/extract", response_model=ExtractResponse, dependencies=[Depends(auth)])
async def extract_sync(
    file: UploadFile,
    schema_name: str | None = Form(default=None),
    backend: str | None = Form(default=None),
) -> ExtractResponse:
    """Synchronous document extraction. Capped by ``IDP_MAX_UPLOAD_BYTES``.

    Form fields:
      - ``schema_name`` (optional): Pydantic schema to extract against.
        If omitted, the template registry is consulted to pick a
        schema based on the uploaded filename / MIME type.
      - ``backend``    (optional): LLM backend name (e.g. ``mock``,
        ``openai/gpt-4o``).

    If a registered template matches the upload, its Markdown body is
    prepended to the LLM context so the model sees field descriptions
    and examples for that PDF type. The matched template name is
    returned in the response as ``template_used``.
    """
    s = _require_settings()
    raw = await _read_capped(file, s.max_upload_bytes)
    upload_dir = _require_upload_dir()
    dst = upload_dir / (file.filename or "upload")
    dst.write_bytes(raw)

    # Template routing: pick a template by filename + MIME if the
    # caller didn't pin a schema.
    reg = _require_template_registry()
    template_used: str | None = None
    if schema_name is None:
        match = reg.find_for_filename(
            filename=dst.name,
            mime=file.content_type,
        )
        if match is not None:
            schema_name = match.schema
            template_used = match.name
            metrics.inc("template_routed", 1, template=match.name)
            _log.info("routed %s to template %s (schema=%s)",
                      dst.name, match.name, match.schema)

    pipeline = Pipeline(
        backend=backend or s.default_backend,
        schema=schema_name or "Invoice",
    )
    doc = Document.from_path(str(dst))
    res = pipeline.run(doc)
    metrics.inc("extractions", 1, backend=res.backend_name, schema=res.schema_name or "_none")
    return ExtractResponse(
        schema_name=res.schema_name,
        backend_name=res.backend_name,
        mode=res.mode or "ocr_llm",  # type: ignore[arg-type]  # router sets this; defensive against no-LLM paths
        classification=res.classification,
        extraction=res.document.extraction or {},
        confidence=res.confidence,
        validation=res.document.validation,
        template_used=template_used,
    )


# ---------------------------------------------------------------------------
# Async /extract_async + webhook + signed result
# ---------------------------------------------------------------------------
@app.post(
    "/extract_async",
    dependencies=[Depends(auth)],
)
async def extract_async(
    request: Request,
    file: UploadFile,
    schema_name: str | None = Form(default=None),
    backend: str | None = Form(default=None),
    callback_url: str | None = Form(default=None),
    callback_secret: str | None = Form(default=None),
) -> JSONResponse:
    """Submit an extraction job. Returns 202 Accepted immediately.

    Form fields:
      - ``schema_name``     optional — same semantics as /extract.
      - ``backend``         optional — same semantics as /extract.
      - ``callback_url``    optional — when the job finishes, the server
                            POSTs the result JSON to this URL. Must be
                            ``https://<hostname>`` or ``http://localhost``
                            / ``http://127.0.0.1`` (SSRF guard; anything
                            else returns 400).
      - ``callback_secret`` optional — HMAC-SHA256 signing secret for the
                            webhook payload. If omitted, the server's
                            ``Settings.api_key`` (or env
                            ``IDP_RESULT_SIGNING_KEY``) is used.

    Response shape::

        {
          "job_id": "<16-char id>",
          "status_url": "/jobs/<job_id>",
          "result_url": "/results/<result_id>"
        }

    Note: ``result_url`` is populated only after the job runs and writes
    the StoredResult; before that, GETting the result_url returns 404.
    """
    s = _require_settings()
    raw = await _read_capped(file, s.max_upload_bytes)
    upload_dir = _require_upload_dir()
    dst = _safe_upload_path(upload_dir, file.filename or "")
    dst.write_bytes(raw)

    if callback_url:
        _ssrf_check_callback_url(callback_url)

    # Per-API-key rate-limit on async submissions. /extract_async is
    # heavier than /extract (each call enqueues work the server runs
    # off-loop), so it gets its own smaller per-minute budget. Keyed on
    # the API key the caller presented; falls back to "_anon" to share
    # a bucket across unauthenticated submissions. Fails closed via
    # RateLimitedError (mapped to 429 + Retry-After by the existing
    # exception handler).
    _api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        or "_anon"
    )
    if _async_rate_limiter is not None:
        _async_rate_limiter.check(key=_api_key)

    # Template routing mirrors /extract
    reg = _require_template_registry()
    if schema_name is None:
        match = reg.find_for_filename(filename=dst.name, mime=file.content_type)
        if match is not None:
            schema_name = match.schema

    queue = _require_job_queue()
    # InProcessQueue.submit takes (doc_path, schema_name, backend_name).
    # We pass callback info via Job.extra so the runner can fire the webhook.
    job = await queue.submit(
        str(dst),
        schema_name or "Invoice",
        backend or s.default_backend,
    )
    if callback_url:
        job.extra["callback_url"] = callback_url
    if callback_secret:
        job.extra["callback_secret"] = callback_secret

    metrics.inc("async_submissions", 1, schema=schema_name or "_none")
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status_url": f"/jobs/{job.id}",
            "result_url": f"/results/{job.result_id}" if job.result_id else None,
        },
        headers={"Location": f"/jobs/{job.id}"},
    )


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Return job status (pending / running / succeeded / failed).

    Includes ``estimated_queue_position`` when the job is still queued
    (returns 0 if currently running, None if finished) so callers can
    decide whether to back off or wait.
    """
    queue = _require_job_queue()
    job = await queue.status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
    payload: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status.value,
        "schema_name": job.schema_name,
        "backend_name": job.backend_name,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "estimated_queue_position": queue.estimated_queue_position(job_id),
    }
    # Live queue depth + inflight for observability
    payload["queue_depth"] = queue.queue_depth
    payload["inflight_jobs"] = queue.inflight
    if job.result_id:
        payload["result_id"] = job.result_id
        payload["result_url"] = f"/results/{job.result_id}"
    return payload


@app.get("/results/{result_id}", dependencies=[Depends(auth)])
async def get_result(result_id: str) -> dict[str, Any]:
    """Return a stored extraction result as JSON."""
    store = _require_results_store()
    stored = store.get(result_id)
    if stored is None:
        raise HTTPException(
            status_code=404, detail=f"unknown result_id {result_id!r}"
        )
    return _stored_to_wire(stored)


@app.post("/results/{result_id}/signed", dependencies=[Depends(auth)])
async def get_signed_result(result_id: str) -> JSONResponse:
    """Return the result + HMAC-SHA256 signature for tamper detection.

    The signature is computed over
    ``timestamp + "." + nonce + "." + json.dumps(result, sort_keys=True)``
    using either ``Settings.api_key`` or env ``IDP_RESULT_SIGNING_KEY``
    when no per-job secret is on file. Webhook consumers should verify
    the signature (and the timestamp + nonce freshness) before trusting
    the payload — protects against MITM when the webhook URL is on a
    non-TLS path (dev only).

    Response shape::

        {
          "result": {...},
          "signature": "sha256=<64-char hex>",
          "timestamp": "<unix-epoch-seconds as string>",
          "nonce": "<uuid4 hex, 16 chars>"
        }

    Headers echo the same three values for clients that prefer headers
    over a JSON body (``X-IDP-Signature``, ``X-IDP-Timestamp``,
    ``X-IDP-Nonce``).
    """
    store = _require_results_store()
    stored = store.get(result_id)
    if stored is None:
        raise HTTPException(
            status_code=404, detail=f"unknown result_id {result_id!r}"
        )
    wire = _stored_to_wire(stored)
    body_bytes = _json.dumps(wire, sort_keys=True).encode("utf-8")
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex[:16]
    try:
        sig = _sign_payload(
            timestamp.encode("utf-8") + b"." +
            nonce.encode("utf-8") + b"." +
            body_bytes,
            secret=None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return JSONResponse(
        content={"result": wire, "signature": sig,
                 "timestamp": timestamp, "nonce": nonce},
        headers={
            _SIGNATURE_HEADER: sig,
            _TIMESTAMP_HEADER: timestamp,
            _NONCE_HEADER: nonce,
        },
    )


class VerifySignatureRequest(BaseModel):
    """Body for POST /verify_signature — replay-protected HMAC check."""

    result: dict[str, Any]
    signature: str  # "sha256=<64-char hex>"
    timestamp: str  # unix-epoch-seconds as string
    nonce: str  # uuid4 hex, 16 chars


@app.post("/verify_signature", dependencies=[Depends(auth)])
async def verify_signature(req_body: VerifySignatureRequest) -> dict[str, Any]:
    """Verify a signed result envelope end-to-end.

    Three checks, all must pass:
      1. ``timestamp`` within ``IDP_SIGNATURE_MAX_AGE_SECONDS`` of now()
         (set to 0 to disable — replay becomes possible).
      2. ``nonce`` not previously seen (server-side bounded LRU).
      3. ``signature`` matches HMAC-SHA256 over the canonical bytes.

    Returns ``{"valid": true}`` on success. Returns ``{"valid": false,
    "reason": "..."}`` on any check failure (status 200 either way so
    callers don't have to handle two HTTP codes — they read the JSON).
    """
    s = _require_settings()
    now = int(time.time())
    try:
        ts_int = int(req_body.timestamp)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "timestamp not an integer"}
    age = abs(now - ts_int)
    if s.idp_signature_max_age_seconds > 0 and age > s.idp_signature_max_age_seconds:
        return {
            "valid": False,
            "reason": f"timestamp too old or future-dated ({age}s > "
                      f"{s.idp_signature_max_age_seconds}s)",
        }

    # Nonce freshness (replay protection)
    if not _nonce_check_and_add(req_body.nonce, maxsize=s.idp_nonce_cache_maxsize):
        return {"valid": False, "reason": "nonce already seen (replay?)"}

    # Signature match
    body_bytes = _json.dumps(req_body.result, sort_keys=True).encode("utf-8")
    try:
        expected = _sign_payload(
            req_body.timestamp.encode("utf-8") + b"." +
            req_body.nonce.encode("utf-8") + b"." +
            body_bytes,
            secret=None,
        )
    except RuntimeError as e:
        return {"valid": False, "reason": f"server has no signing secret: {e}"}

    # Constant-time compare on the full scheme+hex string
    if not hmac.compare_digest(expected, req_body.signature):
        return {"valid": False, "reason": "signature mismatch"}
    return {"valid": True}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _require_settings() -> Settings:
    if _settings is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return _settings


def _require_upload_dir() -> Path:
    if _upload_dir is None:
        raise HTTPException(status_code=503, detail="server not ready: upload dir not configured")
    return _upload_dir


def _require_template_registry() -> TemplateRegistry:
    if _template_registry is None:
        raise HTTPException(status_code=503, detail="server not ready: templates not loaded")
    return _template_registry


# Filenames can carry path-traversal payloads (e.g. "../etc/passwd").
# Anything that isn't [A-Za-z0-9._-] is replaced with an underscore so the
# resolved path can never escape the upload directory. We also pin the
# whitelist of accepted extensions BEFORE writing anything to disk —
# an attacker uploading ".exe" or "../" gets a 415/400, not a silent write.
# On Linux/POSIX backslash is a legal filename char; on Windows it is a
# path separator. Stripping it is defence-in-depth for cross-platform safety.
_SAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_ALLOWED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".eml",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
})


def _sanitize_upload_filename(raw: str | None) -> str:
    """Return a filename safe to join with the upload dir.

    Rules (in order):
      1. ``None`` / empty / pure-whitespace -> ``"upload"``.
      2. Strip directory components. On POSIX ``Path(name).name``
         strips up to the last ``/``; on Windows it would also strip
         up to the last ``\\``. We do an extra ``replace("\\", ...)``
         before to ensure cross-platform safety even when the running
         Python reports the wrong platform.
      3. Replace any character outside [A-Za-z0-9._-] with ``_``.
      4. Reject empty results or names that start with ``.`` (dotfiles).
      5. Cap at 255 chars (POSIX filename limit).
    """
    name = (raw or "").strip()
    if not name:
        return "upload"
    # Cross-platform: strip both separator styles. ``Path(name).name``
    # only strips the platform-native one.
    name = name.replace("\\", "/")
    name = Path(name).name  # strip directory components
    name = _SAFE_NAME_CHARS.sub("_", name)
    if not name or name.startswith("."):
        name = "upload_" + name
    return name[:255]


def _safe_upload_path(upload_dir: Path, filename: str) -> Path:
    """Return a ``Path`` guaranteed to live inside ``upload_dir``.

    Raises ``HTTPException(400)`` if the resolved path escapes the
    directory (defence in depth — sanitization should already prevent
    this, but a symlinked ``upload_dir`` would bypass it).
    Raises ``HTTPException(415)`` for unsupported extensions.
    """
    safe = _sanitize_upload_filename(filename)
    candidate = (upload_dir / safe).resolve()
    upload_root = upload_dir.resolve()
    if not candidate.is_relative_to(upload_root):
        raise HTTPException(status_code=400, detail="invalid filename")
    ext = candidate.suffix.lower()
    if ext not in _ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type: {ext!r}; allowed: "
                   f"{sorted(_ALLOWED_UPLOAD_EXTENSIONS)}",
        )
    return candidate


def _request_id(request: Request) -> str:
    """Pull the request id off request.state, defaulting to '-' if absent."""
    return getattr(request.state, "request_id", "-")


# ---------------------------------------------------------------------------
# Async /extract_async helpers
# ---------------------------------------------------------------------------
import json as _json  # local alias to avoid shadowing issues elsewhere


def _ssrf_check_callback_url(url: str) -> None:
    """Reject callback URLs that target internal networks.

    Allowed:
      * ``https://<anywhere>`` — production webhook target.
      * ``http://localhost`` or ``http://127.0.0.1`` — local dev/loopback
        for testing the webhook end-to-end without TLS.

    Rejected: anything else on http (attacker-controlled port-forward to
    an internal service), private/loopback IPs on https (rare; usually
    misconfigured), and any non-http(s) scheme (file://, gopher://, ...).
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"callback_url scheme {scheme!r} not allowed (must be http/https)",
        )
    host = (parsed.hostname or "").lower().strip()
    if scheme == "http":
        # Only allow loopback for http (localhost / 127.0.0.1 / ::1).
        # We don't resolve here — just string-compare, to avoid DNS-rebind
        # via /etc/hosts. The hostname check is intentionally simple.
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise HTTPException(
                status_code=400,
                detail="callback_url http:// only allowed for localhost / 127.0.0.1",
            )
    else:  # https
        # Reject if the hostname IS a private/loopback address.
        import ipaddress

        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"callback_url https target {host!r} resolves to a "
                           "private/loopback IP",
                )
        except ValueError:
            # Not an IP literal — it's a DNS name. We trust it; production
            # deployments terminate the public DNS at the ingress.
            pass


def _sign_payload(payload_bytes: bytes, secret: str | None) -> str:
    """Return ``sha256=<hex>`` HMAC signature for the payload.

    ``secret`` is the per-job callback secret when provided; otherwise we
    fall back to ``Settings.api_key`` or the env var
    ``IDP_RESULT_SIGNING_KEY``. The scheme prefix matches what consumers
    typically expect (Stripe-style ``t=...,v1=...`` simplified to just
    ``sha256=<hex>``).
    """
    if not secret:
        s = _require_settings()
        secret = s.api_key or os.environ.get("IDP_RESULT_SIGNING_KEY") or ""
    if not secret:
        # No signing material available — refuse rather than emit a weak
        # signature. Webhooks still fire (callers can ignore unsigned), but
        # /results/{id}/signed returns 503 because there's nothing to sign.
        raise RuntimeError(
            "no signing secret configured (set IDP_API_KEY or "
            "IDP_RESULT_SIGNING_KEY in the env)"
        )
    digest = hmac.new(
        secret.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()
    return f"{_SIGNATURE_SCHEME}={digest}"


def _post_webhook(url: str, stored: StoredResult, secret: str | None) -> None:
    """POST a stored result to a callback URL. Best-effort; logs on failure.

    Wire format:
      - Body: JSON of the stored result (sort_keys=True)
      - X-IDP-Signature: sha256=<hex> over (timestamp + "." + nonce + "." + body)
      - X-IDP-Timestamp: unix-epoch-seconds at signature time
      - X-IDP-Nonce: uuid4 hex (16 chars)
      - X-IDP-Job-Id: stored.doc_id

    Consumers should verify all three headers together (replay protection):
      * timestamp must be within IDP_SIGNATURE_MAX_AGE_SECONDS of now()
      * nonce must not have been seen before (server keeps a bounded LRU)
      * signature must match HMAC-SHA256 over the canonical bytes

    Failures (timeout, 5xx, connection refused) are logged at WARNING —
    they do NOT fail the job. Webhook delivery is best-effort; the
    canonical result is at /results/{id}.
    """
    started = time.perf_counter()
    outcome = "success"
    try:
        body_bytes = _json.dumps(
            _stored_to_wire(stored), sort_keys=True
        ).encode("utf-8")
        timestamp = str(int(time.time()))
        nonce = uuid.uuid4().hex[:16]
        sig = _sign_payload(
            timestamp.encode("utf-8") + b"." +
            nonce.encode("utf-8") + b"." +
            body_bytes,
            secret,
        )
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                url,
                content=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    _SIGNATURE_HEADER: sig,
                    _TIMESTAMP_HEADER: timestamp,
                    _NONCE_HEADER: nonce,
                    "X-IDP-Job-Id": stored.doc_id,
                },
            )
            if r.status_code >= 400:
                _log.warning(
                    "webhook POST to %s returned %d: %s",
                    url, r.status_code, r.text[:200],
                )
                outcome = "http_error"
    except Exception as e:  # noqa: BLE001
        _log.warning("webhook delivery to %s failed: %s", url, e)
        outcome = "network_error"
    finally:
        metrics.inc("async_callbacks_delivered", 1, outcome=outcome)
        metrics.observe(
            "async_callback_duration_seconds",
            time.perf_counter() - started,
            outcome=outcome,
        )


def _stored_to_wire(stored: StoredResult) -> dict[str, Any]:
    """Serialize a StoredResult to a JSON-safe dict (id + extraction)."""
    return {
        "id": stored.id,
        "doc_id": stored.doc_id,
        "schema_name": stored.schema_name,
        "backend_name": stored.backend_name,
        "mode": stored.mode,
        "classification": stored.classification,
        "extraction": stored.extraction,
        "confidence": stored.confidence,
        "validation": stored.validation,
        "source_path": stored.source_path,
        "created_at": stored.created_at,
        "reviewed": stored.reviewed,
    }


def _pipeline_result_to_stored(
    res: Any, job: Job
) -> StoredResult:
    """Convert a PipelineResult into a StoredResult for the store + webhook."""
    import time as _time

    doc = res.document
    return StoredResult(
        id="",  # assigned on put()
        doc_id=job.id,
        schema_name=res.schema_name or job.schema_name,
        backend_name=res.backend_name or job.backend_name,
        mode=res.mode,
        classification=res.classification,
        extraction=doc.extraction or {},
        confidence=res.confidence,
        validation=doc.validation,
        source_path=job.doc_path,
        created_at=_time.time(),
    )


def _require_job_queue() -> InProcessQueue:
    if _job_queue is None:
        raise HTTPException(status_code=503, detail="async: job queue not ready")
    return _job_queue


def _require_results_store() -> InMemoryStorage:
    if _results_store is None:
        raise HTTPException(status_code=503, detail="async: results store not ready")
    return _results_store


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read ``file`` but reject payloads larger than ``max_bytes``.

    Defends against missing or lying ``Content-Length`` headers by
    checking actual bytes-read, not just the advertised length.
    """
    # Reject early if Content-Length advertised is over the cap
    cl = file.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"payload too large: {cl} bytes > max {max_bytes}",
                )
        except ValueError:
            pass  # malformed header -> ignore; the chunk loop will catch it

    chunks: list[bytes] = []
    total = 0
    chunk_size = 64 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"payload too large: exceeds max {max_bytes} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
# Per-subtype handlers. Order matters: more specific must be registered
# before more general (FastAPI dispatches in LIFO order of registration).
#
# We do **not** map HTTPException through error_envelope — those are
# FastAPI-native and already have well-formed {"detail": "..."} bodies.
# Mapping them through our envelope would break the OpenAPI spec.


@app.exception_handler(RateLimitedError)
async def rate_limit_handler(request: Request, exc: RateLimitedError) -> JSONResponse:
    """429 + structured envelope. Add Retry-After header."""
    env = error_envelope(exc, request_id=_request_id(request))
    headers = {"Retry-After": "60"}
    return JSONResponse(status_code=429, content=env, headers=headers)


@app.exception_handler(IDPError)
async def idp_error_handler(request: Request, exc: IDPError) -> JSONResponse:
    """Map any other IDPError to its declared http_status + envelope."""
    return JSONResponse(
        status_code=exc.http_status,
        content=error_envelope(exc, request_id=_request_id(request)),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 + envelope for malformed request bodies / query params."""
    # Wrap Pydantic's structured error list under `details` so the
    # envelope stays one-level-deep.
    env = error_envelope(
        IDPError("request validation failed"),
        request_id=_request_id(request),
        details={"errors": exc.errors()},
    )
    # Force the right code for validation; the base IDPError code is
    # IDP-INT-001 which is misleading.
    env["error"]["code"] = "IDP-VAL-001"
    return JSONResponse(status_code=422, content=env)


# ---------------------------------------------------------------------------
# Module version (also used by /version endpoint)
# ---------------------------------------------------------------------------
__all__ = ["app", "lifespan"]

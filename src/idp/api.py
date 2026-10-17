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

import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from idp.ratelimit import RateLimiter
from idp.templates import TemplateRegistry

# Module-level logger; configured at lifespan startup.
_log = get_logger("idp.api")
_settings: Settings | None = None
_rate_limiter: RateLimiter | None = None
_upload_dir: Path | None = None
_template_registry: TemplateRegistry | None = None

# Request-ID header name (lowercase; FastAPI normalizes).
_REQUEST_ID_HEADER = "x-request-id"


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
    global _settings, _rate_limiter, _upload_dir, _template_registry

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

    # NOTE: CORS middleware cannot be added inside lifespan() — Starlette
    # forbids ``add_middleware`` after the app has started. We install it
    # at module-import time below (see ``_install_cors_initial()``), so
    # Settings.cors_origins env var must be set before uvicorn imports
    # ``idp.api:app``. The legacy deferred-install function is kept for
    # test compatibility but is now a no-op (logs a warning if called).
    if os.environ.get("IDP_CORS_ORIGINS"):
        _log.info(
            "CORS configured for: %s — middleware installed at import time",
            _settings.cors_origins,
        )

    _rate_limiter = RateLimiter(
        per_key_per_minute=_settings.rate_limit_per_minute,
        global_per_minute=0,  # disabled by default; set via env if needed
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

    yield

    # Shutdown: clean up, log final metrics
    _log.info("py-idp API shutting down. metrics: %s", metrics.snapshot())
    _settings = None
    _rate_limiter = None
    _upload_dir = None
    _template_registry = None


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


# CORS (installed at module-import time)
#
# Important: Starlette forbids ``app.add_middleware(...)`` after the app
# has started, so we cannot install CORS from inside the lifespan handler.
# Instead, we read IDP_CORS_ORIGINS once at import time and install
# CORSMiddleware BEFORE the ``app = FastAPI(...)`` line below. The
# setting is fixed at import — change requires re-import (which uvicorn
# does on every reload). The legacy ``_install_cors()`` function is kept
# as a documented no-op for back-compat (any old code calling it gets a
# warning explaining the new constraint).
_CORS_INSTALLED = False


def _install_cors() -> None:
    """DEPRECATED no-op for back-compat.

    CORS must now be installed at module-import time via
    ``_install_cors_initial()`` (called automatically below the ``app =
    FastAPI(...)`` line is impossible, so we install above it instead).
    Calling this function at runtime raises ``RuntimeError`` because
    Starlette forbids ``add_middleware`` after app start.
    """
    raise RuntimeError(
        "_install_cors() is a no-op in this version of py-idp. "
        "CORS middleware is installed at module-import time. "
        "Set IDP_CORS_ORIGINS in the environment BEFORE starting "
        "uvicorn, or rebuild the process after changing it."
    )


def _install_cors_initial() -> None:
    """Install CORSMiddleware on the app. Called exactly once at import."""
    global _CORS_INSTALLED
    if _CORS_INSTALLED:
        return
    s = _require_settings_or_default()
    if s.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(s.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        _log.info("CORS middleware installed for origins=%s", list(s.cors_origins))
    _CORS_INSTALLED = True


def _require_settings_or_default():
    """Return Settings.load() if the env has IDP_CORS_ORIGINS, else
    fall back to a Settings() instance with default cors_origins."""
    if os.environ.get("IDP_CORS_ORIGINS"):
        return Settings.load()
    return Settings()


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
    # Filename is attacker-controlled. Sanitize, pin to upload_dir, and
    # reject any extension we don't accept BEFORE writing to disk.
    dst = _safe_upload_path(upload_dir, file.filename or "")
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
    # arun() dispatches the blocking LLM call (5-15s on Nanonets, 1-30s
    # on hosted backends) to a worker thread so the event loop stays
    # responsive. Without this, /extract freezes /healthz, /readyz, and
    # any other concurrent requests. (#P0 audit fix.)
    res = await pipeline.arun(doc)
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
         up to the last ``\\\\``. We do an extra ``replace("\\\\", ...)``
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


# -------------------------------------------------------------------
# Module version (also used by /version endpoint)
# -------------------------------------------------------------------
__all__ = ["app", "lifespan"]

# Install CORS now that all helper functions are defined and ``app`` is
# fully constructed. Must run AFTER ``app = FastAPI(...)`` (line ~218)
# and AFTER all helper functions. Starlette forbids ``add_middleware``
# after the app has started (i.e. inside lifespan), so this is the
# ONLY correct install point. IDP_CORS_ORIGINS must be set in the env
# before uvicorn imports ``idp.api:app``.
_install_cors_initial()

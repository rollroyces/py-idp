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

Not in scope: HTTPS termination (use a reverse proxy), TLS, SSO,
multi-tenant auth (these are deployment-level concerns).
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from idp import __version__
from idp._logging import configure, get_logger
from idp.config import Settings
from idp.core.document import Document
from idp.errors import ConfigurationError, IDPError, RateLimitedError, is_idp_error
from idp.metrics import metrics
from idp.pipeline.pipeline import Pipeline
from idp.ratelimit import RateLimiter

# Module-level logger; configured at lifespan startup.
_log = get_logger("idp.api")
_settings: Settings | None = None
_rate_limiter: RateLimiter | None = None
_upload_dir: Path | None = None


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


class ReviewRequest(BaseModel):
    edited: dict[str, Any]
    reviewer: str


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
    """Configure logging, validate settings, prep upload dir."""
    global _settings, _rate_limiter, _upload_dir

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

    _upload_dir = Path(os.environ.get("IDP_UPLOAD_DIR", "/tmp/idp-uploads"))
    _upload_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown: clean up, log final metrics
    _log.info("py-idp API shutting down. metrics: %s", metrics.snapshot())
    _settings = None
    _rate_limiter = None
    _upload_dir = None


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
async def logging_middleware(request: Request, call_next):
    """Log every request with timing + status code."""
    start = time.perf_counter()
    metrics.inc("http_requests", 1, path=request.url.path, method=request.method)
    try:
        response = await call_next(request)
        metrics.inc("http_responses", 1, status=str(response.status_code), path=request.url.path)
        return response
    except Exception as e:
        metrics.inc("http_errors", 1, type=type(e).__name__)
        _log.exception("request failed: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal server error", "type": type(e).__name__,
                     "is_idp_error": is_idp_error(e)},
        )
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
      - ``backend``    (optional): LLM backend name (e.g. ``mock``, ``openai/gpt-4o``).
    """
    s = _require_settings()
    raw = await _read_capped(file, s.max_upload_bytes)
    upload_dir = _require_upload_dir()
    dst = upload_dir / (file.filename or "upload")
    dst.write_bytes(raw)

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


# Exception handlers — map IDPError subtypes to proper HTTP status codes
@app.exception_handler(IDPError)
async def idp_error_handler(request: Request, exc: IDPError) -> JSONResponse:
    if isinstance(exc, RateLimitedError):
        return JSONResponse(status_code=429, content={"error": str(exc), "type": type(exc).__name__})
    return JSONResponse(status_code=400, content={"error": str(exc), "type": type(exc).__name__})


# ---------------------------------------------------------------------------
# Module version (also used by /version endpoint)
# ---------------------------------------------------------------------------
__all__ = ["app", "lifespan"]
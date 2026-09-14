"""py-idp public exception hierarchy.

All py-idp-specific exceptions inherit from ``IDPError`` so callers can
catch the framework without catching unrelated Python exceptions. Each
subclass documents a recovery hint in its docstring and carries a
**machine-readable error code** (e.g. ``IDP-RATE-001``) for use in HTTP
API responses and log aggregation.
"""
from __future__ import annotations

from typing import Any, ClassVar


class IDPError(Exception):
    """Base class for all py-idp-specific errors.

    Catch this to handle any error from the framework without
    accidentally swallowing unrelated Python exceptions (e.g.
    ``KeyboardInterrupt``).

    Subclasses set ``code`` (e.g. ``"IDP-RATE-001"``) and ``http_status``
    (e.g. ``429``) as class attributes. The ``/api/v1`` exception
    handlers in ``idp.api`` map these to JSON error envelopes.
    """

    code: ClassVar[str] = "IDP-INT-001"  # default if subclass forgets
    http_status: ClassVar[int] = 500


class DocumentParseError(IDPError):
    """Raised when no parser can extract text from a document.

    Common causes:
      - file extension not supported
      - file is corrupt / encrypted
      - Docling is not installed for PDF
    """

    code = "IDP-PARSE-001"
    http_status = 422


class SchemaValidationError(IDPError):
    """Raised when an extraction dict fails Pydantic schema validation
    and ``strict=True`` was passed to ``extract()``.

    By default validation failures are returned as a soft warning in
    ``doc.errors`` — raise this only when the caller explicitly opts
    in to strict mode.
    """

    code = "IDP-SCHEMA-001"
    http_status = 422


class BackendUnavailableError(IDPError):
    """Raised when a configured LLM backend cannot be reached.

    The original exception is chained via ``raise ... from e`` so
    callers can inspect the cause.
    """

    code = "IDP-BACKEND-001"
    http_status = 502


class StorageError(IDPError):
    """Raised on unrecoverable storage failures (disk full, permission
    denied, schema migration failed).
    """

    code = "IDP-STORE-001"
    http_status = 500


class ConfigurationError(IDPError):
    """Raised when a required configuration value is missing or invalid
    (e.g. ``DATABASE_URL`` set but unparseable).
    """

    code = "IDP-CONF-001"
    http_status = 503


class RateLimitedError(IDPError, RuntimeError):  # intentionally inherits both
    """Raised when an API request exceeds the configured rate limit.

    Inherits both ``IDPError`` and ``RuntimeError`` so callers can
    ``except IDPError`` and ``except RuntimeError`` independently.
    """

    code = "IDP-RATE-001"
    http_status = 429


class TemplateNotFoundError(IDPError):
    """Raised when a template name is requested that isn't registered.

    Distinct from ``ConfigurationError`` (which is about server-level
    misconfig) — this is a client-side "you asked for a template that
    doesn't exist" error and should return 404.
    """

    code = "IDP-TMPL-404"
    http_status = 404


class TemplateParseError(IDPError):
    """Raised when a template file is malformed (bad YAML frontmatter,
    missing required field, invalid schema reference).
    """

    code = "IDP-TMPL-001"
    http_status = 500


__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "DocumentParseError",
    "IDPError",
    "RateLimitedError",
    "SchemaValidationError",
    "StorageError",
    "TemplateNotFoundError",
    "TemplateParseError",
]


def is_idp_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is any py-idp-specific exception."""
    return isinstance(exc, IDPError)


def error_envelope(
    exc: BaseException,
    *,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured error envelope for HTTP responses and logs.

    Shape::

        {
            "error": {
                "code": "IDP-RATE-001",
                "message": "rate limit exceeded",
                "type": "RateLimitedError",
                "request_id": "uuid-or-None",
                "details": {...}    # only when caller passes them
            }
        }

    The ``code`` and ``type`` are both included so clients can switch on
    a stable string (code) without having to import the Python class.
    """
    code = getattr(exc, "code", "IDP-INT-001") if is_idp_error(exc) else "IDP-INT-001"
    env: dict[str, Any] = {
        "code": code,
        "message": str(exc) or type(exc).__name__,
        "type": type(exc).__name__,
    }
    if request_id is not None:
        env["request_id"] = request_id
    if details:
        env["details"] = details
    return {"error": env}


def format_error(exc: BaseException) -> dict[str, Any]:
    """Format an exception for structured-log output.

    Returns a dict safe to ``json.dumps`` — no non-serializable values.
    Useful for emitting error events to a log aggregator.
    """
    env: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "module": type(exc).__module__,
        "is_idp_error": is_idp_error(exc),
    }
    if is_idp_error(exc):
        env["code"] = getattr(exc, "code", "IDP-INT-001")
    return env

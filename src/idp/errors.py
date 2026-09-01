"""py-idp public exception hierarchy.

All py-idp-specific exceptions inherit from ``IDPError`` so callers can
catch the framework without catching unrelated Python exceptions. Each
subclass documents a recovery hint in its docstring.
"""
from __future__ import annotations

from typing import Any


class IDPError(Exception):
    """Base class for all py-idp-specific errors.

    Catch this to handle any error from the framework without
    accidentally swallowing unrelated Python exceptions (e.g.
    ``KeyboardInterrupt``).
    """


class DocumentParseError(IDPError):
    """Raised when no parser can extract text from a document.

    Common causes:
      - file extension not supported
      - file is corrupt / encrypted
      - Docling is not installed for PDF
    """


class SchemaValidationError(IDPError):
    """Raised when an extraction dict fails Pydantic schema validation
    and ``strict=True`` was passed to ``extract()``.

    By default validation failures are returned as a soft warning in
    ``doc.errors`` — raise this only when the caller explicitly opts
    in to strict mode.
    """


class BackendUnavailableError(IDPError):
    """Raised when a configured LLM backend cannot be reached.

    The original exception is chained via ``raise ... from e`` so
    callers can inspect the cause.
    """


class StorageError(IDPError):
    """Raised on unrecoverable storage failures (disk full, permission
    denied, schema migration failed).
    """


class ConfigurationError(IDPError):
    """Raised when a required configuration value is missing or invalid
    (e.g. ``DATABASE_URL`` set but unparseable).
    """


class RateLimitedError(IDPError, RuntimeError):  # intentionally inherits both
    """Raised when an API request exceeds the configured rate limit.

    Inherits both ``IDPError`` and ``RuntimeError`` so callers can
    ``except IDPError`` and ``except RuntimeError`` independently.
    """


__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "DocumentParseError",
    "IDPError",
    "RateLimitedError",
    "SchemaValidationError",
    "StorageError",
]


def is_idp_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is any py-idp-specific exception."""
    return isinstance(exc, IDPError)


def format_error(exc: BaseException) -> dict[str, Any]:
    """Format an exception for structured-log output.

    Returns a dict safe to ``json.dumps`` — no non-serializable values.
    Useful for emitting error events to a log aggregator.
    """
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "module": type(exc).__module__,
        "is_idp_error": is_idp_error(exc),
    }
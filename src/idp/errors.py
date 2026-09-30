"""py-idp public exception hierarchy.

All py-idp-specific exceptions inherit from ``IDPError`` so callers can
catch the framework without catching unrelated Python exceptions. Each
subclass documents a recovery hint in its docstring and carries a
**machine-readable error code** (e.g. ``IDP-RATE-001``) for use in HTTP
API responses and log aggregation.

In addition to the exception classes, this module exposes:

* :func:`error_envelope`  -- build a structured JSON envelope for HTTP
  responses / structured logs.
* :func:`format_error`    -- JSON-safe dict for one exception (log aggregator).
* :func:`wrap`            -- decorator that turns arbitrary raised
  exceptions into a chosen :class:`IDPError` subclass while preserving
  the original via ``__cause__``.
* :func:`safe_call`       -- context manager that catches + logs + re-raises
  as an :class:`IDPError` (use when ``@wrap`` doesn't fit, e.g. inside loops).
* :func:`is_transient`    -- best-effort guess whether an exception is a
  transient infrastructure failure worth retrying.
* :func:`classify_status` -- map an HTTP status code (int) to a code
  prefix used in error envelopes.
"""
from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Iterator, TypeVar

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base hierarchy
# ---------------------------------------------------------------------------
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
    ``doc.errors`` -- raise this only when the caller explicitly opts
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


class OCRError(IDPError):
    """Raised when the OCR stage fails to produce usable text.

    Distinct from :class:`DocumentParseError` (which is about the format
    being unrecognizable) -- ``OCRError`` means the document parsed but
    the OCR engine itself returned an error, no text, or garbage.
    """

    code = "IDP-OCR-001"
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


class TimeoutError_(IDPError):  # noqa: N801 -- disambiguate from builtin
    """Raised when an operation exceeds its time budget.

    Used both by HTTP request timeouts (when callers don't already
    surface ``httpx.TimeoutException``) and by in-process operations
    like OCR inference. Typically transient -- see :func:`is_transient`.
    """

    code = "IDP-TIME-001"
    http_status = 504


class TemplateNotFoundError(IDPError):
    """Raised when a template name is requested that isn't registered.

    Distinct from :class:`ConfigurationError` (which is about server-level
    misconfig) -- this is a client-side "you asked for a template that
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


class FileTooLargeError(IDPError):
    """Raised when an uploaded file exceeds the configured size cap.

    Maps to HTTP 413 (Payload Too Large). Distinct from generic
    validation errors because the fix is client-side (upload smaller
    file or raise server cap), not a code change.
    """

    code = "IDP-UPLOAD-413"
    http_status = 413


# Backwards-compat alias: ``TimeoutError`` clashes with the builtin in
# Python 3.11+ on stdin-like paths, so prefer ``TimeoutError_``.
TimeoutError = TimeoutError_  # type: ignore[misc]


__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "DocumentParseError",
    "FileTooLargeError",
    "IDPError",
    "OCRError",
    "RateLimitedError",
    "SchemaValidationError",
    "StorageError",
    "TemplateNotFoundError",
    "TemplateParseError",
    "TimeoutError",
    "TimeoutError_",
    "classify_status",
    "error_envelope",
    "format_error",
    "is_idp_error",
    "is_transient",
    "safe_call",
    "wrap",
]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------
def is_idp_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is any py-idp-specific exception."""
    return isinstance(exc, IDPError)


# HTTP status codes that are typically transient and worth retrying.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 599})


def classify_status(status: int) -> str:
    """Map an HTTP status code (int) to a stable error code prefix.

    Used by API wrappers that want to build an envelope without going
    through a py-idp exception. Falls back to ``"IDP-INT-{status}"``
    for unknown codes so the envelope is always populated.
    """
    if status == 429:
        return "IDP-RATE-001"
    if status == 413:
        return "IDP-UPLOAD-413"
    if 400 <= status < 500:
        return f"IDP-CLIENT-{status:03d}"
    if 500 <= status < 600:
        return f"IDP-SERVER-{status:03d}"
    return f"IDP-INT-{status:03d}"


def is_transient(exc: BaseException) -> bool:
    """Best-effort guess: is this exception a transient infrastructure
    failure worth retrying?

    Used by ``RetryingBackend`` and ad-hoc retry loops. Never treats
    :class:`IDPError` subclasses with non-retryable semantics
    (``ConfigurationError``, ``SchemaValidationError``,
    ``TemplateNotFoundError``, ``FileTooLargeError``) as transient.

    The check is intentionally conservative: if we don't recognize
    the exception, we say it IS transient (the caller can decide).
    """
    # Non-transient py-idp errors: retrying them just delays the failure.
    if isinstance(exc, (ConfigurationError, SchemaValidationError,
                        TemplateNotFoundError, FileTooLargeError,
                        TemplateParseError)):
        return False

    # Built-in transient types
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError):
        # OSError covers a wide family (ConnectionRefusedError,
        # BrokenPipeError, etc.) -- most are worth one retry.
        return True

    # httpx / requests / urllib3 status codes we couldn't classify earlier.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        if status is not None and int(status) in _TRANSIENT_HTTP_STATUSES:
            return True
    except (TypeError, ValueError):
        pass

    # Message-based heuristic as a last resort.
    msg = str(exc).lower()
    transient_markers = (
        "timeout", "timed out", "connection reset", "connection refused",
        "network", "unreachable", "rate limit", "rate_limit", "temporarily",
        "try again", "service unavailable", "502", "503", "504",
    )
    return any(m in msg for m in transient_markers)


# ---------------------------------------------------------------------------
# Envelope builders
# ---------------------------------------------------------------------------
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

    When the code is recognized, a user-facing ``hint`` and ``docs_url``
    are attached so API consumers can render actionable guidance next
    to the error (rather than only a stack-trace-y message).
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
    return _envelope_with_hint({"error": env})


def format_error(exc: BaseException) -> dict[str, Any]:
    """Format an exception for structured-log output.

    Returns a dict safe to ``json.dumps`` -- no non-serializable values.
    Useful for emitting error events to a log aggregator.
    """
    env: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "module": type(exc).__module__,
        "is_idp_error": is_idp_error(exc),
        "is_transient": is_transient(exc),
    }
    if is_idp_error(exc):
        env["code"] = getattr(exc, "code", "IDP-INT-001")
    return env


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
_F = TypeVar("_F", bound=Callable[..., Any])


def wrap(
    *target: type[IDPError],
    default: type[IDPError] | None = None,
    logger: logging.Logger | None = None,
    reraise: bool = True,
) -> Callable[[_F], _F]:
    """Decorator: convert non-IDP exceptions into an :class:`IDPError`.

    Use on risky call sites where you want one unified exception type
    flowing up the stack (HTTP handlers, batch processing, etc.).

    Args:
        target: explicit exception types to convert. If omitted, any
            caught exception is converted to ``default``.
        default: fallback :class:`IDPError` subclass used when ``target``
            is empty or the raised exception doesn't match any of
            them. Defaults to :class:`BackendUnavailableError`.
        logger: optional logger for ``.exception`` call. If ``None``,
            uses this module's logger.
        reraise: if False, swallow the exception and return ``None``.
            Useful for best-effort hooks where the caller can't act on
            the failure.

    Example::

        @idp.errors.wrap(idp.errors.DocumentParseError)
        def parse_pdf(path: str) -> Document:
            ...

    The original exception is preserved via ``raise New(...) from e``
    so tracebacks stay complete.
    """
    log = logger or _log
    fallback = default or BackendUnavailableError

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except IDPError:
                # Already typed -- let it through.
                raise
            except Exception as e:
                log.exception(
                    "%s raised %s; converting to IDPError",
                    fn.__qualname__, type(e).__name__,
                )
                if not reraise:
                    return None
                raise fallback(str(e) or type(e).__name__) from e

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def safe_call(
    op: str,
    *,
    default: type[IDPError] = BackendUnavailableError,
    logger: logging.Logger | None = None,
    reraise: bool = True,
) -> Iterator[dict[str, Any]]:
    """Context manager: catch + log + convert non-IDP exceptions to IDPError.

    Use inside loops or blocks where ``@wrap`` doesn't fit::

        with safe_call("extract_chunk", default=DocumentParseError):
            chunk = parse(path)

    Yields a ``state`` dict the caller can populate (e.g. ``"doc_id"``)
    to be included in the resulting exception's ``__cause__``. Always
    logs the original traceback at ``.exception`` level on failure.
    """
    log = logger or _log
    state: dict[str, Any] = {}
    try:
        yield state
    except IDPError:
        raise
    except Exception as e:
        log.exception("safe_call(%s) caught %s", op, type(e).__name__)
        if not reraise:
            return
        # Preserve state in __cause__ chain via the from-clause.
        raise default(f"{op}: {e}") from e


# ---------------------------------------------------------------------------
# User-facing hints
# ---------------------------------------------------------------------------
# Each ``IDP-error`` code maps to a one-line, action-oriented message that
# end-users (API consumers) see in the response envelope. The hint is
# stable per code -- clients can i18n against it or branch on it -- and
# tells the user what they can do next ("retry", "fix the API key", "use
# a smaller file"), not just what went wrong.
#
# Format: short sentence, plain English, second person. No jargon. If
# you can't fit the advice in one sentence, the message doesn't belong
# in this table -- link to docs instead via the ``docs_url`` field.
_USER_HINTS: dict[str, str] = {
    # --- Server-defined codes (declared on IDPError subclasses) -------
    "IDP-PARSE-001":    "We couldn't read your document. Make sure it's a supported file type (PDF, image, plain text) and isn't password-protected.",
    "IDP-SCHEMA-001":   "The extracted data didn't match the expected schema. Try a clearer document or a different schema.",
    "IDP-BACKEND-001":  "The language model is unavailable right now. Retry in a few seconds; if it keeps failing, check the backend status page.",
    "IDP-OCR-001":      "OCR couldn't read the document. Try a higher-resolution scan or a different document.",
    "IDP-STORE-001":    "We couldn't save the result. Check disk space and file permissions on the server, then retry.",
    "IDP-CONF-001":     "The server is misconfigured. Contact the operator -- this isn't something you can fix from the client side.",
    "IDP-RATE-001":     "Too many requests. Slow down and retry after the time indicated by the Retry-After header.",
    "IDP-TIME-001":     "The operation took too long. Retry with a smaller document or fewer pages.",
    "IDP-TMPL-404":     "That template doesn't exist. Check the /templates endpoint for the list of available templates.",
    "IDP-TMPL-001":     "The template file is malformed. Contact the operator to fix the template.",
    "IDP-UPLOAD-413":   "Your upload is too large. Reduce the file size or raise the server's max_upload_bytes setting.",
    "IDP-INT-001":      "Something went wrong on our end. Retry in a few seconds; if it keeps failing, contact support with the request_id.",
    # --- HTTP-status-derived codes (set by classify_status) -------------
    "IDP-CLIENT-400":   "Your request was malformed. Check the API docs for the expected fields and formats.",
    "IDP-CLIENT-401":   "Authentication required. Provide a valid X-API-Key header.",
    "IDP-CLIENT-403":   "Your API key is not authorized for this operation. Check that the key is correct and has the right permissions.",
    "IDP-CLIENT-404":   "That endpoint or resource doesn't exist. Check the URL.",
    "IDP-CLIENT-405":   "That HTTP method isn't supported on this endpoint. Check the API docs.",
    "IDP-CLIENT-413":   "Your request body is too large. Reduce the payload size.",
    "IDP-CLIENT-415":   "Unsupported media type. Use Content-Type matching what the endpoint expects.",
    "IDP-CLIENT-422":   "Your request body has validation errors. Check the response details for the specific field errors.",
    "IDP-CLIENT-429":   "Too many requests. Slow down; respect the Retry-After header.",
    "IDP-SERVER-500":   "Something went wrong on our end. Retry in a few seconds; if it keeps failing, contact support with the request_id.",
    "IDP-SERVER-502":   "An upstream service is unavailable. Retry shortly; if the issue persists, check the status page.",
    "IDP-SERVER-503":   "The service is starting up or temporarily unavailable. Retry shortly.",
    "IDP-SERVER-504":   "An upstream service timed out. Retry shortly.",
}


def hint_for(code: str) -> str | None:
    """Return the user-facing hint for a given ``IDP-error`` code.

    Returns ``None`` when the code isn't recognized -- callers can fall
    back to the raw ``message`` field in that case.
    """
    return _USER_HINTS.get(code)


def docs_url_for(code: str) -> str | None:
    """Return a docs URL anchor for a given code, or ``None`` if not mapped.

    Clients can render this as a link next to the hint in their UI.
    Anchors are the same as the code (e.g. ``#IDP-RATE-001``) so the
    USER_ERRORS.md catalog can deep-link to a section per code.
    """
    if code in _USER_HINTS:
        return f"https://py-idp.dev/errors#{code}"
    return None


# Patch error_envelope + format_error to attach the hint + docs_url.
def _envelope_with_hint(envelope: dict[str, Any]) -> dict[str, Any]:
    """Attach ``hint`` + ``docs_url`` to an error envelope in place."""
    code = envelope.get("error", {}).get("code")
    if code and code in _USER_HINTS:
        envelope["error"]["hint"] = _USER_HINTS[code]
        envelope["error"]["docs_url"] = f"https://py-idp.dev/errors#{code}"
    return envelope

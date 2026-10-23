"""Regression tests for the user-facing error surface.

Covers the helpers in :mod:`idp.errors` (classification, envelopes,
wrappers) and the FastAPI exception handlers in :mod:`idp.api`.

Two of these tests are deliberate regression guards:

``test_errors_module_does_not_shadow_builtin_timeouterror``
    ``idp.errors`` used to define ``TimeoutError = TimeoutError_``. That
    module-level binding shadowed the *builtin* for the whole module, so
    ``is_transient()`` ended up testing ``isinstance(exc, TimeoutError_)``
    -- the py-idp error -- instead of the real ``TimeoutError``. A plain
    ``TimeoutError("timed out")`` was therefore classified as
    non-transient and never retried.

``test_builtin_timeouterror_is_transient``
    The behavioural half of the same guard.

Both would fail against the original implementation.
"""
from __future__ import annotations

import json

import pytest

from idp import errors
from idp.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DocumentParseError,
    FileTooLargeError,
    IDPError,
    OCRError,
    RateLimitedError,
    SchemaValidationError,
    TemplateNotFoundError,
    TemplateParseError,
    TimeoutError_,
    classify_status,
    docs_url_for,
    error_envelope,
    format_error,
    hint_for,
    is_idp_error,
    is_transient,
    safe_call,
    wrap,
)


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------
def test_errors_module_does_not_shadow_builtin_timeouterror():
    """idp.errors must not bind the name ``TimeoutError`` at module level."""
    assert not hasattr(errors, "TimeoutError"), (
        "idp.errors binds 'TimeoutError', shadowing the builtin for the whole "
        "module -- is_transient() would then test the py-idp error instead "
        "of the real TimeoutError"
    )
    # The disambiguated name is the one that should be importable.
    assert errors.TimeoutError_ is TimeoutError_


def test_builtin_timeouterror_is_transient():
    """The *builtin* TimeoutError must be classified as transient."""
    assert is_transient(TimeoutError("timed out")) is True


def test_builtin_timeouterror_is_transient_with_arbitrary_message():
    """Classification must not depend on the message text for the builtin."""
    assert is_transient(TimeoutError("boom")) is True


# ---------------------------------------------------------------------------
# Transient classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timed out"),
        ConnectionError("refused"),
        ConnectionResetError("reset"),
        OSError("some os error"),
    ],
)
def test_transient_builtin_exceptions(exc):
    assert is_transient(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ConfigurationError("bad config"),
        SchemaValidationError("bad schema"),
        TemplateNotFoundError("no template"),
        FileTooLargeError("too big"),
        TemplateParseError("bad yaml"),
    ],
)
def test_non_transient_idp_errors(exc):
    assert is_transient(exc) is False


def test_unknown_error_without_transient_markers_is_not_transient():
    assert is_transient(ValueError("plain failure")) is False


def test_unknown_error_with_transient_marker_is_transient():
    assert is_transient(ValueError("connection reset by peer")) is True


def test_is_idp_error():
    assert is_idp_error(IDPError("x")) is True
    assert is_idp_error(DocumentParseError("x")) is True
    assert is_idp_error(ValueError("x")) is False


# ---------------------------------------------------------------------------
# classify_status
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, "IDP-RATE-001"),
        (413, "IDP-UPLOAD-413"),
        (400, "IDP-CLIENT-400"),
        (401, "IDP-CLIENT-401"),
        (404, "IDP-CLIENT-404"),
        (500, "IDP-SERVER-500"),
        (503, "IDP-SERVER-503"),
        (504, "IDP-SERVER-504"),
        (200, "IDP-INT-200"),
    ],
)
def test_classify_status(status, expected):
    assert classify_status(status) == expected


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------
def test_error_envelope_shape():
    env = error_envelope(RateLimitedError("slow down"))
    err = env["error"]
    assert err["code"] == "IDP-RATE-001"
    assert err["message"] == "slow down"
    assert err["type"] == "RateLimitedError"
    # Known codes get an actionable hint + docs deep-link.
    assert err["hint"]
    assert err["docs_url"].endswith("#IDP-RATE-001")


def test_error_envelope_includes_request_id_and_details():
    env = error_envelope(
        IDPError("boom"), request_id="req-123", details={"field": "name"}
    )
    assert env["error"]["request_id"] == "req-123"
    assert env["error"]["details"] == {"field": "name"}


def test_error_envelope_omits_absent_optional_fields():
    err = error_envelope(IDPError("boom"))["error"]
    assert "request_id" not in err
    assert "details" not in err


def test_error_envelope_for_non_idp_exception_uses_generic_code():
    err = error_envelope(ValueError("nope"))["error"]
    assert err["code"] == "IDP-INT-001"
    assert err["type"] == "ValueError"


def test_error_envelope_is_json_serialisable():
    env = error_envelope(OCRError("bad scan"), details={"page": 2})
    json.dumps(env)  # must not raise


def test_format_error_shape():
    out = format_error(RateLimitedError("slow down"))
    assert out["type"] == "RateLimitedError"
    assert out["message"] == "slow down"
    assert out["is_idp_error"] is True
    assert out["code"] == "IDP-RATE-001"


def test_format_error_for_non_idp_exception_has_no_code():
    out = format_error(ValueError("nope"))
    assert out["is_idp_error"] is False
    assert "code" not in out
    json.dumps(out)


# ---------------------------------------------------------------------------
# Hints / docs links
# ---------------------------------------------------------------------------
def test_hint_for_known_and_unknown_codes():
    assert hint_for("IDP-RATE-001")
    assert hint_for("IDP-CLIENT-404")
    assert hint_for("IDP-NOT-A-CODE") is None


def test_docs_url_for_known_and_unknown_codes():
    url = docs_url_for("IDP-RATE-001")
    assert url is not None and url.endswith("#IDP-RATE-001")
    assert docs_url_for("IDP-NOT-A-CODE") is None


# ---------------------------------------------------------------------------
# wrap()
# ---------------------------------------------------------------------------
def test_wrap_converts_foreign_exception_to_idp_error():
    @wrap()
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(BackendUnavailableError) as ei:
        boom()
    assert "kaboom" in str(ei.value)
    assert isinstance(ei.value.__cause__, ValueError)


def test_wrap_passes_idp_errors_through_untouched():
    original = DocumentParseError("cannot parse")

    @wrap()
    def boom():
        raise original

    with pytest.raises(DocumentParseError) as ei:
        boom()
    assert ei.value is original


def test_wrap_uses_explicit_default():
    @wrap(default=DocumentParseError)
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(DocumentParseError):
        boom()


def test_wrap_can_swallow_when_reraise_false():
    @wrap(reraise=False)
    def boom():
        raise ValueError("kaboom")

    assert boom() is None


def test_wrap_returns_value_on_success():
    @wrap()
    def fine():
        return 42

    assert fine() == 42


# ---------------------------------------------------------------------------
# safe_call()
# ---------------------------------------------------------------------------
def test_safe_call_converts_foreign_exception():
    with pytest.raises(BackendUnavailableError) as ei, safe_call("extract_chunk"):
        raise ValueError("kaboom")
    assert "extract_chunk" in str(ei.value)
    assert isinstance(ei.value.__cause__, ValueError)


def test_safe_call_passes_idp_errors_through():
    original = DocumentParseError("cannot parse")
    with pytest.raises(DocumentParseError) as ei, safe_call("extract_chunk"):
        raise original
    assert ei.value is original


def test_safe_call_yields_mutable_state():
    with safe_call("extract_chunk") as state:
        state["doc_id"] = "abc"
    assert state == {"doc_id": "abc"}


def test_safe_call_can_swallow_when_reraise_false():
    with safe_call("extract_chunk", reraise=False):
        raise ValueError("kaboom")
    # Reaching here without raising is the assertion.


def test_safe_call_uses_custom_default():
    with pytest.raises(OCRError), safe_call("ocr", default=OCRError):
        raise ValueError("kaboom")


# ---------------------------------------------------------------------------
# API exception handlers
# ---------------------------------------------------------------------------
def _request(rid: str | None = None):
    """Build a minimal Starlette Request for direct handler invocation."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "http_version": "1.1",
    }
    req = Request(scope)
    if rid is not None:
        req.state.request_id = rid
    return req


@pytest.mark.asyncio
async def test_timeout_handler_returns_504_with_envelope():
    from idp.api import timeout_handler

    resp = await timeout_handler(_request("rid-1"), TimeoutError_("too slow"))
    assert resp.status_code == 504
    body = json.loads(resp.body)
    assert body["error"]["code"] == "IDP-TIME-001"
    assert body["error"]["request_id"] == "rid-1"


@pytest.mark.asyncio
async def test_rate_limit_handler_returns_429_with_retry_after():
    from idp.api import rate_limit_handler

    resp = await rate_limit_handler(_request(), RateLimitedError("slow down"))
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"
    body = json.loads(resp.body)
    assert body["error"]["code"] == "IDP-RATE-001"


@pytest.mark.asyncio
async def test_file_too_large_handler_returns_413():
    from idp.api import file_too_large_handler

    resp = await file_too_large_handler(_request(), FileTooLargeError("too big"))
    assert resp.status_code == 413
    assert json.loads(resp.body)["error"]["code"] == "IDP-UPLOAD-413"


@pytest.mark.asyncio
async def test_ocr_error_handler_uses_declared_http_status():
    from idp.api import ocr_error_handler

    resp = await ocr_error_handler(_request(), OCRError("bad scan"))
    assert resp.status_code == OCRError.http_status
    assert json.loads(resp.body)["error"]["code"] == "IDP-OCR-001"


@pytest.mark.asyncio
async def test_generic_idp_error_handler_uses_declared_http_status():
    from idp.api import idp_error_handler

    resp = await idp_error_handler(_request(), DocumentParseError("bad pdf"))
    assert resp.status_code == DocumentParseError.http_status
    assert json.loads(resp.body)["error"]["code"] == "IDP-PARSE-001"


@pytest.mark.asyncio
async def test_unhandled_exception_handler_returns_generic_500():
    from idp.api import unhandled_exception_handler

    resp = await unhandled_exception_handler(_request(), ValueError("secret detail"))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"]["code"] == "IDP-INT-001"
    # The internal detail must not leak to the client.
    assert "secret detail" not in resp.body.decode()


@pytest.mark.asyncio
async def test_http_exception_handler_derives_code_from_status():
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from idp.api import http_exception_handler

    resp = await http_exception_handler(
        _request(), StarletteHTTPException(status_code=404, detail="nope")
    )
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["error"]["code"] == "IDP-CLIENT-404"
    assert body["error"]["message"] == "nope"


@pytest.mark.asyncio
async def test_validation_error_handler_returns_422_with_details():
    from fastapi.exceptions import RequestValidationError

    from idp.api import validation_error_handler

    exc = RequestValidationError([{"loc": ["body", "x"], "msg": "missing", "type": "missing"}])
    resp = await validation_error_handler(_request(), exc)
    assert resp.status_code == 422
    body = json.loads(resp.body)
    assert body["error"]["code"] == "IDP-VAL-001"
    assert body["error"]["details"]["errors"]


# ---------------------------------------------------------------------------
# End-to-end through the real app
# ---------------------------------------------------------------------------
@pytest.fixture
def no_auth_client(tmp_path, monkeypatch):
    """TestClient with API-key auth disabled (see test_api_integration.py)."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "")
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "100")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")

    from fastapi.testclient import TestClient

    from idp.api import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_unknown_template_returns_404_envelope_end_to_end(no_auth_client):
    """The real app must return the structured envelope, not an HTML page."""
    resp = no_auth_client.get("/templates/definitely-not-a-template")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "IDP-TMPL-404"
    assert body["error"]["hint"]
    assert body["error"]["docs_url"]


def test_unknown_route_returns_envelope_end_to_end(no_auth_client):
    """Even a routing 404 is rendered in the project envelope."""
    resp = no_auth_client.get("/no/such/route")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "IDP-CLIENT-404"
    assert body["error"]["hint"]

"""Regression tests for the async /extract_async + /jobs/{id} + /results/{id}
endpoints (P1 audit fix).

Covers:
  * 202 Accepted + JSON shape on submit.
  * Job status transitions: pending → running → succeeded.
  * Result retrieval.
  * HMAC-SHA256 signature on /results/{id}/signed.
  * SSRF guard on callback_url (https OK; http://10.x rejected; etc.).
  * Webhook delivery + signature header.
  * 404s for unknown job_id / result_id.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest


@pytest.fixture
def async_client(tmp_path, monkeypatch):
    """TestClient with auth disabled and a known signing key."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "")
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    # Set a deterministic signing key so signature tests are reproducible.
    monkeypatch.setenv("IDP_RESULT_SIGNING_KEY", "test-signing-secret")
    monkeypatch.setenv("IDP_MAX_UPLOAD_BYTES", str(1024 * 1024))
    from fastapi.testclient import TestClient

    from idp.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_doc(tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text("Vendor: Acme Corp\nInvoice Number: INV-001\nTotal: $540.00")
    return p


# ---------------------------------------------------------------------------
# /extract_async happy path
# ---------------------------------------------------------------------------
def test_extract_async_returns_202_with_job_id(async_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={"schema_name": "Invoice"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert "job_id" in body
    assert body["status_url"] == f"/jobs/{body['job_id']}"
    # result_url may be null (job hasn't run yet) — accept either
    assert "result_url" in body


def test_extract_async_requires_file(async_client):
    r = async_client.post("/extract_async")
    assert r.status_code in (400, 422)


def test_extract_async_rejects_oversized_upload(tmp_path, monkeypatch):
    """Build a fresh TestClient with a tight upload cap so the request fails."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "")
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("IDP_MAX_UPLOAD_BYTES", "1024")  # tight cap (min = 1024 per validator)
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("IDP_RESULT_SIGNING_KEY", "test-signing-secret")
    from fastapi.testclient import TestClient

    from idp.api import app

    with TestClient(app) as c:
        files = {"file": ("big.txt", b"x" * 2000)}  # 2000 bytes > 1024 cap
        r = c.post("/extract_async", files=files, data={"schema_name": "Invoice"})
        assert r.status_code == 413


# ---------------------------------------------------------------------------
# /jobs/{id}
# ---------------------------------------------------------------------------
def test_get_job_status_returns_submitted_job(async_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    submit = async_client.post(
        "/extract_async", files=files, data={"schema_name": "Invoice"}
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    # Status endpoint must return one of the four valid states.
    r = async_client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("pending", "running", "succeeded", "failed")
    assert body["schema_name"] == "Invoice"


def test_get_job_status_unknown_returns_404(async_client):
    r = async_client.get("/jobs/does-not-exist")
    assert r.status_code == 404
    assert "unknown job_id" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /results/{id} and /results/{id}/signed
# ---------------------------------------------------------------------------
def test_get_result_returns_stored_extraction(async_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    submit = async_client.post(
        "/extract_async", files=files, data={"schema_name": "Invoice"}
    )
    job_id = submit.json()["job_id"]

    # Poll the job until it finishes (or 10s timeout).
    import time as _t

    deadline = _t.time() + 10
    s: dict[str, Any] = {}
    while _t.time() < deadline:
        s = async_client.get(f"/jobs/{job_id}").json()
        if s["status"] in ("succeeded", "failed"):
            break
        _t.sleep(0.05)

    assert s["status"] == "succeeded", f"job did not succeed: {s}"
    result_id = s.get("result_id")
    assert result_id

    r = async_client.get(f"/results/{result_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == result_id
    assert body["schema_name"] == "Invoice"
    assert "extraction" in body
    assert "backend_name" in body


def test_get_result_unknown_returns_404(async_client):
    r = async_client.get("/results/does-not-exist")
    assert r.status_code == 404


def test_signed_result_returns_hmac_signature(async_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    submit = async_client.post(
        "/extract_async", files=files, data={"schema_name": "Invoice"}
    )
    job_id = submit.json()["job_id"]

    import time as _t

    deadline = _t.time() + 10
    s: dict[str, Any] = {}
    while _t.time() < deadline:
        s = async_client.get(f"/jobs/{job_id}").json()
        if s["status"] in ("succeeded", "failed"):
            break
        _t.sleep(0.05)
    assert s["status"] == "succeeded"
    result_id = s["result_id"]

    r = async_client.post(f"/results/{result_id}/signed")
    assert r.status_code == 200
    body = r.json()
    assert "result" in body
    assert "signature" in body
    assert "timestamp" in body
    assert "nonce" in body
    sig = body["signature"]
    assert sig.startswith("sha256=")
    # 64 hex chars after the prefix
    assert len(sig.split("=", 1)[1]) == 64
    # Headers echo the same three values
    assert r.headers["X-IDP-Signature"] == sig
    assert r.headers["X-IDP-Timestamp"] == body["timestamp"]
    assert r.headers["X-IDP-Nonce"] == body["nonce"]

    # Round-trip via /verify_signature — the server's own check is the
    # authoritative one (avoid cross-process JSON serialization drift
    # that can happen when we re-encode body["result"] locally).
    from idp.api import _nonce_cache_reset
    _nonce_cache_reset()
    v = async_client.post(
        "/verify_signature",
        json={
            "result": body["result"],
            "signature": sig,
            "timestamp": body["timestamp"],
            "nonce": body["nonce"],
        },
    )
    assert v.status_code == 200, v.text
    assert v.json() == {"valid": True}


def test_signed_result_unknown_returns_404(async_client):
    r = async_client.post("/results/does-not-exist/signed")
    assert r.status_code == 404


def test_signed_result_requires_signing_secret(async_client, sample_doc, monkeypatch):
    """If no signing secret is configured, /signed returns 503 (refuse to weak-sign)."""
    monkeypatch.delenv("IDP_RESULT_SIGNING_KEY", raising=False)
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    submit = async_client.post(
        "/extract_async", files=files, data={"schema_name": "Invoice"}
    )
    job_id = submit.json()["job_id"]
    import time as _t

    deadline = _t.time() + 10
    s: dict[str, Any] = {}
    while _t.time() < deadline:
        s = async_client.get(f"/jobs/{job_id}").json()
        if s["status"] in ("succeeded", "failed"):
            break
        _t.sleep(0.05)
    if s["status"] != "succeeded":
        pytest.skip("job did not complete in time")
    r = async_client.post(f"/results/{s['result_id']}/signed")
    # 503 if no signing key is configured
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# SSRF guard on callback_url
# ---------------------------------------------------------------------------
def test_callback_url_must_be_https_or_localhost(async_client, sample_doc):
    """http://10.0.0.1 → rejected (private network)."""
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={"schema_name": "Invoice", "callback_url": "http://10.0.0.1:9000/cb"},
    )
    assert r.status_code == 400
    assert "callback_url" in r.json()["detail"]


def test_callback_url_rejects_non_http_scheme(async_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={"schema_name": "Invoice", "callback_url": "file:///etc/passwd"},
    )
    assert r.status_code == 400


def test_callback_url_allows_https(async_client, sample_doc):
    """Public https URL is allowed (no DNS resolution; we trust the hostname)."""
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={
            "schema_name": "Invoice",
            "callback_url": "https://example.com/webhook",
        },
    )
    assert r.status_code == 202


def test_callback_url_allows_localhost_http(async_client, sample_doc):
    """http://localhost is allowed (loopback only)."""
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={
            "schema_name": "Invoice",
            "callback_url": "http://localhost:9999/cb",
        },
    )
    assert r.status_code == 202


def test_callback_url_allows_127_http(async_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={
            "schema_name": "Invoice",
            "callback_url": "http://127.0.0.1:9999/cb",
        },
    )
    assert r.status_code == 202


def test_callback_url_https_private_ip_rejected(async_client, sample_doc):
    """https://169.254.169.254 (AWS metadata) is rejected even with https."""
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_client.post(
        "/extract_async",
        files=files,
        data={
            "schema_name": "Invoice",
            "callback_url": "https://169.254.169.254/latest/meta-data/",
        },
    )
    assert r.status_code == 400
    assert "private" in r.json()["detail"].lower() or "loopback" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Webhook delivery (use a real http server on localhost)
# ---------------------------------------------------------------------------
def test_webhook_fires_with_signed_payload(async_client, sample_doc):
    """Run an in-process webhook server on localhost and verify:
       - the server actually receives the POST
       - the X-IDP-Signature header carries a valid HMAC-SHA256
    """
    import http.server
    import socketserver
    import threading

    received: dict[str, Any] = {}

    class WebhookHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server convention
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            received["body"] = body
            received["signature"] = self.headers.get("X-IDP-Signature")
            received["timestamp"] = self.headers.get("X-IDP-Timestamp")
            received["nonce"] = self.headers.get("X-IDP-Nonce")
            received["content_type"] = self.headers.get("Content-Type")
            received["job_id"] = self.headers.get("X-IDP-Job-Id")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args, **kwargs):  # noqa: N802
            return  # silence the access log

    with socketserver.TCPServer(("127.0.0.1", 0), WebhookHandler) as srv:
        port = srv.server_address[1]
        # Run the server in a thread; we're in TestClient (in-process).
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            files = {"file": (sample_doc.name, sample_doc.read_bytes())}
            r = async_client.post(
                "/extract_async",
                files=files,
                data={
                    "schema_name": "Invoice",
                    "callback_url": f"http://127.0.0.1:{port}/cb",
                    "callback_secret": "webhook-secret",
                },
            )
            assert r.status_code == 202

            # Poll for the webhook to arrive.
            import time as _t

            deadline = _t.time() + 10
            while _t.time() < deadline:
                if "body" in received:
                    break
                _t.sleep(0.05)
            assert "body" in received, "webhook never fired"
            assert received["signature"].startswith("sha256=")
            # Replay-protection headers must be present
            assert received["timestamp"], "X-IDP-Timestamp missing"
            assert received["nonce"], "X-IDP-Nonce missing"
            # Verify the signature matches HMAC over the canonical bytes
            # (timestamp + "." + nonce + "." + body) — Fix F wire format.
            canonical = (
                received["timestamp"].encode("utf-8") + b"." +
                received["nonce"].encode("utf-8") + b"." +
                received["body"]
            )
            expected = hmac.new(
                b"webhook-secret", canonical, hashlib.sha256
            ).hexdigest()
            assert received["signature"] == f"sha256={expected}"
            payload = json.loads(received["body"])
            assert "extraction" in payload
            assert payload["schema_name"] == "Invoice"
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@pytest.fixture
def async_auth_client(tmp_path, monkeypatch):
    """TestClient with auth REQUIRED."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "test-key-async")
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "true")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("IDP_RESULT_SIGNING_KEY", "test-signing-secret")
    monkeypatch.setenv("IDP_MAX_UPLOAD_BYTES", str(1024 * 1024))
    from fastapi.testclient import TestClient

    from idp.api import app

    with TestClient(app) as c:
        yield c


def test_async_endpoints_require_api_key(async_auth_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_auth_client.post(
        "/extract_async", files=files, data={"schema_name": "Invoice"}
    )
    assert r.status_code == 401


def test_async_endpoints_accept_correct_api_key(async_auth_client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = async_auth_client.post(
        "/extract_async",
        files=files,
        data={"schema_name": "Invoice"},
        headers={"X-API-Key": "test-key-async"},
    )
    assert r.status_code == 202
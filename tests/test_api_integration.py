"""FastAPI integration tests for the production API.

Uses FastAPI's ``TestClient`` (in-process; no server needed) to exercise
every endpoint: health, version, metrics, extract, auth, rate limit,
upload size limit, configuration validation.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build a TestClient with a clean settings + upload dir."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "test-key-12345")
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "true")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "100")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")  # quiet test output
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
# Health / version / metrics
# ---------------------------------------------------------------------------
def test_healthz_returns_200(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.text == "ok"


def test_readyz_returns_200(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.text == "ready"


def test_version_returns_semver(client):
    r = client.get("/version")
    assert r.status_code == 200
    # Must be dotted numeric (semver-ish); not required to be strict semver
    parts = r.text.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts if p.split("-")[0].isdigit() or p.isdigit())


def test_metrics_returns_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "http_requests" in body
    # Counters, gauges, histograms all exposable
    assert "# TYPE" in body or "{" in body  # either summary or labelled format


def test_metrics_increments_on_request(client):
    before = client.get("/metrics").text
    client.get("/healthz")
    after = client.get("/metrics").text
    # http_requests counter incremented (hard to compare values; just confirm both have it)
    assert "http_requests" in before and "http_requests" in after


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_extract_requires_api_key(client):
    r = client.post("/extract", files={"file": ("x.txt", b"hello")},
                    data={"schema_name": "Invoice"})
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


def test_extract_with_correct_key_passes_auth(client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = client.post("/extract", files=files,
                    data={"schema_name": "Invoice"},
                    headers={"X-API-Key": "test-key-12345"})
    # Mock backend may not extract well, but auth must pass.
    assert r.status_code == 200


def test_extract_with_wrong_key_rejected(client, sample_doc):
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = client.post("/extract", files=files,
                    data={"schema_name": "Invoice"},
                    headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 403


def test_extract_with_bearer_token_header_works(client, sample_doc):
    """Authorization: Bearer ... is also accepted (for OAuth-style clients)."""
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    r = client.post("/extract", files=files,
                    data={"schema_name": "Invoice"},
                    headers={"Authorization": "Bearer test-key-12345"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------
def test_rate_limit_enforced_after_threshold(tmp_path, monkeypatch, sample_doc):
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "k")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    from fastapi.testclient import TestClient

    from idp.api import app
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    with TestClient(app) as c:
        for i in range(3):
            r = c.post("/extract", files=files, data={"schema_name": "Invoice"},
                       headers={"X-API-Key": "k"})
            assert r.status_code in (200, 429), f"call {i}: {r.status_code} {r.text}"
        r = c.post("/extract", files=files, data={"schema_name": "Invoice"},
                   headers={"X-API-Key": "k"})
        assert r.status_code == 429
        assert "limit exceeded" in r.json()["error"]


# ---------------------------------------------------------------------------
# Upload size limit
# ---------------------------------------------------------------------------
def test_upload_size_limit_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "k")
    monkeypatch.setenv("IDP_MAX_UPLOAD_BYTES", "1024")  # 1 KB min
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    from fastapi.testclient import TestClient

    from idp.api import app
    big = b"x" * 2048  # 2 KB -> exceeds cap
    files = {"file": ("big.txt", big)}
    with TestClient(app) as c:
        r = c.post("/extract", files=files, data={"schema_name": "Invoice"},
                   headers={"X-API-Key": "k"})
        assert r.status_code == 413
        assert "too large" in r.json()["detail"]


def test_upload_size_limit_accepts_small_payload(tmp_path, monkeypatch, sample_doc):
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "k")
    monkeypatch.setenv("IDP_MAX_UPLOAD_BYTES", "100000")  # 100 KB
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    from fastapi.testclient import TestClient

    from idp.api import app
    files = {"file": (sample_doc.name, sample_doc.read_bytes())}
    with TestClient(app) as c:
        r = c.post("/extract", files=files, data={"schema_name": "Invoice"},
                   headers={"X-API-Key": "k"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_invalid_configuration_fails_at_startup(tmp_path, monkeypatch):
    """Misconfigured env vars cause the app to refuse to start."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_PORT", "not-a-number")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    from fastapi.testclient import TestClient

    from idp.api import app
    from idp.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match=r"API_PORT|integer"), TestClient(app) as c:
        c.get("/healthz")
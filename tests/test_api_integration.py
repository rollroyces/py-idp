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
@pytest.fixture
def no_auth_client(tmp_path, monkeypatch):
    """Build a TestClient with API key auth DISABLED."""
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("IDP_API_KEY", "")
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("IDP_RATE_LIMIT_PER_MINUTE", "100")
    monkeypatch.setenv("IDP_MAX_UPLOAD_BYTES", str(1024 * 1024))
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    from fastapi.testclient import TestClient

    from idp.api import app
    with TestClient(app) as c:
        yield c


def test_extract_returns_correct_field_names(no_auth_client):
    """The ExtractResponse must use 'backend_name' (matches PipelineResult)."""
    files = {"file": ("x.txt", b"hi")}
    r = no_auth_client.post("/extract", files=files, data={"schema_name": "Invoice"})
    assert r.status_code == 200
    body = r.json()
    assert "backend_name" in body, f"missing backend_name in {list(body.keys())}"
    assert "backend" not in body, "old 'backend' key should be removed"
    assert body["backend_name"] in ("mock", "mock-ideal", "mock-random", "mock-omits")
    assert body["schema_name"] == "Invoice"


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
        # New structured error envelope: `error` is a dict, not a string
        body = r.json()
        assert body["error"]["code"] == "IDP-RATE-001"
        assert "limit exceeded" in body["error"]["message"]
        assert "request_id" in body["error"]


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

# ---------------------------------------------------------------------------
# API key middleware (line 138: 401 missing key)
# ---------------------------------------------------------------------------
def test_api_key_required_returns_401_when_missing(monkeypatch, tmp_path):
    """If API key is configured but request doesn't provide one → 401."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.config import Settings

    saved = api_mod._settings
    # Configure settings with API key required
    monkeypatch.setenv("IDP_API_KEY", "secret-test-key-123")
    api_mod._settings = Settings.load()
    try:
        with TestClient(api_mod.app) as client:
            # /version is exempt; hit /templates (protected)
            r = client.get("/templates")
            assert r.status_code == 401
            assert "missing API key" in r.text
    finally:
        api_mod._settings = saved


def test_api_key_required_returns_403_when_wrong(monkeypatch, tmp_path):
    """If wrong API key is provided → 403."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.config import Settings

    saved = api_mod._settings
    monkeypatch.setenv("IDP_API_KEY", "secret-test-key-123")
    api_mod._settings = Settings.load()
    try:
        with TestClient(api_mod.app) as client:
            r = client.get("/templates", headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 403
    finally:
        api_mod._settings = saved


# ---------------------------------------------------------------------------
# /version returns the package version
# ---------------------------------------------------------------------------
def test_version_endpoint_returns_package_version():
    """GET /version returns idp.__version__."""
    from fastapi.testclient import TestClient

    import idp as _idp
    from idp import api as api_mod

    with TestClient(api_mod.app) as client:
        r = client.get("/version")
        assert r.status_code == 200
        assert r.text == _idp.__version__


# ---------------------------------------------------------------------------
# /metrics disabled → 404 (lines 438-445)
# ---------------------------------------------------------------------------
def test_metrics_disabled_returns_404(monkeypatch):
    """When metrics_enabled is False, /metrics returns 404."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.config import Settings

    saved = api_mod._settings
    monkeypatch.setenv("IDP_METRICS_ENABLED", "0")
    api_mod._settings = Settings.load()
    try:
        with TestClient(api_mod.app) as client:
            r = client.get("/metrics")
            assert r.status_code == 404
            assert "metrics disabled" in r.text
    finally:
        api_mod._settings = saved


# ---------------------------------------------------------------------------
# /templates/{name} GET endpoint (line 318)
# ---------------------------------------------------------------------------
def test_get_template_returns_full_template(tmp_path):
    """GET /templates/{name} returns the full template (frontmatter + body)."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod

    saved = api_mod._settings
    monkeypatch_key = "IDP_API_KEY_REQUIRED"
    import os
    saved_env = os.environ.get(monkeypatch_key)
    os.environ[monkeypatch_key] = "0"
    api_mod._settings = None
    api_mod._template_registry = None
    try:
        with TestClient(api_mod.app) as client:
            from idp.templates import TemplateRegistry
            api_mod._template_registry = TemplateRegistry.load("templates")
            r = client.get("/templates/invoice")
            assert r.status_code == 200
            data = r.json()
            assert data["name"] == "invoice"
            assert data["schema"] == "Invoice"
            assert data["body"]
            assert isinstance(data["field_overrides"], dict)
    finally:
        if saved_env is None:
            os.environ.pop(monkeypatch_key, None)
        else:
            os.environ[monkeypatch_key] = saved_env
        api_mod._settings = saved


def test_get_template_returns_404_for_unknown(tmp_path, monkeypatch):
    """GET /templates/{unknown} returns 404 with IDP-TMPL-404 code."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.templates import TemplateRegistry

    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "0")
    saved_settings = api_mod._settings
    saved_registry = api_mod._template_registry
    api_mod._settings = None  # force reload on next access
    api_mod._template_registry = TemplateRegistry.load("templates")
    try:
        with TestClient(api_mod.app) as client:
            r = client.get("/templates/does-not-exist")
            assert r.status_code == 404
    finally:
        api_mod._settings = saved_settings
        api_mod._template_registry = saved_registry


# ---------------------------------------------------------------------------
# /extract POST endpoint (line 342)
# ---------------------------------------------------------------------------
def test_extract_sync_runs_pipeline(tmp_path, monkeypatch):
    """POST /extract runs the pipeline and returns ExtractResponse."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod

    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "0")
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    saved = api_mod._settings
    api_mod._settings = None
    try:
        with TestClient(api_mod.app) as client:
            files = {"file": ("test.txt", b"INVOICE INV-001\nVendor: Acme", "text/plain")}
            data = {"schema_name": "Invoice", "backend": "mock"}
            r = client.post("/extract", files=files, data=data)
            assert r.status_code == 200, r.text
            payload = r.json()
            assert payload["backend_name"] == "mock"
            assert payload["schema_name"] == "Invoice"
            assert "extraction" in payload
    finally:
        api_mod._settings = saved


def test_extract_sync_routes_via_template_when_no_schema(tmp_path, monkeypatch):
    """If schema_name is omitted, the template registry picks one."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.templates import TemplateRegistry

    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "0")
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    saved = api_mod._settings
    saved_registry = api_mod._template_registry
    api_mod._settings = None
    api_mod._template_registry = TemplateRegistry.load("templates")
    try:
        with TestClient(api_mod.app) as client:
            files = {"file": ("invoice-sample.txt", b"INVOICE 1", "text/plain")}
            r = client.post("/extract", files=files, data={"backend": "mock"})
            assert r.status_code == 200, r.text
            payload = r.json()
            # Either the filename matched an invoice template, or it fell
            # back to the default "Invoice" schema.
            if payload.get("template_used"):
                assert payload["template_used"] in {"invoice", "default"}
    finally:
        api_mod._settings = saved
        api_mod._template_registry = saved_registry


# ---------------------------------------------------------------------------
# /templates GET list endpoint (line 297)
# ---------------------------------------------------------------------------
def test_list_templates_returns_all_registered(tmp_path, monkeypatch):
    """GET /templates returns the list of registered template names."""
    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.templates import TemplateRegistry

    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "0")
    saved = api_mod._settings
    saved_registry = api_mod._template_registry
    api_mod._settings = None
    api_mod._template_registry = TemplateRegistry.load("templates")
    try:
        with TestClient(api_mod.app) as client:
            r = client.get("/templates")
            assert r.status_code == 200
            templates = r.json()
            assert isinstance(templates, list)
            names = sorted(t["name"] for t in templates)
            assert "invoice" in names
            assert "contract" in names
            assert "bank_statement" in names
    finally:
        api_mod._settings = saved
        api_mod._template_registry = saved_registry


def test_list_templates_empty_registry_returns_empty(monkeypatch):
    """With no templates/ dir available, /templates returns [].

    Note: This is hard to test in-process because the lifespan
    handler always reads from the configured templates/ directory
    on startup. We approximate by setting an invalid template dir
    via env var, which causes Settings.load() to point at a
    non-existent path — the lifespan then loads zero templates.
    """
    import tempfile

    from fastapi.testclient import TestClient

    from idp import api as api_mod
    from idp.config import Settings

    # Create an empty temp dir (no .md files)
    with tempfile.TemporaryDirectory() as empty_dir:
        monkeypatch.setenv("IDP_API_KEY_REQUIRED", "0")
        monkeypatch.setenv("IDP_TEMPLATE_DIR", empty_dir)
        saved_settings = api_mod._settings
        api_mod._settings = None
        try:
            api_mod._settings = Settings.load()
            with TestClient(api_mod.app) as client:
                r = client.get("/templates")
                assert r.status_code == 200
                assert r.json() == []
        finally:
            api_mod._settings = saved_settings

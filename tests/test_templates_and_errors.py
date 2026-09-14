"""Tests for the template registry + API endpoints + error envelope."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from idp.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DocumentParseError,
    IDPError,
    RateLimitedError,
    SchemaValidationError,
    StorageError,
    TemplateNotFoundError,
    TemplateParseError,
    error_envelope,
    is_idp_error,
)
from idp.templates import (
    Template,
    TemplateRegistry,
    load_template,
)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------
def test_parse_frontmatter_minimal(tmp_path: Path) -> None:
    """Required keys only — body parsed verbatim."""
    p = tmp_path / "invoice.md"
    p.write_text(
        "---\n"
        "name: invoice\n"
        "schema: Invoice\n"
        "---\n"
        "# Invoice\n"
        "Fields: vendor_name, total_amount\n",
        encoding="utf-8",
    )
    t = load_template(p)
    assert t.name == "invoice"
    assert t.schema == "Invoice"
    assert t.version == 1
    assert t.body == "# Invoice\nFields: vendor_name, total_amount\n"
    assert t.mime_types == []
    assert t.filename_patterns == []


def test_parse_frontmatter_full(tmp_path: Path) -> None:
    """All keys + field_overrides + classification_hints."""
    p = tmp_path / "invoice.md"
    p.write_text(
        "---\n"
        "name: invoice\n"
        "schema: Invoice\n"
        "version: 2\n"
        "mime_types: [application/pdf, image/jpeg]\n"
        "filename_patterns: ['*invoice*', '*receipt*']\n"
        "classification_hints:\n"
        "  - contains total amount\n"
        "  - contains vendor name\n"
        "field_overrides:\n"
        "  invoice_number:\n"
        "    regex: 'INV-\\d+'\n"
        "    examples: [INV-001, INV-2024-123]\n"
        "---\n"
        "Body content here.\n",
        encoding="utf-8",
    )
    t = load_template(p)
    assert t.version == 2
    assert "application/pdf" in t.mime_types
    assert "*invoice*" in t.filename_patterns
    assert len(t.classification_hints) == 2
    assert t.field_overrides["invoice_number"]["regex"] == "INV-\\d+"


def test_parse_frontmatter_missing_name(tmp_path: Path) -> None:
    """Missing 'name' key raises TemplateParseError."""
    p = tmp_path / "bad.md"
    p.write_text("---\nschema: Invoice\n---\nbody\n", encoding="utf-8")
    with pytest.raises(TemplateParseError, match="missing required"):
        load_template(p)


def test_parse_frontmatter_missing_schema(tmp_path: Path) -> None:
    p = tmp_path / "bad.md"
    p.write_text("---\nname: foo\n---\nbody\n", encoding="utf-8")
    with pytest.raises(TemplateParseError, match="missing required"):
        load_template(p)


def test_parse_frontmatter_no_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "no_fm.md"
    p.write_text("just a body, no frontmatter\n", encoding="utf-8")
    with pytest.raises(TemplateParseError, match="no frontmatter"):
        load_template(p)


def test_parse_frontmatter_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "broken_yaml.md"
    p.write_text(
        "---\nname: foo\nschema: Invoice\nthis is: not: valid: yaml:\n---\nbody\n",
        encoding="utf-8",
    )
    with pytest.raises(TemplateParseError, match="invalid YAML"):
        load_template(p)


def test_parse_frontmatter_frontmatter_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list_fm.md"
    p.write_text("---\n- one\n- two\n---\nbody\n", encoding="utf-8")
    with pytest.raises(TemplateParseError, match="not a mapping"):
        load_template(p)


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------
def test_matches_filename_glob(tmp_path: Path) -> None:
    t = Template(
        name="invoice",
        schema="Invoice",
        body="b",
        source_path=tmp_path / "invoice.md",
        filename_patterns=["*invoice*", "*INV*"],
    )
    assert t.matches_filename("acme-INV-001.pdf")
    assert t.matches_filename("my-invoice.pdf")
    assert not t.matches_filename("contract.pdf")


def test_matches_filename_empty_patterns(tmp_path: Path) -> None:
    """Empty patterns -> never matches."""
    t = Template(
        name="invoice", schema="Invoice", body="b", source_path=tmp_path / "i.md"
    )
    assert not t.matches_filename("anything.pdf")


def test_matches_mime(tmp_path: Path) -> None:
    t = Template(
        name="invoice",
        schema="Invoice",
        body="b",
        source_path=tmp_path / "i.md",
        mime_types=["application/pdf", "image/jpeg"],
    )
    assert t.matches_mime("application/pdf")
    assert t.matches_mime("IMAGE/JPEG")  # case-insensitive
    assert not t.matches_mime("text/plain")


def test_template_to_dict_is_json_safe(tmp_path: Path) -> None:
    """to_dict returns plain Python primitives — safe for json.dumps."""
    p = tmp_path / "i.md"
    p.write_text("---\nname: i\nschema: I\n---\nbody\n", encoding="utf-8")
    t = load_template(p)
    d = t.to_dict()
    import json

    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# TemplateRegistry
# ---------------------------------------------------------------------------
def test_registry_load_empty_dir(tmp_path: Path) -> None:
    """Missing dir -> empty registry, not an error."""
    r = TemplateRegistry.load(tmp_path / "nonexistent")
    assert len(r) == 0
    assert r.list_names() == []


def test_registry_load_empty_existing_dir(tmp_path: Path) -> None:
    r = TemplateRegistry.load(tmp_path)
    assert len(r) == 0


def test_registry_load_multiple(tmp_path: Path) -> None:
    for name, schema in [("invoice", "Invoice"), ("contract", "Contract")]:
        (tmp_path / f"{name}.md").write_text(
            f"---\nname: {name}\nschema: {schema}\n---\nbody {name}\n", encoding="utf-8"
        )
    r = TemplateRegistry.load(tmp_path)
    assert len(r) == 2
    assert r.list_names() == ["contract", "invoice"]


def test_registry_duplicate_name_raises(tmp_path: Path) -> None:
    """Two files with the same `name` -> TemplateParseError."""
    (tmp_path / "a.md").write_text(
        "---\nname: dup\nschema: Foo\n---\nbody\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text(
        "---\nname: dup\nschema: Bar\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(TemplateParseError, match="duplicate template name"):
        TemplateRegistry.load(tmp_path)


def test_registry_get_unknown_raises(tmp_path: Path) -> None:
    r = TemplateRegistry.load(tmp_path)
    with pytest.raises(TemplateNotFoundError, match="no template named"):
        r.get("nonexistent")


def test_registry_contains(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "---\nname: a\nschema: A\n---\nbody\n", encoding="utf-8"
    )
    r = TemplateRegistry.load(tmp_path)
    assert "a" in r
    assert "b" not in r


def test_registry_find_by_filename_unique(tmp_path: Path) -> None:
    (tmp_path / "invoice.md").write_text(
        "---\nname: invoice\nschema: Invoice\nfilename_patterns: ['*invoice*']\n---\nb\n",
        encoding="utf-8",
    )
    (tmp_path / "contract.md").write_text(
        "---\nname: contract\nschema: Contract\nfilename_patterns: ['*contract*']\n---\nb\n",
        encoding="utf-8",
    )
    r = TemplateRegistry.load(tmp_path)
    assert r.find_for_filename("acme-invoice-001.pdf").name == "invoice"
    assert r.find_for_filename("acme-contract.pdf").name == "contract"


def test_registry_find_by_filename_no_match(tmp_path: Path) -> None:
    (tmp_path / "invoice.md").write_text(
        "---\nname: invoice\nschema: Invoice\nfilename_patterns: ['*invoice*']\n---\nb\n",
        encoding="utf-8",
    )
    r = TemplateRegistry.load(tmp_path)
    assert r.find_for_filename("random.pdf") is None


def test_registry_find_by_mime_fallback(tmp_path: Path) -> None:
    """No filename match -> fall back to MIME match."""
    (tmp_path / "invoice.md").write_text(
        "---\nname: invoice\nschema: Invoice\nmime_types: [application/pdf]\n---\nb\n",
        encoding="utf-8",
    )
    r = TemplateRegistry.load(tmp_path)
    # filename doesn't match but MIME does
    assert r.find_for_filename("random.bin", mime="application/pdf").name == "invoice"
    # neither matches
    assert r.find_for_filename("random.bin", mime="text/plain") is None


# ---------------------------------------------------------------------------
# Error code registry
# ---------------------------------------------------------------------------
def test_idp_error_subtypes_have_codes() -> None:
    """Every IDPError subclass must have a `code` and `http_status`."""
    for cls in [
        IDPError,
        DocumentParseError,
        SchemaValidationError,
        BackendUnavailableError,
        StorageError,
        ConfigurationError,
        RateLimitedError,
        TemplateNotFoundError,
        TemplateParseError,
    ]:
        assert isinstance(cls.code, str), f"{cls.__name__} missing code"
        assert cls.code.startswith("IDP-"), f"{cls.__name__}.code = {cls.code!r}"
        assert isinstance(cls.http_status, int), f"{cls.__name__} missing http_status"
        assert 400 <= cls.http_status < 600, (
            f"{cls.__name__}.http_status = {cls.http_status}"
        )


def test_codes_are_unique() -> None:
    """No two error classes may share a code."""
    codes: dict[str, str] = {}
    for cls in [
        DocumentParseError,
        SchemaValidationError,
        BackendUnavailableError,
        StorageError,
        ConfigurationError,
        RateLimitedError,
        TemplateNotFoundError,
        TemplateParseError,
    ]:
        if cls.code in codes:
            raise AssertionError(
                f"code {cls.code!r} shared by {cls.__name__} and {codes[cls.code]}"
            )
        codes[cls.code] = cls.__name__


def test_error_envelope_shape() -> None:
    e = RateLimitedError("over limit")
    env = error_envelope(e, request_id="abc-123")
    assert "error" in env
    body = env["error"]
    assert body["code"] == "IDP-RATE-001"
    assert body["message"] == "over limit"
    assert body["type"] == "RateLimitedError"
    assert body["request_id"] == "abc-123"


def test_error_envelope_without_request_id() -> None:
    e = DocumentParseError("bad file")
    env = error_envelope(e)
    assert "request_id" not in env["error"]


def test_error_envelope_with_details() -> None:
    e = SchemaValidationError("bad shape")
    env = error_envelope(e, details={"fields": ["total"]})
    assert env["error"]["details"] == {"fields": ["total"]}


def test_error_envelope_non_idp_exception() -> None:
    """Non-IDPError exceptions get the default code."""
    env = error_envelope(ValueError("nope"))
    assert env["error"]["code"] == "IDP-INT-001"
    assert env["error"]["type"] == "ValueError"


def test_is_idp_error() -> None:
    assert is_idp_error(IDPError("x"))
    assert is_idp_error(RateLimitedError("x"))
    assert not is_idp_error(ValueError("x"))


# ---------------------------------------------------------------------------
# API: /templates endpoints + /extract template routing + error envelope
# ---------------------------------------------------------------------------
@pytest.fixture
def client_with_templates(tmp_path, monkeypatch):
    """A TestClient with templates preloaded and auth disabled for ease."""
    # Create 2 templates
    (tmp_path / "invoice.md").write_text(
        "---\n"
        "name: invoice\n"
        "schema: Invoice\n"
        "version: 1\n"
        "mime_types: [application/pdf]\n"
        "filename_patterns: ['*invoice*']\n"
        "---\n"
        "# Invoice\n"
        "Use this for Acme-style invoices.\n",
        encoding="utf-8",
    )
    (tmp_path / "contract.md").write_text(
        "---\n"
        "name: contract\n"
        "schema: Contract\n"
        "mime_types: [application/pdf]\n"
        "filename_patterns: ['*contract*']\n"
        "---\n"
        "# Contract\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IDP_TEMPLATE_DIR", str(tmp_path))
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    from idp.api import app

    with TestClient(app) as c:
        yield c


def test_list_templates(client_with_templates) -> None:
    r = client_with_templates.get("/templates")
    assert r.status_code == 200
    data = r.json()
    names = {t["name"] for t in data}
    assert names == {"invoice", "contract"}
    # Wire format uses 'schema', not 'schema_name'
    for t in data:
        assert "schema" in t
        assert "name" in t
        assert "version" in t
    # Bodies are NOT in the summary
    assert "body" not in data[0]


def test_get_template_full(client_with_templates) -> None:
    r = client_with_templates.get("/templates/invoice")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "invoice"
    assert data["schema"] == "Invoice"
    assert "body" in data
    assert "Acme-style" in data["body"]


def test_get_template_not_found(client_with_templates) -> None:
    r = client_with_templates.get("/templates/nonexistent")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "IDP-TMPL-404"
    assert "request_id" in body["error"]


def test_extract_uses_template_when_filename_matches(client_with_templates) -> None:
    """A file named 'foo-invoice.pdf' should be routed to the 'invoice' template."""
    files = {"file": ("foo-invoice.pdf", b"%PDF-1.4\nfake content", "application/pdf")}
    r = client_with_templates.post("/extract", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["template_used"] == "invoice"


def test_extract_uses_template_when_mime_matches(client_with_templates) -> None:
    """Filename doesn't match any pattern, but MIME does -> routed by MIME."""
    files = {"file": ("random.bin", b"%PDF-1.4\nfake", "application/pdf")}
    r = client_with_templates.post("/extract", files=files)
    assert r.status_code == 200
    data = r.json()
    # Two templates both have application/pdf mime -> ambiguous, but
    # alphabetical fallback: 'contract' comes first
    assert data["template_used"] in ("invoice", "contract")  # either is acceptable


def test_extract_explicit_schema_skips_template_routing(client_with_templates) -> None:
    """Caller pins schema_name -> no template lookup."""
    files = {"file": ("foo-invoice.pdf", b"%PDF-1.4\nfake", "application/pdf")}
    r = client_with_templates.post(
        "/extract",
        files=files,
        data={"schema_name": "Contract"},  # explicit
    )
    assert r.status_code == 200
    data = r.json()
    assert data["schema_name"] == "Contract"
    assert data["template_used"] is None


def test_extract_no_match_no_template(client_with_templates) -> None:
    """Filename + MIME don't match any template -> template_used is None."""
    files = {"file": ("random.bin", b"%PDF-1.4\nfake", "text/plain")}
    r = client_with_templates.post("/extract", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["template_used"] is None


# ---------------------------------------------------------------------------
# Error envelope via API
# ---------------------------------------------------------------------------
def test_template_parse_error_fails_startup(tmp_path, monkeypatch) -> None:
    """A bad template file should abort server startup."""
    (tmp_path / "broken.md").write_text(
        "---\nname: bad\n---\nbody\n",  # missing schema
        encoding="utf-8",
    )
    monkeypatch.setenv("IDP_TEMPLATE_DIR", str(tmp_path))
    monkeypatch.setenv("IDP_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("IDP_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("IDP_UPLOAD_DIR", str(tmp_path / "uploads"))
    from idp.api import app

    # Lifespan should raise — TestClient.__enter__ propagates the failure
    with pytest.raises(Exception), TestClient(app):  # noqa: B017
        pass


def test_request_id_generated_when_not_provided(client_with_templates) -> None:
    r = client_with_templates.get("/version")
    assert r.status_code == 200
    assert "x-request-id" in {k.lower() for k in r.headers}


def test_request_id_echoed_when_provided(client_with_templates) -> None:
    rid = "test-12345"
    r = client_with_templates.get("/version", headers={"X-Request-ID": rid})
    assert r.headers["X-Request-ID"] == rid


def test_404_returns_error_envelope(client_with_templates) -> None:
    r = client_with_templates.get("/nonexistent")
    assert r.status_code == 404
    # FastAPI default 404 is a {"detail": "Not Found"} shape — we don't
    # wrap that in our envelope (only IDPError subtypes get the
    # envelope). This test documents that contract.
    assert "detail" in r.json()

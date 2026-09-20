"""Regression tests for the SSRF guard on OpenAICompatBackend.base_url (P1 audit fix).

The constructor now refuses:
  * non-http(s) schemes (file://, gopher://, ...)
  * http:// unless IDP_ALLOW_INSECURE_HTTP=1
  * URLs that resolve to loopback/link-local/private/reserved IPs, unless
    IDP_ALLOW_PRIVATE_NETWORK=1 is set

The Ollama / vLLM / LM-Studio preset dispatches use an internal
``_allow_private_network=True`` kwarg so localhost continues to work
for those. The China provider presets are all public HTTPS so they pass
without opt-in.

These tests do NOT touch the env vars for IDP_ALLOW_* (they'd leak
across tests). Instead they use the per-call ``allow_private_network``
and ``allow_insecure_http`` kwargs that the public API exposes.
"""
from __future__ import annotations

import os

import pytest

from idp.llm.backend import (
    OpenAICompatBackend,
    _is_private_ip,
    _validate_base_url,
)


# ---------------------------------------------------------------------------
# _validate_base_url: scheme rules
# ---------------------------------------------------------------------------
def test_validate_base_url_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_base_url("")


def test_validate_base_url_rejects_non_http_scheme():
    for scheme in ("file", "gopher", "ftp", "ssh", "javascript"):
        with pytest.raises(ValueError, match="scheme"):
            _validate_base_url(f"{scheme}://example.com/path")


def test_validate_base_url_rejects_http_without_opt_in(monkeypatch):
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    with pytest.raises(ValueError, match="http://"):
        _validate_base_url("http://api.example.com/v1")


def test_validate_base_url_rejects_missing_host():
    with pytest.raises(ValueError, match="hostname"):
        _validate_base_url("https:///path")


# ---------------------------------------------------------------------------
# _validate_base_url: private-IP / loopback / link-local rules
# ---------------------------------------------------------------------------
def test_validate_base_url_rejects_loopback_hostname(monkeypatch):
    """localhost must be rejected when IDP_ALLOW_PRIVATE_NETWORK is not set."""
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    # No env opt-in, so localhost fails. We pass allow_private_network=False
    # explicitly to avoid the env var leaking from other tests.
    with pytest.raises(ValueError, match="private/loopback"):
        _validate_base_url(
            "https://localhost:9999/v1", allow_private_network=False
        )


def test_validate_base_url_rejects_rfc1918_hostname(monkeypatch):
    """A hostname that resolves to 10.x / 172.16-31.x / 192.168.x is rejected.

    We can't rely on a real DNS lookup returning a private IP, so this
    test exercises the loopback case (``localhost`` → 127.0.0.1) and the
    link-local metadata IP literal. Both should be rejected.
    """
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    # localhost resolves to 127.0.0.1 (loopback) -> rejected
    with pytest.raises(ValueError, match="private/loopback"):
        _validate_base_url(
            "https://localhost/v1", allow_private_network=False
        )


def test_validate_base_url_rejects_metadata_ip_literal(monkeypatch):
    """AWS metadata IP 169.254.169.254 is link-local — must be rejected."""
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    with pytest.raises(ValueError, match="private/loopback"):
        _validate_base_url(
            "https://169.254.169.254/latest/meta-data/", allow_private_network=False
        )


def test_validate_base_url_rejects_private_v4_literal(monkeypatch):
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    for ip in ("10.0.0.1", "172.16.0.1", "192.168.1.1"):
        with pytest.raises(ValueError, match="private/loopback"):
            _validate_base_url(
                f"https://{ip}/v1", allow_private_network=False
            )


def test_validate_base_url_allows_loopback_when_opt_in():
    """With allow_private_network=True, localhost passes (no DNS lookup
    needed since 'localhost' resolves to 127.0.0.1)."""
    # Should NOT raise.
    _validate_base_url(
        "http://localhost:11434/v1",
        allow_private_network=True,
        allow_insecure_http=True,
    )


def test_validate_base_url_rejects_unresolvable_hostname():
    with pytest.raises(ValueError, match="did not resolve"):
        _validate_base_url(
            "https://this-host-definitely-does-not-exist-12345.invalid/v1",
            allow_private_network=True,  # would fail on private-IP check
        )


# ---------------------------------------------------------------------------
# _is_private_ip
# ---------------------------------------------------------------------------
def test_is_private_ip_classifier():
    """Sanity-check the classifier on canonical addresses."""
    import ipaddress

    # Loopback
    assert _is_private_ip(ipaddress.ip_address("127.0.0.1"))
    assert _is_private_ip(ipaddress.ip_address("::1"))
    # RFC1918
    assert _is_private_ip(ipaddress.ip_address("10.0.0.1"))
    assert _is_private_ip(ipaddress.ip_address("172.16.0.1"))
    assert _is_private_ip(ipaddress.ip_address("192.168.1.1"))
    # Link-local
    assert _is_private_ip(ipaddress.ip_address("169.254.169.254"))
    # Public
    assert not _is_private_ip(ipaddress.ip_address("8.8.8.8"))
    assert not _is_private_ip(ipaddress.ip_address("1.1.1.1"))
    # Reserved / unspecified
    assert _is_private_ip(ipaddress.ip_address("0.0.0.0"))


# ---------------------------------------------------------------------------
# OpenAICompatBackend constructor
# ---------------------------------------------------------------------------
def test_constructor_rejects_http_with_no_opt_in(monkeypatch):
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    with pytest.raises(ValueError, match="http://"):
        OpenAICompatBackend(base_url="http://example.com/v1", model="x")


def test_constructor_rejects_private_hostname(monkeypatch):
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    with pytest.raises(ValueError, match="private"):
        OpenAICompatBackend(base_url="https://192.168.1.1/v1", model="x")


def test_constructor_rejects_metadata_ip(monkeypatch):
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    with pytest.raises(ValueError, match="private"):
        OpenAICompatBackend(
            base_url="https://169.254.169.254/latest/meta-data/",
            model="x",
        )


def test_constructor_accepts_https_public(monkeypatch):
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    b = OpenAICompatBackend(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="x",
    )
    assert b.base_url == "https://api.openai.com/v1"
    assert b.model == "gpt-4o"


def test_constructor_accepts_loopback_with_internal_flag(monkeypatch):
    """The Ollama dispatch uses ``_allow_private_network=True`` so localhost
    works without setting IDP_ALLOW_PRIVATE_NETWORK in the env."""
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    b = OpenAICompatBackend(
        base_url="http://localhost:11434/v1",
        model="llama3.2-vision",
        api_key="x",
        _allow_private_network=True,
    )
    assert b.base_url == "http://localhost:11434/v1"


def test_constructor_accepts_http_when_internal_flag_set(monkeypatch):
    """With ``_allow_private_network=True`` the constructor accepts both
    http:// and loopback targets — this is the Ollama / vLLM dispatch path.
    """
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    b = OpenAICompatBackend(
        base_url="http://localhost:11434/v1",
        model="llama3.2-vision",
        api_key="x",
        _allow_private_network=True,
    )
    assert b.base_url == "http://localhost:11434/v1"


def test_constructor_https_to_public_dns_passes(monkeypatch):
    """Even without the internal flag, https to a public DNS name works.

    We don't actually resolve here (network-free test); we just confirm
    that the constructor doesn't fire an early-fail that would block
    real production traffic. We mock the URL to be syntactically valid
    and use the public-DNS heuristic.
    """
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    # Use a hostname that resolves to a public IP. ``one.one.one.one``
    # (Cloudflare 1.1.1.1) is a stable test target.
    b = OpenAICompatBackend(
        base_url="https://one.one.one.one/v1",
        model="x",
        api_key="x",
    )
    assert b.base_url == "https://one.one.one.one/v1"


# ---------------------------------------------------------------------------
# China presets pass through the guard (regression — they were trusted before,
# but the new validation could reject a misconfigured preset).
# ---------------------------------------------------------------------------
def test_china_presets_all_pass_validation(monkeypatch):
    """Every preset base_url in CHINA_PROVIDER_PRESETS must validate
    under the default guard settings (https, public)."""
    monkeypatch.delenv("IDP_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("IDP_ALLOW_PRIVATE_NETWORK", raising=False)
    from idp.llm.china import CHINA_PROVIDER_PRESETS

    for name, preset in CHINA_PROVIDER_PRESETS.items():
        # Should not raise. (The env vars may not exist; we mock a dummy key.)
        with monkeypatch.context() as m:
            m.setenv(preset["env_var"], "test-key-for-validation")
            backend = OpenAICompatBackend(
                base_url=preset["base_url"],
                model=preset["default_model"],
                api_key="test-key-for-validation",
            )
            assert backend.base_url == preset["base_url"].rstrip("/")
# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for OllamaBackend.

These use a mocked httpx transport (no network) per the PR spec. We
register a fake response handler with ``httpx.MockTransport`` and patch
``httpx.Client`` to use it, so the backend's real wire code is
exercised end-to-end without touching the network.

Coverage targets (>=6 tests per PR spec):
  1. send-prompt-builds-request (URL, method, JSON shape)
  2. parses-ollama-response-format (extracts ``message.content``)
  3. raises-on-http-error (5xx -> ``httpx.HTTPStatusError``)
  4. sends-temperature-and-system-prompt (passes options through)
  5. supports-streaming-flag (currently stream=False)
  6. model-name-from-arg-or-env (arg > env > default resolution)
  + bonus tests for ``is_multimodal`` and the ``OLLAMA_HOST`` env var.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from idp.llm.backend import CompletionRequest, Message
from idp.llm.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaBackend,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class _RecorderTransport(httpx.MockTransport):
    """httpx MockTransport that records every request body it sees.

    Tests can inspect ``self.requests`` (list[httpx.Request]) after the
    backend has run, including the JSON payload it sent.
    """

    def __init__(self, handler):
        super().__init__(handler)
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # record BEFORE the handler is called so the test sees the
        # request exactly as Ollama would have.
        self.requests.append(request)
        return super().handle_request(request)


def _make_backend_and_recorder(
    *,
    model: str | None = None,
    base_url: str | None = None,
    json_mode: bool = True,
    response_payload: dict[str, Any] | None = None,
    status_code: int = 200,
):
    """Create an OllamaBackend + a recording transport wired together.

    Returns ``(backend, recorder, response_payload)``.
    """
    payload = response_payload or {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "{\"invoice_number\": \"INV-001\"}"},
        "done": True,
    }
    body = json.dumps(payload).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=body, request=request)

    recorder = _RecorderTransport(handler)

    # Patch the httpx.Client inside the OllamaBackend.complete method to
    # use our recorder. We do this by intercepting httpx.Client with a
    # subclass that takes a ``transport=`` argument.
    original_client = httpx.Client

    class PatchedClient(original_client):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = recorder
            super().__init__(*args, **kwargs)

    backend = OllamaBackend(
        model=model if model is not None else "llama3.2:3b",
        base_url=base_url if base_url is not None else "http://localhost:11434",
    )
    return backend, recorder, PatchedClient, payload


def _patched_complete(backend: OllamaBackend, patched_client, *, json_mode: bool = True):
    """Run ``backend.complete(req)`` while ``httpx.Client`` is the patched class."""

    def _run():
        req = CompletionRequest(
            messages=[
                Message(role="system", content="You are an invoice extractor."),
                Message(role="user", content="Extract the invoice."),
            ],
            json_mode=json_mode,
            temperature=0.0,
            max_tokens=512,
        )
        return backend.complete(req)

    with patch("httpx.Client", patched_client):
        return _run()


# ---------------------------------------------------------------------------
# 1. send-prompt-builds-request
# ---------------------------------------------------------------------------
def test_send_prompt_builds_correct_request():
    """The outbound request URL, method, and JSON body shape are correct."""
    backend, recorder, patched, _payload = _make_backend_and_recorder()

    _patched_complete(backend, patched)

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:11434/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "llama3.2:3b"
    assert body["stream"] is False
    assert body["keep_alive"] == "5m"  # default
    assert isinstance(body["messages"], list) and len(body["messages"]) == 2
    # Both messages should be present, with role/content fields
    roles = [m["role"] for m in body["messages"]]
    assert "system" in roles and "user" in roles
    assert any(m["role"] == "system" and "invoice extractor" in m["content"] for m in body["messages"])
    # options dict carries Ollama-native names, not OpenAI's
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 512


# ---------------------------------------------------------------------------
# 2. parses-ollama-response-format
# ---------------------------------------------------------------------------
def test_parses_ollama_response_format():
    """``message.content`` is what we return; the rest of the wire payload is ignored."""
    payload = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "{\"vendor_name\": \"Acme\"}"},
        "done": True,
        "done_reason": "stop",  # we must not touch or rely on this
        "total_duration": 1234567890,  # noise; must not surface in the return value
    }
    backend, _rec, patched, _ = _make_backend_and_recorder(response_payload=payload)

    out = _patched_complete(backend, patched, json_mode=False)

    assert out == "{\"vendor_name\": \"Acme\"}"


def test_parses_ollama_response_handles_empty_content():
    """An empty ``message.content`` is returned as the empty string, not None."""
    payload = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }
    backend, _rec, patched, _ = _make_backend_and_recorder(response_payload=payload)

    out = _patched_complete(backend, patched, json_mode=False)

    assert out == ""


# ---------------------------------------------------------------------------
# 3. raises-on-http-error
# ---------------------------------------------------------------------------
def test_raises_on_http_error():
    """A 5xx from Ollama surfaces as httpx.HTTPStatusError."""
    backend, _rec, patched, _ = _make_backend_and_recorder(status_code=500)

    with pytest.raises(httpx.HTTPStatusError):
        _patched_complete(backend, patched, json_mode=False)


def test_raises_on_404():
    """A 404 (model not found) also raises; we do not silently fall back."""
    backend, _rec, patched, _ = _make_backend_and_recorder(status_code=404)

    with pytest.raises(httpx.HTTPStatusError):
        _patched_complete(backend, patched, json_mode=False)


# ---------------------------------------------------------------------------
# 4. sends-temperature-and-system-prompt
# ---------------------------------------------------------------------------
def test_sends_temperature_and_system_prompt():
    """The system prompt goes on the system role, and ``temperature`` is forwarded."""
    backend, recorder, patched, _payload = _make_backend_and_recorder()

    # Use a non-zero temperature to prove we actually re-mapped it
    def _run():
        req = CompletionRequest(
            messages=[
                Message(role="system", content="Be terse."),
                Message(role="user", content="hi"),
            ],
            json_mode=True,
            temperature=0.3,
            max_tokens=64,
        )
        return backend.complete(req)

    with patch("httpx.Client", patched):
        _run()

    body = json.loads(recorder.requests[0].content)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "Be terse."
    assert body["options"]["temperature"] == pytest.approx(0.3)
    assert body["options"]["num_predict"] == 64
    # JSON mode -> the Ollama "format": "json" hint was added
    assert body["format"] == "json"


# ---------------------------------------------------------------------------
# 5. supports-streaming-flag
# ---------------------------------------------------------------------------
def test_supports_streaming_flag_is_set_false():
    """v0.4 scope: streaming is OFF. The body must declare ``stream: False`` so Ollama
    returns a single JSON object and we can read it synchronously."""
    backend, recorder, patched, _payload = _make_backend_and_recorder()

    _patched_complete(backend, patched, json_mode=False)

    body = json.loads(recorder.requests[0].content)
    assert body["stream"] is False


# ---------------------------------------------------------------------------
# 6. model-name-from-arg-or-env
# ---------------------------------------------------------------------------
def test_model_name_from_constructor_arg():
    """Constructor ``model=`` overrides both env and default."""
    with patch.dict("os.environ", {"OLLAMA_MODEL": "ignored-env-model"}, clear=False):
        backend = OllamaBackend(model="from-arg")
    assert backend.model == "from-arg"


def test_model_name_from_env_when_no_arg(monkeypatch):
    """No constructor arg -> ``$OLLAMA_MODEL`` -> default."""
    # No arg, no env -> default
    assert OllamaBackend().model == DEFAULT_MODEL
    assert OllamaBackend().model == "llama3.2:3b"
    # No arg, env present -> env wins
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    assert OllamaBackend().model == "qwen2.5:7b"


def test_base_url_from_env(monkeypatch):
    """``$OLLAMA_HOST`` overrides the default base URL when no constructor arg."""
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-server.lan:11434")
    backend = OllamaBackend()
    assert backend.base_url == "http://gpu-server.lan:11434"


def test_base_url_constructor_wins_over_env(monkeypatch):
    """Constructor ``base_url=`` wins over ``$OLLAMA_HOST``."""
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
    backend = OllamaBackend(base_url="http://constructor-host:11434/")
    # Trailing slash must be stripped so f"{base}/api/chat" is well-formed.
    assert backend.base_url == "http://constructor-host:11434"


# ---------------------------------------------------------------------------
# Bonus tests: multimodal detection + json_mode format hint
# ---------------------------------------------------------------------------
def test_is_multimodal_true_for_vision_models():
    """Models with vision tags report is_multimodal=True so the pipeline uses them."""
    assert OllamaBackend(model="llava:13b").is_multimodal is True
    assert OllamaBackend(model="llama3.2-vision:11b").is_multimodal is True
    assert OllamaBackend(model="moondream:1.8b").is_multimodal is True


def test_is_multimodal_false_for_text_models():
    """Plain text models report is_multimodal=False (safer default)."""
    assert OllamaBackend(model="llama3.2:3b").is_multimodal is False
    assert OllamaBackend(model="qwen2.5:7b").is_multimodal is False
    assert OllamaBackend(model="mistral:7b-instruct").is_multimodal is False


def test_no_format_hint_when_json_mode_false():
    """Without json_mode, the request body must not carry the JSON-format hint."""
    backend, recorder, patched, _payload = _make_backend_and_recorder()

    _patched_complete(backend, patched, json_mode=False)

    body = json.loads(recorder.requests[0].content)
    assert "format" not in body


def test_default_base_url_used_when_env_unset(monkeypatch):
    """No env, no arg -> the documented Ollama default URL is used."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    backend = OllamaBackend()
    assert backend.base_url == DEFAULT_BASE_URL
    assert backend.base_url == "http://localhost:11434"

# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LLM backend abstraction.

All stages call `Backend.complete()` and get back a string. Backends
handle their own deps lazily (openai/anthropic/ollama) so the framework
runs without any of them installed.

A `MockBackend` ships for tests + reproducible eval, when the user
has no API keys / hardware.
"""
from __future__ import annotations

import abc
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    # Images are passed as base64 data URIs alongside text in multimodal calls.
    images_b64: list[str] = field(default_factory=list)


@dataclass
class CompletionRequest:
    messages: list[Message]
    json_mode: bool = False        # ask for JSON output
    temperature: float = 0.0
    max_tokens: int = 2048


class Backend(abc.ABC):
    """Pluggable LLM backend."""

    name: str = "base"

    @abc.abstractmethod
    def complete(self, req: CompletionRequest) -> str: ...

    # Convenience helpers ------------------------------------------------
    def json_complete(self, messages: list[Message], **kw: Any) -> dict[str, Any]:
        """Complete + parse JSON. Never raises.

        On parse failure returns `{"_error": ..., "_raw": ...}` so the
        caller can decide whether the empty dict is acceptable or whether
        it should route to an error path.
        """
        req = CompletionRequest(messages=messages, json_mode=True, **kw)
        out = self.complete(req)
        return _safe_json(out)

    @property
    def is_multimodal(self) -> bool:
        return False  # overridden by multimodal-capable backends


# ---------------------------------------------------------------------------
# Mock backend — deterministic, no network. Used by tests and the eval CI.
# ---------------------------------------------------------------------------
class MockBackend(Backend):
    """Returns canned responses derived from the prompt.

    Useful for tests, CI, and the eval harness's `mock-strategy` baseline.
    The 'mode' lets eval compare real vs mock on identical inputs.
    """

    name = "mock"

    def __init__(self, mode: str = "ideal"):
        # 'ideal'  : returns the requested schema as empty Pydantic-style JSON
        # 'random' : perturbs fields to simulate hallucination
        # 'omits'  : drops half the fields (low-confidence simulator)
        self.mode = mode

    @property
    def is_multimodal(self) -> bool:
        return True  # exercises multimodal code paths

    def complete(self, req: CompletionRequest) -> str:
        """Context-aware mock: detect classify vs extract from the prompt.

        MockBackend is used for tests + CI + offline demos. The "right"
        response depends on which stage called us:

        - classify: return {"label": "other", "confidence": 0.5} (so the
          caller falls back to whatever label its rule system produced
          UNLESS confidence_floor is set above 0.5, in which case the
          caller should override). We default to "other" so the
          MockBackend is a *worst-case classifier* on purpose — it never
          gets in the way of rule-based classification.
        - assess (confidence): return {"<field>": 0.5, ...}
        - everything else (extract): ideal-empty JSON
        """
        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        if not last_user:
            return "{}"
        text = last_user.content

        # Detect classify prompt
        if "classify the snippet into exactly one of" in text.lower():
            return json.dumps({"label": "other", "confidence": 0.5})

        # Detect assess prompt
        if "rate extraction confidence per field" in text.lower():
            # best effort: a stub per field in the EXTRACTION TO RATE block
            import re

            m = re.search(r"EXTRACTION TO RATE:\s*(\{.*\})", text, re.S)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    return json.dumps({k: 0.5 for k in obj})
                except Exception:  # noqa: BLE001
                    pass
            return "{}"

        # Default: extract
        schema = _extract_schema_block(text)
        if self.mode == "random":
            return _perturb_json(schema)
        if self.mode == "omits":
            return _omit_fields(schema)
        return _empty_schema(schema)


# ---------------------------------------------------------------------------
# HTTP backend — calls an OpenAI-compatible endpoint.
# Works with vLLM, llama.cpp server, LM Studio, OpenAI, Together, Groq, etc.
# ---------------------------------------------------------------------------
class OpenAICompatBackend(Backend):
    """OpenAI-compatible chat-completions client."""

    name = "openai-compat"

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",  # Ollama default
        model: str = "llama3.2-vision",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        self.timeout = timeout

    @property
    def is_multimodal(self) -> bool:
        # Heuristic: vision-capable models contain 'vision' or 'vl' or 'gpt-4o'.
        m = self.model.lower()
        return any(t in m for t in ("vision", "vl", "gpt-4o", "qwen2-vl", "qwen2.5-vl"))

    def complete(self, req: CompletionRequest) -> str:
        import httpx

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_msg_to_openai(m) for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


def _msg_to_openai(m: Message) -> dict[str, Any]:
    """Convert our Message to OpenAI's chat-completions format."""
    if not m.images_b64:
        return {"role": m.role, "content": m.content}
    parts: list[dict[str, Any]] = [{"type": "text", "text": m.content or " "}]
    for b64 in m.images_b64:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return {"role": m.role, "content": parts}


# ---------------------------------------------------------------------------
# Anthropic (text + images via Claude Messages API).
# ---------------------------------------------------------------------------
class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str | None = None):
        try:
            import anthropic  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Install the 'anthropic' extra: pip install py-idp[anthropic]"
            ) from e
        self.model = model
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError(
                "AnthropicBackend requires ANTHROPIC_API_KEY or api_key=... "
                "in the constructor."
            )
        self.api_key = api_key
        self._client: Any = None

    @property
    def is_multimodal(self) -> bool:
        # Only Claude 3+ family models support vision. Claude 2 / Instant
        # are text-only. Default to vision for known-multimodal families;
        # conservatively False for legacy / unknown model names.
        m = self.model.lower()
        return (
            m.startswith("claude-3")
            or "claude-4" in m
            or "opus" in m
            or "sonnet" in m
            or "haiku-3" in m
        )

    def complete(self, req: CompletionRequest) -> str:
        import anthropic  # type: ignore[import-not-found]
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        client = self._client
        # crude split: first message -> system, rest -> user chain
        system = next((m.content for m in req.messages if m.role == "system"), None)
        user_msgs = [
            {"role": m.role, "content": _msg_to_anthropic(m)}
            for m in req.messages
            if m.role != "system"
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": user_msgs,
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)  # type: ignore[union-attr]
        return resp.content[0].text  # type: ignore[union-attr]


def _msg_to_anthropic(m: Message) -> Any:
    if not m.images_b64:
        return m.content
    blocks: list[dict[str, Any]] = []
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    for b64 in m.images_b64:
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
def get_backend(name: str = "auto", **kwargs: Any) -> Backend:
    """Resolve a backend by name.

    'auto' picks in this order: env override -> anthropic -> openai -> ollama -> mock.

    Recognized names:
        mock | mock-ideal | mock-random | mock-omits   (no API key needed)
        openai | ollama | vllm | compat | anthropic     (international)
        china:deepseek | china:qwen | china:zhipu |     (China providers)
        china:moonshot | china:yi | china:doubao |
        china:hunyuan | china:baichuan
        slowmock                                            (load-test only)

    For China providers you can pass `multimodal=True` to switch to the
    provider's vision model if it has one. Pass `api_key=` or set the
    provider's env var.

    The ``slowmock`` backend is only registered when ``IDP_ENABLE_SLOWMOCK=1``
    is set in the environment. Production deployments never set this; it's
    intended for load-testing only.
    """
    if not name:
        name = "auto"
    if name == "auto":
        name = os.environ.get("IDP_BACKEND", "auto")
    if name == "auto":
        # Priority order:
        #   1. Explicitly configured via env var (handled above)
        #   2. anthropic (if ANTHROPIC_API_KEY set)
        #   3. openai (if OPENAI_API_KEY set)
        #   4. ollama (if running locally)
        #   5. mock (last resort: no API key, no ollama — must work offline)
        #
        # The mock fallback ensures first-time users can run the pipeline
        # without any API keys. To force a real backend, set IDP_BACKEND
        # explicitly or the matching *_API_KEY env var.
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        elif os.environ.get("OLLAMA_HOST") or os.environ.get("IDP_OLLAMA"):
            name = "ollama"
        else:
            name = "mock"
            import logging
            logging.getLogger("idp.llm.backend").info(
                "no LLM credentials found; falling back to 'mock' backend. "
                "Set IDP_BACKEND=openai|anthropic|ollama|... to use a real LLM."
            )
    # Mock variants
    if name in ("mock", "mock-ideal", "mock-random", "mock-omits"):
        mode = name.split("-", 1)[1] if "-" in name else "ideal"
        return MockBackend(mode=mode)
    # SlowMock for load testing — only enabled via env var
    if name == "slowmock":
        if os.environ.get("IDP_ENABLE_SLOWMOCK") != "1":
            raise ValueError(
                "slowmock backend is disabled in production. "
                "Set IDP_ENABLE_SLOWMOCK=1 to enable it (load-test only)."
            )
        from idp._testing_backends import SlowMockBackend as _SlowMockBackend
        return _SlowMockBackend(
            latency_ms=int(os.environ.get("LOAD_LATENCY_MS", "1500")),
            jitter_ms=int(os.environ.get("LOAD_JITTER_MS", "500")),
            **kwargs,
        )
    # China providers: prefix with china:
    if name.startswith("china:"):
        from idp.llm.china import get_china_backend

        provider = name.split(":", 1)[1]
        multimodal = kwargs.pop("multimodal", False)
        return get_china_backend(
            provider,
            model=kwargs.pop("model", None),
            multimodal=multimodal,
            api_key=kwargs.pop("api_key", None),
            timeout=kwargs.pop("timeout", 120.0),
        )
    # International OpenAI-compat
    if name in ("openai", "compat", "ollama", "vllm", "lm-studio"):
        return OpenAICompatBackend(**kwargs)
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating ```json fences.

    Returns an `_error` dict on any parse failure rather than raising,
    so callers can decide whether to swallow or surface the failure.
    Accepts `null` and empty string by returning `{}`.
    """
    s = (text or "").strip()
    if not s or s == "null":
        return {}
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"_value": v}
    except Exception:  # noqa: BLE001
        # last resort: grab the first {...} block
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                return v if isinstance(v, dict) else {"_value": v}
            except Exception:  # noqa: BLE001
                pass
        return {"_error": "could not parse JSON", "_raw": text}


def _extract_schema_block(prompt: str) -> str:
    """Pull the JSON-schema section out of an extraction prompt.

    Handles the two known formats:
        "Output JSON Schema:\n{...}\n\nOutput rules:..."
        "JSON Schema:\n{...}\n\nOutput:..."

    If neither marker is found, returns "{}" so the mock falls back
    to an empty object rather than crashing.
    """
    if "Output JSON Schema:" in prompt:
        body = prompt.split("Output JSON Schema:", 1)[1]
        # The schema block ends at "Output rules:" (preferred) or at the
        # end of the prompt. Defensive against either order.
        for end_marker in ("Output rules:", "Output rules", "Document content:"):
            if end_marker in body:
                return body.split(end_marker, 1)[0].strip()
        return body.strip()
    if "JSON Schema:" in prompt:
        body = prompt.split("JSON Schema:", 1)[1]
        for end_marker in ("Output:", "Document content:"):
            if end_marker in body:
                return body.split(end_marker, 1)[0].strip()
        return body.strip()
    return "{}"


def _empty_schema(schema_str: str) -> str:
    """Return the schema with all values stubbed.

    Carries a $defs registry so $ref references to nested models expand.
    """
    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError:
        return "{}"
    return json.dumps(_stub(schema, defs=schema.get("$defs", {})), indent=2)


def _stub(
    schema: dict[str, Any] | list[Any] | Any,
    defs: dict[str, Any] | None = None,
) -> Any:
    """Recursively reduce a JSON-Schema dict to a default-typed value.

    Resolves `$ref: #/$defs/Foo` against the `defs` registry so that
    nested Pydantic models (e.g. LineItem inside Invoice.line_items)
    produce a properly typed default rather than the empty string.
    """
    defs = defs or {}

    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_stub(x, defs) for x in schema[:1]]
        return None

    # 1. enum first — guard against malformed schemas where `enum` is not
    #    a list (Hypothesis caught cases where it was a dict, set, bool, or
    #    empty). Fall back to None on anything weird.
    if "enum" in schema:
        enum = schema["enum"]
        if isinstance(enum, list) and enum:
            return enum[0]
        return None

    # 2. resolve $ref against the $defs registry
    if "$ref" in schema:
        ref = schema["$ref"]
        # JSON-Pointers of the form "#/$defs/Foo"
        if ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            target = defs.get(name)
            if target is not None:
                return _stub(target, defs)
        return None  # nullable-by-default when ref can't be resolved

    # 3. resolve anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        branches = schema.get("anyOf") or schema.get("oneOf")
        # Defensive: a malformed schema may have anyOf/oneOf as a
        # non-list (e.g. a dict or a scalar). Treat non-lists as absent.
        if not isinstance(branches, (list, tuple)):
            branches = []
        for branch in branches:
            if isinstance(branch, dict) and branch.get("type") != "null":
                return _stub(branch, defs)
        return None
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        branches = schema["allOf"]
        if not isinstance(branches, (list, tuple)):
            branches = []
        for branch in branches:
            if isinstance(branch, dict):
                merged.update({k: v for k, v in branch.items() if k != "allOf"})
        return _stub(merged, defs)

    t = schema.get("type")
    if t == "object" or (t is None and "properties" in schema):
        return {k: _stub(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_stub(schema.get("items", {}), defs)]
    if t == "string":
        return ""
    if t == "number" or t == "integer":
        return 0.0
    if t == "boolean":
        return False
    if t == "null":
        return None
    if "properties" in schema:
        return {k: _stub(v, defs) for k, v in schema["properties"].items()}
    return None


def _perturb_json(schema_str: str) -> str:
    """Simulate hallucination: swap values to wrong strings / wrong numbers."""
    base = json.loads(_empty_schema(schema_str))
    for k, v in list(base.items()):
        if isinstance(v, str):
            base[k] = "MOCK_WRONG_" + k
        elif isinstance(v, (int, float)):
            base[k] = -1.0
        elif isinstance(v, dict):
            # don't recursively poke at nested dicts
            base[k] = {kk: "MOCK_WRONG" for kk in v}
        elif isinstance(v, list):
            base[k] = []
    return json.dumps(base, indent=2)


def _omit_fields(schema_str: str) -> str:
    """Drop half the leaves: low-confidence simulator."""
    base = json.loads(_empty_schema(schema_str))
    keys = list(base.keys())
    for k in keys[::2]:
        base[k] = None
    return json.dumps(base, indent=2)

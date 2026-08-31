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
from dataclasses import dataclass, field
from pathlib import Path
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
        """Complete + parse JSON. Raises if model returns malformed JSON."""
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
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Install the 'anthropic' extra: pip install py-idp[anthropic]"
            ) from e
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self._client = None

    @property
    def is_multimodal(self) -> bool:
        return True

    def complete(self, req: CompletionRequest) -> str:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
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
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text


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

    For China providers you can pass `multimodal=True` to switch to the
    provider's vision model if it has one. Pass `api_key=` or set the
    provider's env var.
    """
    if not name:
        name = "auto"
    if name == "auto":
        name = os.environ.get("IDP_BACKEND", "auto")
    if name == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "ollama"
    # Mock variants
    if name in ("mock", "mock-ideal", "mock-random", "mock-omits"):
        mode = name.split("-", 1)[1] if "-" in name else "ideal"
        return MockBackend(mode=mode)
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
import re
import json


def _safe_json(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating ```json fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\n--- raw ---\n{text}") from e


def _extract_schema_block(prompt: str) -> str:
    """Pull the JSON-schema section out of an extraction prompt."""
    if "Output JSON Schema:" in prompt:
        return prompt.split("Output JSON Schema:", 1)[1].split("Output rules:", 1)[0].strip()
    if "JSON Schema:" in prompt:
        return prompt.split("JSON Schema:", 1)[1].split("Output:", 1)[0].strip()
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

    # 1. enum first
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

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
        for branch in schema.get("anyOf") or schema.get("oneOf") or []:
            if isinstance(branch, dict) and branch.get("type") != "null":
                return _stub(branch, defs)
        return None
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for branch in schema["allOf"]:
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

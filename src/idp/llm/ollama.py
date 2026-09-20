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

"""Backend for a locally-running Ollama server.

Talks to ``http://localhost:11434`` over plain HTTP via :mod:`httpx`
(already a core dependency). We do NOT depend on the optional
``ollama`` SDK package — keeps the install footprint and dependency
graph unchanged for users who just want this backend.

Install::

    # 1. Install Ollama itself from https://ollama.com/download
    # 2. Pull a model:   ollama pull llama3.2:3b
    # 3. Start the server (it runs as a service by default)

Usage::

    from idp.llm.ollama import OllamaBackend

    backend = OllamaBackend(model="llama3.2:3b")
    out = backend.complete(req)

    # Or via the factory:
    from idp.llm.backend import get_backend
    backend = get_backend("ollama")  # uses OLLAMA_HOST / default localhost

Typical pair with :class:`idp.llm.tiered.TieredBackend` to get a
2-line cost win — Ollama for the cheap tier, an API backend for the
expensive tier::

    from idp.llm.backend import AnthropicBackend
    from idp.llm.ollama import OllamaBackend
    from idp.llm.tiered import TieredBackend

    Pipeline(backend=TieredBackend(
        cheap=OllamaBackend(model="llama3.2:3b"),
        expensive=AnthropicBackend(),
    ), schema=Invoice).run(doc)

v0.4 scope (per ``docs/ROADMAP_v0.4.md`` Q6): local-only, no streaming.
"""
from __future__ import annotations

import os
from typing import Any

from idp.llm.backend import Backend, CompletionRequest, Message

DEFAULT_BASE_URL = "http://localhost:11434"
"""Default Ollama server URL — Ollama's documented default on install."""

DEFAULT_MODEL = "llama3.2:3b"
"""A sensible default. Override per-call with ``model=...``."""


class OllamaBackend(Backend):
    """Thin :class:`Backend` over the Ollama HTTP ``/api/chat`` endpoint.

    Args:
        model: model tag as known to Ollama (e.g. ``llama3.2:3b``,
            ``qwen2.5:7b``, ``mistral:7b-instruct``). Defaults to
            :data:`DEFAULT_MODEL`. Can also be supplied via the
            ``OLLAMA_MODEL`` environment variable; the constructor
            argument takes precedence.
        base_url: full URL of the Ollama server. Defaults to
            :data:`DEFAULT_BASE_URL`. The ``OLLAMA_HOST`` environment
            variable is honored when the constructor argument is not
            passed.
        timeout: per-request HTTP timeout in seconds. Defaults to 120s
            — small local models are fast but a cold model load can
            take several seconds on first call.
        keep_alive: Ollama-side ``keep_alive`` parameter. ``"5m"``
            keeps the model warm in VRAM between calls; pass a number
            of seconds (``"0"``) or ``"-1"`` to control eviction.
            Defaults to ``"5m"`` because TieredBackend fans the same
            model out across many ``complete()`` calls.

    The constructor does NOT contact the Ollama server — it just
    stores config. The first :meth:`complete` call is the one that
    hits the network.

    Multimodal: models whose tag contains ``vision``, ``vl``, or
    ``llava`` are flagged multimodal so the pipeline takes the
    multimodal extract path.
    """

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        keep_alive: str = "5m",
    ) -> None:
        # Resolve model: arg > env > default.
        if model is None:
            model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        # Resolve base_url: arg > env > default.
        if base_url is None:
            env_host = os.environ.get("OLLAMA_HOST")
            base_url = env_host or DEFAULT_BASE_URL
        # Strip trailing slashes so ``f"{base}/api/chat"`` is well-formed
        # no matter how the user configured the env var.
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.keep_alive = keep_alive

    @property
    def is_multimodal(self) -> bool:
        m = self.model.lower()
        # Ollama tag conventions for vision-capable models. We are
        # deliberately conservative: passing a tag we don't recognize
        # defaults to text-only, which is the safer mode (a vision
        # model loaded as text-only works; a text model mis-used as
        # vision fails noisily).
        return any(t in m for t in ("vision", "vl", "llava", "-v", "moondream"))

    def complete(self, req: CompletionRequest) -> str:
        """POST to ``/api/chat`` and return the message content.

        Maps our :class:`Message` dataclass to Ollama's ``messages``
        format (role/content strings; Ollama accepts base64 image
        bytes in ``images`` but our pipeline currently sends text-only
        — multimodal path is out of scope for v0.4 per roadmap Q6).

        Raises:
            httpx.HTTPStatusError: any non-2xx response from Ollama.
            httpx.RequestError: connectivity / timeout issues.
            KeyError: Ollama returned a payload missing the expected
                ``message.content`` field.
        """
        import httpx

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_msg_to_ollama(m) for m in req.messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(req.temperature),
                # NOTE: ``num_predict`` is Ollama's name for max tokens.
                "num_predict": int(req.max_tokens),
            },
        }
        if req.json_mode:
            # Ollama honors a JSON-formatter option on its own models
            # and most instruct/chat-tuned models accept the prompt
            # cue "format: json". We use the structured "format": "json"
            # hint; Ollama applies it when the model supports JSON.
            body["format"] = "json"

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()

        # Ollama's /api/chat response: {"message": {"role": "assistant",
        # "content": "..."}, "done": true, ...}
        msg = data.get("message") or {}
        content = msg.get("content", "")
        return content if isinstance(content, str) else str(content)


def _msg_to_ollama(m: Message) -> dict[str, Any]:
    """Convert our :class:`Message` to Ollama's chat-message format.

    If the message carries base64 images (multimodal path), attach them
    under ``images`` as Ollama expects. The pipeline currently does
    not populate ``images_b64``; the wire format is here so a future
    multimodal pipeline call Just Works.
    """
    out: dict[str, Any] = {"role": m.role, "content": m.content or ""}
    if m.images_b64:
        out["images"] = list(m.images_b64)
    return out


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "OllamaBackend",
]

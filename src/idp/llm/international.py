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

"""International LLM providers.

All providers exposed here speak the OpenAI Chat-Completions protocol,
so a single `OpenAICompatBackend(base_url=..., model=..., api_key=...)`
handles them. This module just provides:

  1. `INTERNATIONAL_PROVIDER_PRESETS` — canonical (base_url, default_model,
     vision_model, env_var) tuples so users don't have to remember endpoints.
  2. `get_international_backend(name)` — convenience factory.
  3. `list_international_providers()` — for CLI / docs.

The international vendors realistically needed (2025-2026):

| provider   | family / model            | endpoint                                       | env var            |
|------------|---------------------------|------------------------------------------------|--------------------|
| Groq       | llama-3.x, mixtral        | https://api.groq.com/openai/v1                 | GROQ_API_KEY       |
| Mistral    | mistral-large, mistral-small | https://api.mistral.ai/v1                    | MISTRAL_API_KEY    |
| Together   | meta-llama, qwen, mixtral | https://api.together.xyz/v1                    | TOGETHER_API_KEY   |
| Cohere     | command-r-plus            | https://api.cohere.ai/compatibility/v1         | COHERE_API_KEY     |
| Fireworks  | llama, qwen, deepseek     | https://api.fireworks.ai/inference/v1          | FIREWORKS_API_KEY  |

A multimodal-capable default is also picked for the OCR-VLM use case
(llama-3.2 vision family / pixtral). Cohere stays text-only as of 2025.
"""
from __future__ import annotations

import os
from typing import Any

from idp.llm.backend import OpenAICompatBackend

INTERNATIONAL_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-70b-versatile",
        "vision_model": "llama-3.2-90b-vision-preview",
        "env_var": "GROQ_API_KEY",
        "notes": "Groq (ultra-low-latency LPU inference). Llama 3.x + Mixtral.",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "vision_model": "pixtral-12b-2409",
        "env_var": "MISTRAL_API_KEY",
        "notes": "Mistral AI. mistral-large / -small / -codestral. Vision via pixtral.",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "vision_model": "meta-llama/llama-3.2-90b-vision-instruct",
        "env_var": "TOGETHER_API_KEY",
        "notes": "Together AI. Open-weights catalog (meta-llama, qwen, mixtral, deepseek).",
    },
    "cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "default_model": "command-r-plus",
        "vision_model": None,  # text-only
        "env_var": "COHERE_API_KEY",
        "notes": "Cohere command-r / command-r-plus via the OpenAI-compat endpoint. Text-only.",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "vision_model": "accounts/fireworks/models/llama-3.2-90b-vision-instruct",
        "env_var": "FIREWORKS_API_KEY",
        "notes": "Fireworks AI. Fast open-weights inference (llama, qwen, deepseek).",
    },
}


def list_international_providers() -> list[dict[str, Any]]:
    """For CLI / docs: pretty-list of international providers with their env vars."""
    out = []
    for name, p in INTERNATIONAL_PROVIDER_PRESETS.items():
        out.append(
            {
                "name": name,
                "env_var": p["env_var"],
                "default_model": p["default_model"],
                "vision_model": p["vision_model"],
                "notes": p["notes"],
            }
        )
    return out


def get_international_backend(
    provider: str,
    model: str | None = None,
    multimodal: bool = False,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> OpenAICompatBackend:
    """Build an OpenAICompatBackend pinned to an international provider.

    Args:
        provider:    key in INTERNATIONAL_PROVIDER_PRESETS ('groq', 'mistral', ...)
        model:       model id override; defaults to the preset's default
        multimodal:  if True and the provider has a vision_model, switch to it
        api_key:     pass explicitly; else pulled from the preset's env_var
        timeout:     HTTP timeout
    """
    preset = INTERNATIONAL_PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(
            f"Unknown international provider '{provider}'. "
            f"Known: {list(INTERNATIONAL_PROVIDER_PRESETS)}"
        )
    chosen_model = (
        model
        or (preset["vision_model"] if multimodal else None)
        or preset["default_model"]
    )
    key = api_key or os.environ.get(preset["env_var"])
    if not key:
        raise OSError(
            f"No API key for provider '{provider}'. "
            f"Set env var {preset['env_var']} or pass api_key=..."
        )
    return OpenAICompatBackend(
        base_url=preset["base_url"],
        model=chosen_model,
        api_key=key,
        timeout=timeout,
    )

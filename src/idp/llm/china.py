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

"""China LLM providers.

All providers exposed here speak the OpenAI Chat-Completions protocol,
so a single `OpenAICompatBackend(base_url=..., model=..., api_key=...)`
handles them. This module just provides:

  1. `CHINA_PROVIDER_PRESETS` — canonical (base_url, default_model, env_var)
     tuples so users don't have to remember endpoints.
  2. `get_china_backend(name)` — convenience factory.
  3. `list_china_providers()` — for CLI / docs.

The frameworks markets realistically need (2025-2026):

| provider      | family          | endpoint                                 | env var          |
|---------------|-----------------|------------------------------------------|------------------|
| DeepSeek      | deepseek-chat   | https://api.deepseek.com/v1              | DEEPSEEK_API_KEY |
| Qwen/DashScope| qwen-*          | https://dashscope.aliyuncs.com/compatible-mode/v1 | DASHSCOPE_API_KEY |
| Zhipu GLM     | glm-4           | https://open.bigmodel.cn/api/paas/v4     | ZHIPUAI_API_KEY  |
| Moonshot Kimi | moonshot-v1-*   | https://api.moonshot.cn/v1               | MOONSHOT_API_KEY |
| Yi (01.AI)    | yi-*            | https://api.lingyiwanwu.com/v1           | YI_API_KEY       |
| Doubao        | doubao-*        | https://ark.cn-beijing.volces.com/api/v3 | ARK_API_KEY      |
| Hunyuan       | hunyuan-*       | https://api.hunyuan.cloud.tencent.com/v1 | HUNYUAN_API_KEY  |
| Baichuan      | baichuan-*      | https://api.baichuan-ai.com/v1           | BAICHUAN_API_KEY |

A multimodal-capable default is also picked for the OCR-VLM use case
(Qwen-VL / Qwen2.5-VL via DashScope).
"""
from __future__ import annotations

import os
from typing import Any

from idp.llm.backend import OpenAICompatBackend

CHINA_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "vision_model": None,  # text-only
        "env_var": "DEEPSEEK_API_KEY",
        "notes": "DeepSeek-V3 / R1. Strong reasoning. Text-only.",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "vision_model": "qwen2.5-vl-72b-instruct",
        "env_var": "DASHSCOPE_API_KEY",
        "notes": "Alibaba Qwen (DashScope). Multimodal via qwen-vl / qwen2.5-vl.",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-plus",
        "vision_model": "glm-4v-plus",
        "env_var": "ZHIPUAI_API_KEY",
        "notes": "Zhipu GLM-4. Vision via glm-4v-plus.",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-128k",
        "vision_model": "moonshot-v1-128k-vision-preview",
        "env_var": "MOONSHOT_API_KEY",
        "notes": "Moonshot Kimi. Long context (128k+).",
    },
    "yi": {
        "base_url": "https://api.lingyiwanwu.com/v1",
        "default_model": "yi-large",
        "vision_model": "yi-vision",
        "env_var": "YI_API_KEY",
        "notes": "01.AI Yi. Vision via yi-vision.",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro-32k",
        "vision_model": "doubao-1-5-vision-pro-32k",
        "env_var": "ARK_API_KEY",
        "notes": "ByteDance Doubao. Vision via 1.5-vision-pro.",
    },
    "hunyuan": {
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "default_model": "hunyuan-pro",
        "vision_model": "hunyuan-vision",
        "env_var": "HUNYUAN_API_KEY",
        "notes": "Tencent Hunyuan.",
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "default_model": "baichuan4",
        "vision_model": None,
        "env_var": "BAICHUAN_API_KEY",
        "notes": "Baichuan4. Text-only.",
    },
}


def list_china_providers() -> list[dict[str, Any]]:
    """For CLI / docs: pretty-list of China providers with their env vars."""
    out = []
    for name, p in CHINA_PROVIDER_PRESETS.items():
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


def get_china_backend(
    provider: str,
    model: str | None = None,
    multimodal: bool = False,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> OpenAICompatBackend:
    """Build an OpenAICompatBackend pinned to a China provider.

    Args:
        provider:    key in CHINA_PROVIDER_PRESETS ('qwen', 'deepseek', ...)
        model:       model id override; defaults to the preset's default
        multimodal:  if True and the provider has a vision_model, switch to it
        api_key:     pass explicitly; else pulled from the preset's env_var
        timeout:     HTTP timeout
    """
    preset = CHINA_PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(
            f"Unknown China provider '{provider}'. "
            f"Known: {list(CHINA_PROVIDER_PRESETS)}"
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

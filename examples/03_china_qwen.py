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

"""Example: run the pipeline against a real China LLM (Qwen via DashScope).

This is one of the eight China-LLM providers supported by py-idp out of
the box: deepseek, qwen, zhipu, moonshot, yi, doubao, hunyuan, baichuan.
All speak the OpenAI Chat-Completions protocol, so the API key path is
the same; only the environment variable name and the model differ.

What this shows:
  - Calling get_china_backend("qwen", multimodal=True) gives you
    qwen2.5-vl-72b-instruct (vision-capable).
  - The same `Pipeline(...)` interface works for every backend.
  - If DASHSCOPE_API_KEY is unset, the script falls back to MockBackend
    so the pipeline structure stays visible offline.

Run with a real key:
    export DASHSCOPE_API_KEY=...
    python examples/03_china_qwen.py

Run offline (mock fallback):
    python examples/03_china_qwen.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from idp._util import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm import get_backend
from idp.pipeline import Pipeline

SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"

# Quiet the validation-error spam that MockBackend's empty defaults produce.
logging.getLogger("idp").setLevel(logging.CRITICAL)


def main() -> None:
    try:
        backend = get_backend("china:qwen", multimodal=True)  # qwen2.5-vl-72b
        print("Using DashScope Qwen-VL")
    except Exception as e:
        print(f"  ! Falling back to mock backend: {e}")
        from idp.llm import get_backend as _gb

        backend = _gb("mock")

    pipeline = Pipeline(backend=backend, schema=Invoice)
    res = pipeline.run(Document.from_path(SAMPLE))
    pretty_print_result(res)
    print(
        "\nNote: MockBackend returns empty defaults — Qwen-VL with a valid\n"
        "  DASHSCOPE_API_KEY returns a real, valid extraction and validates\n"
        "  cleanly. Run with a key to see the real path."
    )


if __name__ == "__main__":
    main()


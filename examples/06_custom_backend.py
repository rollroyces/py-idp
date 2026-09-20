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

"""Example: registering a custom LLM backend.

The 12+ backends shipped with py-idp cover OpenAI, Anthropic, Ollama,
vLLM, and 8 China providers. When you need something else — an internal
model gateway, a fixed-response test fixture, a non-standard protocol —
you can register your own Backend subclass with a single decorator.

What this example shows:
  1. Subclassing ``Backend`` and implementing ``complete()``.
  2. Decorating with ``@register_backend("echo")`` so the rest of py-idp
     can resolve it by name (``get_backend("echo")``).
  3. Plugging it into a Pipeline and extracting fields from a real
     document.

The example uses the shipped sample invoice (``inv-001.txt``) so it
runs offline with no API keys.

Run:
    python examples/06_custom_backend.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.backend import Backend, CompletionRequest, get_backend, register_backend
from idp.pipeline.pipeline import Pipeline

logging.getLogger("idp").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# 1. Define your custom backend.
# ---------------------------------------------------------------------------
@register_backend("echo")
class EchoBackend(Backend):
    """Demo backend that echoes the last user message back as JSON.

    Useful for:
      - Smoke-testing pipelines without an API key or model server.
      - Stubbing deterministic test fixtures.
      - Demonstrating the register_backend() extension point.

    Real backends should implement `complete()` to call an LLM and
    return the model output as a JSON string. The rest of py-idp parses
    it the same way regardless of which Backend produced it.
    """

    name = "echo"

    def complete(self, req: CompletionRequest) -> str:
        # Find the last user message and echo its content as JSON.
        last_user = next(
            (m for m in reversed(req.messages) if m.role == "user"), None
        )
        content = last_user.content if last_user else ""
        # Wrap in a JSON object so downstream parsers don't choke.
        import json
        return json.dumps({"echo": content})


# ---------------------------------------------------------------------------
# 2. Resolve it through get_backend (same path any other backend uses).
# ---------------------------------------------------------------------------
backend = get_backend("echo")
print(f"Resolved backend: {backend.name!r} ({type(backend).__name__})")

# ---------------------------------------------------------------------------
# 3. Run a real pipeline end-to-end against the bundled sample invoice.
# ---------------------------------------------------------------------------
SAMPLE = (
    Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"
)
doc = Document.from_path(SAMPLE)

pipeline = Pipeline(backend=backend, schema=Invoice)
result = pipeline.run(doc)

extraction = result.document.extraction or {}
print("\nExtraction result:")
print(f"  fields: {list(extraction.keys())}")
print(f"  validation passed: {result.validation_passed}")
print(f"  errors: {len(result.document.errors)}")

# ---------------------------------------------------------------------------
# 4. (Optional) Clean up: deregister the backend if it was a one-off.
# ---------------------------------------------------------------------------
# from idp.llm.backend import _REGISTRY
# _REGISTRY.pop("echo", None)
#
# We don't actually pop it because the rest of this script (and any
# subsequent code in the same process) might still want to resolve
# "echo". In production, register your backends at import-time and let
# them live for the process lifetime.

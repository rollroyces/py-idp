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

"""One-call extraction — the shortest path from a file path to a dict of fields.

This module exists so a brand-new user can write::

    from idp.easy import extract_one
    fields = extract_one("invoice.pdf")

without learning the full Pipeline / backend / schema vocabulary. It is
deliberately thin: one public function, no new dependencies.

Behavior
--------
1. ``backend`` defaults to the ``IDP_BACKEND`` env var, then ``"mock"``.
2. ``schema`` is resolved by name against the built-in ``SCHEMA_REGISTRY``
   (Invoice, Contract, BankStatement, Receipt). A string is the common
   case; a Pydantic ``BaseModel`` subclass is also accepted for power
   users.
3. The whole six-stage pipeline runs (parse -> classify -> extract ->
   assess -> validate). We return ``result.document.extraction`` — the
   validated dict that matches the chosen schema — so the caller gets
   *just the fields* they asked for.

What this is not
----------------
- Not a replacement for :class:`idp.pipeline.Pipeline`. If you need
  custom business rules, retry/cache wrapping, or per-stage timing, use
  ``Pipeline`` directly.
- Not a batch runner. ``extract_one`` is intentionally singular. Loop in
  Python for batches.

Errors
------
Raises ``FileNotFoundError`` if the path doesn't exist, ``KeyError``
if the schema name is unknown, or any backend-specific exception (e.g.
authentication failures) the underlying LLM backend raises.
"""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from idp.core.document import Document
from idp.core.schemas import get_schema
from idp.pipeline.pipeline import Pipeline

__all__ = ["extract_one"]


def extract_one(
    path: str,
    backend: str | None = None,
    schema: str | type[BaseModel] = "Invoice",
) -> dict[str, Any]:
    """Run the full pipeline on one document and return its extracted fields.

    Args:
        path:    Filesystem path to a ``.pdf`` or ``.txt`` document. The
                 extension drives parser selection (pdfplumber / Docling
                 for PDFs, plain-text fallback for ``.txt``).
        backend: LLM backend name (e.g. ``"mock"``, ``"ollama"``,
                 ``"openai"``, ``"anthropic"``, ``"china:qwen"``). When
                 ``None``, falls back to ``$IDP_BACKEND`` and then to
                 ``"mock"`` (which requires no API key).
        schema:  Either a schema name registered in
                 :data:`idp.core.schemas.SCHEMA_REGISTRY` (``"Invoice"``,
                 ``"Contract"``, ``"BankStatement"``, ``"Receipt"``), or
                 a Pydantic ``BaseModel`` subclass for custom schemas.
                 Default ``"Invoice"``.

    Returns:
        The validated extraction dict — keys match the chosen schema's
        field names. Values are JSON-friendly (str, float, list of
        dicts, etc.).

    Raises:
        FileNotFoundError:   ``path`` doesn't exist.
        KeyError:            ``schema`` is a string not in the registry
                             (with a helpful list of valid names).
        IsADirectoryError:   ``path`` is a directory (no silent empty doc).
    """
    # 1. Resolve backend name. We prefer the explicit kwarg, then the
    #    IDP_BACKEND env var, then mock — same priority as
    #    ``get_backend("auto")`` but without that function's credential
    #    sniffing, which can mask user intent (e.g. when a stale
    #    OPENAI_API_KEY sits in the shell but the user asked for mock).
    backend = backend or os.environ.get("IDP_BACKEND") or "mock"

    # 2. Resolve schema. ``get_schema`` only accepts strings; a Pydantic
    #    class passes through unchanged.
    schema_cls = get_schema(schema) if isinstance(schema, str) else schema

    # 3. Build the pipeline. ``Pipeline(backend=..., schema=...)``
    #    accepts both strings and objects — no need to call
    #    ``get_backend()`` ourselves.
    pipeline = Pipeline(backend=backend, schema=schema_cls)

    # 4. Run and return the validated extraction dict.
    result = pipeline.run(Document.from_path(path))
    return result.document.extraction or {}
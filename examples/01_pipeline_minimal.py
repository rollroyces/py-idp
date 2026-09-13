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

"""Example: the smallest possible end-to-end pipeline run.

This is what you copy first when trying py-idp for the first time.
Uses the in-tree MockBackend so no API key is required.

What this shows:
  - Loading a Document from a file path.
  - Building a Pipeline with backend + schema.
  - Reading result.extraction, result.confidence, result.validation.

Run:
    python examples/01_pipeline_minimal.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from idp._util import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline

# Locally-shipped sample so this works offline / in CI
SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"

# Quiet the validation-error spam that MockBackend's empty defaults produce.
# Real backends (openai, anthropic, etc.) produce valid schemas, so this
# only affects offline / CI runs.
logging.getLogger("idp").setLevel(logging.CRITICAL)


def main() -> None:
    """Run the full pipeline on the sample invoice using the mock backend."""
    backend = get_backend("mock")
    pipeline = Pipeline(backend=backend, schema=Invoice)
    doc = Document.from_path(SAMPLE)
    res = pipeline.run(doc)
    pretty_print_result(res)
    print(
        "\nNote: MockBackend returns empty defaults — real backends (openai,\n"
        "  anthropic, china:*) return valid schemas and validate cleanly.\n"
        "  Run with an API key to see the real path."
    )


if __name__ == "__main__":
    main()

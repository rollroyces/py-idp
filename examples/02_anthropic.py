# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products or hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Example: extract an Invoice with Anthropic Claude 3.5 Sonnet.

What this shows:
  - Setting up the Anthropic backend via get_backend('anthropic').
  - The same Pipeline(...) interface as every other backend.
  - Graceful fallback to MockBackend if ANTHROPIC_API_KEY is unset
    (so this script always prints *something* useful).

Run with a real key:
    pip install py-idp[anthropic]
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/02_anthropic.py

Run offline (mock fallback):
    python examples/02_anthropic.py
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
# Real backends (openai, anthropic, etc.) produce valid schemas, so this
# only affects offline / CI runs.
logging.getLogger("idp").setLevel(logging.CRITICAL)


def main() -> None:
    try:
        backend = get_backend("anthropic")
        print("Using Anthropic Claude")
    except Exception as e:
        print(f"  ! Falling back to MockBackend: {e}")
        backend = get_backend("mock")

    pipeline = Pipeline(backend=backend, schema=Invoice)
    result = pipeline.run(Document.from_path(SAMPLE))
    pretty_print_result(result)
    print(
        "\nNote: MockBackend returns empty defaults — Anthropic Claude with\n"
        "  a valid ANTHROPIC_API_KEY returns a real, valid extraction and\n"
        "  validates cleanly. Run with a key to see the real path."
    )


if __name__ == "__main__":
    main()
